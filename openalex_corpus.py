from __future__ import annotations

import argparse
import contextlib
import hashlib
import html
import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


API_BASE_URL = "https://api.openalex.org"
DEFAULT_OUTPUT_DIR = Path("data/openalex_stats_probability")
DEFAULT_DB_PATH = DEFAULT_OUTPUT_DIR / "corpus.sqlite"
DEFAULT_PDF_DIR = DEFAULT_OUTPUT_DIR / "pdfs"
DEFAULT_PER_PAGE = 100
DEFAULT_SORT = "publication_date:desc"
DEFAULT_SUBFIELD_ID = "2613"
DEFAULT_USER_AGENT = "tfm-openalex-corpus/1.0"

SELECT_FIELDS = [
    "id",
    "doi",
    "title",
    "display_name",
    "publication_year",
    "publication_date",
    "ids",
    "language",
    "primary_location",
    "best_oa_location",
    "open_access",
    "authorships",
    "type",
    "type_crossref",
    "indexed_in",
    "has_fulltext",
    "fulltext_origin",
    "cited_by_count",
    "is_retracted",
    "is_paratext",
    "primary_topic",
    "topics",
    "locations_count",
    "has_content",
    "content_urls",
    "referenced_works_count",
    "abstract_inverted_index",
    "updated_date",
    "created_date",
]

CORE_FILTERS = [
    "has_abstract:true",
    "type:article",
    "language:en",
    "has_references:true",
    "is_retracted:false",
    "is_paratext:false",
]

DOWNLOADABLE_FILTERS = [
    "is_oa:true",
    "has_pdf_url:true",
    "has_fulltext:true",
    "has_content.pdf:true",
]

TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class CorpusConfig:
    api_key: str
    output_dir: Path
    db_path: Path
    pdf_dir: Path
    subfield_id: str
    subfield_scope: str
    profile: str
    per_page: int
    sort: str
    max_works: int | None
    download_pdfs: bool
    max_pdfs: int
    reset_sync_state: bool
    user_agent: str


def parse_args() -> CorpusConfig:
    parser = argparse.ArgumentParser(
        description="Build a local OpenAlex Statistics and Probability corpus in SQLite."
    )
    parser.add_argument("--api-key", help="OpenAlex API key. Defaults to OPENALEX_API_KEY.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory that will contain the database and downloaded PDFs.",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Optional explicit SQLite path. Defaults to <output-dir>/corpus.sqlite.",
    )
    parser.add_argument(
        "--pdf-dir",
        default=None,
        help="Optional explicit PDF directory. Defaults to <output-dir>/pdfs.",
    )
    parser.add_argument(
        "--subfield-id",
        default=DEFAULT_SUBFIELD_ID,
        help="OpenAlex subfield id. Statistics and Probability is 2613.",
    )
    parser.add_argument(
        "--subfield-scope",
        choices=("primary", "any"),
        default="primary",
        help="Use primary_topic.subfield.id or topics.subfield.id.",
    )
    parser.add_argument(
        "--profile",
        choices=("downloadable", "core"),
        default="downloadable",
        help="downloadable adds OA/fulltext/PDF filters; core uses only the base corpus filters.",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=DEFAULT_PER_PAGE,
        help="OpenAlex page size. Current API limit is 100.",
    )
    parser.add_argument(
        "--sort",
        default=DEFAULT_SORT,
        help="OpenAlex sort expression, for example publication_date:desc.",
    )
    parser.add_argument(
        "--max-works",
        type=int,
        default=None,
        help="Optional metadata cap for testing. Leave unset to sync the full result set.",
    )
    parser.add_argument(
        "--download-pdfs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download PDFs via content.openalex.org after metadata sync.",
    )
    parser.add_argument(
        "--max-pdfs",
        type=int,
        default=None,
        help="How many pending PDFs to download this run. Defaults to 10 when downloads are enabled.",
    )
    parser.add_argument(
        "--reset-sync-state",
        action="store_true",
        help="Forget the saved cursor for this filter profile and sync from scratch.",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="HTTP User-Agent header for API calls.",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENALEX_API_KEY")
    if not api_key:
        parser.error("Provide --api-key or set OPENALEX_API_KEY in the environment.")

    output_dir = Path(args.output_dir).resolve()
    db_path = Path(args.db_path).resolve() if args.db_path else output_dir / DEFAULT_DB_PATH.name
    pdf_dir = Path(args.pdf_dir).resolve() if args.pdf_dir else output_dir / DEFAULT_PDF_DIR.name
    max_pdfs = args.max_pdfs if args.max_pdfs is not None else (10 if args.download_pdfs else 0)

    return CorpusConfig(
        api_key=api_key,
        output_dir=output_dir,
        db_path=db_path,
        pdf_dir=pdf_dir,
        subfield_id=str(args.subfield_id),
        subfield_scope=args.subfield_scope,
        profile=args.profile,
        per_page=min(max(args.per_page, 1), 100),
        sort=args.sort,
        max_works=args.max_works,
        download_pdfs=args.download_pdfs,
        max_pdfs=max_pdfs,
        reset_sync_state=args.reset_sync_state,
        user_agent=args.user_agent,
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def short_openalex_id(value: str | None) -> str | None:
    if not value:
        return None
    return value.rstrip("/").rsplit("/", 1)[-1]


def strip_markup(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = TAG_RE.sub("", html.unescape(value)).strip()
    return cleaned or None


def json_text(value: Any) -> str | None:
    if value in (None, [], {}):
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def bool_int(value: Any) -> int | None:
    if value is None:
        return None
    return 1 if bool(value) else 0


def reconstruct_abstract(abstract_inverted_index: dict[str, list[int]] | None) -> str | None:
    if not abstract_inverted_index:
        return None
    positions: dict[int, str] = {}
    for token, indexes in abstract_inverted_index.items():
        for index in indexes:
            positions[index] = token
    if not positions:
        return None
    words = [positions[i] for i in sorted(positions)]
    return html.unescape(" ".join(words)).strip() or None


class OpenAlexClient:
    def __init__(self, api_key: str, user_agent: str) -> None:
        self.api_key = api_key
        self.user_agent = user_agent

    def _request(self, url: str, *, retries: int = 5, timeout: int = 60) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    content_type = response.headers.get("Content-Type", "")
                    payload = response.read()
                if "application/json" in content_type or payload.startswith(b"{"):
                    return json.loads(payload)
                return payload
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                retriable = exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
                if not retriable or attempt == retries:
                    raise RuntimeError(f"HTTP {exc.code} for {url}\n{body}") from exc
                last_error = exc
                time.sleep(min(2**attempt, 30))
            except urllib.error.URLError as exc:
                last_error = exc
                if attempt == retries:
                    break
                time.sleep(min(2**attempt, 30))
        raise RuntimeError(f"Request failed for {url}: {last_error}") from last_error

    def get_json(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        query = dict(params)
        query["api_key"] = self.api_key
        encoded = urllib.parse.urlencode(query)
        url = f"{API_BASE_URL}{path}?{encoded}"
        return self._request(url)

    def get_subfield(self, subfield_id: str) -> dict[str, Any]:
        path = f"/subfields/{subfield_id}"
        return self.get_json(path, {})

    def download_pdf(self, content_url: str, destination: Path) -> dict[str, Any]:
        separator = "&" if "?" in content_url else "?"
        signed_url = f"{content_url}{separator}api_key={urllib.parse.quote(self.api_key)}"
        request = urllib.request.Request(
            signed_url,
            headers={"User-Agent": self.user_agent},
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        last_error: Exception | None = None
        for attempt in range(1, 6):
            sha = hashlib.sha256()
            destination_tmp = destination.with_suffix(".tmp")
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    content_type = response.headers.get("Content-Type", "")
                    total_bytes = 0
                    with destination_tmp.open("wb") as handle:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            handle.write(chunk)
                            sha.update(chunk)
                            total_bytes += len(chunk)
                    destination_tmp.replace(destination)
                return {
                    "file_path": str(destination),
                    "file_size_bytes": total_bytes,
                    "content_type": content_type,
                    "sha256": sha.hexdigest(),
                }
            except urllib.error.HTTPError as exc:
                if destination_tmp.exists():
                    destination_tmp.unlink(missing_ok=True)
                body = exc.read().decode("utf-8", errors="replace")
                retriable = exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
                if not retriable or attempt == 5:
                    raise RuntimeError(f"HTTP {exc.code} while downloading {content_url}\n{body}") from exc
                last_error = exc
                time.sleep(min(2**attempt, 30))
            except urllib.error.URLError as exc:
                if destination_tmp.exists():
                    destination_tmp.unlink(missing_ok=True)
                last_error = exc
                if attempt == 5:
                    break
                time.sleep(min(2**attempt, 30))
        raise RuntimeError(f"Download failed for {content_url}: {last_error}") from last_error


class SQLiteStore:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self._create_schema()

    def close(self) -> None:
        self.conn.close()

    @contextlib.contextmanager
    def transaction(self):
        try:
            self.conn.execute("BEGIN")
            yield
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS works (
                work_id TEXT PRIMARY KEY,
                openalex_id TEXT NOT NULL UNIQUE,
                doi TEXT,
                title TEXT,
                display_name TEXT,
                publication_year INTEGER,
                publication_date TEXT,
                language TEXT,
                type TEXT,
                type_crossref TEXT,
                cited_by_count INTEGER,
                referenced_works_count INTEGER,
                locations_count INTEGER,
                has_abstract INTEGER,
                has_references INTEGER,
                has_pdf_url INTEGER,
                has_fulltext INTEGER,
                has_content_pdf INTEGER,
                has_content_grobid_xml INTEGER,
                is_oa INTEGER,
                oa_status TEXT,
                oa_url TEXT,
                any_repository_has_fulltext INTEGER,
                is_retracted INTEGER,
                is_paratext INTEGER,
                fulltext_origin TEXT,
                primary_topic_id TEXT,
                primary_topic_name TEXT,
                primary_subfield_id TEXT,
                primary_subfield_name TEXT,
                primary_field_id TEXT,
                primary_field_name TEXT,
                primary_domain_id TEXT,
                primary_domain_name TEXT,
                primary_source_id TEXT,
                primary_source_name TEXT,
                primary_source_type TEXT,
                primary_source_issn_l TEXT,
                primary_source_is_oa INTEGER,
                primary_source_is_in_doaj INTEGER,
                primary_source_is_core INTEGER,
                primary_source_host_org_id TEXT,
                primary_source_host_org_name TEXT,
                license TEXT,
                license_id TEXT,
                version TEXT,
                landing_page_url TEXT,
                pdf_url TEXT,
                content_pdf_url TEXT,
                content_grobid_xml_url TEXT,
                abstract_text TEXT,
                indexed_in_json TEXT,
                ids_json TEXT,
                best_oa_location_json TEXT,
                primary_location_json TEXT,
                has_content_json TEXT,
                content_urls_json TEXT,
                created_date TEXT,
                updated_date TEXT,
                inserted_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS authors (
                author_id TEXT PRIMARY KEY,
                openalex_id TEXT NOT NULL UNIQUE,
                display_name TEXT,
                orcid TEXT,
                last_seen_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS work_authorships (
                work_id TEXT NOT NULL REFERENCES works(work_id) ON DELETE CASCADE,
                author_index INTEGER NOT NULL,
                author_id TEXT REFERENCES authors(author_id),
                author_position TEXT,
                is_corresponding INTEGER,
                raw_author_name TEXT,
                countries_json TEXT,
                institutions_json TEXT,
                affiliations_json TEXT,
                raw_affiliation_strings_json TEXT,
                PRIMARY KEY (work_id, author_index)
            );

            CREATE TABLE IF NOT EXISTS topics (
                topic_id TEXT PRIMARY KEY,
                openalex_id TEXT NOT NULL UNIQUE,
                display_name TEXT,
                subfield_id TEXT,
                subfield_name TEXT,
                field_id TEXT,
                field_name TEXT,
                domain_id TEXT,
                domain_name TEXT,
                last_seen_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS work_topics (
                work_id TEXT NOT NULL REFERENCES works(work_id) ON DELETE CASCADE,
                topic_id TEXT NOT NULL REFERENCES topics(topic_id),
                rank_index INTEGER NOT NULL,
                score REAL,
                is_primary INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (work_id, topic_id)
            );

            CREATE TABLE IF NOT EXISTS pdf_downloads (
                work_id TEXT PRIMARY KEY REFERENCES works(work_id) ON DELETE CASCADE,
                status TEXT NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                file_path TEXT,
                file_size_bytes INTEGER,
                content_type TEXT,
                sha256 TEXT,
                http_status INTEGER,
                downloaded_at TEXT,
                last_attempt_at TEXT NOT NULL,
                last_error TEXT
            );

            CREATE TABLE IF NOT EXISTS sync_jobs (
                job_key TEXT PRIMARY KEY,
                filter_scope TEXT NOT NULL,
                filter_profile TEXT NOT NULL,
                filter_string TEXT NOT NULL,
                subfield_id TEXT NOT NULL,
                subfield_name TEXT,
                sort TEXT NOT NULL,
                per_page INTEGER NOT NULL,
                next_cursor TEXT,
                total_expected INTEGER,
                works_synced INTEGER NOT NULL DEFAULT 0,
                completed INTEGER NOT NULL DEFAULT 0,
                started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_works_publication_year ON works(publication_year);
            CREATE INDEX IF NOT EXISTS idx_works_publication_date ON works(publication_date);
            CREATE INDEX IF NOT EXISTS idx_works_primary_subfield ON works(primary_subfield_id);
            CREATE INDEX IF NOT EXISTS idx_works_cited_by_count ON works(cited_by_count);
            CREATE INDEX IF NOT EXISTS idx_works_has_content_pdf ON works(has_content_pdf);
            CREATE INDEX IF NOT EXISTS idx_works_primary_source ON works(primary_source_name);
            CREATE INDEX IF NOT EXISTS idx_authors_display_name ON authors(display_name);
            CREATE INDEX IF NOT EXISTS idx_work_authorships_author ON work_authorships(author_id);
            CREATE INDEX IF NOT EXISTS idx_work_topics_topic ON work_topics(topic_id);
            CREATE INDEX IF NOT EXISTS idx_pdf_downloads_status ON pdf_downloads(status);

            CREATE VIEW IF NOT EXISTS download_queue AS
            SELECT
                w.work_id,
                w.title,
                w.publication_date,
                w.cited_by_count,
                w.primary_source_name,
                w.license,
                w.pdf_url,
                w.content_pdf_url,
                COALESCE(d.status, 'pending') AS download_status,
                d.file_path,
                d.file_size_bytes,
                d.downloaded_at,
                d.last_error
            FROM works w
            LEFT JOIN pdf_downloads d ON d.work_id = w.work_id
            WHERE w.has_content_pdf = 1;
            """
        )
        self.conn.commit()

    def _upsert(self, table: str, row: dict[str, Any], conflict_columns: list[str]) -> None:
        columns = list(row.keys())
        placeholders = ", ".join(["?"] * len(columns))
        assignments = ", ".join(
            f"{column}=excluded.{column}" for column in columns if column not in conflict_columns
        )
        sql = (
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders}) "
            f"ON CONFLICT({', '.join(conflict_columns)}) DO UPDATE SET {assignments}"
        )
        self.conn.execute(sql, [row[column] for column in columns])

    def get_sync_job(self, job_key: str) -> sqlite3.Row | None:
        row = self.conn.execute(
            "SELECT * FROM sync_jobs WHERE job_key = ?",
            (job_key,),
        ).fetchone()
        return row

    def upsert_sync_job(
        self,
        *,
        job_key: str,
        filter_scope: str,
        filter_profile: str,
        filter_string: str,
        subfield_id: str,
        subfield_name: str,
        sort: str,
        per_page: int,
        next_cursor: str | None,
        total_expected: int | None,
        works_synced: int,
        completed: bool,
    ) -> None:
        existing = self.get_sync_job(job_key)
        now = utc_now()
        row = {
            "job_key": job_key,
            "filter_scope": filter_scope,
            "filter_profile": filter_profile,
            "filter_string": filter_string,
            "subfield_id": subfield_id,
            "subfield_name": subfield_name,
            "sort": sort,
            "per_page": per_page,
            "next_cursor": next_cursor,
            "total_expected": total_expected,
            "works_synced": works_synced,
            "completed": 1 if completed else 0,
            "started_at": existing["started_at"] if existing else now,
            "updated_at": now,
            "completed_at": now if completed else (existing["completed_at"] if existing else None),
        }
        self._upsert("sync_jobs", row, ["job_key"])
        self.conn.commit()

    def reset_sync_job(self, job_key: str) -> None:
        self.conn.execute("DELETE FROM sync_jobs WHERE job_key = ?", (job_key,))
        self.conn.commit()

    def upsert_work_bundle(self, work: dict[str, Any]) -> None:
        now = utc_now()
        work_id = short_openalex_id(work.get("id"))
        if not work_id:
            raise ValueError("Work without an OpenAlex id.")

        primary_topic = work.get("primary_topic") or {}
        primary_subfield = primary_topic.get("subfield") or {}
        primary_field = primary_topic.get("field") or {}
        primary_domain = primary_topic.get("domain") or {}
        best_oa_location = work.get("best_oa_location") or {}
        primary_location = work.get("primary_location") or {}
        open_access = work.get("open_access") or {}
        has_content = work.get("has_content") or {}
        content_urls = work.get("content_urls") or {}

        source = (primary_location.get("source") or best_oa_location.get("source") or {})
        pdf_url = (
            best_oa_location.get("pdf_url")
            or primary_location.get("pdf_url")
            or open_access.get("oa_url")
        )

        work_row = {
            "work_id": work_id,
            "openalex_id": work.get("id"),
            "doi": work.get("doi"),
            "title": strip_markup(work.get("title")),
            "display_name": strip_markup(work.get("display_name")),
            "publication_year": work.get("publication_year"),
            "publication_date": work.get("publication_date"),
            "language": work.get("language"),
            "type": work.get("type"),
            "type_crossref": work.get("type_crossref"),
            "cited_by_count": work.get("cited_by_count"),
            "referenced_works_count": work.get("referenced_works_count"),
            "locations_count": work.get("locations_count"),
            "has_abstract": bool_int(bool(work.get("abstract_inverted_index"))),
            "has_references": bool_int((work.get("referenced_works_count") or 0) > 0),
            "has_pdf_url": bool_int(bool(pdf_url)),
            "has_fulltext": bool_int(work.get("has_fulltext")),
            "has_content_pdf": bool_int(has_content.get("pdf")),
            "has_content_grobid_xml": bool_int(has_content.get("grobid_xml")),
            "is_oa": bool_int(open_access.get("is_oa")),
            "oa_status": open_access.get("oa_status"),
            "oa_url": open_access.get("oa_url"),
            "any_repository_has_fulltext": bool_int(open_access.get("any_repository_has_fulltext")),
            "is_retracted": bool_int(work.get("is_retracted")),
            "is_paratext": bool_int(work.get("is_paratext")),
            "fulltext_origin": work.get("fulltext_origin"),
            "primary_topic_id": short_openalex_id(primary_topic.get("id")),
            "primary_topic_name": primary_topic.get("display_name"),
            "primary_subfield_id": short_openalex_id(primary_subfield.get("id")),
            "primary_subfield_name": primary_subfield.get("display_name"),
            "primary_field_id": short_openalex_id(primary_field.get("id")),
            "primary_field_name": primary_field.get("display_name"),
            "primary_domain_id": short_openalex_id(primary_domain.get("id")),
            "primary_domain_name": primary_domain.get("display_name"),
            "primary_source_id": short_openalex_id(source.get("id")),
            "primary_source_name": source.get("display_name"),
            "primary_source_type": source.get("type"),
            "primary_source_issn_l": source.get("issn_l"),
            "primary_source_is_oa": bool_int(source.get("is_oa")),
            "primary_source_is_in_doaj": bool_int(source.get("is_in_doaj")),
            "primary_source_is_core": bool_int(source.get("is_core")),
            "primary_source_host_org_id": short_openalex_id(source.get("host_organization")),
            "primary_source_host_org_name": source.get("host_organization_name"),
            "license": best_oa_location.get("license") or primary_location.get("license"),
            "license_id": best_oa_location.get("license_id") or primary_location.get("license_id"),
            "version": best_oa_location.get("version") or primary_location.get("version"),
            "landing_page_url": best_oa_location.get("landing_page_url") or primary_location.get("landing_page_url"),
            "pdf_url": pdf_url,
            "content_pdf_url": content_urls.get("pdf"),
            "content_grobid_xml_url": content_urls.get("grobid_xml"),
            "abstract_text": reconstruct_abstract(work.get("abstract_inverted_index")),
            "indexed_in_json": json_text(work.get("indexed_in")),
            "ids_json": json_text(work.get("ids")),
            "best_oa_location_json": json_text(best_oa_location),
            "primary_location_json": json_text(primary_location),
            "has_content_json": json_text(has_content),
            "content_urls_json": json_text(content_urls),
            "created_date": work.get("created_date"),
            "updated_date": work.get("updated_date"),
            "inserted_at": now,
            "last_seen_at": now,
        }
        self._upsert("works", work_row, ["work_id"])

        self.conn.execute("DELETE FROM work_authorships WHERE work_id = ?", (work_id,))
        for author_index, authorship in enumerate(work.get("authorships") or [], start=1):
            author = authorship.get("author") or {}
            author_id = short_openalex_id(author.get("id"))
            if author_id:
                author_row = {
                    "author_id": author_id,
                    "openalex_id": author.get("id"),
                    "display_name": author.get("display_name"),
                    "orcid": author.get("orcid"),
                    "last_seen_at": now,
                }
                self._upsert("authors", author_row, ["author_id"])

            authorship_row = {
                "work_id": work_id,
                "author_index": author_index,
                "author_id": author_id,
                "author_position": authorship.get("author_position"),
                "is_corresponding": bool_int(authorship.get("is_corresponding")),
                "raw_author_name": authorship.get("raw_author_name"),
                "countries_json": json_text(authorship.get("countries")),
                "institutions_json": json_text(authorship.get("institutions")),
                "affiliations_json": json_text(authorship.get("affiliations")),
                "raw_affiliation_strings_json": json_text(authorship.get("raw_affiliation_strings")),
            }
            self.conn.execute(
                """
                INSERT INTO work_authorships (
                    work_id, author_index, author_id, author_position, is_corresponding,
                    raw_author_name, countries_json, institutions_json, affiliations_json,
                    raw_affiliation_strings_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tuple(authorship_row.values()),
            )

        self.conn.execute("DELETE FROM work_topics WHERE work_id = ?", (work_id,))
        topics = work.get("topics") or []
        primary_topic_id = short_openalex_id(primary_topic.get("id"))
        seen_topic_ids: set[str] = set()
        for rank_index, topic in enumerate(topics, start=1):
            topic_id = short_openalex_id(topic.get("id"))
            if not topic_id or topic_id in seen_topic_ids:
                continue
            seen_topic_ids.add(topic_id)
            subfield = topic.get("subfield") or {}
            field = topic.get("field") or {}
            domain = topic.get("domain") or {}
            topic_row = {
                "topic_id": topic_id,
                "openalex_id": topic.get("id"),
                "display_name": topic.get("display_name"),
                "subfield_id": short_openalex_id(subfield.get("id")),
                "subfield_name": subfield.get("display_name"),
                "field_id": short_openalex_id(field.get("id")),
                "field_name": field.get("display_name"),
                "domain_id": short_openalex_id(domain.get("id")),
                "domain_name": domain.get("display_name"),
                "last_seen_at": now,
            }
            self._upsert("topics", topic_row, ["topic_id"])
            self.conn.execute(
                """
                INSERT INTO work_topics (work_id, topic_id, rank_index, score, is_primary)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    work_id,
                    topic_id,
                    rank_index,
                    topic.get("score"),
                    1 if topic_id == primary_topic_id else 0,
                ),
            )

    def get_database_summary(self) -> dict[str, int]:
        keys = [
            ("works", "SELECT COUNT(*) FROM works"),
            ("authors", "SELECT COUNT(*) FROM authors"),
            ("authorship_rows", "SELECT COUNT(*) FROM work_authorships"),
            ("topics", "SELECT COUNT(*) FROM topics"),
            ("topic_links", "SELECT COUNT(*) FROM work_topics"),
            ("downloaded_pdfs", "SELECT COUNT(*) FROM pdf_downloads WHERE status = 'success'"),
            ("pending_pdfs", "SELECT COUNT(*) FROM download_queue WHERE download_status = 'pending'"),
        ]
        summary: dict[str, int] = {}
        for key, query in keys:
            summary[key] = int(self.conn.execute(query).fetchone()[0])
        return summary

    def get_pdf_candidates(self, limit: int) -> list[sqlite3.Row]:
        query = """
            SELECT w.work_id, w.content_pdf_url, w.publication_date, w.cited_by_count
            FROM works w
            LEFT JOIN pdf_downloads d ON d.work_id = w.work_id
            WHERE w.has_content_pdf = 1
              AND d.work_id IS NULL
            ORDER BY COALESCE(w.cited_by_count, 0) DESC, w.publication_date ASC, w.work_id ASC
            LIMIT ?
        """
        return list(self.conn.execute(query, (limit,)).fetchall())

    def record_pdf_success(self, work_id: str, download_info: dict[str, Any]) -> None:
        existing = self.conn.execute(
            "SELECT attempt_count FROM pdf_downloads WHERE work_id = ?",
            (work_id,),
        ).fetchone()
        attempt_count = (existing["attempt_count"] if existing else 0) + 1
        row = {
            "work_id": work_id,
            "status": "success",
            "attempt_count": attempt_count,
            "file_path": download_info["file_path"],
            "file_size_bytes": download_info["file_size_bytes"],
            "content_type": download_info["content_type"],
            "sha256": download_info["sha256"],
            "http_status": 200,
            "downloaded_at": utc_now(),
            "last_attempt_at": utc_now(),
            "last_error": None,
        }
        self._upsert("pdf_downloads", row, ["work_id"])
        self.conn.commit()

    def record_pdf_error(self, work_id: str, message: str, http_status: int | None = None) -> None:
        existing = self.conn.execute(
            "SELECT attempt_count FROM pdf_downloads WHERE work_id = ?",
            (work_id,),
        ).fetchone()
        attempt_count = (existing["attempt_count"] if existing else 0) + 1
        row = {
            "work_id": work_id,
            "status": "error",
            "attempt_count": attempt_count,
            "file_path": None,
            "file_size_bytes": None,
            "content_type": None,
            "sha256": None,
            "http_status": http_status,
            "downloaded_at": None,
            "last_attempt_at": utc_now(),
            "last_error": message[:4000],
        }
        self._upsert("pdf_downloads", row, ["work_id"])
        self.conn.commit()


def build_filter_string(config: CorpusConfig) -> str:
    scope_field = "primary_topic.subfield.id" if config.subfield_scope == "primary" else "topics.subfield.id"
    filters = [f"{scope_field}:{config.subfield_id}", *CORE_FILTERS]
    if config.profile == "downloadable":
        filters.extend(DOWNLOADABLE_FILTERS)
    return ",".join(filters)


def build_job_key(config: CorpusConfig, filter_string: str) -> str:
    fingerprint = f"{filter_string}|sort={config.sort}"
    digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:8]
    return f"subfield-{config.subfield_id}-{config.subfield_scope}-{config.profile}-{digest}"


def sync_metadata(
    config: CorpusConfig,
    client: OpenAlexClient,
    store: SQLiteStore,
    *,
    filter_string: str,
    subfield_name: str,
    job_key: str,
) -> None:
    if config.reset_sync_state:
        store.reset_sync_job(job_key)

    existing_job = store.get_sync_job(job_key)
    if existing_job and existing_job["completed"]:
        print(
            f"Metadata sync already completed for job {job_key} "
            f"({existing_job['works_synced']} works)."
        )
        return

    cursor = existing_job["next_cursor"] if existing_job and existing_job["next_cursor"] else "*"
    works_synced = int(existing_job["works_synced"]) if existing_job else 0
    total_expected = int(existing_job["total_expected"]) if existing_job and existing_job["total_expected"] else None
    page_count = 0

    while True:
        remaining = None if config.max_works is None else max(config.max_works - works_synced, 0)
        if remaining == 0:
            store.upsert_sync_job(
                job_key=job_key,
                filter_scope=config.subfield_scope,
                filter_profile=config.profile,
                filter_string=filter_string,
                subfield_id=config.subfield_id,
                subfield_name=subfield_name,
                sort=config.sort,
                per_page=config.per_page,
                next_cursor=cursor,
                total_expected=total_expected,
                works_synced=works_synced,
                completed=False,
            )
            print(f"Reached max_works={config.max_works}; metadata sync paused at {works_synced} works.")
            return

        page_size = min(config.per_page, remaining) if remaining is not None else config.per_page
        params = {
            "filter": filter_string,
            "sort": config.sort,
            "cursor": cursor,
            "per_page": page_size,
            "select": ",".join(SELECT_FIELDS),
        }
        response = client.get_json("/works", params)
        results = response.get("results") or []
        meta = response.get("meta") or {}

        if total_expected is None and meta.get("count") is not None:
            total_expected = int(meta["count"])

        if not results:
            store.upsert_sync_job(
                job_key=job_key,
                filter_scope=config.subfield_scope,
                filter_profile=config.profile,
                filter_string=filter_string,
                subfield_id=config.subfield_id,
                subfield_name=subfield_name,
                sort=config.sort,
                per_page=config.per_page,
                next_cursor=None,
                total_expected=total_expected,
                works_synced=works_synced,
                completed=True,
            )
            print(f"Metadata sync complete: {works_synced} works.")
            return

        with store.transaction():
            for work in results:
                store.upsert_work_bundle(work)

        works_synced += len(results)
        page_count += 1
        next_cursor = meta.get("next_cursor")
        completed = not bool(next_cursor)
        store.upsert_sync_job(
            job_key=job_key,
            filter_scope=config.subfield_scope,
            filter_profile=config.profile,
            filter_string=filter_string,
            subfield_id=config.subfield_id,
            subfield_name=subfield_name,
            sort=config.sort,
            per_page=config.per_page,
            next_cursor=next_cursor,
            total_expected=total_expected,
            works_synced=works_synced,
            completed=completed,
        )

        if page_count == 1 or page_count % 10 == 0 or completed:
            total_label = total_expected if total_expected is not None else "?"
            print(f"Synced {works_synced}/{total_label} works...")

        if completed:
            print(f"Metadata sync complete: {works_synced} works.")
            return

        cursor = next_cursor


def download_pdfs(config: CorpusConfig, client: OpenAlexClient, store: SQLiteStore) -> None:
    if not config.download_pdfs or config.max_pdfs <= 0:
        print("PDF download step skipped.")
        return

    candidates = store.get_pdf_candidates(config.max_pdfs)
    if not candidates:
        print("No pending PDFs found.")
        return

    print(f"Downloading up to {len(candidates)} PDFs...")
    for index, row in enumerate(candidates, start=1):
        work_id = row["work_id"]
        content_pdf_url = row["content_pdf_url"]
        if not content_pdf_url:
            store.record_pdf_error(work_id, "Missing content_pdf_url in local metadata.")
            continue

        try:
            destination = config.pdf_dir / f"{work_id}.pdf"
            result = client.download_pdf(content_pdf_url, destination)
            store.record_pdf_success(work_id, result)
            print(f"[{index}/{len(candidates)}] Downloaded {work_id} -> {destination.name}")
        except Exception as exc:
            store.record_pdf_error(work_id, str(exc))
            print(f"[{index}/{len(candidates)}] Failed {work_id}: {exc}")


def main() -> int:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.pdf_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAlexClient(api_key=config.api_key, user_agent=config.user_agent)
    store = SQLiteStore(config.db_path)
    try:
        subfield = client.get_subfield(config.subfield_id)
        subfield_name = subfield.get("display_name") or "Unknown subfield"
        filter_string = build_filter_string(config)
        job_key = build_job_key(config, filter_string)

        print(f"Subfield: {subfield_name} ({config.subfield_id})")
        print(f"Profile: {config.profile}")
        print(f"Scope: {config.subfield_scope}")
        print(f"Filter: {filter_string}")
        print(f"Database: {config.db_path}")
        print(f"PDF directory: {config.pdf_dir}")

        sync_metadata(
            config,
            client,
            store,
            filter_string=filter_string,
            subfield_name=subfield_name,
            job_key=job_key,
        )
        download_pdfs(config, client, store)

        summary = store.get_database_summary()
        print("Local database summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
