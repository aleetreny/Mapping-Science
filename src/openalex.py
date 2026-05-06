from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests
from dotenv import load_dotenv
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_repo_env(env_path: str | Path | None = None) -> bool:
    """Load environment variables from the repo .env file without overriding shell vars."""
    path = Path(env_path) if env_path is not None else REPO_ROOT / ".env"
    return load_dotenv(path, override=False)


load_repo_env()


DEFAULT_SELECT_FIELDS = [
    "id",
    "doi",
    "title",
    "display_name",
    "abstract_inverted_index",
    "publication_year",
    "publication_date",
    "type",
    "language",
    "primary_topic",
    "topics",
    "cited_by_count",
    "referenced_works_count",
    "is_retracted",
    "is_paratext",
]


def build_filter(parts: Iterable[str | None] | str | None) -> str:
    """Build an OpenAlex comma-separated filter string."""
    if parts is None:
        return ""
    if isinstance(parts, str):
        return parts.strip().strip(",")

    cleaned: list[str] = []
    for part in parts:
        if part is None:
            continue
        text = str(part).strip().strip(",")
        if text:
            cleaned.append(text)
    return ",".join(cleaned)


def short_openalex_id(value: Any) -> str | None:
    """Return the final OpenAlex id segment, preserving values already short."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.rstrip("/").split("/")[-1]


def safe_url_for_logging(url: str) -> str:
    """Redact secrets and direct contact details from a URL before logging."""
    parts = urlsplit(url)
    sensitive_names = {"api_key", "apikey", "key", "api-key", "mailto", "email"}
    query = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        normalized_key = key.lower()
        if normalized_key in sensitive_names or "email" in normalized_key:
            query.append((key, "REDACTED"))
        else:
            query.append((key, value))
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
    )


def build_text_query_params(
    subfield_id: str,
    start_year: int,
    end_year: int,
    work_types: Iterable[str],
    language: str = "en",
    per_page: int = 200,
    select_fields: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build the main works query for the morphology text corpus."""
    filters = [
        f"primary_topic.subfield.id:{short_openalex_id(subfield_id)}",
        f"from_publication_date:{start_year}-01-01",
        f"to_publication_date:{end_year}-12-31",
        "has_abstract:true",
        f"language:{language}",
        f"type:{'|'.join(work_types)}",
        "is_retracted:false",
        "is_paratext:false",
    ]
    return {
        "filter": build_filter(filters),
        "select": ",".join(select_fields or DEFAULT_SELECT_FIELDS),
        "per-page": per_page,
        "sort": "publication_date:asc",
    }


def build_year_text_query_params(
    subfield_id: str,
    year: int,
    work_types: Iterable[str],
    language: str = "en",
    per_page: int = 200,
    select_fields: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build a year-specific works query for the morphology text corpus."""
    filters = [
        f"primary_topic.subfield.id:{short_openalex_id(subfield_id)}",
        f"publication_year:{year}",
        "has_abstract:true",
        f"language:{language}",
        f"type:{'|'.join(work_types)}",
        "is_retracted:false",
        "is_paratext:false",
    ]
    return {
        "filter": build_filter(filters),
        "select": ",".join(select_fields or DEFAULT_SELECT_FIELDS),
        "per-page": per_page,
        "sort": "publication_date:asc",
    }


def build_sampled_text_query_params(
    subfield_id: str,
    year: int,
    sample_size: int,
    seed: int,
    work_types: Iterable[str],
    language: str = "en",
    select_fields: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build an OpenAlex sampled works query for one subfield-year cell."""
    params = build_year_text_query_params(
        subfield_id=subfield_id,
        year=year,
        work_types=work_types,
        language=language,
        per_page=min(int(sample_size), 200),
        select_fields=select_fields or DEFAULT_SELECT_FIELDS,
    )
    params["sample"] = sample_size
    params["seed"] = seed
    params.pop("sort", None)
    return params


def build_count_query_params(
    year: int,
    group_by: str | None = None,
    extra_filters: Iterable[str | None] | None = None,
    per_page: int = 200,
) -> dict[str, Any]:
    """Build a count or group-by count query for OpenAlex works."""
    filters = [
        f"publication_year:{year}",
        "is_retracted:false",
        "is_paratext:false",
    ]
    if extra_filters:
        filters.extend(extra_filters)

    params: dict[str, Any] = {
        "filter": build_filter(filters),
        "per-page": per_page,
    }
    if group_by:
        params["group_by"] = group_by
    return params


class OpenAlexClient:
    """Small OpenAlex client with retries, polite auth params, and pagination."""

    def __init__(
        self,
        base_url: str = "https://api.openalex.org",
        email_env: str = "OPENALEX_EMAIL",
        api_key_env: str = "OPENALEX_API_KEY",
        timeout_seconds: int = 30,
        max_retries: int = 5,
        backoff_seconds: int = 2,
        max_requests_per_second: float = 5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.email = os.getenv(email_env, "").strip()
        self.api_key = os.getenv(api_key_env, "").strip()
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.max_requests_per_second = max_requests_per_second
        self._last_request_at = 0.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OpenAlexClient":
        openalex_config = config.get("openalex", {})
        return cls(
            base_url=openalex_config.get("base_url", "https://api.openalex.org"),
            email_env=openalex_config.get("email_env", "OPENALEX_EMAIL"),
            api_key_env=openalex_config.get("api_key_env", "OPENALEX_API_KEY"),
            timeout_seconds=int(openalex_config.get("timeout_seconds", 30)),
            max_retries=int(openalex_config.get("max_retries", 5)),
            backoff_seconds=int(openalex_config.get("backoff_seconds", 2)),
            max_requests_per_second=float(
                openalex_config.get("max_requests_per_second", 5)
            ),
        )

    def _add_auth_params(self, params: dict[str, Any]) -> dict[str, Any]:
        result = dict(params)
        if self.email:
            result.setdefault("mailto", self.email)
        if self.api_key:
            result.setdefault("api_key", self.api_key)
        return result

    def _rate_limit(self) -> None:
        if self.max_requests_per_second <= 0:
            return
        min_interval = 1.0 / self.max_requests_per_second
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_at = time.monotonic()

    def get_json(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        endpoint_path = endpoint.strip("/")
        url = f"{self.base_url}/{endpoint_path}"
        request_params = self._add_auth_params(params or {})

        retryer = Retrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(
                multiplier=self.backoff_seconds,
                min=self.backoff_seconds,
                max=60,
            ),
            retry=retry_if_exception_type(requests.RequestException),
            reraise=True,
        )

        for attempt in retryer:
            with attempt:
                self._rate_limit()
                response = requests.get(
                    url,
                    params=request_params,
                    timeout=self.timeout_seconds,
                    headers={"User-Agent": "tfm-openalex-subfield-pipeline/1.0"},
                )
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise requests.HTTPError(
                        f"{response.status_code} for {safe_url_for_logging(response.url)}",
                        response=response,
                    )
                if response.status_code >= 400:
                    raise requests.HTTPError(
                        f"{response.status_code} for {safe_url_for_logging(response.url)}",
                        response=response,
                    )
                return response.json()

        raise RuntimeError("OpenAlex request failed without returning a response")

    def get_count(self, endpoint: str, filters: Iterable[str] | str) -> int:
        params = {
            "filter": build_filter(filters),
            "per-page": 1,
        }
        data = self.get_json(endpoint, params)
        return int((data.get("meta") or {}).get("count") or 0)

    def paginate(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        results_key: str = "results",
    ):
        page_params = dict(params or {})
        page_params.setdefault("cursor", "*")

        while True:
            data = self.get_json(endpoint, page_params)
            rows = data.get(results_key) or []
            for row in rows:
                yield row

            next_cursor = (data.get("meta") or {}).get("next_cursor")
            if not next_cursor:
                break
            page_params["cursor"] = next_cursor


def project_root() -> Path:
    return REPO_ROOT
