from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.abstracts import reconstruct_abstract
from src.openalex import (
    DEFAULT_SELECT_FIELDS,
    OpenAlexClient,
    build_text_query_params,
    short_openalex_id,
)
from src.sampling import stratified_sample_by_year
from src.storage import connect_duckdb, ensure_dirs, load_parquet, save_parquet, write_table


TOKEN_RE = re.compile(r"\b\w+\b")

WORKS_TEXT_COLUMNS = [
    "work_id",
    "doi",
    "title",
    "abstract",
    "text_for_embedding",
    "publication_year",
    "publication_date",
    "type",
    "language",
    "subfield_id",
    "subfield_display_name",
    "field_id",
    "field_display_name",
    "domain_id",
    "domain_display_name",
    "cited_by_count",
    "referenced_works_count",
]


def load_config() -> dict[str, Any]:
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download sampled OpenAlex text corpus.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned downloads.")
    parser.add_argument(
        "--limit-subfields",
        type=int,
        default=None,
        help="Limit eligible subfields for testing.",
    )
    return parser.parse_args()


def token_count(text: str | None) -> int:
    if not text:
        return 0
    return len(TOKEN_RE.findall(text))


def topic_ref(topic: dict[str, Any], name: str) -> tuple[str | None, str | None]:
    entity = topic.get(name)
    if not isinstance(entity, dict):
        return None, None
    return short_openalex_id(entity.get("id")), entity.get("display_name")


def validate_and_normalize_work(
    work: dict[str, Any],
    config: dict[str, Any],
    start_year: int,
    end_year: int,
) -> dict[str, Any] | None:
    filters = config["filters"]
    title = work.get("title") or work.get("display_name")
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
    publication_year = work.get("publication_year")
    work_type = work.get("type")
    language = work.get("language")
    primary_topic = work.get("primary_topic") or {}

    if not isinstance(publication_year, int) or not (
        start_year <= publication_year <= end_year
    ):
        return None
    if language != filters["language"]:
        return None
    if work_type not in set(filters["work_types"]):
        return None
    if bool(work.get("is_retracted")):
        return None
    if bool(work.get("is_paratext")):
        return None
    if token_count(title) < int(filters["min_title_tokens"]):
        return None
    if token_count(abstract) < int(filters["min_abstract_tokens"]):
        return None

    subfield_id, subfield_name = topic_ref(primary_topic, "subfield")
    field_id, field_name = topic_ref(primary_topic, "field")
    domain_id, domain_name = topic_ref(primary_topic, "domain")
    if not subfield_id:
        return None

    work_id = short_openalex_id(work.get("id"))
    if not work_id:
        return None

    return {
        "work_id": work_id,
        "doi": work.get("doi"),
        "title": title,
        "abstract": abstract,
        "text_for_embedding": f"{title}\n\n{abstract}",
        "publication_year": publication_year,
        "publication_date": work.get("publication_date"),
        "type": work_type,
        "language": language,
        "subfield_id": subfield_id,
        "subfield_display_name": subfield_name,
        "field_id": field_id,
        "field_display_name": field_name,
        "domain_id": domain_id,
        "domain_display_name": domain_name,
        "cited_by_count": work.get("cited_by_count"),
        "referenced_works_count": work.get("referenced_works_count"),
    }


def selected_plan(plan: pd.DataFrame, limit_subfields: int | None) -> pd.DataFrame:
    selected = plan[plan["eligible_for_text_corpus"]].copy()
    selected = selected.sort_values(["subfield_id"]).reset_index(drop=True)
    if limit_subfields is not None:
        selected = selected.head(limit_subfields)
    return selected


def print_dry_run(
    plan: pd.DataFrame, config: dict[str, Any], limit_subfields: int | None
) -> None:
    selected = selected_plan(plan, limit_subfields)
    per_page = int(config["openalex"].get("per_page", 200))
    planned_works = int(selected["planned_sample_size"].sum()) if not selected.empty else 0
    estimated_work_pages = math.ceil(planned_works / per_page) if planned_works else 0
    estimated_count_requests = len(selected) * 11

    print(f"Eligible subfields selected: {len(selected)}")
    print(f"Planned sampled works: {planned_works}")
    print(
        "Estimated requests: "
        f"{estimated_count_requests} count requests plus about "
        f"{estimated_work_pages} work pages."
    )
    if not selected.empty:
        preview = selected[
            ["subfield_id", "subfield_display_name", "planned_sample_size"]
        ].head(10)
        print(preview.to_string(index=False))


def count_filters_for_year(
    subfield_id: str, year: int, config: dict[str, Any]
) -> str:
    params = build_text_query_params(
        subfield_id=subfield_id,
        start_year=year,
        end_year=year,
        work_types=config["filters"]["work_types"],
        language=config["filters"]["language"],
        per_page=int(config["openalex"].get("per_page", 200)),
        select_fields=DEFAULT_SELECT_FIELDS,
    )
    return params["filter"]


def candidate_cap_by_year(
    year_estimates: dict[int, int],
    target_per_year: int,
    target_total: int,
) -> dict[int, int]:
    shortage = sum(max(0, target_per_year - count) for count in year_estimates.values())
    rich_years = [year for year, count in year_estimates.items() if count > target_per_year]
    extra_per_rich = math.ceil(shortage / len(rich_years)) if rich_years else 0

    caps = {}
    for year, count in year_estimates.items():
        if count <= target_per_year:
            caps[year] = count
        else:
            buffered = max(target_per_year * 3, target_per_year + extra_per_rich * 2)
            caps[year] = min(count, buffered, target_total)
    return caps


def fetch_candidates_for_subfield(
    client: OpenAlexClient,
    row: pd.Series,
    config: dict[str, Any],
    years: list[int],
) -> pd.DataFrame:
    per_page = int(config["openalex"].get("per_page", 200))
    target_total = int(row["planned_sample_size"])
    target_per_year = int(config["sampling"]["target_per_year_per_subfield"])
    subfield_id = str(row["subfield_id"])

    total_filters = build_text_query_params(
        subfield_id=subfield_id,
        start_year=years[0],
        end_year=years[-1],
        work_types=config["filters"]["work_types"],
        language=config["filters"]["language"],
        per_page=per_page,
        select_fields=DEFAULT_SELECT_FIELDS,
    )["filter"]
    total_estimate = client.get_count("works", total_filters)
    print(
        f"{subfield_id} {row.get('subfield_display_name', '')}: "
        f"{total_estimate} candidate works before local validation."
    )

    year_estimates = {
        year: client.get_count("works", count_filters_for_year(subfield_id, year, config))
        for year in years
    }
    caps = candidate_cap_by_year(year_estimates, target_per_year, target_total)

    records = []
    for year in years:
        cap = caps.get(year, 0)
        if cap <= 0:
            continue

        params = build_text_query_params(
            subfield_id=subfield_id,
            start_year=year,
            end_year=year,
            work_types=config["filters"]["work_types"],
            language=config["filters"]["language"],
            per_page=per_page,
            select_fields=DEFAULT_SELECT_FIELDS,
        )

        fetched = 0
        for work in client.paginate("works", params):
            normalized = validate_and_normalize_work(
                work,
                config=config,
                start_year=years[0],
                end_year=years[-1],
            )
            fetched += 1
            if normalized is not None:
                records.append(normalized)
            if fetched >= cap:
                break

    if not records:
        return pd.DataFrame(columns=WORKS_TEXT_COLUMNS)
    return pd.DataFrame(records, columns=WORKS_TEXT_COLUMNS)


def main() -> None:
    args = parse_args()
    config = load_config()
    ensure_dirs(config)

    interim_dir = ROOT / config["storage"]["interim_dir"]
    processed_dir = ROOT / config["storage"]["processed_dir"]
    duckdb_path = ROOT / config["storage"]["duckdb_path"]
    plan_path = interim_dir / "corpus_plan.parquet"
    if not plan_path.exists():
        raise FileNotFoundError(
            "Missing data/interim/corpus_plan.parquet. Run scripts/02_build_corpus_plan.py first."
        )

    plan = load_parquet(plan_path)
    if args.dry_run:
        print_dry_run(plan, config, args.limit_subfields)
        return

    selected = selected_plan(plan, args.limit_subfields)
    windows = config["windows"]
    years = list(
        range(
            int(windows["morphology_start_year"]),
            int(windows["morphology_end_year"]) + 1,
        )
    )
    random_seed = int(config["sampling"]["random_seed"])
    target_per_year = int(config["sampling"]["target_per_year_per_subfield"])

    client = OpenAlexClient.from_config(config)
    sampled_parts = []

    for _, row in tqdm(
        selected.iterrows(), total=len(selected), desc="Downloading subfields"
    ):
        candidates = fetch_candidates_for_subfield(client, row, config, years)
        sampled = stratified_sample_by_year(
            candidates,
            year_col="publication_year",
            target_total=int(row["planned_sample_size"]),
            target_per_year=target_per_year,
            random_seed=random_seed,
        )
        sampled_parts.append(sampled)

    if sampled_parts:
        works_text = pd.concat(sampled_parts, ignore_index=True)
    else:
        works_text = pd.DataFrame(columns=WORKS_TEXT_COLUMNS)

    works_text = works_text.drop_duplicates("work_id", keep="first").reset_index(drop=True)
    save_parquet(works_text, processed_dir / "works_text.parquet")

    con = connect_duckdb(duckdb_path)
    try:
        write_table(con, "works_text", works_text)
    finally:
        con.close()

    summary = {
        "selected_subfields": len(selected),
        "downloaded_works": len(works_text),
        "limit_subfields": args.limit_subfields,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
