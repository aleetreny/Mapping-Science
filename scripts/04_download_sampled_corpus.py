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
    build_sampled_text_query_params,
    build_year_text_query_params,
    short_openalex_id,
)
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
    parser = argparse.ArgumentParser(description="Download OpenAlex sampled text corpus.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned downloads.")
    parser.add_argument(
        "--limit-subfields",
        type=int,
        default=None,
        help="Limit unique subfields for testing.",
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


def selected_sample_plan(
    sample_plan: pd.DataFrame, limit_subfields: int | None
) -> pd.DataFrame:
    selected = sample_plan.copy()
    selected["subfield_id"] = selected["subfield_id"].astype(str)
    selected = selected.sort_values(["subfield_id", "publication_year"]).reset_index(
        drop=True
    )
    if limit_subfields is None:
        return selected
    keep_ids = selected["subfield_id"].drop_duplicates().head(limit_subfields)
    return selected[selected["subfield_id"].isin(set(keep_ids))].reset_index(drop=True)


def validate_and_normalize_work(
    work: dict[str, Any],
    config: dict[str, Any],
    expected_subfield_id: str,
    expected_year: int,
) -> dict[str, Any] | None:
    filters = config["filters"]
    title = work.get("title") or work.get("display_name")
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
    publication_year = work.get("publication_year")
    work_type = work.get("type")
    language = work.get("language")
    primary_topic = work.get("primary_topic") or {}

    if publication_year != int(expected_year):
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
    if not subfield_id or subfield_id != str(expected_subfield_id):
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


def query_params_for_row(row: pd.Series, config: dict[str, Any]) -> dict[str, Any]:
    work_types = config["filters"]["work_types"]
    language = config["filters"]["language"]
    per_page = int(config["openalex"].get("per_page", 200))

    if row["sampling_method"] == "sample_api":
        return build_sampled_text_query_params(
            subfield_id=str(row["subfield_id"]),
            year=int(row["publication_year"]),
            sample_size=int(row["planned_sample_size"]),
            seed=int(row["seed"]),
            work_types=work_types,
            language=language,
            select_fields=DEFAULT_SELECT_FIELDS,
        )

    return build_year_text_query_params(
        subfield_id=str(row["subfield_id"]),
        year=int(row["publication_year"]),
        work_types=work_types,
        language=language,
        per_page=per_page,
        select_fields=DEFAULT_SELECT_FIELDS,
    )


def print_dry_run(sample_plan: pd.DataFrame, config: dict[str, Any]) -> None:
    planned = sample_plan[sample_plan["planned_sample_size"] > 0].copy()
    per_page = int(config["openalex"].get("per_page", 200))
    sample_rows = planned[planned["sampling_method"] == "sample_api"]
    all_rows = planned[planned["sampling_method"] == "download_all_available"]
    estimated_requests = int(
        sum(
            math.ceil(size / min(size, per_page))
            for size in sample_rows["planned_sample_size"]
            if size > 0
        )
    ) + int(
        sum(math.ceil(size / per_page) for size in all_rows["planned_sample_size"])
    )

    print(f"subfields selected: {sample_plan['subfield_id'].nunique()}")
    print(f"planned works: {int(sample_plan['planned_sample_size'].sum())}")
    print(f"estimated API requests: {estimated_requests}")
    print("planned works by sampling method:")
    if planned.empty:
        print("(none)")
    else:
        print(
            planned.groupby("sampling_method")["planned_sample_size"]
            .sum()
            .astype(int)
            .to_string()
        )

    examples = planned.head(3)
    if not examples.empty:
        print("example query params:")
        for _, row in examples.iterrows():
            print(json.dumps(query_params_for_row(row, config), indent=2))


def fetch_works_for_row(
    client: OpenAlexClient,
    row: pd.Series,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    planned_sample_size = int(row["planned_sample_size"])
    if planned_sample_size <= 0:
        return []

    params = query_params_for_row(row, config)
    records = []

    if row["sampling_method"] == "sample_api":
        page = 1
        while len(records) < planned_sample_size:
            page_params = dict(params)
            page_params["page"] = page
            data = client.get_json("works", page_params)
            works = data.get("results") or []
            if not works:
                break
            for work in works:
                normalized = validate_and_normalize_work(
                    work,
                    config=config,
                    expected_subfield_id=str(row["subfield_id"]),
                    expected_year=int(row["publication_year"]),
                )
                if normalized is not None:
                    records.append(normalized)
                if len(records) >= planned_sample_size:
                    break
            page += 1
        return records

    if row["sampling_method"] == "download_all_available":
        fetched = 0
        for work in client.paginate("works", params):
            fetched += 1
            normalized = validate_and_normalize_work(
                work,
                config=config,
                expected_subfield_id=str(row["subfield_id"]),
                expected_year=int(row["publication_year"]),
            )
            if normalized is not None:
                records.append(normalized)
            if fetched >= planned_sample_size:
                break

    return records


def main() -> None:
    args = parse_args()
    config = load_config()
    ensure_dirs(config)

    interim_dir = ROOT / config["storage"]["interim_dir"]
    processed_dir = ROOT / config["storage"]["processed_dir"]
    duckdb_path = ROOT / config["storage"]["duckdb_path"]
    sample_plan_path = interim_dir / "sample_plan.parquet"
    if not sample_plan_path.exists():
        raise FileNotFoundError(
            "Missing data/interim/sample_plan.parquet. Run scripts/03_build_sample_plan.py first."
        )

    sample_plan = load_parquet(sample_plan_path)
    sample_plan = selected_sample_plan(sample_plan, args.limit_subfields)

    if args.dry_run:
        print_dry_run(sample_plan, config)
        return

    client = OpenAlexClient.from_config(config)
    records = []
    planned_rows = sample_plan[sample_plan["planned_sample_size"] > 0]

    for _, row in tqdm(
        planned_rows.iterrows(), total=len(planned_rows), desc="Downloading sample plan"
    ):
        records.extend(fetch_works_for_row(client, row, config))

    if records:
        works_text = pd.DataFrame(records, columns=WORKS_TEXT_COLUMNS)
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
        "selected_subfields": int(sample_plan["subfield_id"].nunique()),
        "planned_works": int(sample_plan["planned_sample_size"].sum()),
        "downloaded_works": len(works_text),
        "limit_subfields": args.limit_subfields,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
