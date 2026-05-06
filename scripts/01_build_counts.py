from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.openalex import OpenAlexClient, build_count_query_params, short_openalex_id
from src.storage import connect_duckdb, ensure_dirs, load_parquet, save_parquet, write_table


COUNT_SPECS = [
    ("n_works_total", []),
    ("n_works_article_preprint", ["type:article|preprint"]),
    ("n_works_article_preprint_en", ["type:article|preprint", "language:en"]),
    (
        "n_works_article_preprint_en_with_abstract",
        ["type:article|preprint", "language:en", "has_abstract:true"],
    ),
]

LEVELS = [
    ("subfield", "primary_topic.subfield.id", "subfield_id"),
    ("field", "primary_topic.field.id", "field_id"),
    ("domain", "primary_topic.domain.id", "domain_id"),
]


def load_config() -> dict[str, Any]:
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OpenAlex count tables.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned queries.")
    return parser.parse_args()


def year_range(config: dict[str, Any]) -> list[int]:
    windows = config["windows"]
    return list(
        range(
            int(windows["morphology_start_year"]),
            int(windows["growth_end_year"]) + 1,
        )
    )


def print_dry_run(config: dict[str, Any]) -> None:
    years = year_range(config)
    total_queries = len(years) * len(COUNT_SPECS) * len(LEVELS)
    print(f"Would run {total_queries} OpenAlex group-by count queries.")
    for level_name, group_by, _ in LEVELS:
        metric_name, extra_filters = COUNT_SPECS[1]
        params = build_count_query_params(
            years[0],
            group_by=group_by,
            extra_filters=extra_filters,
            per_page=int(config["openalex"].get("per_page", 200)),
        )
        print(f"{level_name} {metric_name}: /works?{params}")


def load_taxonomy(interim_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    domains_path = interim_dir / "domains.parquet"
    fields_path = interim_dir / "fields.parquet"
    subfields_path = interim_dir / "subfields.parquet"
    missing = [
        str(path.relative_to(ROOT))
        for path in [domains_path, fields_path, subfields_path]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing taxonomy files. Run scripts/00_fetch_taxonomy.py first: "
            + ", ".join(missing)
        )
    return load_parquet(domains_path), load_parquet(fields_path), load_parquet(subfields_path)


def make_grid(ids: list[str], years: list[int], id_col: str) -> pd.DataFrame:
    return pd.MultiIndex.from_product(
        [ids, years], names=[id_col, "publication_year"]
    ).to_frame(index=False)


def fetch_metric_counts(
    client: OpenAlexClient,
    years: list[int],
    group_by: str,
    id_col: str,
    metric_name: str,
    extra_filters: list[str],
    per_page: int,
) -> pd.DataFrame:
    rows = []
    for year in tqdm(years, desc=f"{id_col} {metric_name}"):
        params = build_count_query_params(
            year,
            group_by=group_by,
            extra_filters=extra_filters,
            per_page=per_page,
        )
        for group in client.paginate("works", params, results_key="group_by"):
            key = short_openalex_id(group.get("key"))
            if not key or key.lower() == "unknown":
                continue
            rows.append(
                {
                    id_col: key,
                    "publication_year": year,
                    metric_name: int(group.get("count") or 0),
                }
            )
    return pd.DataFrame(rows)


def build_level_counts(
    client: OpenAlexClient,
    ids: list[str],
    years: list[int],
    level_name: str,
    group_by: str,
    id_col: str,
    per_page: int,
) -> pd.DataFrame:
    counts = make_grid(ids, years, id_col)
    for metric_name, extra_filters in COUNT_SPECS:
        metric_counts = fetch_metric_counts(
            client=client,
            years=years,
            group_by=group_by,
            id_col=id_col,
            metric_name=metric_name,
            extra_filters=extra_filters,
            per_page=per_page,
        )
        if metric_counts.empty:
            counts[metric_name] = 0
            continue
        counts = counts.merge(
            metric_counts, on=[id_col, "publication_year"], how="left"
        )
        counts[metric_name] = counts[metric_name].fillna(0).astype("int64")
    print(f"Built {level_name} counts: {len(counts)} rows.")
    return counts


def main() -> None:
    args = parse_args()
    config = load_config()
    ensure_dirs(config)

    if args.dry_run:
        print_dry_run(config)
        return

    interim_dir = ROOT / config["storage"]["interim_dir"]
    duckdb_path = ROOT / config["storage"]["duckdb_path"]
    years = year_range(config)
    per_page = int(config["openalex"].get("per_page", 200))

    domains, fields, subfields = load_taxonomy(interim_dir)
    client = OpenAlexClient.from_config(config)

    subfield_counts = build_level_counts(
        client,
        subfields["subfield_id"].dropna().astype(str).tolist(),
        years,
        "subfield",
        "primary_topic.subfield.id",
        "subfield_id",
        per_page,
    )
    subfield_counts = subfield_counts.merge(
        subfields[["subfield_id", "field_id", "domain_id"]],
        on="subfield_id",
        how="left",
    )
    subfield_counts = subfield_counts[
        [
            "subfield_id",
            "field_id",
            "domain_id",
            "publication_year",
            "n_works_total",
            "n_works_article_preprint",
            "n_works_article_preprint_en",
            "n_works_article_preprint_en_with_abstract",
        ]
    ]

    field_counts = build_level_counts(
        client,
        fields["field_id"].dropna().astype(str).tolist(),
        years,
        "field",
        "primary_topic.field.id",
        "field_id",
        per_page,
    )
    field_counts = field_counts.merge(
        fields[["field_id", "domain_id"]], on="field_id", how="left"
    )
    field_counts = field_counts[
        [
            "field_id",
            "domain_id",
            "publication_year",
            "n_works_total",
            "n_works_article_preprint",
            "n_works_article_preprint_en",
            "n_works_article_preprint_en_with_abstract",
        ]
    ]

    domain_counts = build_level_counts(
        client,
        domains["domain_id"].dropna().astype(str).tolist(),
        years,
        "domain",
        "primary_topic.domain.id",
        "domain_id",
        per_page,
    )
    domain_counts = domain_counts[
        [
            "domain_id",
            "publication_year",
            "n_works_total",
            "n_works_article_preprint",
            "n_works_article_preprint_en",
            "n_works_article_preprint_en_with_abstract",
        ]
    ]

    save_parquet(subfield_counts, interim_dir / "subfield_year_counts.parquet")
    save_parquet(field_counts, interim_dir / "field_year_counts.parquet")
    save_parquet(domain_counts, interim_dir / "domain_year_counts.parquet")

    con = connect_duckdb(duckdb_path)
    try:
        write_table(con, "subfield_year_counts", subfield_counts)
        write_table(con, "field_year_counts", field_counts)
        write_table(con, "domain_year_counts", domain_counts)
    finally:
        con.close()

    print("Count tables written to Parquet and DuckDB.")


if __name__ == "__main__":
    main()
