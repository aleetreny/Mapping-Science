from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sampling import (
    allocate_yearly_sample_sizes,
    api_initial_sample_size,
    sampling_method_for_cell,
    stable_sample_seed,
)
from src.extraction_versions import (
    add_dataset_version_argument,
    analysis_years,
    apply_dataset_version,
    build_no_redistribution_allocations,
    extraction_paths,
    output_path_display,
    target_total_from_config,
    versioned_table_name,
)
from src.storage import connect_duckdb, ensure_dirs, load_parquet, save_parquet, write_table


def load_config() -> dict[str, Any]:
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OpenAlex sampled download plan.")
    add_dataset_version_argument(parser)
    return parser.parse_args()


def read_required_parquet(interim_dir: Path, filename: str) -> pd.DataFrame:
    path = interim_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.relative_to(ROOT)}. Run earlier pipeline scripts first."
        )
    return load_parquet(path)


def years_from_config(config: dict[str, Any]) -> list[int]:
    return analysis_years(config)


def build_sample_plan(
    corpus_plan: pd.DataFrame,
    subfield_year_counts: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    sampling = config["sampling"]
    years = years_from_config(config)
    target_total = int(sampling["target_sample_per_subfield"])
    target_per_year = int(sampling["target_per_year_per_subfield"])
    random_seed = int(sampling["random_seed"])
    use_sample_api = bool(sampling.get("use_openalex_sample_api", True))
    oversample_factor = float(sampling.get("oversample_factor", 1.0))
    redistribute_unused = bool(sampling.get("redistribute_unused_annual_capacity", True))

    eligible = corpus_plan[corpus_plan["eligible_for_text_corpus"]].copy()
    counts = subfield_year_counts.copy()
    counts["subfield_id"] = counts["subfield_id"].astype(str)
    eligible["subfield_id"] = eligible["subfield_id"].astype(str)

    count_lookup = {
        (str(row.subfield_id), int(row.publication_year)): int(
            row.n_works_article_preprint_en_with_abstract
        )
        for row in counts.itertuples(index=False)
    }

    rows = []
    for subfield in eligible.sort_values("subfield_id").itertuples(index=False):
        subfield_id = str(subfield.subfield_id)
        available_by_year = {
            year: count_lookup.get((subfield_id, year), 0) for year in years
        }
        if redistribute_unused:
            allocations = allocate_yearly_sample_sizes(
                available_by_year=available_by_year,
                target_total=target_total,
                target_per_year=target_per_year,
            )
        else:
            allocations = build_no_redistribution_allocations(
                available_by_year,
                target_per_year=target_per_year,
            )

        for year in years:
            available = int(available_by_year.get(year, 0))
            planned = int(allocations.get(year, 0))
            method = sampling_method_for_cell(
                available_valid_works=available,
                planned_sample_size=planned,
                target_per_year=target_per_year,
                use_openalex_sample_api=use_sample_api,
            )
            initial_sample_size = api_initial_sample_size(
                available_valid_works=available,
                planned_sample_size=planned,
                sampling_method=method,
                oversample_factor=oversample_factor,
            )
            rows.append(
                {
                    "subfield_id": subfield_id,
                    "subfield_display_name": subfield.subfield_display_name,
                    "field_id": subfield.field_id,
                    "field_display_name": subfield.field_display_name,
                    "domain_id": subfield.domain_id,
                    "domain_display_name": subfield.domain_display_name,
                    "publication_year": year,
                    "available_valid_works": available,
                    "planned_sample_size": planned,
                    "sampling_method": method,
                    "seed": stable_sample_seed(subfield_id, year, random_seed),
                    "api_initial_sample_size": initial_sample_size,
                    "expected_shortfall_risk": bool(
                        planned > 0 and initial_sample_size == planned
                    ),
                }
            )

    columns = [
        "subfield_id",
        "subfield_display_name",
        "field_id",
        "field_display_name",
        "domain_id",
        "domain_display_name",
        "publication_year",
        "available_valid_works",
        "planned_sample_size",
        "sampling_method",
        "seed",
        "api_initial_sample_size",
        "expected_shortfall_risk",
    ]
    return pd.DataFrame(rows, columns=columns)


def print_summary(sample_plan: pd.DataFrame, per_page: int, target_total: int) -> None:
    if sample_plan.empty:
        print("eligible subfields: 0")
        print("planned total works: 0")
        print(f"subfields with planned_sample_size < {target_total}: 0")
        print("estimated API requests: 0")
        return

    planned_by_subfield = sample_plan.groupby("subfield_id")["planned_sample_size"].sum()
    sample_rows = sample_plan[
        (sample_plan["sampling_method"] == "sample_api")
        & (sample_plan["planned_sample_size"] > 0)
    ]
    all_rows = sample_plan[
        (sample_plan["sampling_method"] == "download_all_available")
        & (sample_plan["planned_sample_size"] > 0)
    ]
    estimated_requests = int(
        sum(
            math.ceil(size / min(size, per_page))
            for size in sample_rows["api_initial_sample_size"]
            if size > 0
        )
    ) + int(
        sum(math.ceil(size / per_page) for size in all_rows["planned_sample_size"])
    )

    print(f"eligible subfields: {sample_plan['subfield_id'].nunique()}")
    print(f"planned total works: {int(sample_plan['planned_sample_size'].sum())}")
    print(
        f"subfields with planned_sample_size < {target_total}: "
        f"{int((planned_by_subfield < target_total).sum())}"
    )
    print(f"estimated API requests: {estimated_requests}")
    print("planned works by sampling method:")
    print(
        sample_plan.groupby("sampling_method")["planned_sample_size"]
        .sum()
        .astype(int)
        .to_string()
    )
    print("initial raw API requests by sampling method:")
    print(
        sample_plan.groupby("sampling_method")["api_initial_sample_size"]
        .sum()
        .astype(int)
        .to_string()
    )
    print(
        "cells with expected shortfall risk: "
        f"{int(sample_plan['expected_shortfall_risk'].sum())}"
    )


def main() -> None:
    args = parse_args()
    config = apply_dataset_version(load_config(), args.dataset_version)
    ensure_dirs(config)

    duckdb_path = ROOT / config["storage"]["duckdb_path"]
    per_page = int(config["openalex"].get("per_page", 200))
    paths = extraction_paths(ROOT, config, args.dataset_version)

    corpus_plan = load_parquet(paths.corpus_plan)
    subfield_year_counts = load_parquet(paths.subfield_year_counts)

    sample_plan = build_sample_plan(corpus_plan, subfield_year_counts, config)
    save_parquet(sample_plan, paths.sample_plan)

    con = connect_duckdb(duckdb_path)
    try:
        write_table(
            con,
            versioned_table_name("sample_plan", args.dataset_version),
            sample_plan,
        )
    finally:
        con.close()

    print(f"Wrote {output_path_display(paths.sample_plan, ROOT)}")
    print_summary(sample_plan, per_page, target_total_from_config(config))


if __name__ == "__main__":
    main()
