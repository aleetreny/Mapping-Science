from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import build_analysis_subfields, build_versioned_analysis_subfields
from src.extraction_versions import (
    add_dataset_version_argument,
    apply_dataset_version,
    extraction_paths,
    n_downloaded_works_column,
    output_path_display,
    versioned_table_name,
)
from src.storage import connect_duckdb, ensure_dirs, load_parquet, save_parquet, write_table


def load_config() -> dict[str, Any]:
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build analysis subfield table.")
    add_dataset_version_argument(parser)
    return parser.parse_args()


def read_required_parquet(path: Path, message: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(message)
    return load_parquet(path)


def read_plan(interim_dir: Path) -> pd.DataFrame:
    sample_plan_path = interim_dir / "sample_plan.parquet"
    corpus_plan_path = interim_dir / "corpus_plan.parquet"
    if sample_plan_path.exists():
        return load_parquet(sample_plan_path)
    if corpus_plan_path.exists():
        corpus_plan = load_parquet(corpus_plan_path)
        return corpus_plan.rename(columns={"planned_sample_size": "planned_works"})
    raise FileNotFoundError(
        "Missing data/interim/sample_plan.parquet or data/interim/corpus_plan.parquet."
    )


def print_summary(analysis_subfields: pd.DataFrame) -> None:
    main = analysis_subfields["main_analysis_eligible_2500"]
    robust = analysis_subfields["robustness_eligible_500"]
    total_works_main = int(analysis_subfields.loc[main, "n_valid_works"].sum())
    total_works_excluded_main = int(analysis_subfields.loc[~main, "n_valid_works"].sum())

    print(f"total subfields: {len(analysis_subfields)}")
    print(f"main analysis subfields >=2500: {int(main.sum())}")
    print(f"robustness subfields >=500: {int(robust.sum())}")
    print(f"excluded from main analysis: {int((~main).sum())}")
    print(f"excluded from robustness: {int((~robust).sum())}")
    print(f"total works in main analysis: {total_works_main}")
    print(f"total works excluded from main analysis: {total_works_excluded_main}")


def main() -> None:
    args = parse_args()
    config = apply_dataset_version(load_config(), args.dataset_version)
    ensure_dirs(config)

    duckdb_path = ROOT / config["storage"]["duckdb_path"]
    paths = extraction_paths(ROOT, config, args.dataset_version)

    works_text = read_required_parquet(
        paths.works_text,
        f"Missing {output_path_display(paths.works_text, ROOT)}. "
        "Run the corpus download first.",
    )
    if args.dataset_version:
        sample_plan = read_required_parquet(
            paths.sample_plan,
            f"Missing {output_path_display(paths.sample_plan, ROOT)}. "
            "Run scripts/03_build_sample_plan.py first.",
        )
        corpus_plan = read_required_parquet(
            paths.corpus_plan,
            f"Missing {output_path_display(paths.corpus_plan, ROOT)}. "
            "Run scripts/02_build_corpus_plan.py first.",
        )
        analysis_subfields = build_versioned_analysis_subfields(
            works_text,
            sample_plan,
            corpus_plan,
            config,
        )
    else:
        interim_dir = ROOT / config["storage"]["interim_dir"]
        plan = read_plan(interim_dir)
        analysis_subfields = build_analysis_subfields(works_text, plan)

    save_parquet(analysis_subfields, paths.analysis_subfields)

    con = connect_duckdb(duckdb_path)
    try:
        write_table(
            con,
            versioned_table_name("analysis_subfields", args.dataset_version),
            analysis_subfields,
        )
    finally:
        con.close()

    print(f"Wrote {output_path_display(paths.analysis_subfields, ROOT)}")
    if args.dataset_version:
        downloaded_col = n_downloaded_works_column(config)
        print(f"total subfields: {len(analysis_subfields)}")
        print(
            "eligible full-period 10,000: "
            f"{int(analysis_subfields['eligible_10000_full_period'].sum())}"
        )
        print(
            "eligible minimum 5,000: "
            f"{int(analysis_subfields['eligible_min_5000_full_period'].sum())}"
        )
        print(
            "eligible temporal exploration: "
            f"{int(analysis_subfields['eligible_for_temporal_5year_exploration'].sum())}"
        )
        print(f"total downloaded works: {int(analysis_subfields[downloaded_col].sum())}")
    else:
        print_summary(analysis_subfields)


if __name__ == "__main__":
    main()
