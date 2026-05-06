from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import build_analysis_subfields
from src.storage import connect_duckdb, ensure_dirs, load_parquet, save_parquet, write_table


def load_config() -> dict[str, Any]:
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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
    config = load_config()
    ensure_dirs(config)

    processed_dir = ROOT / config["storage"]["processed_dir"]
    interim_dir = ROOT / config["storage"]["interim_dir"]
    duckdb_path = ROOT / config["storage"]["duckdb_path"]

    works_text = read_required_parquet(
        processed_dir / "works_text.parquet",
        "Missing data/processed/works_text.parquet. Run the corpus download first.",
    )
    plan = read_plan(interim_dir)
    analysis_subfields = build_analysis_subfields(works_text, plan)

    save_parquet(analysis_subfields, processed_dir / "analysis_subfields.parquet")

    con = connect_duckdb(duckdb_path)
    try:
        write_table(con, "analysis_subfields", analysis_subfields)
    finally:
        con.close()

    print_summary(analysis_subfields)


if __name__ == "__main__":
    main()
