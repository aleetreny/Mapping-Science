from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.storage import connect_duckdb, ensure_dirs, load_parquet, save_parquet, write_table


def load_config() -> dict[str, Any]:
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_required_parquet(interim_dir: Path, filename: str) -> pd.DataFrame:
    path = interim_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.relative_to(ROOT)}. Run earlier pipeline scripts first."
        )
    return load_parquet(path)


def window_sum(
    df: pd.DataFrame,
    group_cols: list[str],
    start_year: int,
    end_year: int,
    value_col: str,
    output_col: str,
) -> pd.DataFrame:
    mask = df["publication_year"].between(start_year, end_year)
    result = (
        df.loc[mask]
        .groupby(group_cols, dropna=False)[value_col]
        .sum()
        .reset_index()
        .rename(columns={value_col: output_col})
    )
    return result


def add_safe_share(
    df: pd.DataFrame, numerator_col: str, denominator_col: str, output_col: str
) -> None:
    df[output_col] = 0.0
    mask = df[denominator_col] > 0
    df.loc[mask, output_col] = (
        df.loc[mask, numerator_col] / df.loc[mask, denominator_col]
    )


def main() -> None:
    config = load_config()
    ensure_dirs(config)

    interim_dir = ROOT / config["storage"]["interim_dir"]
    duckdb_path = ROOT / config["storage"]["duckdb_path"]
    windows = config["windows"]
    sampling = config["sampling"]

    analysis_start = int(windows["morphology_start_year"])
    analysis_end = int(windows["morphology_end_year"])
    target_sample = int(sampling["target_sample_per_subfield"])
    min_texts = int(sampling["min_valid_texts_per_subfield"])

    subfields = read_required_parquet(interim_dir, "subfields.parquet")
    sub_counts = read_required_parquet(interim_dir, "subfield_year_counts.parquet")
    field_counts = read_required_parquet(interim_dir, "field_year_counts.parquet")
    domain_counts = read_required_parquet(interim_dir, "domain_year_counts.parquet")

    sub_group = ["subfield_id", "field_id", "domain_id"]
    sub_total = window_sum(
        sub_counts,
        sub_group,
        analysis_start,
        analysis_end,
        "n_works_article_preprint",
        "analysis_count_2010_2025",
    )
    sub_text = window_sum(
        sub_counts,
        sub_group,
        analysis_start,
        analysis_end,
        "n_works_article_preprint_en_with_abstract",
        "analysis_text_count_2010_2025",
    )

    field_group = ["field_id", "domain_id"]
    field_total = window_sum(
        field_counts,
        field_group,
        analysis_start,
        analysis_end,
        "n_works_article_preprint",
        "field_analysis_count_2010_2025",
    )

    domain_group = ["domain_id"]
    domain_total = window_sum(
        domain_counts,
        domain_group,
        analysis_start,
        analysis_end,
        "n_works_article_preprint",
        "domain_analysis_count_2010_2025",
    )

    plan = sub_total.merge(sub_text, on=sub_group, how="outer")
    plan = plan.merge(field_total, on=field_group, how="left")
    plan = plan.merge(domain_total, on=domain_group, how="left")

    display_cols = [
        "subfield_id",
        "subfield_display_name",
        "field_id",
        "field_display_name",
        "domain_id",
        "domain_display_name",
    ]
    plan = plan.merge(subfields[display_cols], on=sub_group, how="left")

    numeric_cols = [
        "analysis_count_2010_2025",
        "analysis_text_count_2010_2025",
        "field_analysis_count_2010_2025",
        "domain_analysis_count_2010_2025",
    ]
    for column in numeric_cols:
        plan[column] = plan[column].fillna(0).astype("int64")

    add_safe_share(
        plan,
        "analysis_count_2010_2025",
        "field_analysis_count_2010_2025",
        "analysis_share_within_field",
    )
    add_safe_share(
        plan,
        "analysis_count_2010_2025",
        "domain_analysis_count_2010_2025",
        "analysis_share_within_domain",
    )

    plan["eligible_for_text_corpus"] = (
        (plan["analysis_text_count_2010_2025"] >= min_texts)
        & plan["field_id"].notna()
        & plan["domain_id"].notna()
    )
    plan["planned_sample_size"] = 0
    eligible_mask = plan["eligible_for_text_corpus"]
    plan.loc[eligible_mask, "planned_sample_size"] = plan.loc[
        eligible_mask, "analysis_text_count_2010_2025"
    ].clip(upper=target_sample)

    ordered_cols = [
        "subfield_id",
        "subfield_display_name",
        "field_id",
        "field_display_name",
        "domain_id",
        "domain_display_name",
        "analysis_count_2010_2025",
        "analysis_text_count_2010_2025",
        "field_analysis_count_2010_2025",
        "domain_analysis_count_2010_2025",
        "analysis_share_within_field",
        "analysis_share_within_domain",
        "eligible_for_text_corpus",
        "planned_sample_size",
    ]
    plan = plan[ordered_cols].sort_values("subfield_id").reset_index(drop=True)

    save_parquet(plan, interim_dir / "corpus_plan.parquet")
    con = connect_duckdb(duckdb_path)
    try:
        write_table(con, "corpus_plan", plan)
    finally:
        con.close()

    print(
        "Wrote corpus_plan for the "
        f"{analysis_start}-{analysis_end} analysis period with "
        f"{int(plan['eligible_for_text_corpus'].sum())} eligible subfields."
    )


if __name__ == "__main__":
    main()
