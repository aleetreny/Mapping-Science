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

from src.extraction_versions import (
    add_dataset_version_argument,
    apply_dataset_version,
    extraction_paths,
    n_valid_works_column,
    output_path_display,
    period_label,
    target_per_year_from_config,
    versioned_table_name,
)
from src.storage import connect_duckdb, ensure_dirs, load_parquet, save_parquet, write_table


def load_config() -> dict[str, Any]:
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OpenAlex corpus plan.")
    add_dataset_version_argument(parser)
    return parser.parse_args()


def read_required_parquet(path: Path) -> pd.DataFrame:
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
    return (
        df.loc[mask]
        .groupby(group_cols, dropna=False)[value_col]
        .sum()
        .reset_index()
        .rename(columns={value_col: output_col})
    )


def add_safe_share(
    df: pd.DataFrame, numerator_col: str, denominator_col: str, output_col: str
) -> None:
    df[output_col] = 0.0
    mask = df[denominator_col] > 0
    df.loc[mask, output_col] = (
        df.loc[mask, numerator_col] / df.loc[mask, denominator_col]
    )


def annual_coverage_summary(
    sub_counts: pd.DataFrame,
    *,
    target_per_year: int,
) -> pd.DataFrame:
    valid = sub_counts.copy()
    valid_count = pd.to_numeric(
        valid["n_works_article_preprint_en_with_abstract"],
        errors="coerce",
    ).fillna(0)
    valid["valid_count"] = valid_count
    summary = (
        valid.groupby("subfield_id")["valid_count"]
        .agg(
            n_years_with_any_valid_works=lambda values: int((values > 0).sum()),
            n_years_reaching_400=lambda values: int(
                (values >= int(target_per_year)).sum()
            ),
            min_annual_valid_works="min",
            median_annual_valid_works="median",
        )
        .reset_index()
    )
    summary["min_annual_valid_works"] = (
        summary["min_annual_valid_works"].fillna(0).astype("int64")
    )
    return summary


def corpus_plan_column_names(
    config: dict[str, Any],
    *,
    dataset_version: str | None,
) -> dict[str, str]:
    label = period_label(config)
    return {
        "total": f"analysis_count_{label}",
        "valid": n_valid_works_column(config)
        if dataset_version
        else f"analysis_text_count_{label}",
        "field_total": f"field_analysis_count_{label}",
        "domain_total": f"domain_analysis_count_{label}",
    }


def build_corpus_plan(
    *,
    subfields: pd.DataFrame,
    sub_counts: pd.DataFrame,
    field_counts: pd.DataFrame,
    domain_counts: pd.DataFrame,
    config: dict[str, Any],
    dataset_version: str | None,
) -> pd.DataFrame:
    windows = config["windows"]
    sampling = config["sampling"]
    analysis_start = int(windows["morphology_start_year"])
    analysis_end = int(windows["morphology_end_year"])
    target_sample = int(sampling["target_sample_per_subfield"])
    target_per_year = target_per_year_from_config(config)
    min_texts = int(sampling["min_valid_texts_per_subfield"])
    names = corpus_plan_column_names(config, dataset_version=dataset_version)

    sub_group = ["subfield_id", "field_id", "domain_id"]
    field_group = ["field_id", "domain_id"]
    domain_group = ["domain_id"]
    sub_total = window_sum(
        sub_counts,
        sub_group,
        analysis_start,
        analysis_end,
        "n_works_article_preprint",
        names["total"],
    )
    sub_text = window_sum(
        sub_counts,
        sub_group,
        analysis_start,
        analysis_end,
        "n_works_article_preprint_en_with_abstract",
        names["valid"],
    )
    field_total = window_sum(
        field_counts,
        field_group,
        analysis_start,
        analysis_end,
        "n_works_article_preprint",
        names["field_total"],
    )
    domain_total = window_sum(
        domain_counts,
        domain_group,
        analysis_start,
        analysis_end,
        "n_works_article_preprint",
        names["domain_total"],
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

    for column in names.values():
        plan[column] = plan[column].fillna(0).astype("int64")

    add_safe_share(
        plan,
        names["total"],
        names["field_total"],
        "analysis_share_within_field",
    )
    add_safe_share(
        plan,
        names["total"],
        names["domain_total"],
        "analysis_share_within_domain",
    )

    plan["eligible_for_text_corpus"] = (
        (plan[names["valid"]] >= min_texts)
        & plan["field_id"].notna()
        & plan["domain_id"].notna()
    )
    plan["reason_if_ineligible"] = ""
    plan.loc[plan[names["valid"]] < min_texts, "reason_if_ineligible"] = (
        f"below_{min_texts}_valid_texts"
    )
    plan.loc[plan["field_id"].isna(), "reason_if_ineligible"] = "missing_field_id"
    plan.loc[plan["domain_id"].isna(), "reason_if_ineligible"] = "missing_domain_id"

    plan["planned_sample_size"] = 0
    eligible_mask = plan["eligible_for_text_corpus"]
    plan.loc[eligible_mask, "planned_sample_size"] = plan.loc[
        eligible_mask, names["valid"]
    ].clip(upper=target_sample)

    ordered_cols = [
        "subfield_id",
        "subfield_display_name",
        "field_id",
        "field_display_name",
        "domain_id",
        "domain_display_name",
        names["total"],
        names["valid"],
        names["field_total"],
        names["domain_total"],
        "analysis_share_within_field",
        "analysis_share_within_domain",
        "eligible_for_text_corpus",
        "planned_sample_size",
    ]
    if dataset_version:
        coverage = annual_coverage_summary(
            sub_counts,
            target_per_year=target_per_year,
        )
        plan = plan.merge(coverage, on="subfield_id", how="left")
        plan["n_years_with_any_valid_works"] = (
            plan["n_years_with_any_valid_works"].fillna(0).astype("int64")
        )
        plan["n_years_reaching_400"] = (
            plan["n_years_reaching_400"].fillna(0).astype("int64")
        )
        plan["min_annual_valid_works"] = (
            plan["min_annual_valid_works"].fillna(0).astype("int64")
        )
        plan["median_annual_valid_works"] = plan[
            "median_annual_valid_works"
        ].fillna(0.0)
        ordered_cols.insert(14, "reason_if_ineligible")
        ordered_cols.extend(
            [
                "n_years_with_any_valid_works",
                "n_years_reaching_400",
                "min_annual_valid_works",
                "median_annual_valid_works",
            ]
        )

    return plan[ordered_cols].sort_values("subfield_id").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    config = apply_dataset_version(load_config(), args.dataset_version)
    ensure_dirs(config)

    interim_dir = ROOT / config["storage"]["interim_dir"]
    duckdb_path = ROOT / config["storage"]["duckdb_path"]
    paths = extraction_paths(ROOT, config, args.dataset_version)

    subfields = read_required_parquet(interim_dir / "subfields.parquet")
    sub_counts = read_required_parquet(paths.subfield_year_counts)
    field_counts = read_required_parquet(paths.field_year_counts)
    domain_counts = read_required_parquet(paths.domain_year_counts)
    plan = build_corpus_plan(
        subfields=subfields,
        sub_counts=sub_counts,
        field_counts=field_counts,
        domain_counts=domain_counts,
        config=config,
        dataset_version=args.dataset_version,
    )

    save_parquet(plan, paths.corpus_plan)
    con = connect_duckdb(duckdb_path)
    try:
        write_table(
            con,
            versioned_table_name("corpus_plan", args.dataset_version),
            plan,
        )
    finally:
        con.close()

    windows = config["windows"]
    print(
        f"Wrote {output_path_display(paths.corpus_plan, ROOT)} for the "
        f"{windows['morphology_start_year']}-{windows['morphology_end_year']} "
        "analysis period with "
        f"{int(plan['eligible_for_text_corpus'].sum())} eligible subfields."
    )


if __name__ == "__main__":
    main()
