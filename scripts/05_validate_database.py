from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.extraction_versions import (
    add_dataset_version_argument,
    apply_dataset_version,
    extraction_filter_summary,
    extraction_paths,
    output_path_display,
    safe_json_loads,
    target_per_year_from_config,
    versioned_table_name,
    year_min_max,
)
from src.storage import connect_duckdb, ensure_dirs, load_parquet


def load_config() -> dict[str, Any]:
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate OpenAlex extraction outputs.")
    add_dataset_version_argument(parser)
    return parser.parse_args()


def read_table_or_empty(con: duckdb.DuckDBPyConnection | None, table: str) -> pd.DataFrame:
    if con is None:
        return pd.DataFrame()
    try:
        return con.execute(f'SELECT * FROM "{table}"').fetchdf()
    except duckdb.CatalogException:
        return pd.DataFrame()


def read_parquet_or_empty(path: Path) -> pd.DataFrame:
    return load_parquet(path) if path.exists() else pd.DataFrame()


def describe_counts(counts: pd.Series) -> dict[str, Any]:
    if counts.empty:
        return {}
    cleaned = {}
    for key, value in counts.describe().to_dict().items():
        number = float(value)
        cleaned[key] = None if math.isnan(number) else number
    return cleaned


def missing_text_count(df: pd.DataFrame, column: str) -> int:
    if column not in df:
        return len(df)
    values = df[column]
    return int(values.isna().sum() + (values.astype(str).str.strip() == "").sum())


def below_token_threshold_count(df: pd.DataFrame, column: str, threshold: int) -> int:
    if column not in df:
        return 0
    values = pd.to_numeric(df[column], errors="coerce")
    return int((values < int(threshold)).sum())


def load_pipeline_tables(
    config: dict[str, Any],
    dataset_version: str | None,
) -> dict[str, pd.DataFrame]:
    paths = extraction_paths(ROOT, config, dataset_version)
    interim_dir = ROOT / config["storage"]["interim_dir"]
    duckdb_path = ROOT / config["storage"]["duckdb_path"]

    con = connect_duckdb(duckdb_path) if duckdb_path.exists() else None
    try:
        tables = {
            "domains": read_table_or_empty(con, "domains"),
            "fields": read_table_or_empty(con, "fields"),
            "subfields": read_table_or_empty(con, "subfields"),
            "corpus_plan": read_table_or_empty(
                con, versioned_table_name("corpus_plan", dataset_version)
            ),
            "sample_plan": read_table_or_empty(
                con, versioned_table_name("sample_plan", dataset_version)
            ),
            "download_manifest": read_table_or_empty(
                con, versioned_table_name("download_manifest", dataset_version)
            ),
            "works_text": read_table_or_empty(
                con, versioned_table_name("works_text", dataset_version)
            ),
            "analysis_subfields": read_table_or_empty(
                con, versioned_table_name("analysis_subfields", dataset_version)
            ),
        }
    finally:
        if con is not None:
            con.close()

    fallback_paths = {
        "domains": interim_dir / "domains.parquet",
        "fields": interim_dir / "fields.parquet",
        "subfields": interim_dir / "subfields.parquet",
        "corpus_plan": paths.corpus_plan,
        "sample_plan": paths.sample_plan,
        "download_manifest": paths.download_manifest,
        "works_text": paths.works_text,
        "analysis_subfields": paths.analysis_subfields,
    }
    for name, path in fallback_paths.items():
        if tables[name].empty:
            tables[name] = read_parquet_or_empty(path)
    return tables


def downloaded_by_cell(works_text: pd.DataFrame) -> pd.DataFrame:
    columns = ["subfield_id", "publication_year", "downloaded_works"]
    if works_text.empty or not {"subfield_id", "publication_year"}.issubset(works_text.columns):
        return pd.DataFrame(columns=columns)
    works = works_text.copy()
    works["subfield_id"] = works["subfield_id"].astype(str)
    works["publication_year"] = pd.to_numeric(
        works["publication_year"], errors="coerce"
    ).astype("Int64")
    return (
        works.dropna(subset=["publication_year"])
        .groupby(["subfield_id", "publication_year"])
        .size()
        .reset_index(name="downloaded_works")
    )


def build_subfield_year_coverage(
    sample_plan: pd.DataFrame,
    works_text: pd.DataFrame,
) -> pd.DataFrame:
    downloaded = downloaded_by_cell(works_text)
    if sample_plan.empty:
        return downloaded.assign(
            planned_sample_size=0,
            available_valid_works=0,
            shortfall=0,
        )

    planned_cols = [
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
    ]
    planned = sample_plan.copy()
    planned["subfield_id"] = planned["subfield_id"].astype(str)
    planned["publication_year"] = pd.to_numeric(
        planned["publication_year"], errors="coerce"
    ).astype("Int64")
    for column in planned_cols:
        if column not in planned.columns:
            planned[column] = pd.NA
    coverage = planned[planned_cols].merge(
        downloaded,
        on=["subfield_id", "publication_year"],
        how="left",
    )
    coverage["downloaded_works"] = coverage["downloaded_works"].fillna(0).astype("int64")
    coverage["planned_sample_size"] = (
        pd.to_numeric(coverage["planned_sample_size"], errors="coerce")
        .fillna(0)
        .astype("int64")
    )
    coverage["available_valid_works"] = (
        pd.to_numeric(coverage["available_valid_works"], errors="coerce")
        .fillna(0)
        .astype("int64")
    )
    coverage["shortfall"] = (
        coverage["planned_sample_size"] - coverage["downloaded_works"]
    ).clip(lower=0)
    return coverage.sort_values(["subfield_id", "publication_year"]).reset_index(drop=True)


def build_subfield_coverage_summary(coverage: pd.DataFrame, *, target_per_year: int) -> pd.DataFrame:
    if coverage.empty:
        return pd.DataFrame()
    summary = (
        coverage.groupby(
            [
                "subfield_id",
                "subfield_display_name",
                "field_id",
                "field_display_name",
                "domain_id",
                "domain_display_name",
            ],
            dropna=False,
        )
        .agg(
            planned_works=("planned_sample_size", "sum"),
            downloaded_works=("downloaded_works", "sum"),
            shortfall=("shortfall", "sum"),
            n_years_planned=("planned_sample_size", lambda values: int((values > 0).sum())),
            n_years_reaching_target=(
                "downloaded_works",
                lambda values: int((values >= int(target_per_year)).sum()),
            ),
            n_years_below_target=(
                "downloaded_works",
                lambda values: int((values < int(target_per_year)).sum()),
            ),
        )
        .reset_index()
    )
    return summary


def build_year_coverage_summary(coverage: pd.DataFrame, *, target_per_year: int) -> pd.DataFrame:
    if coverage.empty:
        return pd.DataFrame()
    return (
        coverage.groupby("publication_year")
        .agg(
            planned_works=("planned_sample_size", "sum"),
            downloaded_works=("downloaded_works", "sum"),
            shortfall=("shortfall", "sum"),
            planned_cells=("planned_sample_size", lambda values: int((values > 0).sum())),
            cells_reaching_target=(
                "downloaded_works",
                lambda values: int((values >= int(target_per_year)).sum()),
            ),
            cells_below_target=(
                "downloaded_works",
                lambda values: int((values < int(target_per_year)).sum()),
            ),
        )
        .reset_index()
        .sort_values("publication_year")
    )


def validate_works_text(
    works_text: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    filters = config["filters"]
    year_min, year_max = year_min_max(config)
    allowed_types = set(filters["work_types"])
    language = filters["language"]
    years = (
        pd.to_numeric(works_text["publication_year"], errors="coerce")
        if "publication_year" in works_text
        else pd.Series(dtype=float)
    )
    topics_valid = (
        works_text["topics_json"].apply(safe_json_loads)
        if "topics_json" in works_text
        else pd.Series([False] * len(works_text))
    )
    summary = {
        "year_min_expected": year_min,
        "year_max_expected": year_max,
        "observed_year_min": int(years.min()) if not years.dropna().empty else None,
        "observed_year_max": int(years.max()) if not years.dropna().empty else None,
        "rows_outside_expected_year_window": int(
            (~years.between(year_min, year_max, inclusive="both")).sum()
        )
        if len(years)
        else 0,
        "rows_in_2025_or_2026": int(years.isin([2025, 2026]).sum()) if len(years) else 0,
        "duplicate_work_ids": int(works_text["work_id"].duplicated().sum())
        if "work_id" in works_text
        else len(works_text),
        "missing_work_id": missing_text_count(works_text, "work_id"),
        "missing_subfield_id": missing_text_count(works_text, "subfield_id"),
        "missing_field_id": missing_text_count(works_text, "field_id"),
        "missing_domain_id": missing_text_count(works_text, "domain_id"),
        "missing_primary_topic_id": missing_text_count(works_text, "primary_topic_id"),
        "missing_primary_topic_display_name": missing_text_count(
            works_text, "primary_topic_display_name"
        ),
        "missing_topics_json": missing_text_count(works_text, "topics_json"),
        "unparsable_topics_json": int((~topics_valid).sum()) if len(topics_valid) else 0,
        "missing_title": missing_text_count(works_text, "title"),
        "missing_abstract": missing_text_count(works_text, "abstract"),
        "short_title_rows": below_token_threshold_count(
            works_text, "title_token_count", int(filters["min_title_tokens"])
        ),
        "short_abstract_rows": below_token_threshold_count(
            works_text, "abstract_token_count", int(filters["min_abstract_tokens"])
        ),
        "non_english_rows": int((works_text["language"] != language).sum())
        if "language" in works_text
        else len(works_text),
        "disallowed_type_rows": int((~works_text["type"].isin(allowed_types)).sum())
        if "type" in works_text
        else len(works_text),
    }
    return summary


def validation_failed(checks: dict[str, Any]) -> bool:
    critical_keys = [
        "rows_outside_expected_year_window",
        "rows_in_2025_or_2026",
        "duplicate_work_ids",
        "missing_work_id",
        "missing_subfield_id",
        "missing_field_id",
        "missing_domain_id",
        "missing_primary_topic_id",
        "missing_topics_json",
        "unparsable_topics_json",
        "missing_title",
        "missing_abstract",
        "short_title_rows",
        "short_abstract_rows",
        "non_english_rows",
        "disallowed_type_rows",
    ]
    return any(int(checks.get(key) or 0) > 0 for key in critical_keys)


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def build_validation_summary(
    tables: dict[str, pd.DataFrame],
    config: dict[str, Any],
    dataset_version: str | None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    works_text = tables["works_text"]
    sample_plan = tables["sample_plan"]
    manifest = tables["download_manifest"]
    coverage = build_subfield_year_coverage(sample_plan, works_text)
    target_per_year = target_per_year_from_config(config)
    subfield_summary = build_subfield_coverage_summary(
        coverage,
        target_per_year=target_per_year,
    )
    year_summary = build_year_coverage_summary(
        coverage,
        target_per_year=target_per_year,
    )
    checks = validate_works_text(works_text, config)
    planned_total = (
        int(sample_plan["planned_sample_size"].sum())
        if "planned_sample_size" in sample_plan
        else 0
    )
    downloaded_total = int(len(works_text))
    shortfall_total = int(coverage["shortfall"].sum()) if "shortfall" in coverage else 0
    works_per_subfield = (
        works_text.groupby("subfield_id").size()
        if "subfield_id" in works_text and not works_text.empty
        else pd.Series(dtype=int)
    )
    works_per_year = (
        works_text.groupby("publication_year").size()
        if "publication_year" in works_text and not works_text.empty
        else pd.Series(dtype=int)
    )
    manifest_status_counts = (
        manifest["status"].value_counts(dropna=False).astype(int)
        if "status" in manifest
        else pd.Series(dtype=int)
    )
    summary = {
        "dataset_version": dataset_version,
        "filters": extraction_filter_summary(config),
        "sampling": {
            "target_per_year_per_subfield": target_per_year,
            "target_sample_per_subfield": int(
                config["sampling"]["target_sample_per_subfield"]
            ),
            "max_sample_per_subfield": int(
                config["sampling"]["max_sample_per_subfield"]
            ),
            "oversample_factor": float(config["sampling"].get("oversample_factor", 1.0)),
            "max_backfill_rounds": int(config["sampling"].get("max_backfill_rounds", 0)),
        },
        "n_rows": downloaded_total,
        "n_subfields": int(works_text["subfield_id"].nunique())
        if "subfield_id" in works_text
        else 0,
        "total_planned_works": planned_total,
        "total_downloaded_works": downloaded_total,
        "total_shortfall": shortfall_total,
        "subfield_year_cells_below_target": int(
            (coverage["downloaded_works"] < target_per_year).sum()
        )
        if "downloaded_works" in coverage
        else 0,
        "manifest_status_counts": {
            str(key): int(value) for key, value in manifest_status_counts.to_dict().items()
        },
        "works_per_subfield_distribution": describe_counts(works_per_subfield),
        "works_per_year": {str(key): int(value) for key, value in works_per_year.to_dict().items()},
        "checks": checks,
        "validation_failed": validation_failed(checks),
    }
    return summary, coverage, subfield_summary, year_summary


def validation_report_markdown(
    summary: dict[str, Any],
    paths_display: dict[str, str],
) -> str:
    checks = summary["checks"]
    lines = [
        "# OpenAlex Extraction Validation",
        "",
        f"- Dataset version: {summary.get('dataset_version') or 'canonical'}",
        f"- Rows: {summary['n_rows']}",
        f"- Subfields: {summary['n_subfields']}",
        f"- Planned works: {summary['total_planned_works']}",
        f"- Downloaded works: {summary['total_downloaded_works']}",
        f"- Total shortfall: {summary['total_shortfall']}",
        f"- Validation failed: {summary['validation_failed']}",
        "",
        "## Year Checks",
        "",
        f"- Expected year min: {checks['year_min_expected']}",
        f"- Expected year max: {checks['year_max_expected']}",
        f"- Observed year min: {checks['observed_year_min']}",
        f"- Observed year max: {checks['observed_year_max']}",
        f"- Rows outside expected window: {checks['rows_outside_expected_year_window']}",
        f"- Rows in 2025 or 2026: {checks['rows_in_2025_or_2026']}",
        "",
        "## Corpus Checks",
        "",
        f"- Duplicate work IDs: {checks['duplicate_work_ids']}",
        f"- Missing subfield IDs: {checks['missing_subfield_id']}",
        f"- Missing field IDs: {checks['missing_field_id']}",
        f"- Missing domain IDs: {checks['missing_domain_id']}",
        f"- Missing primary topic IDs: {checks['missing_primary_topic_id']}",
        f"- Missing topics_json: {checks['missing_topics_json']}",
        f"- Unparsable topics_json: {checks['unparsable_topics_json']}",
        f"- Missing titles: {checks['missing_title']}",
        f"- Missing abstracts: {checks['missing_abstract']}",
        f"- Short titles: {checks['short_title_rows']}",
        f"- Short abstracts: {checks['short_abstract_rows']}",
        f"- Non-English rows: {checks['non_english_rows']}",
        f"- Disallowed type rows: {checks['disallowed_type_rows']}",
        "",
        "## Diagnostic Files",
        "",
        f"- Subfield-year coverage: {paths_display['subfield_year_download_coverage']}",
        f"- Subfield coverage summary: {paths_display['subfield_coverage_summary']}",
        f"- Year coverage summary: {paths_display['year_coverage_summary']}",
    ]
    return "\n".join(lines) + "\n"


def write_validation_outputs(
    *,
    summary: dict[str, Any],
    coverage: pd.DataFrame,
    subfield_summary: pd.DataFrame,
    year_summary: pd.DataFrame,
    config: dict[str, Any],
    dataset_version: str | None,
) -> None:
    paths = extraction_paths(ROOT, config, dataset_version)
    paths.validation_report.parent.mkdir(parents=True, exist_ok=True)
    paths.validation_summary.parent.mkdir(parents=True, exist_ok=True)
    paths.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(paths.subfield_year_download_coverage, index=False)
    subfield_summary.to_csv(paths.subfield_coverage_summary, index=False)
    year_summary.to_csv(paths.year_coverage_summary, index=False)

    paths_display = {
        "subfield_year_download_coverage": output_path_display(
            paths.subfield_year_download_coverage, ROOT
        ),
        "subfield_coverage_summary": output_path_display(
            paths.subfield_coverage_summary, ROOT
        ),
        "year_coverage_summary": output_path_display(paths.year_coverage_summary, ROOT),
    }
    report = validation_report_markdown(summary, paths_display)
    paths.validation_report.write_text(report, encoding="utf-8")
    paths.extraction_summary.write_text(report, encoding="utf-8")
    paths.validation_summary.write_text(
        json.dumps(make_json_safe(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Wrote {output_path_display(paths.validation_report, ROOT)}")
    print(f"Wrote {output_path_display(paths.validation_summary, ROOT)}")
    print(f"Wrote {output_path_display(paths.diagnostics_dir, ROOT)}")


def main() -> None:
    args = parse_args()
    config = apply_dataset_version(load_config(), args.dataset_version)
    ensure_dirs(config)

    tables = load_pipeline_tables(config, args.dataset_version)
    summary, coverage, subfield_summary, year_summary = build_validation_summary(
        tables,
        config,
        args.dataset_version,
    )
    write_validation_outputs(
        summary=summary,
        coverage=coverage,
        subfield_summary=subfield_summary,
        year_summary=year_summary,
        config=config,
        dataset_version=args.dataset_version,
    )


if __name__ == "__main__":
    main()
