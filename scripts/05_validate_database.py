from __future__ import annotations

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

from src.storage import connect_duckdb, ensure_dirs, load_parquet


def load_config() -> dict[str, Any]:
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_table_or_empty(con: duckdb.DuckDBPyConnection | None, table: str) -> pd.DataFrame:
    if con is None:
        return pd.DataFrame()
    try:
        return con.execute(f'SELECT * FROM "{table}"').fetchdf()
    except duckdb.CatalogException:
        return pd.DataFrame()


def read_parquet_or_empty(path: Path) -> pd.DataFrame:
    if path.exists():
        return load_parquet(path)
    return pd.DataFrame()


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
    return int(df[column].isna().sum() + (df[column] == "").sum())


def describe_numeric_column(df: pd.DataFrame, column: str) -> dict[str, Any]:
    if column not in df or df.empty:
        return {}
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return describe_counts(values)


def bytes_to_mb(value: int) -> float:
    return round(value / (1024 * 1024), 2)


def load_pipeline_tables(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    interim_dir = ROOT / config["storage"]["interim_dir"]
    processed_dir = ROOT / config["storage"]["processed_dir"]
    duckdb_path = ROOT / config["storage"]["duckdb_path"]

    con = connect_duckdb(duckdb_path) if duckdb_path.exists() else None
    try:
        tables = {
            "domains": read_table_or_empty(con, "domains"),
            "fields": read_table_or_empty(con, "fields"),
            "subfields": read_table_or_empty(con, "subfields"),
            "corpus_plan": read_table_or_empty(con, "corpus_plan"),
            "sample_plan": read_table_or_empty(con, "sample_plan"),
            "download_manifest": read_table_or_empty(con, "download_manifest"),
            "works_text": read_table_or_empty(con, "works_text"),
            "analysis_subfields": read_table_or_empty(con, "analysis_subfields"),
        }
    finally:
        if con is not None:
            con.close()

    fallback_paths = {
        "domains": interim_dir / "domains.parquet",
        "fields": interim_dir / "fields.parquet",
        "subfields": interim_dir / "subfields.parquet",
        "corpus_plan": interim_dir / "corpus_plan.parquet",
        "sample_plan": interim_dir / "sample_plan.parquet",
        "download_manifest": interim_dir / "download_manifest.parquet",
        "works_text": processed_dir / "works_text.parquet",
        "analysis_subfields": processed_dir / "analysis_subfields.parquet",
    }
    for name, path in fallback_paths.items():
        if tables[name].empty:
            tables[name] = read_parquet_or_empty(path)
    return tables


def bucket_counts(works_per_subfield: pd.Series) -> dict[str, int]:
    return {
        "subfields_with_3000_valid_works": int((works_per_subfield >= 3000).sum())
        if not works_per_subfield.empty
        else 0,
        "subfields_with_2500_2999_valid_works": int(
            ((works_per_subfield >= 2500) & (works_per_subfield < 3000)).sum()
        )
        if not works_per_subfield.empty
        else 0,
        "subfields_with_500_2499_valid_works": int(
            ((works_per_subfield >= 500) & (works_per_subfield < 2500)).sum()
        )
        if not works_per_subfield.empty
        else 0,
        "subfields_below_500_valid_works": int((works_per_subfield < 500).sum())
        if not works_per_subfield.empty
        else 0,
    }


def analysis_eligibility_counts(analysis_subfields: pd.DataFrame) -> dict[str, int]:
    required = {
        "n_valid_works",
        "main_analysis_eligible_2500",
        "robustness_eligible_500",
    }
    if analysis_subfields.empty or not required.issubset(analysis_subfields.columns):
        return {}

    n_valid_works = pd.to_numeric(
        analysis_subfields["n_valid_works"], errors="coerce"
    ).fillna(0)
    main = analysis_subfields["main_analysis_eligible_2500"].fillna(False).astype(bool)
    robust = analysis_subfields["robustness_eligible_500"].fillna(False).astype(bool)

    return {
        "main_analysis_subfields_2500": int(main.sum()),
        "robustness_subfields_500": int(robust.sum()),
        "main_analysis_works": int(n_valid_works[main].sum()),
        "excluded_from_main_analysis_subfields": int((~main).sum()),
        "excluded_from_main_analysis_works": int(n_valid_works[~main].sum()),
        "excluded_from_robustness_subfields": int((~robust).sum()),
        "excluded_from_robustness_works": int(n_valid_works[~robust].sum()),
    }


def main() -> None:
    config = load_config()
    ensure_dirs(config)

    interim_dir = ROOT / config["storage"]["interim_dir"]
    tables = load_pipeline_tables(config)
    domains = tables["domains"]
    fields = tables["fields"]
    subfields = tables["subfields"]
    corpus_plan = tables["corpus_plan"]
    sample_plan = tables["sample_plan"]
    manifest = tables["download_manifest"]
    works_text = tables["works_text"]
    analysis_subfields = tables["analysis_subfields"]

    n_works = len(works_text)
    n_eligible = (
        int(corpus_plan["eligible_for_text_corpus"].sum())
        if "eligible_for_text_corpus" in corpus_plan
        else 0
    )
    total_planned_works = (
        int(sample_plan["planned_sample_size"].sum())
        if "planned_sample_size" in sample_plan
        else 0
    )
    downloaded_ratio = (
        n_works / total_planned_works if total_planned_works > 0 else None
    )

    planned_by_method = (
        sample_plan.groupby("sampling_method")["planned_sample_size"].sum().astype(int)
        if {"sampling_method", "planned_sample_size"}.issubset(sample_plan.columns)
        else pd.Series(dtype=int)
    )

    works_per_subfield = (
        works_text.groupby("subfield_id").size() if not works_text.empty else pd.Series(dtype=int)
    )
    works_per_field = (
        works_text.groupby("field_id").size() if not works_text.empty else pd.Series(dtype=int)
    )
    works_per_year = (
        works_text.groupby("publication_year").size()
        if not works_text.empty
        else pd.Series(dtype=int)
    )

    full_planned_by_subfield = (
        sample_plan.groupby("subfield_id")["planned_sample_size"].sum().astype(int)
        if {"subfield_id", "planned_sample_size"}.issubset(sample_plan.columns)
        else pd.Series(dtype=int)
    )
    manifest_planned_by_subfield = (
        manifest.groupby("subfield_id")["planned_sample_size"].sum().astype(int)
        if {"subfield_id", "planned_sample_size"}.issubset(manifest.columns)
        else pd.Series(dtype=int)
    )
    assessed_planned_by_subfield = (
        manifest_planned_by_subfield
        if not manifest_planned_by_subfield.empty
        else full_planned_by_subfield
    )
    manifest_planned_works = int(manifest_planned_by_subfield.sum()) if not manifest_planned_by_subfield.empty else 0
    manifest_downloaded_ratio = (
        n_works / manifest_planned_works if manifest_planned_works > 0 else None
    )
    planned_vs_kept = pd.DataFrame(
        {
            "planned_works": assessed_planned_by_subfield,
            "kept_works": works_per_subfield,
        }
    ).fillna(0)
    if not planned_vs_kept.empty:
        planned_vs_kept["planned_works"] = planned_vs_kept["planned_works"].astype(int)
        planned_vs_kept["kept_works"] = planned_vs_kept["kept_works"].astype(int)
        planned_vs_kept["shortfall"] = (
            planned_vs_kept["planned_works"] - planned_vs_kept["kept_works"]
        ).clip(lower=0)

    if not works_text.empty and not sample_plan.empty:
        method_lookup = sample_plan[
            ["subfield_id", "publication_year", "sampling_method"]
        ].drop_duplicates()
        works_with_method = works_text.merge(
            method_lookup, on=["subfield_id", "publication_year"], how="left"
        )
        kept_by_method = works_with_method.groupby("sampling_method").size().astype(int)
    else:
        kept_by_method = pd.Series(dtype=int)

    manifest_status_counts = (
        manifest["status"].value_counts(dropna=False).astype(int)
        if "status" in manifest
        else pd.Series(dtype=int)
    )
    retention_rate = None
    if {"raw_returned_works", "valid_after_local_filter"}.issubset(manifest.columns):
        raw_total = int(manifest["raw_returned_works"].sum())
        valid_total = int(manifest["valid_after_local_filter"].sum())
        retention_rate = valid_total / raw_total if raw_total > 0 else None

    duplicate_work_ids = (
        int(works_text["work_id"].duplicated().sum()) if "work_id" in works_text else 0
    )
    missing_titles = missing_text_count(works_text, "title")
    missing_abstracts = missing_text_count(works_text, "abstract")
    missing_primary_topic_id = missing_text_count(works_text, "primary_topic_id")
    missing_topics_json = missing_text_count(works_text, "topics_json")
    language_distribution = (
        works_text["language"].value_counts(dropna=False).to_dict()
        if "language" in works_text
        else {}
    )
    type_distribution = (
        works_text["type"].value_counts(dropna=False).to_dict()
        if "type" in works_text
        else {}
    )

    worst_shortfalls = []
    planned_vs_kept_records = {}
    if not planned_vs_kept.empty:
        worst_shortfalls = (
            planned_vs_kept.sort_values("shortfall", ascending=False)
            .head(20)
            .reset_index()
            .rename(columns={"index": "subfield_id"})
            .to_dict(orient="records")
        )
        planned_vs_kept_records = {
            str(index): {
                "planned_works": int(row["planned_works"]),
                "kept_works": int(row["kept_works"]),
                "shortfall": int(row["shortfall"]),
            }
            for index, row in planned_vs_kept.iterrows()
        }

    embedding_float32_mb = bytes_to_mb(n_works * 768 * 4)
    embedding_float16_mb = bytes_to_mb(n_works * 768 * 2)
    buckets = bucket_counts(works_per_subfield)
    eligibility_counts = analysis_eligibility_counts(analysis_subfields)

    summary = {
        "n_domains": len(domains),
        "n_fields": len(fields),
        "n_subfields": len(subfields),
        "total_eligible_subfields": n_eligible,
        "total_planned_works": total_planned_works,
        "manifest_planned_works": manifest_planned_works,
        "total_downloaded_valid_works": n_works,
        "downloaded_valid_works_over_planned_works": downloaded_ratio,
        "downloaded_valid_works_over_manifest_planned_works": manifest_downloaded_ratio,
        **buckets,
        **eligibility_counts,
        "planned_vs_kept_per_subfield": planned_vs_kept_records,
        "worst_shortfalls": worst_shortfalls,
        "planned_works_by_sampling_method": {
            str(k): int(v) for k, v in planned_by_method.to_dict().items()
        },
        "kept_works_by_sampling_method": {
            str(k): int(v) for k, v in kept_by_method.to_dict().items()
        },
        "manifest_status_counts": {
            str(k): int(v) for k, v in manifest_status_counts.to_dict().items()
        },
        "average_local_validation_retention_rate": retention_rate,
        "duplicate_work_ids": duplicate_work_ids,
        "missing_titles": missing_titles,
        "missing_abstracts": missing_abstracts,
        "missing_primary_topic_id": missing_primary_topic_id,
        "missing_topics_json": missing_topics_json,
        "title_token_count_distribution": describe_numeric_column(
            works_text, "title_token_count"
        ),
        "abstract_token_count_distribution": describe_numeric_column(
            works_text, "abstract_token_count"
        ),
        "text_token_count_distribution": describe_numeric_column(
            works_text, "text_token_count"
        ),
        "works_per_subfield_distribution": describe_counts(works_per_subfield),
        "works_per_field_distribution": describe_counts(works_per_field),
        "works_per_year": {str(k): int(v) for k, v in works_per_year.to_dict().items()},
        "language_distribution": {str(k): int(v) for k, v in language_distribution.items()},
        "type_distribution": {str(k): int(v) for k, v in type_distribution.items()},
        "estimated_embedding_size_mb_float32_768d": embedding_float32_mb,
        "estimated_embedding_size_mb_float16_768d": embedding_float16_mb,
    }

    analysis_report_lines = []
    if eligibility_counts:
        analysis_report_lines = [
            "",
            "## Analysis Eligibility",
            "",
            f"- Main analysis subfields >=2,500: {eligibility_counts['main_analysis_subfields_2500']}",
            f"- Robustness subfields >=500: {eligibility_counts['robustness_subfields_500']}",
            f"- Main analysis works: {eligibility_counts['main_analysis_works']}",
            f"- Excluded from main analysis subfields: {eligibility_counts['excluded_from_main_analysis_subfields']}",
            f"- Excluded from main analysis works: {eligibility_counts['excluded_from_main_analysis_works']}",
            f"- Excluded from robustness subfields: {eligibility_counts['excluded_from_robustness_subfields']}",
            f"- Excluded from robustness works: {eligibility_counts['excluded_from_robustness_works']}",
        ]

    report_lines = [
        "# Validation Report",
        "",
        f"- Domains: {summary['n_domains']}",
        f"- Fields: {summary['n_fields']}",
        f"- Subfields: {summary['n_subfields']}",
        f"- Eligible subfields: {n_eligible}",
        f"- Planned works: {total_planned_works}",
        f"- Manifest planned works: {manifest_planned_works}",
        f"- Downloaded valid works: {n_works}",
        f"- Downloaded / planned: {downloaded_ratio:.3f}" if downloaded_ratio is not None else "- Downloaded / planned: n/a",
        f"- Downloaded / manifest planned: {manifest_downloaded_ratio:.3f}" if manifest_downloaded_ratio is not None else "- Downloaded / manifest planned: n/a",
        f"- Subfields with 3,000 valid works: {buckets['subfields_with_3000_valid_works']}",
        f"- Subfields with 2,500-2,999 valid works: {buckets['subfields_with_2500_2999_valid_works']}",
        f"- Subfields with 500-2,499 valid works: {buckets['subfields_with_500_2499_valid_works']}",
        f"- Subfields below 500 valid works: {buckets['subfields_below_500_valid_works']}",
        f"- Average local validation retention rate: {retention_rate:.3f}" if retention_rate is not None else "- Average local validation retention rate: n/a",
        f"- Duplicate work IDs: {duplicate_work_ids}",
        f"- Missing titles: {missing_titles}",
        f"- Missing abstracts: {missing_abstracts}",
        f"- Missing primary_topic_id: {missing_primary_topic_id}",
        f"- Missing topics_json: {missing_topics_json}",
        *analysis_report_lines,
        "",
        "## Planned Works By Sampling Method",
        "",
        json.dumps(summary["planned_works_by_sampling_method"], indent=2),
        "",
        "## Kept Works By Sampling Method",
        "",
        json.dumps(summary["kept_works_by_sampling_method"], indent=2),
        "",
        "## Manifest Status Counts",
        "",
        json.dumps(summary["manifest_status_counts"], indent=2),
        "",
        "## Worst Shortfalls",
        "",
        json.dumps(worst_shortfalls, indent=2),
        "",
        "## Works Per Subfield Distribution",
        "",
        json.dumps(summary["works_per_subfield_distribution"], indent=2),
        "",
        "## Works Per Field Distribution",
        "",
        json.dumps(summary["works_per_field_distribution"], indent=2),
        "",
        "## Token Count Distributions",
        "",
        "title_token_count:",
        json.dumps(summary["title_token_count_distribution"], indent=2),
        "",
        "abstract_token_count:",
        json.dumps(summary["abstract_token_count_distribution"], indent=2),
        "",
        "text_token_count:",
        json.dumps(summary["text_token_count_distribution"], indent=2),
        "",
        "## Works Per Year",
        "",
        json.dumps(summary["works_per_year"], indent=2),
        "",
        "## Language Distribution",
        "",
        json.dumps(summary["language_distribution"], indent=2),
        "",
        "## Type Distribution",
        "",
        json.dumps(summary["type_distribution"], indent=2),
        "",
        "## Later Embedding Size Estimate",
        "",
        f"- 768d float32: {embedding_float32_mb} MB",
        f"- 768d float16: {embedding_float16_mb} MB",
    ]

    report_path = interim_dir / "validation_report.md"
    summary_path = interim_dir / "validation_summary.json"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )

    print(f"Wrote {report_path.relative_to(ROOT)}")
    print(f"Wrote {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
