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
    description = counts.describe().to_dict()
    cleaned = {}
    for key, value in description.items():
        number = float(value)
        cleaned[key] = None if math.isnan(number) else number
    return cleaned


def bytes_to_mb(value: int) -> float:
    return round(value / (1024 * 1024), 2)


def main() -> None:
    config = load_config()
    ensure_dirs(config)

    interim_dir = ROOT / config["storage"]["interim_dir"]
    processed_dir = ROOT / config["storage"]["processed_dir"]
    duckdb_path = ROOT / config["storage"]["duckdb_path"]

    con = connect_duckdb(duckdb_path) if duckdb_path.exists() else None
    try:
        domains = read_table_or_empty(con, "domains")
        fields = read_table_or_empty(con, "fields")
        subfields = read_table_or_empty(con, "subfields")
        corpus_plan = read_table_or_empty(con, "corpus_plan")
        sample_plan = read_table_or_empty(con, "sample_plan")
        works_text = read_table_or_empty(con, "works_text")
    finally:
        if con is not None:
            con.close()

    if domains.empty:
        domains = read_parquet_or_empty(interim_dir / "domains.parquet")
    if fields.empty:
        fields = read_parquet_or_empty(interim_dir / "fields.parquet")
    if subfields.empty:
        subfields = read_parquet_or_empty(interim_dir / "subfields.parquet")
    if corpus_plan.empty:
        corpus_plan = read_parquet_or_empty(interim_dir / "corpus_plan.parquet")
    if sample_plan.empty:
        sample_plan = read_parquet_or_empty(interim_dir / "sample_plan.parquet")
    if works_text.empty:
        works_text = read_parquet_or_empty(processed_dir / "works_text.parquet")

    n_works = len(works_text)
    total_planned_works = (
        int(sample_plan["planned_sample_size"].sum())
        if "planned_sample_size" in sample_plan
        else 0
    )
    planned_by_method = (
        sample_plan.groupby("sampling_method")["planned_sample_size"].sum().astype(int)
        if {"sampling_method", "planned_sample_size"}.issubset(sample_plan.columns)
        else pd.Series(dtype=int)
    )
    planned_by_subfield = (
        sample_plan.groupby("subfield_id")["planned_sample_size"].sum().astype(int)
        if {"subfield_id", "planned_sample_size"}.issubset(sample_plan.columns)
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

    duplicate_work_ids = (
        int(works_text["work_id"].duplicated().sum()) if "work_id" in works_text else 0
    )
    missing_titles = (
        int(works_text["title"].isna().sum() + (works_text["title"] == "").sum())
        if "title" in works_text
        else 0
    )
    missing_abstracts = (
        int(works_text["abstract"].isna().sum() + (works_text["abstract"] == "").sum())
        if "abstract" in works_text
        else 0
    )

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

    downloaded_vs_planned = n_works - total_planned_works
    subfields_planned_below_3000 = (
        int((planned_by_subfield < 3000).sum()) if not planned_by_subfield.empty else 0
    )
    subfields_downloaded_below_3000 = (
        int((works_per_subfield < 3000).sum()) if not works_per_subfield.empty else 0
    )
    subfields_downloaded_below_500 = (
        int((works_per_subfield < 500).sum()) if not works_per_subfield.empty else 0
    )

    embedding_float32_mb = bytes_to_mb(n_works * 768 * 4)
    embedding_float16_mb = bytes_to_mb(n_works * 768 * 2)

    summary = {
        "n_domains": len(domains),
        "n_fields": len(fields),
        "n_subfields": len(subfields),
        "n_eligible_subfields": int(corpus_plan["eligible_for_text_corpus"].sum())
        if "eligible_for_text_corpus" in corpus_plan
        else 0,
        "total_planned_works": total_planned_works,
        "planned_works_by_sampling_method": {
            str(k): int(v) for k, v in planned_by_method.to_dict().items()
        },
        "subfields_planned_below_3000": subfields_planned_below_3000,
        "n_downloaded_works": n_works,
        "downloaded_works_minus_planned_works": downloaded_vs_planned,
        "works_per_subfield_distribution": describe_counts(works_per_subfield),
        "works_per_field_distribution": describe_counts(works_per_field),
        "works_per_year": {str(k): int(v) for k, v in works_per_year.to_dict().items()},
        "missing_titles": missing_titles,
        "missing_abstracts": missing_abstracts,
        "duplicate_work_ids": duplicate_work_ids,
        "language_distribution": {str(k): int(v) for k, v in language_distribution.items()},
        "type_distribution": {str(k): int(v) for k, v in type_distribution.items()},
        "subfields_downloaded_below_3000_papers": subfields_downloaded_below_3000,
        "subfields_downloaded_below_500_papers": subfields_downloaded_below_500,
        "estimated_embedding_size_mb_float32_768d": embedding_float32_mb,
        "estimated_embedding_size_mb_float16_768d": embedding_float16_mb,
    }

    report_lines = [
        "# Validation Report",
        "",
        f"- Domains: {summary['n_domains']}",
        f"- Fields: {summary['n_fields']}",
        f"- Subfields: {summary['n_subfields']}",
        f"- Eligible subfields: {summary['n_eligible_subfields']}",
        f"- Total planned works: {total_planned_works}",
        f"- Downloaded works: {summary['n_downloaded_works']}",
        f"- Downloaded minus planned works: {downloaded_vs_planned}",
        f"- Missing titles: {missing_titles}",
        f"- Missing abstracts: {missing_abstracts}",
        f"- Duplicate work IDs: {duplicate_work_ids}",
        f"- Subfields planned below 3,000: {subfields_planned_below_3000}",
        f"- Subfields downloaded below 3,000: {subfields_downloaded_below_3000}",
        f"- Subfields downloaded below 500: {subfields_downloaded_below_500}",
        "",
        "## Planned Works By Sampling Method",
        "",
        json.dumps(summary["planned_works_by_sampling_method"], indent=2),
        "",
        "## Works Per Subfield Distribution",
        "",
        json.dumps(summary["works_per_subfield_distribution"], indent=2),
        "",
        "## Works Per Field Distribution",
        "",
        json.dumps(summary["works_per_field_distribution"], indent=2),
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
