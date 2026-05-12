from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.embeddings import (
    EMBEDDING_DIM,
    EMBEDDING_DTYPE,
    EMBEDDING_MODEL,
    EMBEDDING_VERSION,
    build_embedding_index,
    read_embedding_metadata_index,
    validate_embedding_relationships,
    validate_embedding_shards,
)
from src.storage import connect_duckdb, ensure_dirs, load_parquet, save_parquet, write_table


def load_config() -> dict[str, Any]:
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Validate local SPECTER2 embeddings.")
    parser.add_argument(
        "--embedding-dir",
        default=os.getenv("LOCAL_EMBEDDINGS_DIR", "embeddings/specter2_v1"),
        help="Local folder containing SPECTER2 embedding shards.",
    )
    parser.add_argument(
        "--expected-shards",
        type=int,
        default=37,
        help="Expected number of embedding, metadata, and summary shards.",
    )
    return parser.parse_args()


def read_required_parquet(path: Path, errors: list[str]) -> pd.DataFrame:
    if not path.exists():
        errors.append(f"Missing required parquet file: {path.relative_to(ROOT)}")
        return pd.DataFrame()
    return load_parquet(path)


def report_lines(summary: dict[str, Any]) -> list[str]:
    errors = summary["errors"]
    lines = [
        "# Embedding Validation Report",
        "",
        f"- Passed: {summary['passed']}",
        f"- Embedding directory: `{summary['embedding_dir']}`",
        f"- Embedding model: `{summary['embedding_model']}`",
        f"- Embedding version: `{summary['embedding_version']}`",
        f"- Expected shards: {summary['expected_shards']}",
        f"- Embedding shards: {summary['file_counts']['embedding_shards']}",
        f"- Metadata shards: {summary['file_counts']['metadata_shards']}",
        f"- Summary files: {summary['file_counts']['summary_files']}",
        f"- Total embedding rows: {summary['total_embedding_rows']}",
        f"- Total metadata rows: {summary['total_metadata_rows']}",
        f"- Works text rows: {summary['works_text_rows']}",
        f"- Embedding index rows: {summary['embedding_index_rows']}",
        f"- Embedding dimension: {summary['embedding_dim']}",
        f"- Embedding dtype: `{summary['embedding_dtype']}`",
        f"- Duplicate metadata work IDs: {summary['duplicate_work_id_rows']}",
        f"- Embedded IDs missing from works_text: {summary['embedded_missing_from_works']}",
        f"- works_text IDs missing from embeddings: {summary['works_missing_from_embeddings']}",
        f"- Subfield mismatch rows: {summary['subfield_mismatch_rows']}",
        f"- Analysis flag missing rows: {summary['analysis_flag_missing_rows']}",
        f"- Main-analysis embedded rows: {summary['main_analysis_embedded_rows']}",
        f"- Robustness embedded rows: {summary['robustness_embedded_rows']}",
        (
            "- Strict full-period embedded rows: "
            f"{summary['strict_full_period_embedded_rows']}"
        ),
        f"- DuckDB table written: {summary['duckdb_table_written']}",
        "",
        "## Errors",
        "",
    ]
    if errors:
        lines.extend(f"- {error}" for error in errors)
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Shards",
            "",
            "| shard | embedding rows | metadata rows | dim | dtype |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for shard in summary["shards"]:
        lines.append(
            "| {shard_id:04d} | {embedding_rows} | {metadata_rows} | "
            "{embedding_dim} | {embedding_dtype} |".format(
                shard_id=shard["shard_id"],
                embedding_rows=shard["embedding_rows"],
                metadata_rows=shard["metadata_rows"],
                embedding_dim=shard["embedding_dim"],
                embedding_dtype=shard["embedding_dtype"],
            )
        )
    return lines


def main() -> None:
    args = parse_args()
    config = load_config()
    ensure_dirs(config)

    processed_dir = ROOT / config["storage"]["processed_dir"]
    duckdb_path = ROOT / config["storage"]["duckdb_path"]
    embedding_dir = ROOT / args.embedding_dir
    embedding_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    shard_validation = validate_embedding_shards(
        embedding_dir,
        expected_shards=args.expected_shards,
        expected_dim=EMBEDDING_DIM,
        expected_dtype=EMBEDDING_DTYPE,
    )
    errors.extend(shard_validation["errors"])

    works_text = read_required_parquet(processed_dir / "works_text.parquet", errors)
    analysis_subfields = read_required_parquet(
        processed_dir / "analysis_subfields.parquet", errors
    )

    metadata_index = pd.DataFrame()
    embedding_index = pd.DataFrame()
    relationship_validation: dict[str, Any] = {
        "errors": [],
        "duplicate_work_id_rows": 0,
        "embedded_missing_from_works": 0,
        "works_missing_from_embeddings": 0,
        "subfield_mismatch_rows": 0,
        "analysis_flag_missing_rows": 0,
        "analysis_subfields_with_flags": 0,
        "main_analysis_embedded_rows": 0,
        "robustness_embedded_rows": 0,
        "strict_full_period_embedded_rows": 0,
        "total_index_rows": 0,
    }
    duckdb_table_written = False

    try:
        metadata_index = read_embedding_metadata_index(embedding_dir)
        if metadata_index.empty:
            errors.append("No embedding metadata rows were found.")
        elif not works_text.empty and not analysis_subfields.empty:
            embedding_index = build_embedding_index(
                metadata_index=metadata_index,
                works_text=works_text,
                analysis_subfields=analysis_subfields,
            )
            relationship_validation = validate_embedding_relationships(
                metadata_index=metadata_index,
                works_text=works_text,
                analysis_subfields=analysis_subfields,
                embedding_index=embedding_index,
            )
            errors.extend(relationship_validation["errors"])

            output_path = processed_dir / "embedding_index.parquet"
            save_parquet(embedding_index, output_path)

            con = connect_duckdb(duckdb_path)
            try:
                write_table(con, "embedding_index", embedding_index)
                duckdb_table_written = True
            finally:
                con.close()
    except Exception as exc:
        errors.append(f"Could not build embedding index: {exc}")

    if len(works_text) != shard_validation["total_embedding_rows"]:
        errors.append(
            "Total embedding rows do not match works_text rows: "
            f"{shard_validation['total_embedding_rows']} vs {len(works_text)}"
        )

    summary = {
        "passed": not errors,
        "errors": errors,
        "embedding_dir": str(embedding_dir.relative_to(ROOT)),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_version": EMBEDDING_VERSION,
        "embedding_dim": EMBEDDING_DIM,
        "embedding_dtype": EMBEDDING_DTYPE,
        "expected_shards": args.expected_shards,
        "file_counts": shard_validation["file_counts"],
        "total_embedding_rows": shard_validation["total_embedding_rows"],
        "total_metadata_rows": shard_validation["total_metadata_rows"],
        "works_text_rows": int(len(works_text)),
        "analysis_subfields_rows": int(len(analysis_subfields)),
        "embedding_index_rows": int(len(embedding_index)),
        "duckdb_table_written": duckdb_table_written,
        "shards": shard_validation["shards"],
        **{
            key: value
            for key, value in relationship_validation.items()
            if key != "errors"
        },
    }

    report_path = embedding_dir / "embedding_validation_report.md"
    summary_path = embedding_dir / "embedding_validation_summary.json"
    report_path.write_text("\n".join(report_lines(summary)) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {report_path.relative_to(ROOT)}")
    print(f"Wrote {summary_path.relative_to(ROOT)}")
    if embedding_index.empty:
        print("Embedding index was not written.")
    else:
        print(f"Wrote {(processed_dir / 'embedding_index.parquet').relative_to(ROOT)}")
        print('Wrote DuckDB table "embedding_index"')

    if errors:
        raise SystemExit(1)
    print("Embedding validation completed successfully.")


if __name__ == "__main__":
    main()
