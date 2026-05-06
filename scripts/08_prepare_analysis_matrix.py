from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis_matrix import (
    ensure_outputs_do_not_exist,
    prepare_analysis_embedding_index,
    validate_analysis_matrix,
    write_analysis_matrix,
)
from src.embeddings import EMBEDDING_DIM, EMBEDDING_DTYPE
from src.storage import connect_duckdb, ensure_dirs, load_parquet, save_parquet, write_table


def load_config() -> dict[str, Any]:
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Build the main-analysis embedding matrix from validated shards."
    )
    parser.add_argument(
        "--embedding-dir",
        default=os.getenv("LOCAL_EMBEDDINGS_DIR", "embeddings/specter2_v1"),
        help="Local folder containing SPECTER2 embedding shards.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing analysis matrix outputs.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config()
    ensure_dirs(config)

    processed_dir = ROOT / config["storage"]["processed_dir"]
    duckdb_path = ROOT / config["storage"]["duckdb_path"]
    embedding_dir = ROOT / args.embedding_dir
    analysis_dir = embedding_dir / "analysis"

    embedding_index_path = processed_dir / "embedding_index.parquet"
    analysis_subfields_path = processed_dir / "analysis_subfields.parquet"
    analysis_index_path = processed_dir / "analysis_embedding_index.parquet"
    matrix_path = analysis_dir / "main_embeddings.float16.npy"
    work_ids_path = analysis_dir / "main_work_ids.parquet"
    summary_path = analysis_dir / "main_matrix_summary.json"

    ensure_outputs_do_not_exist(
        [analysis_index_path, matrix_path, work_ids_path, summary_path],
        force=args.force,
    )

    if not embedding_index_path.exists():
        raise FileNotFoundError(
            "Missing data/processed/embedding_index.parquet. "
            "Run scripts/07_validate_embeddings.py first."
        )
    if not analysis_subfields_path.exists():
        raise FileNotFoundError(
            "Missing data/processed/analysis_subfields.parquet. "
            "Run scripts/06_build_analysis_subfields.py first."
        )
    if not embedding_dir.exists():
        raise FileNotFoundError(f"Missing embedding directory: {embedding_dir}")

    embedding_index = load_parquet(embedding_index_path)
    analysis_subfields = load_parquet(analysis_subfields_path)
    analysis_index = prepare_analysis_embedding_index(embedding_index)

    save_parquet(analysis_index, analysis_index_path)
    save_parquet(analysis_index[["analysis_row_id", "work_id"]], work_ids_path)

    write_analysis_matrix(
        analysis_index=analysis_index,
        embedding_dir=embedding_dir,
        output_path=matrix_path,
        embedding_dim=EMBEDDING_DIM,
        embedding_dtype=EMBEDDING_DTYPE,
    )

    errors = validate_analysis_matrix(
        analysis_index=analysis_index,
        matrix_path=matrix_path,
        embedding_dim=EMBEDDING_DIM,
        embedding_dtype=EMBEDDING_DTYPE,
    )
    if errors:
        raise RuntimeError("Analysis matrix validation failed: " + "; ".join(errors))

    con = connect_duckdb(duckdb_path)
    try:
        write_table(con, "analysis_embedding_index", analysis_index)
    finally:
        con.close()

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_dir": str(embedding_dir.relative_to(ROOT)),
        "analysis_index_path": str(analysis_index_path.relative_to(ROOT)),
        "matrix_path": str(matrix_path.relative_to(ROOT)),
        "work_ids_path": str(work_ids_path.relative_to(ROOT)),
        "n_rows": int(len(analysis_index)),
        "embedding_dim": EMBEDDING_DIM,
        "embedding_dtype": EMBEDDING_DTYPE,
        "n_main_analysis_subfields": int(
            analysis_subfields["main_analysis_eligible_2500"]
            .fillna(False)
            .astype(bool)
            .sum()
        )
        if "main_analysis_eligible_2500" in analysis_subfields
        else None,
        "duplicate_work_ids": int(analysis_index["work_id"].duplicated().sum()),
        "duckdb_table": "analysis_embedding_index",
    }
    write_json(summary_path, summary)

    print(f"Wrote {analysis_index_path.relative_to(ROOT)}")
    print(f"Wrote {matrix_path.relative_to(ROOT)}")
    print(f"Wrote {work_ids_path.relative_to(ROOT)}")
    print(f"Wrote {summary_path.relative_to(ROOT)}")
    print('Wrote DuckDB table "analysis_embedding_index"')


if __name__ == "__main__":
    main()
