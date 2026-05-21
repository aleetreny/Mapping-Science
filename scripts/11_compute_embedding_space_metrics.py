from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.embedding_space_metrics import (
    CONTROL_COLUMNS,
    CORE_EMBEDDING_METRIC_COLUMNS,
    DIAGNOSTIC_COLUMNS,
    compute_subfield_embedding_metric_row,
    embedding_metric_row_frame,
    failed_embedding_metric_row,
    metric_dictionary_frame,
    validate_embedding_index_columns,
    validate_year_window,
)
from src.embedding_metric_diagnostics import (
    diagnostic_output_paths,
    write_embedding_metric_diagnostics,
)
from src.metric_year_windows import resolve_metric_year_window
from src.per_subfield_umap_maps import (
    filter_input_window,
    main_analysis_subfields,
    sample_subfield_rows,
)
from src.storage import load_parquet, save_parquet
from src.subfield_labels import add_subfield_label_columns


def parse_args() -> argparse.Namespace:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Compute embedding-space structure metrics per OpenAlex subfield."
    )
    parser.add_argument(
        "--index-path",
        default="data/processed/analysis_embedding_index.parquet",
        help="Parquet index aligned row-for-row with the main embedding matrix.",
    )
    parser.add_argument(
        "--embedding-dir",
        default=os.getenv(
            "LOCAL_EMBEDDINGS_DIR",
            "embeddings/specter2_v1_2000_2024_400py",
        ),
        help=(
            "Local folder containing SPECTER2 embedding artifacts. Used to derive "
            "--embeddings-path when that argument is not provided."
        ),
    )
    parser.add_argument(
        "--embeddings-path",
        default=None,
        help=(
            "Main SPECTER2 embedding matrix. Defaults to "
            "<embedding-dir>/analysis/main_embeddings.float16.npy."
        ),
    )
    parser.add_argument(
        "--output-parquet",
        default="data/processed/subfield_embedding_space_metrics.parquet",
    )
    parser.add_argument(
        "--output-csv",
        default="data/processed/subfield_embedding_space_metrics.csv",
    )
    parser.add_argument(
        "--summary-path",
        default="outputs/03_embedding_metrics/subfield_embedding_space_metrics_summary.json",
    )
    parser.add_argument(
        "--dictionary-path",
        default="outputs/03_embedding_metrics/subfield_embedding_space_metrics_dictionary.csv",
    )
    parser.add_argument(
        "--diagnostics-output-dir",
        default="outputs/03_embedding_metrics/diagnostics",
        help="Folder for embedding metric histograms, correlations, and distribution diagnostics.",
    )
    parser.add_argument("--year-min", type=int, default=2000)
    parser.add_argument("--year-max", type=int, default=2024)
    parser.add_argument("--min-papers", type=int, default=1)
    parser.add_argument(
        "--max-papers-per-subfield",
        type=int,
        default=None,
        help="Optional deterministic cap per subfield. By default all rows are used.",
    )
    parser.add_argument("--subfield-id", default=None)
    parser.add_argument("--limit-subfields", type=int, default=None)
    parser.add_argument("--k-neighbors", type=int, default=15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def default_embeddings_path(embedding_dir: str | Path) -> Path:
    return Path(embedding_dir) / "analysis" / "main_embeddings.float16.npy"


def resolve_embeddings_path(args: argparse.Namespace) -> Path:
    path = args.embeddings_path or default_embeddings_path(args.embedding_dir)
    return resolve_path(path)


def display_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ensure_outputs_do_not_exist(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        formatted = ", ".join(display_path(path) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing embedding-space metric outputs without "
            f"--overwrite: {formatted}"
        )


def select_attempted_subfields(
    subfields: pd.DataFrame,
    *,
    subfield_id: str | None,
    limit_subfields: int | None,
) -> pd.DataFrame:
    attempted = subfields.copy()
    if subfield_id is not None:
        attempted = attempted[
            attempted["subfield_id"].astype(str) == str(subfield_id)
        ].reset_index(drop=True)
        if attempted.empty:
            raise ValueError(f"No main-analysis subfield matched --subfield-id {subfield_id}")

    if limit_subfields is not None:
        if limit_subfields <= 0:
            raise ValueError("limit_subfields must be positive when provided")
        attempted = attempted.head(limit_subfields).reset_index(drop=True)

    return attempted


def make_subfield_group_lookup(window_index: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        str(subfield_id): group.reset_index(drop=True)
        for subfield_id, group in window_index.groupby("subfield_id", sort=False)
    }


def load_matrix(embeddings_path: Path) -> np.ndarray:
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Missing embedding matrix: {display_path(embeddings_path)}. "
            "Run scripts/08_prepare_analysis_matrix.py first."
        )
    matrix = np.load(embeddings_path, mmap_mode="r")
    if matrix.ndim != 2:
        raise ValueError(
            f"Expected a 2D embedding matrix at {display_path(embeddings_path)}, "
            f"got shape {matrix.shape}"
        )
    return matrix


def validate_matrix_alignment(
    index: pd.DataFrame,
    matrix: np.ndarray,
    *,
    index_path: Path,
    embeddings_path: Path,
) -> None:
    row_ids = pd.to_numeric(index["analysis_row_id"], errors="raise")
    if row_ids.empty:
        raise ValueError("analysis index is empty")
    min_row = int(row_ids.min())
    max_row = int(row_ids.max())
    if min_row < 0 or max_row >= matrix.shape[0]:
        raise ValueError(
            "analysis_embedding_index and embedding matrix appear to belong to "
            "different embedding versions. "
            f"index_path={display_path(index_path)}; "
            f"embeddings_path={display_path(embeddings_path)}; "
            f"analysis_row_id range {min_row}-{max_row}; "
            f"matrix rows {matrix.shape[0]}. Use a matching --embedding-dir or "
            "--embeddings-path."
        )


def maybe_sample_rows(
    rows: pd.DataFrame,
    *,
    max_papers_per_subfield: int | None,
    random_state: int,
    subfield_id: object,
) -> tuple[pd.DataFrame, bool]:
    if max_papers_per_subfield is None:
        return rows.sort_values(
            ["publication_year", "work_id", "analysis_row_id"],
            kind="mergesort",
        ).reset_index(drop=True), False
    if max_papers_per_subfield <= 0:
        raise ValueError("max_papers_per_subfield must be positive when provided")
    sampled = sample_subfield_rows(
        rows,
        max_papers=max_papers_per_subfield,
        random_state=random_state,
        subfield_id=subfield_id,
    )
    return sampled, len(sampled) < len(rows)


def main() -> None:
    args = parse_args()
    validate_year_window(args.year_min, args.year_max)
    metric_window = resolve_metric_year_window(args.year_min, args.year_max)
    if args.min_papers <= 0:
        raise ValueError("min_papers must be positive")
    if args.k_neighbors < 1:
        raise ValueError("k_neighbors must be positive")

    index_path = resolve_path(args.index_path)
    embeddings_path = resolve_embeddings_path(args)
    output_parquet = resolve_path(args.output_parquet)
    output_csv = resolve_path(args.output_csv)
    summary_path = resolve_path(args.summary_path)
    dictionary_path = resolve_path(args.dictionary_path)
    diagnostics_output_dir = resolve_path(args.diagnostics_output_dir)
    diagnostics_paths = diagnostic_output_paths(diagnostics_output_dir)
    ensure_outputs_do_not_exist(
        [
            output_parquet,
            output_csv,
            summary_path,
            dictionary_path,
            *diagnostics_paths.values(),
        ],
        args.overwrite,
    )

    if not index_path.exists():
        raise FileNotFoundError(
            f"Missing analysis index: {display_path(index_path)}. "
            "Run scripts/08_prepare_analysis_matrix.py first."
        )

    print(f"Loading analysis index: {display_path(index_path)}")
    analysis_index = load_parquet(index_path)
    validate_embedding_index_columns(analysis_index)

    subfields = main_analysis_subfields(analysis_index)
    attempted_subfields = select_attempted_subfields(
        subfields,
        subfield_id=args.subfield_id,
        limit_subfields=args.limit_subfields,
    )
    window_index = filter_input_window(
        analysis_index,
        year_min=args.year_min,
        year_max=args.year_max,
    )
    groups = make_subfield_group_lookup(window_index)

    print(f"Memory-mapping embeddings: {display_path(embeddings_path)}")
    matrix = load_matrix(embeddings_path)
    validate_matrix_alignment(
        analysis_index,
        matrix,
        index_path=index_path,
        embeddings_path=embeddings_path,
    )

    rows: list[dict[str, Any]] = []
    total = len(attempted_subfields)
    print(f"Computing embedding-space metrics for {total} subfields")

    for position, subfield in attempted_subfields.iterrows():
        subfield_id = subfield["subfield_id"]
        subfield_name = subfield["subfield_display_name"]
        subfield_key = str(subfield_id)
        available = groups.get(subfield_key, window_index.head(0))
        n_available = int(len(available))
        percent = ((position + 1) / total) * 100 if total else 100.0
        prefix = f"[{position + 1}/{total} | {percent:.1f}%] {subfield_key} - {subfield_name}"
        sampling_applied = False

        try:
            if n_available < args.min_papers:
                raise ValueError(
                    f"Only {n_available} papers available; min_papers is {args.min_papers}"
                )
            sampled, sampling_applied = maybe_sample_rows(
                available,
                max_papers_per_subfield=args.max_papers_per_subfield,
                random_state=args.random_state,
                subfield_id=subfield_id,
            )
            row_ids = sampled["analysis_row_id"].astype(int).to_numpy()
            embeddings = np.asarray(matrix[row_ids], dtype=np.float32)
            row = compute_subfield_embedding_metric_row(
                sampled,
                embeddings,
                n_available=n_available,
                year_min=args.year_min,
                year_max=args.year_max,
                analysis_index_path=display_path(index_path),
                embedding_matrix_path=display_path(embeddings_path),
                k_neighbors=args.k_neighbors,
                random_state=args.random_state,
                sampling_applied=sampling_applied,
                max_papers_per_subfield=args.max_papers_per_subfield,
            )
            rows.append(row)
            print(f"{prefix}: {row['metric_status']}")
        except Exception as exc:
            rows.append(
                failed_embedding_metric_row(
                    subfield_row=subfield,
                    n_available=n_available,
                    year_min=args.year_min,
                    year_max=args.year_max,
                    analysis_index_path=display_path(index_path),
                    embedding_matrix_path=display_path(embeddings_path),
                    k_neighbors=args.k_neighbors,
                    random_state=args.random_state,
                    sampling_applied=sampling_applied,
                    max_papers_per_subfield=args.max_papers_per_subfield,
                    error_message=str(exc),
                )
            )
            print(f"{prefix}: failed ({exc})")

    metrics = add_subfield_label_columns(embedding_metric_row_frame(rows))
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(metrics, output_parquet)
    metrics.to_csv(output_csv, index=False)

    dictionary = metric_dictionary_frame()
    dictionary_path.parent.mkdir(parents=True, exist_ok=True)
    dictionary.to_csv(dictionary_path, index=False)
    diagnostics_summary = write_embedding_metric_diagnostics(
        metrics,
        diagnostics_output_dir,
    )

    status_counts = metrics["metric_status"].value_counts().to_dict()
    failed = metrics.loc[
        metrics["metric_status"] == "failed",
        ["subfield_id", "subfield_display_name", "metric_error_message"],
    ].to_dict("records")
    warnings = metrics.loc[
        metrics["metric_warning_message"].fillna("") != "",
        ["subfield_id", "subfield_display_name", "metric_warning_message"],
    ].to_dict("records")
    n_failed = int(status_counts.get("failed", 0))
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_paths": {
            "analysis_index_path": display_path(index_path),
            "embedding_dir": display_path(resolve_path(args.embedding_dir)),
            "embedding_matrix_path": display_path(embeddings_path),
        },
        "output_paths": {
            "output_parquet": display_path(output_parquet),
            "output_csv": display_path(output_csv),
            "summary_path": display_path(summary_path),
            "dictionary_path": display_path(dictionary_path),
            "diagnostics_output_dir": display_path(diagnostics_output_dir),
            "diagnostic_outputs": {
                key: display_path(path) for key, path in diagnostics_paths.items()
            },
        },
        "year_window": {
            "year_min": int(args.year_min),
            "year_max": int(args.year_max),
            "early_year_min": int(metric_window.early_year_min),
            "early_year_max": int(metric_window.early_year_max),
            "late_year_min": int(metric_window.late_year_min),
            "late_year_max": int(metric_window.late_year_max),
            "supported_window": metric_window.supported_window,
        },
        "n_subfields_attempted": int(len(metrics)),
        "n_completed": int(len(metrics) - n_failed),
        "n_failed": n_failed,
        "core_metric_columns": CORE_EMBEDDING_METRIC_COLUMNS,
        "n_core_metric_columns": len(CORE_EMBEDDING_METRIC_COLUMNS),
        "diagnostic_columns": DIAGNOSTIC_COLUMNS,
        "control_columns": CONTROL_COLUMNS,
        "diagnostics_summary": diagnostics_summary,
        "k_neighbors": int(args.k_neighbors),
        "random_state": int(args.random_state),
        "min_papers": int(args.min_papers),
        "max_papers_per_subfield": args.max_papers_per_subfield,
        "subfield_id": args.subfield_id,
        "limit_subfields": args.limit_subfields,
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "failed_subfields": failed,
        "warnings": warnings,
    }
    write_json(summary_path, summary)

    print(f"Wrote {display_path(output_parquet)}")
    print(f"Wrote {display_path(output_csv)}")
    print(f"Wrote {display_path(summary_path)}")
    print(f"Wrote {display_path(dictionary_path)}")
    print(f"Wrote diagnostics in {display_path(diagnostics_output_dir)}")
    print(
        "Done: "
        f"{summary['n_completed']} completed, "
        f"{summary['n_failed']} failed"
    )


if __name__ == "__main__":
    main()
