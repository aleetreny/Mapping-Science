from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.storage import load_parquet
from src.temporal_common import (
    default_embeddings_path,
    display_path,
    load_embedding_matrix,
    resolve_path,
    validate_matrix_alignment,
)
from src.temporal_embedding_metric_evolution import (
    build_temporal_metric_evolution_outputs,
    ensure_temporal_metric_outputs,
)


def parse_args() -> argparse.Namespace:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(
        description=(
            "Compute temporal evolution of selected embedding-space structural "
            "metrics for OpenAlex subfields."
        )
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
        "--output-dir",
        default="outputs/analysis/temporal_embedding_metric_evolution",
    )
    parser.add_argument("--year-min", type=int, default=2000)
    parser.add_argument("--year-max", type=int, default=2024)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--min-papers-per-window", type=int, default=100)
    parser.add_argument("--k-neighbors", type=int, default=15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--limit-subfields", type=int, default=None)
    parser.add_argument("--subfield-id", default=None)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--figure-top-n", type=int, default=25)
    parser.add_argument("--heatmap-top-n", type=int, default=80)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_embeddings_path(args: argparse.Namespace) -> Path:
    path = args.embeddings_path or default_embeddings_path(args.embedding_dir)
    return resolve_path(path, root=ROOT)


def main() -> None:
    args = parse_args()
    if args.year_min > args.year_max:
        raise ValueError("year_min must be <= year_max")
    if args.window_size <= 0:
        raise ValueError("window_size must be positive")
    if args.min_papers_per_window <= 0:
        raise ValueError("min_papers_per_window must be positive")
    if args.k_neighbors < 1:
        raise ValueError("k_neighbors must be positive")

    index_path = resolve_path(args.index_path, root=ROOT)
    embeddings_path = resolve_embeddings_path(args)
    output_dir = resolve_path(args.output_dir, root=ROOT)
    ensure_temporal_metric_outputs(output_dir, overwrite=args.overwrite, root=ROOT)

    if not index_path.exists():
        raise FileNotFoundError(
            f"Missing analysis index: {display_path(index_path, root=ROOT)}. "
            "Run scripts/08_prepare_analysis_matrix.py first."
        )

    print(f"Loading analysis index: {display_path(index_path, root=ROOT)}")
    analysis_index = load_parquet(index_path)
    print(f"Memory-mapping embeddings: {display_path(embeddings_path, root=ROOT)}")
    matrix = load_embedding_matrix(embeddings_path)
    validate_matrix_alignment(
        analysis_index,
        matrix,
        index_path=index_path,
        embeddings_path=embeddings_path,
    )

    summary = build_temporal_metric_evolution_outputs(
        analysis_index,
        matrix,
        year_min=args.year_min,
        year_max=args.year_max,
        window_size=args.window_size,
        min_papers_per_window=args.min_papers_per_window,
        k_neighbors=args.k_neighbors,
        subfield_id=args.subfield_id,
        limit_subfields=args.limit_subfields,
        analysis_index_path=display_path(index_path, root=ROOT),
        embedding_matrix_path=display_path(embeddings_path, root=ROOT),
        output_dir=output_dir,
        root=ROOT,
        heatmap_top_n=args.heatmap_top_n,
        top_n=args.top_n,
        figure_top_n=args.figure_top_n,
    )
    print(
        "Done: "
        f"{summary['n_completed_window_rows']} completed subfield-windows, "
        f"{summary['n_window_rows']} attempted."
    )
    print(
        "Wrote "
        f"{summary['output_paths']['summary_md']} and "
        f"{summary['output_paths']['top_changes_by_metric_csv']}"
    )


if __name__ == "__main__":
    main()
