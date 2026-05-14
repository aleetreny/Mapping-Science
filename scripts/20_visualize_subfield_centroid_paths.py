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
from src.temporal_centroid_paths import (
    build_centroid_path_outputs,
    ensure_centroid_path_outputs,
)
from src.temporal_common import (
    default_embeddings_path,
    display_path,
    load_embedding_matrix,
    parse_multi_value,
    resolve_path,
    validate_matrix_alignment,
)


def parse_args() -> argparse.Namespace:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(
        description=(
            "Compute and visualize temporal paths of subfield centroids in "
            "SPECTER2 embedding space."
        )
    )
    parser.add_argument(
        "--index-path",
        default="data/processed/analysis_embedding_index.parquet",
    )
    parser.add_argument(
        "--embedding-dir",
        default=os.getenv(
            "LOCAL_EMBEDDINGS_DIR",
            "embeddings/specter2_v1_2000_2024_400py",
        ),
    )
    parser.add_argument("--embeddings-path", default=None)
    parser.add_argument(
        "--output-dir",
        default="outputs/analysis/temporal_centroid_paths",
    )
    parser.add_argument("--year-min", type=int, default=2000)
    parser.add_argument("--year-max", type=int, default=2024)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--min-papers-per-window", type=int, default=100)
    parser.add_argument(
        "--projection",
        nargs="+",
        default=["pca"],
        help=(
            "Projection for the main workflow. Default: pca. "
            "Comma-separated values are also accepted."
        ),
    )
    parser.add_argument(
        "--include-umap-experimental",
        action="store_true",
        help="Also write appendix-only experimental UMAP projection figures.",
    )
    parser.add_argument("--highlight-top-n", type=int, default=15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--limit-subfields", type=int, default=None)
    parser.add_argument("--subfield-id", default=None)
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
    if args.highlight_top_n <= 0:
        raise ValueError("highlight_top_n must be positive")
    projections = parse_multi_value(args.projection) or ["pca"]
    invalid = sorted(set(projections) - {"pca"})
    if invalid:
        raise ValueError(
            "Unsupported main projections: "
            f"{', '.join(invalid)}. Use --include-umap-experimental for UMAP."
        )
    projections = ["pca"]
    if args.include_umap_experimental:
        projections.append("umap")

    index_path = resolve_path(args.index_path, root=ROOT)
    embeddings_path = resolve_embeddings_path(args)
    output_dir = resolve_path(args.output_dir, root=ROOT)
    ensure_centroid_path_outputs(
        output_dir,
        overwrite=args.overwrite,
        root=ROOT,
        include_umap_experimental=args.include_umap_experimental,
    )

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

    summary = build_centroid_path_outputs(
        analysis_index,
        matrix,
        year_min=args.year_min,
        year_max=args.year_max,
        window_size=args.window_size,
        min_papers_per_window=args.min_papers_per_window,
        projection=projections,
        highlight_top_n=args.highlight_top_n,
        random_state=args.random_state,
        subfield_id=args.subfield_id,
        limit_subfields=args.limit_subfields,
        analysis_index_path=display_path(index_path, root=ROOT),
        embedding_matrix_path=display_path(embeddings_path, root=ROOT),
        output_dir=output_dir,
        root=ROOT,
        cleanup_legacy_umap=args.overwrite and not args.include_umap_experimental,
    )
    print(
        "Done: "
        f"{summary['n_completed_centroid_rows']} completed centroid rows, "
        f"{summary['n_centroid_rows']} attempted."
    )
    print(f"Wrote {summary['output_paths']['summary_md']}")


if __name__ == "__main__":
    main()
