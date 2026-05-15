from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.storage import load_parquet, save_parquet
from src.temporal_centroid_paths import compute_subfield_window_centroids
from src.temporal_common import (
    default_embeddings_path,
    default_or_computed_windows,
    display_path,
    load_embedding_matrix,
    parse_multi_value,
    resolve_path,
    validate_matrix_alignment,
)
from src.temporal_semantic_distances import (
    build_semantic_distance_outputs,
    ensure_semantic_distance_outputs,
)
from src.per_subfield_umap_maps import main_analysis_subfields
from src.temporal_centroid_paths import select_attempted_subfields


def parse_args() -> argparse.Namespace:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Compute temporal semantic distances between discipline centroids."
    )
    parser.add_argument(
        "--centroids-path",
        default="data/processed/temporal/subfield_window_centroids.parquet",
    )
    parser.add_argument(
        "--index-path",
        default="data/processed/analysis_embedding_index.parquet",
        help="Used only if centroids are unavailable and must be computed internally.",
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
        default="outputs/analysis/temporal_semantic_distances",
    )
    parser.add_argument(
        "--level",
        nargs="+",
        default=["subfield", "field", "domain"],
        help="Levels to compute: subfield, field, domain. Comma-separated values are accepted.",
    )
    parser.add_argument("--distance", choices=["cosine"], default="cosine")
    parser.add_argument("--cluster-heatmaps", action="store_true")
    parser.add_argument("--top-n-pairs", type=int, default=10)
    parser.add_argument("--heatmap-top-n", type=int, default=40)
    parser.add_argument("--max-label-length", type=int, default=55)
    parser.add_argument(
        "--aggregation-mode",
        choices=["weighted_subfield_centroids", "direct_paper_centroids"],
        default="weighted_subfield_centroids",
        help=(
            "How field/domain centroids are derived. direct_paper_centroids is "
            "registered as an explicit sensitivity mode but is not computed by "
            "this script yet."
        ),
    )
    parser.add_argument("--year-min", type=int, default=2000)
    parser.add_argument("--year-max", type=int, default=2024)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--min-papers-per-window", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_embeddings_path(args: argparse.Namespace) -> Path:
    path = args.embeddings_path or default_embeddings_path(args.embedding_dir)
    return resolve_path(path, root=ROOT)


def load_or_compute_centroids(args: argparse.Namespace, centroids_path: Path) -> tuple[object, dict[str, str]]:
    if centroids_path.exists():
        print(f"Loading centroids: {display_path(centroids_path, root=ROOT)}")
        return load_parquet(centroids_path), {
            "centroids_path": display_path(centroids_path, root=ROOT)
        }

    index_path = resolve_path(args.index_path, root=ROOT)
    embeddings_path = resolve_embeddings_path(args)
    if not index_path.exists():
        raise FileNotFoundError(
            f"Missing analysis index: {display_path(index_path, root=ROOT)}. "
            "Run script 20 first, or provide index/embedding inputs so centroids can be computed."
        )
    print(
        "Centroids were not found; computing internally from "
        f"{display_path(index_path, root=ROOT)}"
    )
    analysis_index = load_parquet(index_path)
    matrix = load_embedding_matrix(embeddings_path)
    validate_matrix_alignment(
        analysis_index,
        matrix,
        index_path=index_path,
        embeddings_path=embeddings_path,
    )
    windows = default_or_computed_windows(
        year_min=args.year_min,
        year_max=args.year_max,
        window_size=args.window_size,
    )
    subfields = main_analysis_subfields(analysis_index)
    attempted = select_attempted_subfields(
        subfields,
        subfield_id=None,
        limit_subfields=None,
    )
    centroids = compute_subfield_window_centroids(
        analysis_index,
        matrix,
        windows=windows,
        attempted_subfields=attempted,
        year_min=args.year_min,
        year_max=args.year_max,
        min_papers_per_window=args.min_papers_per_window,
    )
    centroids_path.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(centroids, centroids_path)
    return centroids, {
        "computed_centroids_path": display_path(centroids_path, root=ROOT),
        "analysis_index_path": display_path(index_path, root=ROOT),
        "embedding_matrix_path": display_path(embeddings_path, root=ROOT),
    }


def main() -> None:
    args = parse_args()
    levels = parse_multi_value(args.level)
    invalid = sorted(set(levels) - {"subfield", "field", "domain"})
    if invalid:
        raise ValueError(f"Unsupported levels: {', '.join(invalid)}")
    if args.top_n_pairs <= 0:
        raise ValueError("top_n_pairs must be positive")
    if args.heatmap_top_n <= 0:
        raise ValueError("heatmap_top_n must be positive")
    if args.max_label_length <= 0:
        raise ValueError("max_label_length must be positive")
    if args.year_min > args.year_max:
        raise ValueError("year_min must be <= year_max")
    if args.aggregation_mode == "direct_paper_centroids":
        print(
            "Warning: --aggregation-mode direct_paper_centroids is recorded as a "
            "sensitivity request, but direct paper-level aggregation is not yet "
            "implemented. Falling back to weighted_subfield_centroids."
        )

    output_dir = resolve_path(args.output_dir, root=ROOT)
    centroids_path = resolve_path(args.centroids_path, root=ROOT)
    ensure_semantic_distance_outputs(
        output_dir,
        levels=levels,
        overwrite=args.overwrite,
        root=ROOT,
    )
    centroids, input_paths = load_or_compute_centroids(args, centroids_path)
    summary = build_semantic_distance_outputs(
        centroids,
        levels=levels,
        output_dir=output_dir,
        root=ROOT,
        top_n_pairs=args.top_n_pairs,
        heatmap_top_n=args.heatmap_top_n,
        max_label_length=args.max_label_length,
        aggregation_mode=args.aggregation_mode,
        input_paths=input_paths,
        year_min=args.year_min,
        year_max=args.year_max,
        window_size=args.window_size,
    )
    print(
        "Done: "
        f"{summary['n_pair_distance_rows']} pair-window rows, "
        f"{summary['n_pair_change_rows']} pair changes."
    )
    print(f"Wrote {summary['output_paths']['summary_md']}")


if __name__ == "__main__":
    main()
