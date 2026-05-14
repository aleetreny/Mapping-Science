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
from src.subfield_temporal_umap_panels import (
    build_subfield_temporal_umap_panels,
    ensure_temporal_umap_outputs,
)
from src.temporal_common import (
    default_embeddings_path,
    display_path,
    load_embedding_matrix,
    resolve_path,
    validate_matrix_alignment,
)


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_args() -> argparse.Namespace:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(
        description=(
            "Build static panels and optional GIFs showing temporal windows on "
            "fixed per-subfield UMAP coordinates."
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
        default="outputs/analysis/subfield_temporal_umap_panels",
    )
    parser.add_argument(
        "--reuse-existing-coordinates",
        type=str_to_bool,
        default=True,
    )
    parser.add_argument(
        "--coordinates-dir",
        default="outputs/maps/per_subfield_umap/coordinates",
    )
    parser.add_argument("--fit-if-missing", action="store_true")
    parser.add_argument("--year-min", type=int, default=2000)
    parser.add_argument("--year-max", type=int, default=2024)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--min-papers", type=int, default=500)
    parser.add_argument("--max-papers-per-subfield", type=int, default=10000)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.05)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--make-gif", action="store_true")
    parser.add_argument("--make-static-panel", dest="make_static_panel", action="store_true", default=True)
    parser.add_argument("--make-static-panels", dest="make_static_panel", action="store_true")
    parser.add_argument("--no-static-panel", dest="make_static_panel", action="store_false")
    parser.add_argument("--make-density-panels", action="store_true")
    parser.add_argument("--make-density-difference-panels", action="store_true")
    parser.add_argument("--make-density-gifs", action="store_true")
    parser.add_argument("--density-grid-size", type=int, default=220)
    parser.add_argument("--density-sigma", type=float, default=2.0)
    parser.add_argument("--min-points-for-density", type=int, default=30)
    parser.add_argument("--density-alpha", type=float, default=0.95)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--limit-subfields", type=int, default=None)
    parser.add_argument("--subfield-id", default=None)
    parser.add_argument(
        "--top-dynamic-from",
        default="data/processed/temporal/subfield_centroid_path_metrics.parquet",
    )
    parser.add_argument("--top-dynamic-n", "--top-n", dest="top_dynamic_n", type=int, default=25)
    parser.add_argument("--dpi", type=int, default=180)
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
    if args.min_papers <= 0:
        raise ValueError("min_papers must be positive")
    if args.max_papers_per_subfield <= 0:
        raise ValueError("max_papers_per_subfield must be positive")
    if args.n_neighbors < 2:
        raise ValueError("n_neighbors must be at least 2")
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if args.top_dynamic_n <= 0:
        raise ValueError("top_n/top_dynamic_n must be positive")
    if args.density_grid_size < 10:
        raise ValueError("density_grid_size must be at least 10")
    if args.density_sigma <= 0:
        raise ValueError("density_sigma must be positive")
    if args.min_points_for_density <= 0:
        raise ValueError("min_points_for_density must be positive")
    if not 0 < args.density_alpha <= 1:
        raise ValueError("density_alpha must be in (0, 1]")

    index_path = resolve_path(args.index_path, root=ROOT)
    embeddings_path = resolve_embeddings_path(args)
    output_dir = resolve_path(args.output_dir, root=ROOT)
    coordinates_dir = resolve_path(args.coordinates_dir, root=ROOT)
    top_dynamic_from = (
        resolve_path(args.top_dynamic_from, root=ROOT)
        if args.top_dynamic_from
        else None
    )
    ensure_temporal_umap_outputs(output_dir, overwrite=args.overwrite, root=ROOT)

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

    summary = build_subfield_temporal_umap_panels(
        analysis_index,
        matrix,
        year_min=args.year_min,
        year_max=args.year_max,
        window_size=args.window_size,
        min_papers=args.min_papers,
        max_papers_per_subfield=args.max_papers_per_subfield,
        reuse_existing_coordinates=args.reuse_existing_coordinates,
        coordinates_dir=coordinates_dir,
        fit_if_missing=args.fit_if_missing,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.random_state,
        make_gif=args.make_gif,
        make_static_panel=args.make_static_panel,
        make_density_panels=args.make_density_panels,
        make_density_difference_panels=args.make_density_difference_panels,
        make_density_gifs=args.make_density_gifs,
        density_grid_size=args.density_grid_size,
        density_sigma=args.density_sigma,
        min_points_for_density=args.min_points_for_density,
        density_alpha=args.density_alpha,
        fps=args.fps,
        subfield_id=args.subfield_id,
        limit_subfields=args.limit_subfields,
        top_dynamic_from=top_dynamic_from,
        top_dynamic_n=args.top_dynamic_n,
        output_dir=output_dir,
        root=ROOT,
        overwrite=args.overwrite,
        dpi=args.dpi,
    )
    print(
        "Done: "
        f"{summary['n_completed']} completed, "
        f"{summary['n_skipped']} skipped, "
        f"{summary['n_failed']} failed."
    )
    print(f"Wrote {summary['output_paths']['summary_md']}")


if __name__ == "__main__":
    main()
