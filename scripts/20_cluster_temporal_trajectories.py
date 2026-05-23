from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.temporal_common import display_path, resolve_path
from src.temporal_trajectory_clustering import (
    build_temporal_trajectory_clustering_outputs,
    ensure_temporal_trajectory_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster subfields by temporal trajectories of windowed morphology metrics."
    )
    parser.add_argument(
        "--window-metrics-path",
        default="data/processed/temporal/subfield_window_embedding_metrics.parquet",
    )
    parser.add_argument(
        "--centroid-path-metrics-path",
        default="data/processed/temporal/subfield_centroid_path_metrics.parquet",
    )
    parser.add_argument(
        "--typology-assignments-path",
        default="outputs/09_morphological_typologies/subfield_typology_assignments.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/09_morphological_typologies/temporal_trajectory_clustering",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    window_metrics_path = resolve_path(args.window_metrics_path, root=ROOT)
    centroid_path_metrics_path = resolve_path(args.centroid_path_metrics_path, root=ROOT)
    typology_assignments_path = resolve_path(args.typology_assignments_path, root=ROOT)
    output_dir = resolve_path(args.output_dir, root=ROOT)

    ensure_temporal_trajectory_outputs(output_dir, overwrite=args.overwrite, root=ROOT)
    for path, label in [
        (window_metrics_path, "window metrics"),
        (typology_assignments_path, "typology assignments"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {display_path(path, root=ROOT)}")

    summary = build_temporal_trajectory_clustering_outputs(
        window_metrics_path=window_metrics_path,
        centroid_path_metrics_path=centroid_path_metrics_path,
        typology_assignments_path=typology_assignments_path,
        output_dir=output_dir,
        root=ROOT,
        random_seed=args.random_seed,
    )
    selected = summary["selected_solution"]
    print(
        "Done: "
        f"{summary['n_subfields']} complete temporal trajectories, "
        f"selected {selected['solution_id']} "
        f"(silhouette={float(selected['silhouette']):.3f})."
    )
    print(f"Wrote {display_path(output_dir, root=ROOT)}")


if __name__ == "__main__":
    main()
