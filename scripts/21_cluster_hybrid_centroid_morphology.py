from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hybrid_morphology_centroid_clustering import (
    build_hybrid_clustering_outputs,
    ensure_hybrid_outputs,
)
from src.temporal_common import display_path, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster hybrid subfield representations combining centroid embeddings and morphology profiles."
    )
    parser.add_argument(
        "--typology-assignments-path",
        default="outputs/09_morphological_typologies/subfield_typology_assignments.csv",
    )
    parser.add_argument(
        "--centroid-embeddings-path",
        default=(
            "outputs/09_morphological_typologies/centroid_embedding_clustering/"
            "subfield_centroid_embeddings_unit.npy"
        ),
    )
    parser.add_argument(
        "--centroid-assignments-path",
        default=(
            "outputs/09_morphological_typologies/centroid_embedding_clustering/"
            "centroid_cluster_assignments.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/09_morphological_typologies/hybrid_centroid_morphology_clustering",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    typology_assignments_path = resolve_path(args.typology_assignments_path, root=ROOT)
    centroid_embeddings_path = resolve_path(args.centroid_embeddings_path, root=ROOT)
    centroid_assignments_path = resolve_path(args.centroid_assignments_path, root=ROOT)
    output_dir = resolve_path(args.output_dir, root=ROOT)

    ensure_hybrid_outputs(output_dir, overwrite=args.overwrite, root=ROOT)
    for path, label in [
        (typology_assignments_path, "typology assignments"),
        (centroid_embeddings_path, "centroid embeddings"),
        (centroid_assignments_path, "centroid assignments"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {display_path(path, root=ROOT)}")

    summary = build_hybrid_clustering_outputs(
        typology_assignments_path=typology_assignments_path,
        centroid_embeddings_path=centroid_embeddings_path,
        centroid_assignments_path=centroid_assignments_path,
        output_dir=output_dir,
        root=ROOT,
        random_seed=args.random_seed,
    )
    selected = summary["selected_solution"]
    print(
        "Done: "
        f"{summary['n_subfields']} hybrid subfield rows, "
        f"selected {selected['solution_id']} "
        f"(silhouette={float(selected['silhouette']):.3f})."
    )
    print(f"Wrote {display_path(output_dir, root=ROOT)}")


if __name__ == "__main__":
    main()
