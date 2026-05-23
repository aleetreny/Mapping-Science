from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.centroid_embedding_clustering import (
    build_centroid_embedding_clustering_outputs,
    ensure_centroid_embedding_clustering_outputs,
)
from src.storage import load_parquet
from src.temporal_common import (
    default_embeddings_path,
    display_path,
    load_embedding_matrix,
    resolve_path,
    validate_matrix_alignment,
)


def parse_args() -> argparse.Namespace:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(
        description=(
            "Cluster subfield centroid embeddings as a sensitivity check for "
            "Chapter 9 morphological typologies."
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
        "--typology-assignments-path",
        default="outputs/09_morphological_typologies/subfield_typology_assignments.csv",
    )
    parser.add_argument(
        "--morphology-pca-path",
        default="outputs/09_morphological_typologies/typology_pca_scores.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/09_morphological_typologies/centroid_embedding_clustering",
    )
    parser.add_argument("--figure-dir", default="memory/figures")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_embeddings_path(args: argparse.Namespace) -> Path:
    path = args.embeddings_path or default_embeddings_path(args.embedding_dir)
    return resolve_path(path, root=ROOT)


def main() -> None:
    args = parse_args()
    index_path = resolve_path(args.index_path, root=ROOT)
    embeddings_path = resolve_embeddings_path(args)
    typology_assignments_path = resolve_path(args.typology_assignments_path, root=ROOT)
    morphology_pca_path = resolve_path(args.morphology_pca_path, root=ROOT)
    output_dir = resolve_path(args.output_dir, root=ROOT)
    figure_dir = resolve_path(args.figure_dir, root=ROOT)

    ensure_centroid_embedding_clustering_outputs(
        output_dir,
        figure_dir,
        overwrite=args.overwrite,
        root=ROOT,
    )

    if not index_path.exists():
        raise FileNotFoundError(f"Missing analysis index: {display_path(index_path, root=ROOT)}")
    if not typology_assignments_path.exists():
        raise FileNotFoundError(
            "Missing morphology typology assignments: "
            f"{display_path(typology_assignments_path, root=ROOT)}"
        )
    if not morphology_pca_path.exists():
        raise FileNotFoundError(
            f"Missing morphology PCA scores: {display_path(morphology_pca_path, root=ROOT)}"
        )

    print(f"Loading analysis index: {display_path(index_path, root=ROOT)}")
    analysis_index = load_parquet(index_path)
    print(f"Memory-mapping embeddings: {display_path(embeddings_path, root=ROOT)}")
    embedding_matrix = load_embedding_matrix(embeddings_path)
    validate_matrix_alignment(
        analysis_index,
        embedding_matrix,
        index_path=index_path,
        embeddings_path=embeddings_path,
    )
    summary = build_centroid_embedding_clustering_outputs(
        analysis_index=analysis_index,
        embedding_matrix=embedding_matrix,
        typology_assignments_path=typology_assignments_path,
        morphology_pca_path=morphology_pca_path,
        output_dir=output_dir,
        figure_dir=figure_dir,
        root=ROOT,
        random_seed=args.random_seed,
        index_path_label=display_path(index_path, root=ROOT),
        embedding_matrix_path_label=display_path(embeddings_path, root=ROOT),
    )
    selected = summary["selected_solution"]
    print(
        "Done: "
        f"{summary['n_subfields']} subfield centroid embeddings, "
        f"selected {selected['solution_id']} "
        f"(silhouette={float(selected['silhouette']):.3f})."
    )
    print(f"Wrote {display_path(output_dir, root=ROOT)}")


if __name__ == "__main__":
    main()
