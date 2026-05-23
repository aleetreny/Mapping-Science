from __future__ import annotations

import numpy as np
import pandas as pd

from src.hybrid_morphology_centroid_clustering import (
    balanced_concat,
    build_hybrid_representations,
    load_hybrid_inputs,
)
from src.morphological_typologies import REDUCED_METRICS


def test_balanced_concat_gives_expected_shape_and_finite_values() -> None:
    first = np.arange(24, dtype=float).reshape(6, 4)
    second = np.arange(66, dtype=float).reshape(6, 11)

    combined = balanced_concat(first, second)

    assert combined.shape == (6, 15)
    assert np.isfinite(combined).all()


def test_hybrid_representations_do_not_use_projection_coordinates() -> None:
    rng = np.random.default_rng(42)
    centroid = rng.normal(size=(8, 6))
    centroid = centroid / np.linalg.norm(centroid, axis=1, keepdims=True)
    morphology = rng.normal(size=(8, len(REDUCED_METRICS)))

    feature_reps, distance_reps, metadata = build_hybrid_representations(centroid, morphology)

    assert "concat_centroid_pca10_morphology" in feature_reps
    assert "distance_alpha0p50" in distance_reps
    assert all(np.isfinite(matrix).all() for matrix in feature_reps.values())
    assert all(np.isfinite(matrix).all() for matrix in distance_reps.values())
    assert not any("umap" in name.lower() for name in [*feature_reps, *distance_reps])
    assert metadata["distance_alpha0p50"]["centroid_distance"] == "cosine"


def test_load_hybrid_inputs_aligns_centroid_rows(tmp_path) -> None:
    rows = []
    for idx in range(4):
        row = {
            "subfield_id": str(1000 + idx),
            "subfield_display_name": f"Subfield {idx}",
            "field_display_name": "Field",
            "domain_display_name": "Domain",
            "typology_id": f"T{idx % 2 + 1}",
            "typology_label": "Label",
        }
        for metric_idx, metric in enumerate(REDUCED_METRICS):
            row[f"scaled_{metric}"] = float(idx + metric_idx)
        rows.append(row)
    typology_path = tmp_path / "typology.csv"
    pd.DataFrame(rows).to_csv(typology_path, index=False)

    centroid_assignments = pd.DataFrame(
        {
            "subfield_id": ["1002", "1000", "1003", "1001"],
            "centroid_cluster_id": [2, 1, 2, 1],
        }
    )
    centroid_assignments_path = tmp_path / "centroid_assignments.csv"
    centroid_assignments.to_csv(centroid_assignments_path, index=False)
    centroid_matrix = np.arange(16, dtype=np.float32).reshape(4, 4)
    centroid_path = tmp_path / "centroids.npy"
    np.save(centroid_path, centroid_matrix)

    metadata, aligned_centroids, morphology = load_hybrid_inputs(
        typology_assignments_path=typology_path,
        centroid_embeddings_path=centroid_path,
        centroid_assignments_path=centroid_assignments_path,
    )

    assert metadata["subfield_id"].tolist() == ["1000", "1001", "1002", "1003"]
    assert np.allclose(aligned_centroids[0], centroid_matrix[1])
    assert morphology.shape == (4, len(REDUCED_METRICS))
