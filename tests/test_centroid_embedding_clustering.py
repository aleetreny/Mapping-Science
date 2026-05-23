from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.centroid_embedding_clustering import (
    compute_subfield_centroids,
    domain_purity,
    l2_normalize_rows,
    select_centroid_solution,
)


def test_l2_normalize_rows_handles_zero_vectors() -> None:
    matrix = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)

    normalized = l2_normalize_rows(matrix)

    assert np.allclose(normalized[0], [0.6, 0.8])
    assert np.allclose(normalized[1], [0.0, 0.0])


def test_compute_subfield_centroids_uses_l2_normalized_papers() -> None:
    index = pd.DataFrame(
        {
            "analysis_row_id": [0, 1, 2],
            "subfield_id": [10, 10, 11],
            "subfield_display_name": ["A", "A", "B"],
            "field_id": [1, 1, 2],
            "field_display_name": ["F1", "F1", "F2"],
            "domain_id": [1, 1, 2],
            "domain_display_name": ["D1", "D1", "D2"],
            "publication_year": [2000, 2001, 2002],
            "main_analysis_eligible": [True, True, True],
        }
    )
    assignments = pd.DataFrame(
        {
            "subfield_id": [10, 11],
            "subfield_display_name": ["A", "B"],
            "field_display_name": ["F1", "F2"],
            "domain_display_name": ["D1", "D2"],
            "typology_id": ["T1", "T2"],
            "typology_label": ["One", "Two"],
        }
    )
    matrix = np.array([[3.0, 0.0], [0.0, 4.0], [0.0, 2.0]], dtype=np.float32)

    metadata, centroids = compute_subfield_centroids(index, matrix, assignments)

    expected_a = np.array([1.0, 1.0]) / math.sqrt(2)
    assert metadata.loc[0, "n_centroid_papers"] == 2
    assert np.allclose(centroids[0], expected_a, atol=1e-6)
    assert np.allclose(centroids[1], [0.0, 1.0], atol=1e-6)


def test_domain_purity_reports_majority_share() -> None:
    clusters = np.array([1, 1, 1, 2, 2])
    domains = np.array(["A", "A", "B", "B", "C"])

    assert domain_purity(clusters, domains) == 3 / 5


def test_select_centroid_solution_prefers_balanced_high_silhouette() -> None:
    comparison = pd.DataFrame(
        [
            {
                "solution_id": "unbalanced",
                "status": "ok",
                "k": 3,
                "silhouette": 0.9,
                "min_cluster_size": 1,
                "max_cluster_size": 239,
                "domain_purity": 0.9,
                "ami_vs_domain": 0.5,
            },
            {
                "solution_id": "balanced",
                "status": "ok",
                "k": 5,
                "silhouette": 0.4,
                "min_cluster_size": 20,
                "max_cluster_size": 80,
                "domain_purity": 0.7,
                "ami_vs_domain": 0.4,
            },
        ]
    )

    selected = select_centroid_solution(comparison)

    assert selected["solution_id"] == "balanced"
