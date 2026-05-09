from __future__ import annotations

import numpy as np
import pandas as pd

from src.metric_family_comparison import (
    analogue_pair_correlations,
    join_metric_families,
    pairwise_correlations,
    top_absolute_correlations,
)


def tiny_umap_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subfield_id": ["S1", "S2", "S3", "S4"],
            "metric_status": ["completed", "completed", "failed", "completed"],
            "knn_median_distance": [1.0, 2.0, 3.0, np.nan],
            "knn_distance_cv": [0.1, 0.2, 0.3, 0.4],
            "radial_tail_index": [2.0, 3.0, 4.0, 5.0],
        }
    )


def tiny_embedding_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subfield_id": ["S1", "S2", "S3", "S4"],
            "metric_status": ["completed", "completed_with_warnings", "completed", "completed"],
            "embedding_knn_median_distance": [10.0, 20.0, 30.0, 40.0],
            "embedding_knn_distance_cv": [0.4, 0.3, 0.2, 0.1],
            "embedding_tail_index_p90_median": [2.1, 3.1, 4.1, np.nan],
        }
    )


def test_join_metric_families_uses_subfield_id_and_valid_statuses() -> None:
    joined = join_metric_families(tiny_umap_metrics(), tiny_embedding_metrics())

    assert joined["subfield_id"].tolist() == ["S1", "S2", "S4"]
    assert "knn_median_distance" in joined.columns
    assert "embedding_knn_median_distance" in joined.columns


def test_pairwise_correlations_shape_and_nan_handling() -> None:
    joined = join_metric_families(tiny_umap_metrics(), tiny_embedding_metrics())
    result = pairwise_correlations(
        joined,
        ["knn_median_distance", "radial_tail_index"],
        ["embedding_knn_median_distance", "embedding_tail_index_p90_median"],
        method="spearman",
    )

    assert len(result) == 4
    assert set(result["method"]) == {"spearman"}
    assert result["n_valid_pairwise"].min() >= 1


def test_pearson_correlations_are_available() -> None:
    joined = join_metric_families(tiny_umap_metrics(), tiny_embedding_metrics())
    result = pairwise_correlations(
        joined,
        ["knn_distance_cv"],
        ["embedding_knn_distance_cv"],
        method="pearson",
    )

    assert len(result) == 1
    assert result.loc[0, "correlation"] < 0


def test_analogue_pairs_tolerate_missing_metrics() -> None:
    joined = join_metric_families(tiny_umap_metrics(), tiny_embedding_metrics())
    result = analogue_pair_correlations(
        joined,
        analogue_pairs={
            "knn_median_distance": "embedding_knn_median_distance",
            "missing_umap": "embedding_knn_median_distance",
        },
    )

    assert len(result) == 2
    assert not bool(result.loc[result["umap_metric"] == "missing_umap", "available"].iloc[0])


def test_top_absolute_correlations_sorts_descending() -> None:
    correlations = pd.DataFrame(
        {
            "umap_metric": ["a", "b", "c"],
            "embedding_metric": ["x", "y", "z"],
            "abs_correlation": [0.2, 0.9, 0.5],
            "correlation": [0.2, -0.9, 0.5],
            "n_valid_pairwise": [4, 4, 4],
        }
    )

    top = top_absolute_correlations(correlations, n=2)

    assert top["umap_metric"].tolist() == ["b", "c"]
