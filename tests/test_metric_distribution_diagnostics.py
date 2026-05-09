from __future__ import annotations

import numpy as np
import pandas as pd

from src.metric_distribution_diagnostics import (
    build_distribution_summary,
    low_information_metrics,
    summarize_metric_frame,
)


def test_distribution_flags_constant_low_unique_high_missing_and_zero_dominated() -> None:
    frame = pd.DataFrame(
        {
            "constant_metric": [1, 1, 1, 1, 1],
            "low_unique_metric": [0, 1, 0, 1, 0],
            "high_missing_metric": [1.0, np.nan, np.nan, np.nan, 5.0],
            "zero_dominated_metric": [0, 0, 0, 0, 1],
            "continuous_metric": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    summary = summarize_metric_frame(
        frame,
        metric_family="test",
        metric_columns=list(frame.columns),
        high_missing_threshold=0.30,
        low_unique_threshold=2,
        low_unique_share_threshold=0.05,
        zero_dominated_threshold=0.80,
    )
    by_metric = summary.set_index("metric_name")

    assert bool(by_metric.loc["constant_metric", "is_constant"])
    assert bool(by_metric.loc["low_unique_metric", "is_low_unique"])
    assert bool(by_metric.loc["high_missing_metric", "is_high_missing"])
    assert bool(by_metric.loc["zero_dominated_metric", "is_zero_dominated"])
    assert not bool(by_metric.loc["continuous_metric", "recommended_review_flag"])


def test_low_information_metrics_filters_flagged_rows() -> None:
    summary = pd.DataFrame(
        {
            "metric_name": ["a", "b"],
            "metric_family": ["test", "test"],
            "recommended_review_flag": [True, False],
            "is_constant": [True, False],
            "is_high_missing": [False, False],
            "is_zero_dominated": [False, False],
        }
    )

    low_info = low_information_metrics(summary)

    assert low_info["metric_name"].tolist() == ["a"]


def test_build_distribution_summary_uses_available_default_metric_columns() -> None:
    umap = pd.DataFrame(
        {
            "radial_tail_index": [1.0, 2.0, 3.0],
            "density_entropy": [0.5, 0.6, 0.7],
        }
    )
    embedding = pd.DataFrame(
        {
            "embedding_centroid_norm": [0.1, 0.2, 0.3],
            "embedding_knn_mean_distance": [0.4, 0.5, 0.6],
        }
    )

    summary = build_distribution_summary(umap, embedding)

    assert set(summary["metric_family"]) == {"umap_projected", "embedding_space"}
    assert set(summary["metric_name"]) == {
        "radial_tail_index",
        "density_entropy",
        "embedding_centroid_norm",
        "embedding_knn_mean_distance",
    }
