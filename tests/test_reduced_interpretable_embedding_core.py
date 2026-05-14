from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.reduced_interpretable_embedding_core import (
    EXCLUDED_FROM_CLEAN_CORE_V1,
    REDUCED_INTERPRETABLE_EMBEDDING_CORE_GROUPS,
    REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS,
    build_reduced_interpretable_embedding_core_outputs,
    output_paths,
)


def synthetic_full_embedding_metrics(n: int = 7) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "subfield_id": [f"S{idx}" for idx in range(n)],
            "subfield_display_name": [f"Subfield {idx}" for idx in range(n)],
            "subfield_label_unique": [f"S{idx} | Subfield {idx}" for idx in range(n)],
            "subfield_label_short": [f"S{idx}" for idx in range(n)],
            "field_id": ["F1"] * n,
            "field_display_name": ["Field"] * n,
            "domain_id": ["D1"] * n,
            "domain_display_name": ["Domain"] * n,
            "n_available": [100] * n,
            "n_used": [100] * n,
            "year_min": [2000] * n,
            "year_max": [2024] * n,
            "metric_status": ["completed"] * (n - 1) + ["failed"],
            "metric_warning_message": [""] * n,
        }
    )
    base = np.linspace(0.1, 0.9, n)
    for idx, metric in enumerate(REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS):
        frame[metric] = base + (idx * 0.013)
    return frame


def test_reduced_interpretable_core_definition_is_exact_and_unique() -> None:
    grouped = [
        metric
        for metrics in REDUCED_INTERPRETABLE_EMBEDDING_CORE_GROUPS.values()
        for metric in metrics
    ]

    assert len(REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS) == 11
    assert len(set(REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS)) == 11
    assert grouped == REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS
    assert not (
        set(REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS)
        & set(EXCLUDED_FROM_CLEAN_CORE_V1)
    )


def test_reduced_metrics_exist_in_current_full_metric_table_if_available() -> None:
    path = Path("data/processed/subfield_embedding_space_metrics.parquet")
    if not path.exists():
        return
    frame = pd.read_parquet(path)

    missing = [
        metric
        for metric in REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS
        if metric not in frame.columns
    ]

    assert missing == []


def test_build_reduced_outputs_writes_expected_files_and_matrices(tmp_path) -> None:
    summary = build_reduced_interpretable_embedding_core_outputs(
        synthetic_full_embedding_metrics(),
        input_path="input.parquet",
        output_dir=tmp_path,
    )
    paths = output_paths(tmp_path)

    assert summary["core_metric_count"] == 11
    assert summary["n_rows_input"] == 7
    assert summary["n_rows_eligible"] == 6
    assert not summary["missing_core_metrics"]
    assert paths["metrics_parquet"].exists()
    assert paths["spearman_heatmap"].exists()
    assert paths["pearson_heatmap"].exists()
    assert paths["histograms"].exists()
    assert paths["summary_md"].exists()

    spearman = pd.read_csv(paths["spearman_matrix"], index_col=0)
    pearson = pd.read_csv(paths["pearson_matrix"], index_col=0)
    assert spearman.shape == (11, 11)
    assert pearson.shape == (11, 11)
    assert spearman.index.tolist() == REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS
    assert spearman.columns.tolist() == REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS
    assert pearson.index.tolist() == REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS
    assert pearson.columns.tolist() == REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS

    top_pairs = pd.read_csv(paths["top_spearman"])
    assert not top_pairs.empty
    assert not (top_pairs["metric_a"] == top_pairs["metric_b"]).any()
