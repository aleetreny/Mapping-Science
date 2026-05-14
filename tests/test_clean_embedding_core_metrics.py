from __future__ import annotations

import numpy as np
import pandas as pd

from src.clean_embedding_core_metrics import (
    CLEAN_CORE_EMBEDDING_METRIC_COLUMNS,
    CLEAN_CORE_EMBEDDING_METRIC_GROUPS,
    EXCLUDED_FROM_CLEAN_CORE,
    build_clean_embedding_core_outputs,
    output_paths,
)


def synthetic_broad_embedding_metrics(n: int = 6) -> pd.DataFrame:
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
            "metric_status": ["completed"] * n,
            "metric_warning_message": [""] * n,
            "n_available": [100] * n,
            "n_used": [100] * n,
            "year_min": [2000] * n,
            "year_max": [2024] * n,
        }
    )
    base = np.linspace(0.1, 0.9, n)
    for idx, metric in enumerate(CLEAN_CORE_EMBEDDING_METRIC_COLUMNS):
        frame[metric] = base + (idx * 0.01)
    return frame


def test_clean_core_definition_has_14_unique_grouped_metrics() -> None:
    grouped = [
        metric
        for metrics in CLEAN_CORE_EMBEDDING_METRIC_GROUPS.values()
        for metric in metrics
    ]

    assert len(CLEAN_CORE_EMBEDDING_METRIC_COLUMNS) == 14
    assert len(set(CLEAN_CORE_EMBEDDING_METRIC_COLUMNS)) == 14
    assert grouped == CLEAN_CORE_EMBEDDING_METRIC_COLUMNS
    assert not (set(CLEAN_CORE_EMBEDDING_METRIC_COLUMNS) & set(EXCLUDED_FROM_CLEAN_CORE))


def test_build_clean_embedding_core_outputs_writes_expected_files(tmp_path) -> None:
    summary = build_clean_embedding_core_outputs(
        synthetic_broad_embedding_metrics(),
        input_path="input.parquet",
        output_dir=tmp_path,
    )
    paths = output_paths(tmp_path)

    assert summary["n_clean_core_metric_columns"] == 14
    assert not summary["missing_final_metrics"]
    assert paths["clean_parquet"].exists()
    assert paths["spearman_heatmap"].exists()
    assert paths["pearson_heatmap"].exists()
    assert paths["summary_md"].exists()

    clean = pd.read_parquet(paths["clean_parquet"])
    assert CLEAN_CORE_EMBEDDING_METRIC_COLUMNS[-1] in clean.columns
