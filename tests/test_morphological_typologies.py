from __future__ import annotations

import numpy as np
import pandas as pd

from src.morphological_typologies import (
    REDUCED_METRICS,
    STATIC_STRUCTURAL_METRICS,
    feature_columns_for,
    load_reduced_metric_core,
    scale_feature_matrix,
)


def tiny_metric_core(n: int = 6) -> pd.DataFrame:
    rows = []
    for idx in range(n):
        row = {
            "subfield_id": f"S{idx}",
            "subfield_display_name": f"Subfield {idx}",
            "field_id": f"F{idx % 2}",
            "field_display_name": f"Field {idx % 2}",
            "domain_id": f"D{idx % 3}",
            "domain_display_name": f"Domain {idx % 3}",
            "metric_status": "completed",
        }
        for metric_idx, metric in enumerate(REDUCED_METRICS):
            row[metric] = float(idx + 1) + metric_idx / 10.0
        rows.append(row)
    return pd.DataFrame(rows)


def test_reduced_typology_features_are_exact_and_do_not_use_umap() -> None:
    assert len(REDUCED_METRICS) == 11
    assert len(STATIC_STRUCTURAL_METRICS) == 8
    assert feature_columns_for("all11") == REDUCED_METRICS
    assert feature_columns_for("static8") == STATIC_STRUCTURAL_METRICS
    assert all("umap" not in metric.lower() for metric in REDUCED_METRICS)


def test_load_reduced_metric_core_filters_failed_rows(tmp_path) -> None:
    frame = tiny_metric_core()
    frame.loc[0, "metric_status"] = "failed"
    path = tmp_path / "core.csv"
    frame.to_csv(path, index=False)

    loaded = load_reduced_metric_core(path)

    assert len(loaded) == len(frame) - 1
    assert loaded[REDUCED_METRICS].isna().sum().sum() == 0


def test_robust_and_family_balanced_scaling_are_finite() -> None:
    frame = tiny_metric_core()
    matrix, params = scale_feature_matrix(
        frame,
        metrics=REDUCED_METRICS,
        scaling="robust",
        family_balanced=True,
    )

    assert matrix.shape == (len(frame), len(REDUCED_METRICS))
    assert np.isfinite(matrix).all()
    assert set(params["family_weight"]) != {1.0}

