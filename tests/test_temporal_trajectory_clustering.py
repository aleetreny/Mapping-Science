from __future__ import annotations

import numpy as np
import pandas as pd

from src.temporal_trajectory_clustering import (
    STATIC_STRUCTURAL_METRICS,
    load_complete_window_metrics,
    temporal_feature_matrices,
)


def tiny_window_metrics(n_subfields: int = 3) -> pd.DataFrame:
    rows = []
    for subfield_idx in range(n_subfields):
        for window_index in range(5):
            row = {
                "subfield_id": str(1000 + subfield_idx),
                "subfield_display_name": f"Subfield {subfield_idx}",
                "field_display_name": f"Field {subfield_idx}",
                "domain_display_name": "Domain A" if subfield_idx % 2 == 0 else "Domain B",
                "window_index": window_index,
                "window_label": f"W{window_index}",
                "metric_status": "completed",
            }
            for metric_idx, metric in enumerate(STATIC_STRUCTURAL_METRICS):
                row[metric] = float(subfield_idx + window_index + metric_idx / 10)
            rows.append(row)
    return pd.DataFrame(rows)


def test_temporal_feature_shapes_and_no_missing_values(tmp_path) -> None:
    path = tmp_path / "window_metrics.parquet"
    tiny_window_metrics().to_parquet(path, index=False)

    complete, dropped = load_complete_window_metrics(path)
    metadata, matrices, columns = temporal_feature_matrices(complete)

    assert dropped.empty
    assert len(metadata) == 3
    assert matrices["first_last_delta"].shape == (3, 8)
    assert matrices["linear_slope"].shape == (3, 8)
    assert matrices["full_trajectory"].shape == (3, 40)
    assert matrices["dynamic_summary"].shape == (3, 24)
    assert all(np.isfinite(matrix).all() for matrix in matrices.values())
    assert not any("umap" in column.lower() for values in columns.values() for column in values)


def test_temporal_loader_drops_incomplete_trajectories(tmp_path) -> None:
    frame = tiny_window_metrics()
    frame = frame.loc[~((frame["subfield_id"] == "1001") & (frame["window_index"] == 0))]
    path = tmp_path / "window_metrics.parquet"
    frame.to_parquet(path, index=False)

    complete, dropped = load_complete_window_metrics(path)

    assert set(complete["subfield_id"]) == {"1000", "1002"}
    assert dropped["subfield_id"].tolist() == ["1001"]
