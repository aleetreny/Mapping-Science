from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.subfield_temporal_umap_panels import (
    plot_density_difference_panel,
    plot_temporal_density_panel,
    relative_density_grid,
    window_density_grids,
)
from src.temporal_common import TemporalWindow


def five_windows() -> list[TemporalWindow]:
    return [
        TemporalWindow(2000, 2004, 0),
        TemporalWindow(2005, 2009, 1),
        TemporalWindow(2010, 2014, 2),
        TemporalWindow(2015, 2019, 3),
        TemporalWindow(2020, 2024, 4),
    ]


def synthetic_temporal_coordinates(points_per_window: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows: list[dict[str, float | int]] = []
    for window in five_windows():
        center = np.array([window.window_index * 0.35, window.window_index * 0.12])
        coordinates = rng.normal(loc=center, scale=0.08, size=(points_per_window, 2))
        for idx, (x, y) in enumerate(coordinates):
            rows.append(
                {
                    "analysis_row_id": window.window_index * points_per_window + idx,
                    "publication_year": window.window_start + (idx % 5),
                    "umap_x": float(x),
                    "umap_y": float(y),
                }
            )
    return pd.DataFrame(rows)


def test_relative_density_grid_is_normalized_within_period() -> None:
    frame = synthetic_temporal_coordinates()
    coordinates = frame[["umap_x", "umap_y"]].to_numpy(dtype=float)

    density = relative_density_grid(
        coordinates,
        xlim=(-1, 3),
        ylim=(-1, 2),
        grid_size=50,
        sigma=1.2,
    )

    assert density.shape == (50, 50)
    assert np.isclose(density.sum(), 1.0)


def test_window_density_grids_mark_insufficient_windows() -> None:
    frame = synthetic_temporal_coordinates(points_per_window=12)
    grids = window_density_grids(
        frame,
        windows=five_windows(),
        xlim=(-1, 3),
        ylim=(-1, 2),
        grid_size=40,
        sigma=1.0,
        min_points_for_density=30,
    )

    assert all(not grid["has_density"] for grid in grids)
    assert all(np.isclose(grid["density"].sum(), 0.0) for grid in grids)


def test_density_panels_write_static_pngs(tmp_path: Path) -> None:
    frame = synthetic_temporal_coordinates()
    density_path = tmp_path / "density_panel.png"
    difference_path = tmp_path / "density_difference_panel.png"

    density_counts = plot_temporal_density_panel(
        frame,
        density_path,
        subfield_label="Synthetic / Temporal UMAP",
        windows=five_windows(),
        density_grid_size=45,
        density_sigma=1.1,
        min_points_for_density=30,
        density_alpha=0.95,
        dpi=80,
    )
    difference_counts = plot_density_difference_panel(
        frame,
        difference_path,
        subfield_label="Synthetic / Temporal UMAP",
        windows=five_windows(),
        density_grid_size=45,
        density_sigma=1.1,
        min_points_for_density=30,
        density_alpha=0.95,
        dpi=80,
    )

    assert density_counts == (5, 0)
    assert difference_counts == (5, 0)
    assert density_path.exists()
    assert density_path.stat().st_size > 0
    assert difference_path.exists()
    assert difference_path.stat().st_size > 0
