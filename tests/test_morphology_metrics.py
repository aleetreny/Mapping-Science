from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.morphology_metrics import (
    CORE_METRIC_COLUMNS_V2,
    DEFAULT_DENSITY_EXTENT,
    DIAGNOSTIC_METRIC_COLUMNS,
    METRIC_COLUMNS,
    anisotropy_ratio,
    build_density_grid,
    component_metrics,
    compute_subfield_metric_row,
    density_concentration_metrics,
    density_entropy,
    detect_density_peaks,
    mass_mask,
    normalize_coordinates,
    outlier_control_metrics,
    radial_metrics,
    temporal_metrics,
)
from src.subfield_labels import (
    add_subfield_label_columns,
    duplicate_subfield_names_report,
)


ROOT = Path(__file__).resolve().parents[1]


def coordinate_frame(
    coords: np.ndarray,
    years: np.ndarray | None = None,
    *,
    subfield_id: str = "1100",
    subfield_name: str = "Synthetic Subfield",
) -> pd.DataFrame:
    if years is None:
        years = np.resize(np.arange(2010, 2020), len(coords))
    records = []
    for row_id, ((x, y), year) in enumerate(zip(coords, years)):
        records.append(
            {
                "work_id": f"W{subfield_id}-{row_id:03d}",
                "analysis_row_id": row_id,
                "subfield_id": subfield_id,
                "subfield_display_name": subfield_name,
                "field_id": f"F{subfield_id}",
                "field_display_name": f"Field {subfield_id}",
                "domain_id": "D1",
                "domain_display_name": "Domain 1",
                "primary_topic_id": f"T{row_id}",
                "primary_topic_display_name": f"Topic {row_id}",
                "publication_year": int(year),
                "umap_x": float(x),
                "umap_y": float(y),
            }
        )
    return pd.DataFrame(records)


def normalize(coords: np.ndarray) -> np.ndarray:
    return normalize_coordinates(coords).normalized


def two_blob_cloud(seed: int = 42, n: int = 120) -> np.ndarray:
    rng = np.random.default_rng(seed)
    left = rng.normal(loc=(-2.5, 0.0), scale=0.18, size=(n // 2, 2))
    right = rng.normal(loc=(2.5, 0.0), scale=0.18, size=(n - n // 2, 2))
    return np.vstack([left, right])


def moving_cloud(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    coords = []
    years = []
    for year in range(2010, 2020):
        center = np.array([(year - 2010) * 0.25, 0.0])
        coords.append(center + rng.normal(scale=0.04, size=(8, 2)))
        years.extend([year] * 8)
    return np.vstack(coords), np.asarray(years)


def radial_trend_cloud(noisy: bool = False) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(55 if not noisy else 56)
    coords = []
    years = []
    for year in range(2010, 2020):
        if noisy:
            radius = rng.uniform(0.25, 1.25)
        else:
            radius = 0.25 + 0.08 * (year - 2010)
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        points = np.column_stack([np.cos(angles) * radius, np.sin(angles) * radius])
        points += rng.normal(scale=0.01, size=points.shape)
        coords.append(points)
        years.extend([year] * len(points))
    return np.vstack(coords), np.asarray(years)


def erratic_moving_cloud(seed: int = 57) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 0.6],
            [1.1, 0.6],
            [2.0, -0.3],
            [2.1, -0.3],
            [3.0, 0.7],
            [3.1, 0.7],
            [4.0, -0.4],
            [4.1, -0.4],
        ]
    )
    coords = []
    years = []
    for year, center in zip(range(2010, 2020), centers):
        coords.append(center + rng.normal(scale=0.01, size=(8, 2)))
        years.extend([year] * 8)
    return np.vstack(coords), np.asarray(years)


def entropy_trend_cloud(increasing: bool = True) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(72 if increasing else 73)
    coords = []
    years = []
    for year in range(2010, 2020):
        step = year - 2010
        scale = 0.05 + 0.05 * step if increasing else 0.50 - 0.045 * step
        coords.append(rng.normal(scale=scale, size=(35, 2)))
        years.extend([year] * 35)
    return np.vstack(coords), np.asarray(years)


def test_core_metric_columns_v2_are_curated_25() -> None:
    assert len(CORE_METRIC_COLUMNS_V2) == 25
    assert METRIC_COLUMNS == CORE_METRIC_COLUMNS_V2
    for demoted in [
        "density_gini",
        "effective_area_50",
        "support_circularity",
        "hole_count",
        "outlier_share_r_gt_1",
        "outlier_share_outside_density_extent",
        "outlier_share_r_gt_1_5",
    ]:
        assert demoted not in CORE_METRIC_COLUMNS_V2
        assert demoted in DIAGNOSTIC_METRIC_COLUMNS
    assert "density_entropy_slope_by_year" in CORE_METRIC_COLUMNS_V2
    assert "density_entropy_slope_r2" in DIAGNOSTIC_METRIC_COLUMNS


def test_coordinate_normalization_is_translation_and_scale_invariant() -> None:
    rng = np.random.default_rng(42)
    coords = rng.normal(size=(80, 2))
    transformed = coords * 7.5 + np.array([100.0, -50.0])

    first = normalize_coordinates(coords)
    second = normalize_coordinates(transformed)

    assert np.allclose(first.normalized, second.normalized)
    assert np.isclose(second.raw_r95, first.raw_r95 * 7.5)


def test_radial_metrics_are_finite_on_simple_cloud() -> None:
    rng = np.random.default_rng(7)
    coords_norm = normalize(rng.normal(size=(120, 2)))
    metrics = radial_metrics(np.linalg.norm(coords_norm, axis=1))

    assert np.isfinite(metrics["radial_tail_index"])
    assert np.isfinite(metrics["radial_iqr_index"])
    assert metrics["radial_tail_index"] > 0


def test_density_grid_sums_to_one() -> None:
    rng = np.random.default_rng(13)
    coords_norm = normalize(rng.normal(size=(100, 2)))
    grid = build_density_grid(coords_norm, grid_size=50)

    assert np.isclose(grid.density.sum(), 1.0)
    assert grid.density.shape == (50, 50)


def test_density_grid_uses_fixed_extent_by_default() -> None:
    rng = np.random.default_rng(14)
    coords_norm = normalize(rng.normal(size=(100, 2)))
    grid = build_density_grid(coords_norm, grid_size=50)

    assert grid.extent == DEFAULT_DENSITY_EXTENT
    assert np.isclose(grid.x_centers[0], -1.5 + (3.0 / 50) / 2)
    assert np.isclose(grid.x_centers[-1], 1.5 - (3.0 / 50) / 2)
    assert np.isclose(grid.cell_area, (3.0 / 50) ** 2)


def test_density_grid_rejects_invalid_extent() -> None:
    rng = np.random.default_rng(15)
    coords_norm = normalize(rng.normal(size=(100, 2)))

    with pytest.raises(ValueError, match="x_min < x_max"):
        build_density_grid(coords_norm, grid_size=50, density_extent=(1, -1, -1, 1))


def test_density_entropy_higher_for_spread_cloud_than_concentrated_cloud() -> None:
    rng = np.random.default_rng(21)
    angles = rng.uniform(0, 2 * np.pi, size=220)
    radii = np.sqrt(rng.uniform(0, 1, size=220))
    spread = np.column_stack([np.cos(angles) * radii, np.sin(angles) * radii])
    center = rng.normal(scale=0.03, size=(200, 2))
    tail_angles = rng.uniform(0, 2 * np.pi, size=20)
    tail = np.column_stack([np.cos(tail_angles), np.sin(tail_angles)])
    concentrated = np.vstack([center, tail])

    spread_entropy = density_entropy(build_density_grid(normalize(spread), grid_size=60).density)
    concentrated_entropy = density_entropy(
        build_density_grid(normalize(concentrated), grid_size=60).density
    )

    assert spread_entropy > concentrated_entropy


def test_single_extreme_outlier_does_not_radically_change_effective_area() -> None:
    rng = np.random.default_rng(22)
    cloud = rng.normal(scale=0.20, size=(200, 2))
    with_outlier = np.vstack([cloud, np.array([[100.0, 100.0]])])

    base_grid = build_density_grid(normalize(cloud), grid_size=80)
    outlier_grid = build_density_grid(normalize(with_outlier), grid_size=80)
    base_metrics = density_concentration_metrics(base_grid)
    outlier_metrics = density_concentration_metrics(outlier_grid)

    for column in ["effective_area_50", "effective_area_90"]:
        ratio = outlier_metrics[column] / base_metrics[column]
        assert 0.5 < ratio < 2.0


def test_outlier_controls_report_extreme_points() -> None:
    rng = np.random.default_rng(23)
    cloud = rng.normal(scale=0.20, size=(200, 2))
    with_outlier = np.vstack([cloud, np.array([[100.0, 100.0]])])
    base_controls = outlier_control_metrics(normalize(cloud))
    outlier_controls = outlier_control_metrics(normalize(with_outlier))

    assert outlier_controls["max_normalized_radius"] > base_controls["max_normalized_radius"]
    assert outlier_controls["outlier_share_r_gt_1_5"] > 0
    assert outlier_controls["outlier_share_outside_density_extent"] > 0


def test_peak_count_detects_two_blob_cloud() -> None:
    grid = build_density_grid(normalize(two_blob_cloud()), grid_size=80)
    _, peak_indices = detect_density_peaks(grid.density)

    assert len(peak_indices) >= 2


def test_dense_component_count_higher_for_separated_blobs_than_one_blob() -> None:
    rng = np.random.default_rng(31)
    one_blob_grid = build_density_grid(normalize(rng.normal(scale=0.4, size=(120, 2))), grid_size=70)
    two_blob_grid = build_density_grid(normalize(two_blob_cloud(seed=31)), grid_size=70)

    one_core = mass_mask(one_blob_grid.density, 0.50)
    two_core = mass_mask(two_blob_grid.density, 0.50)
    _, one_peaks = detect_density_peaks(one_blob_grid.density)
    _, two_peaks = detect_density_peaks(two_blob_grid.density)
    one_metrics = component_metrics(one_blob_grid, peak_indices=one_peaks, core_mask=one_core)
    two_metrics = component_metrics(two_blob_grid, peak_indices=two_peaks, core_mask=two_core)

    assert two_metrics["dense_component_count"] > one_metrics["dense_component_count"]


def test_anisotropy_higher_for_elongated_cloud_than_round_cloud() -> None:
    rng = np.random.default_rng(44)
    round_cloud = normalize(rng.normal(size=(160, 2)))
    elongated = normalize(
        np.column_stack(
            [
                rng.normal(scale=4.0, size=160),
                rng.normal(scale=0.15, size=160),
            ]
        )
    )

    assert anisotropy_ratio(elongated) > anisotropy_ratio(round_cloud)


def test_temporal_drift_positive_for_steadily_moving_centroids() -> None:
    coords, years = moving_cloud()
    coords_norm = normalize(coords)

    metrics, controls, warnings = temporal_metrics(
        coords_norm,
        years,
        year_min=2010,
        year_max=2019,
    )

    assert metrics["centroid_drift_early_late"] > 0
    assert metrics["annual_centroid_path_length"] > metrics["centroid_drift_early_late"]
    assert metrics["directionality_ratio"] > 0
    assert controls["n_years_available"] == 10
    assert warnings == []


def test_annual_centroid_step_cv_is_higher_for_erratic_motion() -> None:
    smooth_coords, smooth_years = moving_cloud(seed=61)
    erratic_coords, erratic_years = erratic_moving_cloud(seed=62)
    smooth_metrics, _, _ = temporal_metrics(
        normalize(smooth_coords),
        smooth_years,
        year_min=2010,
        year_max=2019,
    )
    erratic_metrics, _, _ = temporal_metrics(
        normalize(erratic_coords),
        erratic_years,
        year_min=2010,
        year_max=2019,
    )

    assert smooth_metrics["annual_centroid_step_cv"] < 0.5
    assert erratic_metrics["annual_centroid_step_cv"] > smooth_metrics[
        "annual_centroid_step_cv"
    ]


def test_radial_expansion_r2_tracks_cleaner_radial_trend() -> None:
    clean_coords, clean_years = radial_trend_cloud(noisy=False)
    noisy_coords, noisy_years = radial_trend_cloud(noisy=True)
    clean_metrics, _, _ = temporal_metrics(
        normalize(clean_coords),
        clean_years,
        year_min=2010,
        year_max=2019,
    )
    noisy_metrics, _, _ = temporal_metrics(
        normalize(noisy_coords),
        noisy_years,
        year_min=2010,
        year_max=2019,
    )

    assert clean_metrics["radial_expansion_r2"] > 0.9
    assert noisy_metrics["radial_expansion_r2"] < clean_metrics["radial_expansion_r2"]


def test_density_entropy_slope_tracks_temporal_diversification() -> None:
    expanding_coords, expanding_years = entropy_trend_cloud(increasing=True)
    concentrating_coords, concentrating_years = entropy_trend_cloud(increasing=False)
    expanding_metrics, expanding_controls, expanding_warnings = temporal_metrics(
        normalize(expanding_coords),
        expanding_years,
        year_min=2010,
        year_max=2019,
        grid_size=50,
    )
    concentrating_metrics, concentrating_controls, concentrating_warnings = temporal_metrics(
        normalize(concentrating_coords),
        concentrating_years,
        year_min=2010,
        year_max=2019,
        grid_size=50,
    )

    assert expanding_metrics["density_entropy_slope_by_year"] > 0
    assert concentrating_metrics["density_entropy_slope_by_year"] < 0
    assert expanding_controls["n_density_entropy_years"] == 10
    assert concentrating_controls["n_density_entropy_years"] == 10
    assert not any("density_entropy_slope_by_year unavailable" in warning for warning in expanding_warnings)
    assert not any("density_entropy_slope_by_year unavailable" in warning for warning in concentrating_warnings)


def test_density_entropy_slope_warns_when_too_few_years() -> None:
    rng = np.random.default_rng(74)
    coords = np.vstack(
        [
            rng.normal(scale=0.10, size=(10, 2)),
            rng.normal(scale=0.30, size=(10, 2)),
        ]
    )
    years = np.array([2010] * 10 + [2011] * 10)

    metrics, controls, warnings = temporal_metrics(
        normalize(coords),
        years,
        year_min=2010,
        year_max=2019,
        grid_size=40,
    )

    assert np.isnan(metrics["density_entropy_slope_by_year"])
    assert np.isnan(metrics["density_entropy_slope_r2"])
    assert controls["n_density_entropy_years"] == 2
    assert any("density_entropy_slope_by_year unavailable" in warning for warning in warnings)


def test_cli_runs_on_tiny_manifest_and_coordinate_parquets(tmp_path: Path) -> None:
    first_coords, first_years = moving_cloud(seed=101)
    second_coords = two_blob_cloud(seed=102, n=100)
    second_years = np.resize(np.arange(2010, 2020), len(second_coords))

    first_path = tmp_path / "1100.parquet"
    second_path = tmp_path / "2200.parquet"
    coordinate_frame(first_coords, first_years, subfield_id="1100").to_parquet(
        first_path,
        index=False,
    )
    coordinate_frame(
        second_coords,
        second_years,
        subfield_id="2200",
        subfield_name="Two Blob Subfield",
    ).to_parquet(second_path, index=False)

    manifest = pd.DataFrame(
        [
            {
                "subfield_id": "1100",
                "subfield_name": "Synthetic Subfield",
                "n_available": len(first_coords),
                "n_used": len(first_coords),
                "year_min": 2010,
                "year_max": 2019,
                "status": "completed",
                "coordinate_path": str(first_path),
            },
            {
                "subfield_id": "2200",
                "subfield_name": "Two Blob Subfield",
                "n_available": len(second_coords),
                "n_used": len(second_coords),
                "year_min": 2010,
                "year_max": 2019,
                "status": "completed",
                "coordinate_path": str(second_path),
            },
        ]
    )
    manifest_path = tmp_path / "manifest.parquet"
    output_parquet = tmp_path / "metrics.parquet"
    output_csv = tmp_path / "metrics.csv"
    summary_path = tmp_path / "summary.json"
    dictionary_path = tmp_path / "dictionary.csv"
    manifest.to_parquet(manifest_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "11_compute_subfield_morphology_metrics.py"),
            "--manifest-path",
            str(manifest_path),
            "--output-parquet",
            str(output_parquet),
            "--output-csv",
            str(output_csv),
            "--summary-path",
            str(summary_path),
            "--dictionary-path",
            str(dictionary_path),
            "--limit-subfields",
            "2",
            "--grid-size",
            "50",
            "--k-neighbors",
            "5",
            "--mst-max-points",
            "80",
            "--overwrite",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    metrics = pd.read_parquet(output_parquet)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    dictionary = pd.read_csv(dictionary_path)

    assert len(metrics) == 2
    assert set(METRIC_COLUMNS).issubset(metrics.columns)
    assert metrics["metric_status"].isin(["completed", "completed_with_warnings"]).all()
    assert summary["n_subfields_attempted"] == 2
    assert summary["n_failed"] == 0
    assert set(METRIC_COLUMNS).issubset(set(dictionary["metric_name"]))


def test_compute_row_contains_all_25_metric_columns() -> None:
    coords, years = moving_cloud(seed=88)
    row = compute_subfield_metric_row(
        coordinate_frame(coords, years),
        coordinate_path="synthetic.parquet",
        year_min=2010,
        year_max=2019,
        grid_size=50,
        k_neighbors=5,
        mst_max_points=80,
        random_state=42,
    )

    assert set(METRIC_COLUMNS).issubset(row.keys())
    assert len(METRIC_COLUMNS) == 25
    assert set(DIAGNOSTIC_METRIC_COLUMNS).issubset(row.keys())
    assert "density_entropy_slope_by_year" in row
    assert "density_entropy_slope_r2" in row
    assert "n_density_entropy_years" in row
    assert row["metric_status"] in {"completed", "completed_with_warnings"}
    assert (
        row["subfield_label_unique"]
        == "1100 | Domain 1 / Field 1100 / Synthetic Subfield"
    )
    assert row["subfield_label_short"] == "1100 | Field 1100 / Synthetic Subfield"
    assert row["density_x_min"] == DEFAULT_DENSITY_EXTENT[0]
    assert row["density_x_max"] == DEFAULT_DENSITY_EXTENT[1]
    assert row["density_y_min"] == DEFAULT_DENSITY_EXTENT[2]
    assert row["density_y_max"] == DEFAULT_DENSITY_EXTENT[3]
    assert "max_normalized_radius" in row
    assert "outlier_share_outside_density_extent" in row


def test_duplicate_display_names_do_not_break_metric_rows() -> None:
    coords, years = moving_cloud(seed=90)
    first = compute_subfield_metric_row(
        coordinate_frame(coords, years, subfield_id="1303", subfield_name="Biochemistry"),
        coordinate_path="first.parquet",
        year_min=2010,
        year_max=2019,
        grid_size=50,
        k_neighbors=5,
        mst_max_points=80,
        random_state=42,
    )
    second_frame = coordinate_frame(
        coords + np.array([3.0, 0.0]),
        years,
        subfield_id="2704",
        subfield_name="Biochemistry",
    )
    second_frame["field_display_name"] = "Medicine"
    second_frame["domain_display_name"] = "Health Sciences"
    second = compute_subfield_metric_row(
        second_frame,
        coordinate_path="second.parquet",
        year_min=2010,
        year_max=2019,
        grid_size=50,
        k_neighbors=5,
        mst_max_points=80,
        random_state=42,
    )

    rows = add_subfield_label_columns(pd.DataFrame([first, second]))
    report = duplicate_subfield_names_report(rows)

    assert rows["subfield_display_name_is_duplicated"].tolist() == [True, True]
    assert len(report) == 2
    assert "1303 | Domain 1 / Field 1303 / Biochemistry" in set(
        report["subfield_label_unique"]
    )
    assert "2704 | Health Sciences / Medicine / Biochemistry" in set(
        report["subfield_label_unique"]
    )


def test_compute_row_rejects_year_window_outside_morphology_period() -> None:
    coords, years = moving_cloud(seed=89)

    with pytest.raises(ValueError, match="2010-2019"):
        compute_subfield_metric_row(
            coordinate_frame(coords, years),
            coordinate_path="synthetic.parquet",
            year_min=2010,
            year_max=2020,
            grid_size=50,
            k_neighbors=5,
            mst_max_points=80,
            random_state=42,
        )
