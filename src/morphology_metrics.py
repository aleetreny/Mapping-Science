from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors


REQUIRED_COORDINATE_COLUMNS = {
    "work_id",
    "analysis_row_id",
    "subfield_id",
    "subfield_display_name",
    "field_id",
    "field_display_name",
    "domain_id",
    "domain_display_name",
    "publication_year",
    "umap_x",
    "umap_y",
}

METRIC_COLUMNS = [
    "radial_tail_index",
    "radial_iqr_index",
    "knn_median_distance",
    "knn_distance_cv",
    "density_entropy",
    "density_gini",
    "peak_dominance",
    "effective_area_50",
    "effective_area_90",
    "core_periphery_ratio",
    "density_peak_count",
    "peak_mass_entropy",
    "dense_component_count",
    "largest_component_mass_share",
    "component_separation_index",
    "mst_gap_index",
    "anisotropy_ratio",
    "support_solidity",
    "support_circularity",
    "boundary_complexity",
    "hole_count",
    "centroid_drift_early_late",
    "annual_centroid_path_length",
    "directionality_ratio",
    "radial_expansion_slope",
]

CONTROL_COLUMNS = [
    "subfield_id",
    "subfield_display_name",
    "field_id",
    "field_display_name",
    "domain_id",
    "domain_display_name",
    "n_points",
    "year_min",
    "year_max",
    "metric_status",
    "metric_error_message",
    "metric_warning_message",
    "coordinate_path",
    "raw_umap_x_min",
    "raw_umap_x_max",
    "raw_umap_y_min",
    "raw_umap_y_max",
    "raw_r95",
    "raw_center_x",
    "raw_center_y",
    "normalized_center_x",
    "normalized_center_y",
    "grid_size",
    "k_neighbors",
    "mst_points_used",
    "mst_sampling_applied",
    "n_years_available",
    "n_early_points",
    "n_late_points",
]

OUTPUT_COLUMNS = CONTROL_COLUMNS + METRIC_COLUMNS
TEMPORAL_MIN_POINTS = 3
EPS = 1e-12


@dataclass(frozen=True)
class NormalizedCoordinates:
    raw: np.ndarray
    normalized: np.ndarray
    finite_mask: np.ndarray
    center: np.ndarray
    raw_r95: float
    radial_normalized: np.ndarray
    raw_bounds: dict[str, float]


@dataclass(frozen=True)
class DensityGrid:
    density: np.ndarray
    x_centers: np.ndarray
    y_centers: np.ndarray
    cell_area: float
    method: str


def stable_int_seed(random_state: int, key: object) -> int:
    payload = f"{random_state}:{key}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=4).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def validate_coordinate_frame(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COORDINATE_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            "coordinate parquet is missing required columns: "
            f"{', '.join(sorted(missing))}"
        )


def validate_morphology_year_window(year_min: int, year_max: int) -> None:
    if year_min < 2010 or year_max > 2019 or year_min > year_max:
        raise ValueError(
            "morphology metrics must use a year window inside 2010-2019; "
            f"got {year_min}-{year_max}"
        )


def finite_coordinate_frame(
    frame: pd.DataFrame,
    *,
    year_min: int,
    year_max: int,
) -> pd.DataFrame:
    validate_coordinate_frame(frame)
    validate_morphology_year_window(year_min, year_max)
    years = pd.to_numeric(frame["publication_year"], errors="coerce")
    x = pd.to_numeric(frame["umap_x"], errors="coerce")
    y = pd.to_numeric(frame["umap_y"], errors="coerce")
    mask = (
        years.between(year_min, year_max, inclusive="both")
        & np.isfinite(x)
        & np.isfinite(y)
    )
    filtered = frame.loc[mask].copy()
    filtered["publication_year"] = years.loc[mask].astype("int64")
    filtered["umap_x"] = x.loc[mask].astype(float)
    filtered["umap_y"] = y.loc[mask].astype(float)
    if filtered.empty:
        raise ValueError(
            f"no finite coordinates remain inside year window {year_min}-{year_max}"
        )
    return filtered.reset_index(drop=True)


def normalize_coordinates(coords: np.ndarray) -> NormalizedCoordinates:
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (n_points, 2)")

    finite_mask = np.isfinite(coords).all(axis=1)
    finite_coords = coords[finite_mask]
    if len(finite_coords) < 3:
        raise ValueError("at least three finite coordinate points are required")

    center = np.nanmedian(finite_coords, axis=0)
    radial_raw = np.linalg.norm(finite_coords - center, axis=1)
    raw_r95 = float(np.percentile(radial_raw, 95))
    if not np.isfinite(raw_r95) or raw_r95 <= 0:
        raise ValueError(f"invalid raw_r95 for coordinate normalization: {raw_r95}")

    normalized = (finite_coords - center) / raw_r95
    radial_normalized = np.linalg.norm(normalized, axis=1)
    raw_bounds = {
        "raw_umap_x_min": float(np.min(finite_coords[:, 0])),
        "raw_umap_x_max": float(np.max(finite_coords[:, 0])),
        "raw_umap_y_min": float(np.min(finite_coords[:, 1])),
        "raw_umap_y_max": float(np.max(finite_coords[:, 1])),
    }
    return NormalizedCoordinates(
        raw=finite_coords,
        normalized=normalized,
        finite_mask=finite_mask,
        center=center,
        raw_r95=raw_r95,
        radial_normalized=radial_normalized,
        raw_bounds=raw_bounds,
    )


def density_grid_extent(coords_norm: np.ndarray) -> tuple[float, float, float, float]:
    finite = np.asarray(coords_norm, dtype=float)
    finite = finite[np.isfinite(finite).all(axis=1)]
    max_abs = float(np.max(np.abs(finite))) if len(finite) else 1.0
    half_range = max(1.25, max_abs * 1.05)
    return -half_range, half_range, -half_range, half_range


def build_density_grid(
    coords_norm: np.ndarray,
    *,
    grid_size: int = 160,
    kde_max_points: int = 5000,
) -> DensityGrid:
    if grid_size < 10:
        raise ValueError("grid_size must be at least 10")

    coords_norm = np.asarray(coords_norm, dtype=float)
    finite = coords_norm[np.isfinite(coords_norm).all(axis=1)]
    if len(finite) < 3:
        raise ValueError("at least three finite points are required for density")

    x_min, x_max, y_min, y_max = density_grid_extent(finite)
    x_edges = np.linspace(x_min, x_max, grid_size + 1)
    y_edges = np.linspace(y_min, y_max, grid_size + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    cell_area = float((x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0]))

    method = "kde"
    try:
        if len(finite) > kde_max_points:
            raise ValueError("point count above KDE threshold")
        kde = gaussian_kde(finite.T)
        xx, yy = np.meshgrid(x_centers, y_centers, indexing="ij")
        density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(
            grid_size,
            grid_size,
        )
    except Exception:
        method = "histogram_gaussian_filter"
        hist, _, _ = np.histogram2d(finite[:, 0], finite[:, 1], bins=[x_edges, y_edges])
        density = ndimage.gaussian_filter(hist.astype(float), sigma=1.25)

    density = np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
    density = np.maximum(density, 0.0)
    total = float(density.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("density grid has non-positive total mass")
    density = density / total
    return DensityGrid(
        density=density,
        x_centers=x_centers,
        y_centers=y_centers,
        cell_area=cell_area,
        method=method,
    )


def radial_metrics(radial_normalized: np.ndarray) -> dict[str, float]:
    radial = np.asarray(radial_normalized, dtype=float)
    radial = radial[np.isfinite(radial)]
    if len(radial) == 0:
        return {"radial_tail_index": np.nan, "radial_iqr_index": np.nan}

    r25, r50, r75, r95 = np.percentile(radial, [25, 50, 75, 95])
    if r50 <= 0:
        return {"radial_tail_index": np.nan, "radial_iqr_index": np.nan}
    return {
        "radial_tail_index": float(r95 / r50),
        "radial_iqr_index": float((r75 - r25) / r50),
    }


def knn_metrics(coords_norm: np.ndarray, *, k_neighbors: int) -> dict[str, float]:
    coords_norm = np.asarray(coords_norm, dtype=float)
    if k_neighbors < 1:
        raise ValueError("k_neighbors must be positive")
    if len(coords_norm) < 2:
        return {"knn_median_distance": np.nan, "knn_distance_cv": np.nan}

    effective_k = min(k_neighbors, len(coords_norm) - 1)
    model = NearestNeighbors(n_neighbors=effective_k + 1, metric="euclidean")
    model.fit(coords_norm)
    distances, _ = model.kneighbors(coords_norm)
    kth_distances = distances[:, effective_k]
    mean_distance = float(np.mean(kth_distances))
    cv = float(np.std(kth_distances) / mean_distance) if mean_distance > 0 else np.nan
    return {
        "knn_median_distance": float(np.median(kth_distances)),
        "knn_distance_cv": cv,
    }


def density_entropy(density: np.ndarray) -> float:
    values = np.asarray(density, dtype=float)
    positive = values[values > 0]
    if len(positive) <= 1:
        return 0.0
    entropy = -float(np.sum(positive * np.log(positive)))
    return float(entropy / np.log(len(positive)))


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values >= 0)]
    if len(values) == 0:
        return np.nan
    total = float(values.sum())
    if total <= 0:
        return 0.0
    sorted_values = np.sort(values)
    n = len(sorted_values)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_values) / (n * total)) - ((n + 1) / n))


def mass_mask(density: np.ndarray, mass_fraction: float) -> np.ndarray:
    if not 0 < mass_fraction <= 1:
        raise ValueError("mass_fraction must be in (0, 1]")
    flat = np.asarray(density, dtype=float).ravel()
    if flat.sum() <= 0:
        return np.zeros_like(density, dtype=bool)
    order = np.argsort(flat)[::-1]
    cumulative = np.cumsum(flat[order])
    cutoff = int(np.searchsorted(cumulative, mass_fraction, side="left"))
    cutoff = min(cutoff, len(order) - 1)
    selected = np.zeros_like(flat, dtype=bool)
    selected[order[: cutoff + 1]] = True
    return selected.reshape(density.shape)


def effective_area(density: np.ndarray, *, mass_fraction: float, cell_area: float) -> float:
    mask = mass_mask(density, mass_fraction)
    return float(mask.sum() * cell_area)


def detect_density_peaks(
    density: np.ndarray,
    *,
    relative_threshold: float = 0.10,
    neighbourhood_size: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    density = np.asarray(density, dtype=float)
    if density.size == 0 or np.max(density) <= 0:
        return np.zeros_like(density, dtype=bool), np.empty((0, 2), dtype=int)

    size = max(3, int(neighbourhood_size))
    if size % 2 == 0:
        size += 1
    local_max = ndimage.maximum_filter(density, size=size, mode="nearest")
    candidate_mask = (density == local_max) & (
        density >= relative_threshold * float(np.max(density))
    )
    labels, n_labels = ndimage.label(candidate_mask)
    peak_indices: list[tuple[int, int]] = []
    peak_mask = np.zeros_like(candidate_mask, dtype=bool)
    for label_id in range(1, n_labels + 1):
        positions = np.argwhere(labels == label_id)
        if len(positions) == 0:
            continue
        values = density[positions[:, 0], positions[:, 1]]
        chosen = positions[int(np.argmax(values))]
        peak_mask[chosen[0], chosen[1]] = True
        peak_indices.append((int(chosen[0]), int(chosen[1])))
    if not peak_indices:
        return peak_mask, np.empty((0, 2), dtype=int)
    return peak_mask, np.asarray(peak_indices, dtype=int)


def peak_mass_entropy(
    density: np.ndarray,
    peak_indices: np.ndarray,
    core_mask: np.ndarray,
) -> float:
    if len(peak_indices) == 0:
        return np.nan
    if len(peak_indices) == 1:
        return 0.0

    core_positions = np.argwhere(core_mask)
    if len(core_positions) == 0:
        return np.nan
    distances = np.sum(
        (core_positions[:, None, :] - peak_indices[None, :, :]) ** 2,
        axis=2,
    )
    assignments = np.argmin(distances, axis=1)
    masses = np.zeros(len(peak_indices), dtype=float)
    core_values = density[core_positions[:, 0], core_positions[:, 1]]
    np.add.at(masses, assignments, core_values)
    positive = masses[masses > 0]
    if len(positive) <= 1:
        return 0.0
    probabilities = positive / positive.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return float(entropy / np.log(len(positive)))


def density_concentration_metrics(grid: DensityGrid) -> dict[str, float]:
    density = grid.density
    positive = density[density > 0]
    mean_positive = float(np.mean(positive)) if len(positive) else np.nan
    max_density = float(np.max(density))
    area_50 = effective_area(density, mass_fraction=0.50, cell_area=grid.cell_area)
    area_90 = effective_area(density, mass_fraction=0.90, cell_area=grid.cell_area)
    return {
        "density_entropy": density_entropy(density),
        "density_gini": gini(positive),
        "peak_dominance": float(max_density / mean_positive)
        if mean_positive and mean_positive > 0
        else np.nan,
        "effective_area_50": area_50,
        "effective_area_90": area_90,
        "core_periphery_ratio": float(area_50 / area_90) if area_90 > 0 else np.nan,
    }


def component_metrics(
    grid: DensityGrid,
    *,
    peak_indices: np.ndarray,
    core_mask: np.ndarray,
) -> dict[str, float]:
    density = grid.density
    structure = np.ones((3, 3), dtype=int)
    labels, n_components = ndimage.label(core_mask, structure=structure)
    masses: list[tuple[int, float]] = []
    for label_id in range(1, n_components + 1):
        mass = float(density[labels == label_id].sum())
        if mass > 0:
            masses.append((label_id, mass))

    if not masses:
        largest_share = np.nan
        separation = 0.0
    else:
        total_mass = sum(mass for _, mass in masses)
        largest_share = max(mass for _, mass in masses) / total_mass
        separation = component_separation_index(grid, labels, masses)

    return {
        "density_peak_count": int(len(peak_indices)),
        "peak_mass_entropy": peak_mass_entropy(density, peak_indices, core_mask),
        "dense_component_count": int(len(masses)),
        "largest_component_mass_share": float(largest_share),
        "component_separation_index": float(separation),
    }


def component_separation_index(
    grid: DensityGrid,
    labels: np.ndarray,
    component_masses: list[tuple[int, float]],
) -> float:
    if len(component_masses) < 2:
        return 0.0

    top_two = sorted(component_masses, key=lambda item: item[1], reverse=True)[:2]
    xx, yy = np.meshgrid(grid.x_centers, grid.y_centers, indexing="ij")
    centroids = []
    radii = []
    areas = []
    for label_id, mass in top_two:
        mask = labels == label_id
        weights = grid.density[mask]
        if weights.sum() <= 0:
            continue
        points = np.column_stack([xx[mask], yy[mask]])
        centroid = np.average(points, axis=0, weights=weights)
        distances = np.linalg.norm(points - centroid, axis=1)
        radius = math.sqrt(float(np.average(distances**2, weights=weights)))
        centroids.append(centroid)
        radii.append(radius)
        areas.append(float(mask.sum() * grid.cell_area))

    if len(centroids) < 2:
        return 0.0

    between = float(np.linalg.norm(centroids[0] - centroids[1]))
    average_radius = float(np.mean(radii))
    if average_radius <= EPS:
        average_radius = math.sqrt(max(float(np.mean(areas)), EPS) / math.pi)
    return float(between / max(average_radius, EPS))


def mst_gap_metric(
    coords_norm: np.ndarray,
    *,
    mst_max_points: int,
    random_state: int,
    subfield_id: object,
) -> tuple[float, int, bool]:
    coords_norm = np.asarray(coords_norm, dtype=float)
    if mst_max_points < 2:
        raise ValueError("mst_max_points must be at least 2")
    if len(coords_norm) < 2:
        return np.nan, len(coords_norm), False

    sampled = coords_norm
    sampling_applied = False
    if len(coords_norm) > mst_max_points:
        rng = np.random.default_rng(stable_int_seed(random_state, subfield_id))
        positions = np.sort(
            rng.choice(len(coords_norm), size=mst_max_points, replace=False)
        )
        sampled = coords_norm[positions]
        sampling_applied = True

    distances = squareform(pdist(sampled, metric="euclidean"))
    tree = minimum_spanning_tree(distances)
    edges = np.asarray(tree.data, dtype=float)
    edges = edges[edges > 0]
    if len(edges) == 0:
        return np.nan, len(sampled), sampling_applied
    median = float(np.median(edges))
    if median <= 0:
        return np.nan, len(sampled), sampling_applied
    return float(np.percentile(edges, 95) / median), len(sampled), sampling_applied


def anisotropy_ratio(coords_norm: np.ndarray) -> float:
    coords_norm = np.asarray(coords_norm, dtype=float)
    if len(coords_norm) < 2:
        return np.nan
    covariance = np.cov(coords_norm, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.maximum(eigenvalues, EPS)
    return float(math.sqrt(float(np.max(eigenvalues) / np.min(eigenvalues))))


def support_perimeter(mask: np.ndarray, *, dx: float, dy: float) -> float:
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask, pad_width=1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    left = padded[:-2, 1:-1]
    right = padded[2:, 1:-1]
    down = padded[1:-1, :-2]
    up = padded[1:-1, 2:]
    edges_x = np.count_nonzero(center & ~left) + np.count_nonzero(center & ~right)
    edges_y = np.count_nonzero(center & ~down) + np.count_nonzero(center & ~up)
    return float(edges_x * dy + edges_y * dx)


def hole_count(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    inverse = ~mask
    labels, n_labels = ndimage.label(inverse, structure=np.ones((3, 3), dtype=int))
    if n_labels == 0:
        return 0
    border_labels = set(labels[0, :])
    border_labels.update(labels[-1, :])
    border_labels.update(labels[:, 0])
    border_labels.update(labels[:, -1])
    border_labels.discard(0)
    holes = 0
    for label_id in range(1, n_labels + 1):
        if label_id not in border_labels:
            holes += 1
    return holes


def support_solidity(grid: DensityGrid, support_mask: np.ndarray) -> float:
    support_area = float(support_mask.sum() * grid.cell_area)
    if support_area <= 0:
        return np.nan
    xx, yy = np.meshgrid(grid.x_centers, grid.y_centers, indexing="ij")
    points = np.column_stack([xx[support_mask], yy[support_mask]])
    if len(points) < 3:
        return np.nan
    try:
        hull_area = float(ConvexHull(points).volume)
    except Exception:
        return np.nan
    if hull_area <= 0:
        return np.nan
    return float(min(1.0, support_area / max(hull_area, support_area)))


def support_shape_metrics(
    coords_norm: np.ndarray,
    grid: DensityGrid,
    *,
    support_mask: np.ndarray,
) -> dict[str, float]:
    dx = float(grid.x_centers[1] - grid.x_centers[0]) if len(grid.x_centers) > 1 else 1.0
    dy = float(grid.y_centers[1] - grid.y_centers[0]) if len(grid.y_centers) > 1 else 1.0
    area = float(support_mask.sum() * grid.cell_area)
    perimeter = support_perimeter(support_mask, dx=dx, dy=dy)
    circularity = (
        float((4 * math.pi * area) / (perimeter**2))
        if area > 0 and perimeter > 0
        else np.nan
    )
    return {
        "anisotropy_ratio": anisotropy_ratio(coords_norm),
        "support_solidity": support_solidity(grid, support_mask),
        "support_circularity": circularity,
        "boundary_complexity": float(1.0 / circularity)
        if circularity and circularity > 0
        else np.nan,
        "hole_count": int(hole_count(support_mask)),
    }


def temporal_metrics(
    coords_norm: np.ndarray,
    years: np.ndarray,
    *,
    year_min: int,
    year_max: int,
) -> tuple[dict[str, float], dict[str, int], list[str]]:
    coords_norm = np.asarray(coords_norm, dtype=float)
    years = np.asarray(years, dtype=int)
    warnings: list[str] = []

    available_years = sorted(int(year) for year in np.unique(years))
    early_mask = (years >= 2010) & (years <= 2012)
    late_mask = (years >= 2017) & (years <= 2019)
    n_early = int(np.count_nonzero(early_mask))
    n_late = int(np.count_nonzero(late_mask))

    if n_early >= TEMPORAL_MIN_POINTS and n_late >= TEMPORAL_MIN_POINTS:
        early_centroid = np.median(coords_norm[early_mask], axis=0)
        late_centroid = np.median(coords_norm[late_mask], axis=0)
        drift = float(np.linalg.norm(late_centroid - early_centroid))
    else:
        drift = np.nan
        warnings.append(
            "centroid_drift_early_late unavailable because early or late period "
            f"has fewer than {TEMPORAL_MIN_POINTS} papers"
        )

    annual_centroids: list[tuple[int, np.ndarray]] = []
    annual_median_radius: list[tuple[int, float]] = []
    radial = np.linalg.norm(coords_norm, axis=1)
    for year in range(year_min, year_max + 1):
        year_mask = years == year
        if np.count_nonzero(year_mask) < TEMPORAL_MIN_POINTS:
            continue
        annual_centroids.append((year, np.median(coords_norm[year_mask], axis=0)))
        annual_median_radius.append((year, float(np.median(radial[year_mask]))))

    if len(annual_centroids) >= 2:
        path = 0.0
        for (_, previous), (_, current) in zip(annual_centroids, annual_centroids[1:]):
            path += float(np.linalg.norm(current - previous))
    else:
        path = np.nan
        warnings.append(
            "annual_centroid_path_length unavailable because fewer than two years "
            f"have at least {TEMPORAL_MIN_POINTS} papers"
        )

    if np.isfinite(drift) and np.isfinite(path) and path > 0:
        directionality = float(drift / path)
    else:
        directionality = np.nan
        warnings.append("directionality_ratio unavailable because drift or path is unavailable")

    if len(annual_median_radius) >= 2:
        year_values = np.array([year for year, _ in annual_median_radius], dtype=float)
        radius_values = np.array([radius for _, radius in annual_median_radius], dtype=float)
        year_values = year_values - year_values.mean()
        slope = float(np.polyfit(year_values, radius_values, deg=1)[0])
    else:
        slope = np.nan
        warnings.append(
            "radial_expansion_slope unavailable because fewer than two years "
            f"have at least {TEMPORAL_MIN_POINTS} papers"
        )

    return (
        {
            "centroid_drift_early_late": drift,
            "annual_centroid_path_length": float(path),
            "directionality_ratio": directionality,
            "radial_expansion_slope": slope,
        },
        {
            "n_years_available": int(len(available_years)),
            "n_early_points": n_early,
            "n_late_points": n_late,
        },
        warnings,
    )


def compute_core_metrics(
    coords_norm: np.ndarray,
    years: np.ndarray,
    *,
    subfield_id: object,
    year_min: int,
    year_max: int,
    grid_size: int,
    k_neighbors: int,
    mst_max_points: int,
    random_state: int,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    warnings: list[str] = []
    grid = build_density_grid(coords_norm, grid_size=grid_size)
    support_mask = mass_mask(grid.density, 0.95)
    core_mask = mass_mask(grid.density, 0.50)
    _, peak_indices = detect_density_peaks(
        grid.density,
        neighbourhood_size=max(5, grid_size // 40),
    )

    metrics: dict[str, Any] = {}
    metrics.update(radial_metrics(np.linalg.norm(coords_norm, axis=1)))
    metrics.update(knn_metrics(coords_norm, k_neighbors=k_neighbors))
    metrics.update(density_concentration_metrics(grid))
    metrics.update(
        component_metrics(grid, peak_indices=peak_indices, core_mask=core_mask)
    )
    if not np.isfinite(metrics["peak_mass_entropy"]):
        warnings.append("peak_mass_entropy unavailable because no density peak was detected")

    mst_gap, mst_points_used, mst_sampling_applied = mst_gap_metric(
        coords_norm,
        mst_max_points=mst_max_points,
        random_state=random_state,
        subfield_id=subfield_id,
    )
    metrics["mst_gap_index"] = mst_gap
    metrics.update(
        support_shape_metrics(coords_norm, grid, support_mask=support_mask)
    )
    temporal, temporal_controls, temporal_warnings = temporal_metrics(
        coords_norm,
        years,
        year_min=year_min,
        year_max=year_max,
    )
    metrics.update(temporal)
    warnings.extend(temporal_warnings)

    controls = {
        "grid_size": int(grid_size),
        "k_neighbors": int(k_neighbors),
        "mst_points_used": int(mst_points_used),
        "mst_sampling_applied": bool(mst_sampling_applied),
        **temporal_controls,
    }
    return metrics, controls, warnings


def first_value(frame: pd.DataFrame, column: str, default: str = "") -> Any:
    if column not in frame.columns or frame.empty:
        return default
    value = frame[column].dropna()
    if value.empty:
        return default
    return value.iloc[0]


def compute_subfield_metric_row(
    coordinate_frame: pd.DataFrame,
    *,
    coordinate_path: str | Path,
    year_min: int,
    year_max: int,
    grid_size: int,
    k_neighbors: int,
    mst_max_points: int,
    random_state: int,
) -> dict[str, Any]:
    filtered = finite_coordinate_frame(
        coordinate_frame,
        year_min=year_min,
        year_max=year_max,
    )
    normalized = normalize_coordinates(filtered[["umap_x", "umap_y"]].to_numpy())
    coords_norm = normalized.normalized
    years = filtered["publication_year"].to_numpy(dtype=int)
    subfield_id = first_value(filtered, "subfield_id")

    metrics, controls, warnings = compute_core_metrics(
        coords_norm,
        years,
        subfield_id=subfield_id,
        year_min=year_min,
        year_max=year_max,
        grid_size=grid_size,
        k_neighbors=k_neighbors,
        mst_max_points=mst_max_points,
        random_state=random_state,
    )
    row: dict[str, Any] = {
        "subfield_id": str(first_value(filtered, "subfield_id")),
        "subfield_display_name": str(first_value(filtered, "subfield_display_name")),
        "field_id": str(first_value(filtered, "field_id")),
        "field_display_name": str(first_value(filtered, "field_display_name")),
        "domain_id": str(first_value(filtered, "domain_id")),
        "domain_display_name": str(first_value(filtered, "domain_display_name")),
        "n_points": int(len(filtered)),
        "year_min": int(year_min),
        "year_max": int(year_max),
        "metric_status": "completed_with_warnings" if warnings else "completed",
        "metric_error_message": "",
        "metric_warning_message": "; ".join(dict.fromkeys(warnings)),
        "coordinate_path": str(coordinate_path),
        **normalized.raw_bounds,
        "raw_r95": float(normalized.raw_r95),
        "raw_center_x": float(normalized.center[0]),
        "raw_center_y": float(normalized.center[1]),
        "normalized_center_x": float(np.median(coords_norm[:, 0])),
        "normalized_center_y": float(np.median(coords_norm[:, 1])),
        **controls,
        **metrics,
    }
    return metric_row_frame([row]).iloc[0].to_dict()


def failed_metric_row(
    *,
    manifest_row: pd.Series | dict[str, Any],
    coordinate_path: str | Path,
    year_min: int,
    year_max: int,
    grid_size: int,
    k_neighbors: int,
    error_message: str,
) -> dict[str, Any]:
    getter = manifest_row.get
    row: dict[str, Any] = {
        "subfield_id": str(getter("subfield_id", "")),
        "subfield_display_name": str(getter("subfield_name", "")),
        "field_id": "",
        "field_display_name": "",
        "domain_id": "",
        "domain_display_name": "",
        "n_points": 0,
        "year_min": int(year_min),
        "year_max": int(year_max),
        "metric_status": "failed",
        "metric_error_message": str(error_message),
        "metric_warning_message": "",
        "coordinate_path": str(coordinate_path),
        "grid_size": int(grid_size),
        "k_neighbors": int(k_neighbors),
    }
    for column in CONTROL_COLUMNS:
        row.setdefault(column, np.nan)
    for column in METRIC_COLUMNS:
        row[column] = np.nan
    return metric_row_frame([row]).iloc[0].to_dict()


def metric_row_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in OUTPUT_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[OUTPUT_COLUMNS]


def metric_dictionary_frame() -> pd.DataFrame:
    rows = metric_dictionary_rows()
    return pd.DataFrame(
        rows,
        columns=[
            "metric_name",
            "family",
            "definition",
            "interpretation",
            "higher_means",
            "computed_on",
            "notes",
        ],
    )


def metric_dictionary_rows() -> list[dict[str, str]]:
    metric_rows = [
        (
            "radial_tail_index",
            "Radial dispersion",
            "95th percentile normalized radius divided by median normalized radius.",
            "Captures the strength of long semantic tails relative to the center.",
            "Longer periphery or tails.",
        ),
        (
            "radial_iqr_index",
            "Radial dispersion",
            "Normalized radial interquartile range divided by median radius.",
            "Captures internal radial spread after robust scale normalization.",
            "Broader internal spread.",
        ),
        (
            "knn_median_distance",
            "Local granularity",
            "Median distance to the k-th nearest neighbour; k is 15 by default.",
            "Measures typical local sparsity.",
            "Locally sparser point cloud.",
        ),
        (
            "knn_distance_cv",
            "Local granularity",
            "Standard deviation divided by mean of per-point k-th nearest-neighbour distances.",
            "Measures heterogeneity in local density.",
            "More uneven local density.",
        ),
        (
            "density_entropy",
            "Density concentration",
            "Shannon entropy of positive density grid cells normalized by log positive-cell count.",
            "Measures whether density mass is broadly spread or concentrated.",
            "More evenly spread density.",
        ),
        (
            "density_gini",
            "Density concentration",
            "Gini coefficient of positive density grid values.",
            "Measures inequality in the density surface.",
            "Density dominated by fewer cells.",
        ),
        (
            "peak_dominance",
            "Density concentration",
            "Maximum density divided by mean positive density.",
            "Measures dominance of the strongest density peak.",
            "One very dominant peak.",
        ),
        (
            "effective_area_50",
            "Density concentration",
            "Grid-cell area needed to accumulate 50 percent of density mass.",
            "Estimates the area of the high-density semantic core.",
            "Larger dense core footprint.",
        ),
        (
            "effective_area_90",
            "Density concentration",
            "Grid-cell area needed to accumulate 90 percent of density mass.",
            "Estimates the effective semantic footprint.",
            "Larger occupied semantic footprint.",
        ),
        (
            "core_periphery_ratio",
            "Density concentration",
            "effective_area_50 divided by effective_area_90.",
            "Compares dense-core area to broader footprint.",
            "Core occupies more of the footprint.",
        ),
        (
            "density_peak_count",
            "Multimodality",
            "Count of local density maxima above 10 percent of the maximum density.",
            "Measures the number of semantic foci.",
            "More density peaks.",
        ),
        (
            "peak_mass_entropy",
            "Multimodality",
            "Entropy of core density mass assigned to nearest detected peak.",
            "Measures whether peak-associated masses are balanced.",
            "Several balanced peaks.",
        ),
        (
            "dense_component_count",
            "Fragmentation",
            "Connected components in cells accumulating the top 50 percent of density mass.",
            "Measures fragmentation of the dense semantic core.",
            "More dense islands.",
        ),
        (
            "largest_component_mass_share",
            "Fragmentation",
            "Largest dense-component mass divided by total dense-region mass.",
            "Measures dominance of the largest dense island.",
            "One dominant dense island.",
        ),
        (
            "component_separation_index",
            "Fragmentation",
            "Distance between the two largest dense-component centroids divided by their average radius.",
            "Measures separation of the two largest dense islands.",
            "More separated dense islands.",
        ),
        (
            "mst_gap_index",
            "Fragmentation",
            "95th percentile MST edge length divided by median MST edge length.",
            "Measures large point-cloud gaps using a minimum spanning tree.",
            "Larger discontinuities or gaps.",
        ),
        (
            "anisotropy_ratio",
            "Support shape",
            "Square root of largest covariance eigenvalue divided by smallest eigenvalue.",
            "Measures elongation of the normalized point cloud.",
            "More elongated semantic field.",
        ),
        (
            "support_solidity",
            "Support shape",
            "Occupied support area divided by convex-hull area of support cells.",
            "Measures how filled versus concave or fragmented the support is.",
            "More compact filled support.",
        ),
        (
            "support_circularity",
            "Support shape",
            "4*pi*support_area divided by support_perimeter squared.",
            "Measures compactness and roundness of occupied support.",
            "Rounder, less irregular support.",
        ),
        (
            "boundary_complexity",
            "Support shape",
            "support_perimeter squared divided by 4*pi*support_area.",
            "Inverse of support circularity.",
            "More irregular or elongated boundary.",
        ),
        (
            "hole_count",
            "Support topology",
            "Count of enclosed background components inside the 95 percent support mask.",
            "Measures internal voids in occupied support.",
            "More internal holes.",
        ),
        (
            "centroid_drift_early_late",
            "Temporal morphology",
            "Distance between median centroids for 2010-2012 and 2017-2019.",
            "Measures net semantic displacement before 2020.",
            "Larger early-to-late displacement.",
        ),
        (
            "annual_centroid_path_length",
            "Temporal morphology",
            "Sum of distances between consecutive available annual median centroids.",
            "Measures total semantic movement through the input window.",
            "More internally dynamic trajectory.",
        ),
        (
            "directionality_ratio",
            "Temporal morphology",
            "Early-late centroid distance divided by annual centroid path length.",
            "Measures whether movement is directional or wandering.",
            "More coherent directional movement.",
        ),
        (
            "radial_expansion_slope",
            "Temporal morphology",
            "Least-squares slope of annual median normalized radius on year.",
            "Measures radial expansion or contraction during 2010-2019.",
            "Semantic expansion over time.",
        ),
    ]
    rows = [
        {
            "metric_name": name,
            "family": family,
            "definition": definition,
            "interpretation": interpretation,
            "higher_means": higher,
            "computed_on": "Per-subfield normalized UMAP coordinates",
            "notes": "Coordinates are median-centered and divided by raw r95 before computation.",
        }
        for name, family, definition, interpretation, higher in metric_rows
    ]

    control_rows = [
        (
            "n_points",
            "Quality/control",
            "Number of finite coordinate rows in the manifest year window.",
            "Sample size used for metric computation.",
            "More points available.",
            "Coordinate parquet rows",
            "Rows outside the manifest year window are excluded.",
        ),
        (
            "raw_r95",
            "Quality/control",
            "95th percentile raw UMAP radius around the raw median center.",
            "Raw scale diagnostic used for normalization.",
            "Larger raw UMAP scale before normalization.",
            "Raw UMAP coordinates",
            "Not a core morphology feature because UMAPs were fitted separately.",
        ),
        (
            "normalized_center_x",
            "Quality/control",
            "Median x coordinate after normalization.",
            "Diagnostic check that normalized coordinates are centered.",
            "No substantive interpretation.",
            "Normalized UMAP coordinates",
            "Should be approximately zero.",
        ),
        (
            "normalized_center_y",
            "Quality/control",
            "Median y coordinate after normalization.",
            "Diagnostic check that normalized coordinates are centered.",
            "No substantive interpretation.",
            "Normalized UMAP coordinates",
            "Should be approximately zero.",
        ),
        (
            "mst_points_used",
            "Quality/control",
            "Number of points used for the MST metric.",
            "Shows whether MST was computed on all points or a deterministic sample.",
            "More points used in MST.",
            "Normalized UMAP coordinates",
            "Capped by mst_max_points for runtime safety.",
        ),
        (
            "mst_sampling_applied",
            "Quality/control",
            "Whether deterministic MST subsampling was applied.",
            "Flags expensive-MST runtime protection.",
            "MST used a sample rather than all points.",
            "Normalized UMAP coordinates",
            "Sampling is deterministic for random_state and subfield_id.",
        ),
        (
            "n_years_available",
            "Quality/control",
            "Number of publication years represented after filtering.",
            "Temporal coverage diagnostic.",
            "More years represented.",
            "publication_year in coordinate parquet",
            "Only the morphology input window is used.",
        ),
        (
            "n_early_points",
            "Quality/control",
            "Number of points in 2010-2012.",
            "Early-period temporal sample size.",
            "More early-period papers.",
            "publication_year in coordinate parquet",
            "Used for centroid_drift_early_late.",
        ),
        (
            "n_late_points",
            "Quality/control",
            "Number of points in 2017-2019.",
            "Late-period temporal sample size.",
            "More late-period papers.",
            "publication_year in coordinate parquet",
            "Used for centroid_drift_early_late.",
        ),
    ]
    for name, family, definition, interpretation, higher, computed_on, notes in control_rows:
        rows.append(
            {
                "metric_name": name,
                "family": family,
                "definition": definition,
                "interpretation": interpretation,
                "higher_means": higher,
                "computed_on": computed_on,
                "notes": notes,
            }
        )
    return rows
