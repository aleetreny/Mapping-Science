from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, triu
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from src.embeddings import normalize_eligibility_flags
from src.metric_year_windows import (
    resolve_metric_year_window,
    validate_metric_year_window,
)
from src.subfield_labels import add_subfield_label_columns


TEMPORAL_MIN_POINTS = 3
EPS = 1e-12

REQUIRED_INDEX_COLUMNS = {
    "analysis_row_id",
    "work_id",
    "subfield_id",
    "subfield_display_name",
    "field_id",
    "field_display_name",
    "domain_id",
    "domain_display_name",
    "publication_year",
}

CORE_EMBEDDING_METRIC_COLUMNS = [
    "embedding_centroid_norm",
    "embedding_distance_to_centroid_mean",
    "embedding_distance_to_centroid_median",
    "embedding_distance_to_centroid_std",
    "embedding_distance_to_centroid_iqr",
    "embedding_distance_to_centroid_p90",
    "embedding_tail_index_p90_median",
    "embedding_knn_mean_distance",
    "embedding_knn_median_distance",
    "embedding_knn_p90_distance",
    "embedding_knn_distance_cv",
    "embedding_pca_first_component_share",
    "embedding_pca_top3_variance_share",
    "embedding_pca_top5_variance_share",
    "embedding_pca_dim_50",
    "embedding_pca_dim_80",
    "embedding_pca_participation_ratio",
    "embedding_graph_connected_component_count",
    "embedding_graph_largest_component_share",
    "embedding_graph_edge_distance_median",
    "embedding_graph_edge_distance_p90",
    "embedding_centroid_drift_early_late",
    "embedding_annual_centroid_path_length",
    "embedding_directionality_ratio",
    "embedding_radial_expansion_slope",
]

DIAGNOSTIC_COLUMNS = [
    "embedding_radial_expansion_r2",
    "n_valid_embedding_rows",
    "n_years_available",
    "n_early_points",
    "n_late_points",
    "n_annual_centroids",
    "n_pca_components_used",
]

CONTROL_COLUMNS = [
    "subfield_id",
    "subfield_display_name",
    "subfield_label_unique",
    "subfield_label_short",
    "subfield_display_name_is_duplicated",
    "field_id",
    "field_display_name",
    "domain_id",
    "domain_display_name",
    "n_available",
    "n_used",
    "year_min",
    "year_max",
    "early_year_min",
    "early_year_max",
    "late_year_min",
    "late_year_max",
    "metric_status",
    "metric_error_message",
    "metric_warning_message",
    "embedding_matrix_path",
    "analysis_index_path",
    "k_neighbors",
    "effective_k_neighbors",
    "random_state",
    "sampling_applied",
    "max_papers_per_subfield",
]

OUTPUT_COLUMNS = list(
    dict.fromkeys(CONTROL_COLUMNS + CORE_EMBEDDING_METRIC_COLUMNS + DIAGNOSTIC_COLUMNS)
)


def validate_year_window(year_min: int, year_max: int) -> None:
    validate_metric_year_window(year_min, year_max)


def validate_embedding_index_columns(index: pd.DataFrame) -> None:
    missing = REQUIRED_INDEX_COLUMNS - set(index.columns)
    if missing:
        raise ValueError(
            "analysis_embedding_index.parquet is missing required columns: "
            f"{', '.join(sorted(missing))}"
        )
    try:
        normalize_eligibility_flags(index, require_robustness=False)
    except ValueError as exc:
        raise ValueError(
            "analysis_embedding_index.parquet eligibility flags could not be resolved: "
            f"{exc}"
        ) from exc


def stable_int_seed(random_state: int, key: object) -> int:
    payload = f"{random_state}:{key}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=4).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def first_value(frame: pd.DataFrame, column: str, default: object = "") -> object:
    if column not in frame.columns or frame.empty:
        return default
    values = frame[column].dropna()
    if values.empty:
        return default
    return values.iloc[0]


def finite_or_nan(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return np.nan
    return numeric if np.isfinite(numeric) else np.nan


def nan_core_metrics() -> dict[str, float]:
    return {column: np.nan for column in CORE_EMBEDDING_METRIC_COLUMNS}


def l2_normalize_rows(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"embeddings must be a 2D matrix, got shape {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1)
    valid = np.isfinite(norms) & (norms > EPS)
    normalized = np.zeros_like(matrix, dtype=np.float32)
    if valid.any():
        normalized[valid] = matrix[valid] / norms[valid, None]
    return normalized, valid


def cosine_distance_to_direction(
    normalized: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    direction_norm = float(np.linalg.norm(direction))
    if not np.isfinite(direction_norm) or direction_norm <= EPS:
        return np.full(len(normalized), np.nan, dtype=float)
    unit = direction / direction_norm
    distances = 1.0 - np.asarray(normalized, dtype=float) @ unit
    return np.clip(distances, 0.0, 2.0)


def cosine_distance_between_centroids(first: np.ndarray, second: np.ndarray) -> float:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= EPS or second_norm <= EPS:
        return np.nan
    distance = 1.0 - float(np.dot(first / first_norm, second / second_norm))
    return float(np.clip(distance, 0.0, 2.0))


def concentration_metrics(normalized: np.ndarray) -> tuple[dict[str, float], list[str]]:
    warnings: list[str] = []
    centroid = np.mean(normalized, axis=0)
    centroid_norm = finite_or_nan(np.linalg.norm(centroid))
    distances = cosine_distance_to_direction(normalized, centroid)
    distances = distances[np.isfinite(distances)]
    if len(distances) == 0:
        warnings.append("centroid distance metrics unavailable because centroid norm is zero")
        return {
            "embedding_centroid_norm": centroid_norm,
            "embedding_distance_to_centroid_mean": np.nan,
            "embedding_distance_to_centroid_median": np.nan,
            "embedding_distance_to_centroid_std": np.nan,
            "embedding_distance_to_centroid_iqr": np.nan,
            "embedding_distance_to_centroid_p90": np.nan,
            "embedding_tail_index_p90_median": np.nan,
        }, warnings

    q25, q75, p90 = np.percentile(distances, [25, 75, 90])
    median = float(np.median(distances))
    return {
        "embedding_centroid_norm": centroid_norm,
        "embedding_distance_to_centroid_mean": float(np.mean(distances)),
        "embedding_distance_to_centroid_median": median,
        "embedding_distance_to_centroid_std": float(np.std(distances)),
        "embedding_distance_to_centroid_iqr": float(q75 - q25),
        "embedding_distance_to_centroid_p90": float(p90),
        "embedding_tail_index_p90_median": float(p90 / median) if median > EPS else np.nan,
    }, warnings


def knn_metrics(
    normalized: np.ndarray,
    *,
    k_neighbors: int,
) -> tuple[dict[str, float], int, list[str]]:
    if k_neighbors < 1:
        raise ValueError("k_neighbors must be positive")
    if len(normalized) < 2:
        return {
            "embedding_knn_mean_distance": np.nan,
            "embedding_knn_median_distance": np.nan,
            "embedding_knn_p90_distance": np.nan,
            "embedding_knn_distance_cv": np.nan,
        }, 0, ["kNN metrics unavailable because fewer than two embeddings are available"]

    effective_k = min(int(k_neighbors), len(normalized) - 1)
    model = NearestNeighbors(n_neighbors=effective_k + 1, metric="cosine")
    model.fit(normalized)
    distances, _ = model.kneighbors(normalized)
    neighbor_distances = np.clip(distances[:, 1:], 0.0, 2.0).ravel()
    mean_distance = float(np.mean(neighbor_distances))
    cv = (
        float(np.std(neighbor_distances) / mean_distance)
        if mean_distance > EPS
        else np.nan
    )
    return {
        "embedding_knn_mean_distance": mean_distance,
        "embedding_knn_median_distance": float(np.median(neighbor_distances)),
        "embedding_knn_p90_distance": float(np.percentile(neighbor_distances, 90)),
        "embedding_knn_distance_cv": cv,
    }, effective_k, []


def dimension_for_variance_share(ratios: np.ndarray, threshold: float) -> float:
    if len(ratios) == 0:
        return np.nan
    cumulative = np.cumsum(ratios)
    positions = np.flatnonzero(cumulative >= threshold)
    if len(positions) == 0:
        return np.nan
    return float(int(positions[0]) + 1)


def pca_metrics(normalized: np.ndarray) -> tuple[dict[str, float], int, list[str]]:
    if len(normalized) < 2:
        return {
            "embedding_pca_first_component_share": np.nan,
            "embedding_pca_top3_variance_share": np.nan,
            "embedding_pca_top5_variance_share": np.nan,
            "embedding_pca_dim_50": np.nan,
            "embedding_pca_dim_80": np.nan,
            "embedding_pca_participation_ratio": np.nan,
        }, 0, ["PCA metrics unavailable because fewer than two embeddings are available"]

    n_components = min(len(normalized) - 1, normalized.shape[1])
    try:
        pca = PCA(n_components=n_components, svd_solver="full")
        pca.fit(normalized)
        ratios = np.asarray(pca.explained_variance_ratio_, dtype=float)
    except Exception as exc:
        return {
            "embedding_pca_first_component_share": np.nan,
            "embedding_pca_top3_variance_share": np.nan,
            "embedding_pca_top5_variance_share": np.nan,
            "embedding_pca_dim_50": np.nan,
            "embedding_pca_dim_80": np.nan,
            "embedding_pca_participation_ratio": np.nan,
        }, 0, [f"PCA metrics unavailable: {exc}"]

    ratios = ratios[np.isfinite(ratios) & (ratios >= 0)]
    if len(ratios) == 0 or float(np.sum(ratios)) <= EPS:
        return {
            "embedding_pca_first_component_share": np.nan,
            "embedding_pca_top3_variance_share": np.nan,
            "embedding_pca_top5_variance_share": np.nan,
            "embedding_pca_dim_50": np.nan,
            "embedding_pca_dim_80": np.nan,
            "embedding_pca_participation_ratio": np.nan,
        }, n_components, ["PCA metrics unavailable because explained variance is zero"]

    participation = float(1.0 / np.sum(ratios**2)) if np.sum(ratios**2) > EPS else np.nan
    return {
        "embedding_pca_first_component_share": float(ratios[0]),
        "embedding_pca_top3_variance_share": float(np.sum(ratios[:3])),
        "embedding_pca_top5_variance_share": float(np.sum(ratios[:5])),
        "embedding_pca_dim_50": dimension_for_variance_share(ratios, 0.50),
        "embedding_pca_dim_80": dimension_for_variance_share(ratios, 0.80),
        "embedding_pca_participation_ratio": participation,
    }, n_components, []


def graph_metrics(
    normalized: np.ndarray,
    *,
    k_neighbors: int,
) -> tuple[dict[str, float], list[str]]:
    if len(normalized) < 2:
        return {
            "embedding_graph_connected_component_count": np.nan,
            "embedding_graph_largest_component_share": np.nan,
            "embedding_graph_edge_distance_median": np.nan,
            "embedding_graph_edge_distance_p90": np.nan,
        }, ["graph metrics unavailable because fewer than two embeddings are available"]

    effective_k = min(int(k_neighbors), len(normalized) - 1)
    model = NearestNeighbors(n_neighbors=effective_k + 1, metric="cosine")
    model.fit(normalized)
    distances, indices = model.kneighbors(normalized)
    distances = np.clip(distances[:, 1:], 0.0, 2.0)
    indices = indices[:, 1:]

    rows = np.repeat(np.arange(len(normalized)), effective_k)
    cols = indices.ravel()
    data = distances.ravel()
    distance_graph = csr_matrix(
        (data, (rows, cols)),
        shape=(len(normalized), len(normalized)),
    )
    connectivity = csr_matrix(
        (np.ones_like(data), (rows, cols)),
        shape=(len(normalized), len(normalized)),
    )
    graph = connectivity.maximum(connectivity.T)
    n_components, labels = connected_components(graph, directed=False, return_labels=True)
    component_sizes = np.bincount(labels, minlength=n_components)
    largest_share = float(component_sizes.max() / len(normalized)) if len(component_sizes) else np.nan
    edge_distances = triu(distance_graph.maximum(distance_graph.T), k=1).data
    edge_distances = edge_distances[np.isfinite(edge_distances)]

    return {
        "embedding_graph_connected_component_count": int(n_components),
        "embedding_graph_largest_component_share": largest_share,
        "embedding_graph_edge_distance_median": float(np.median(edge_distances))
        if len(edge_distances)
        else np.nan,
        "embedding_graph_edge_distance_p90": float(np.percentile(edge_distances, 90))
        if len(edge_distances)
        else np.nan,
    }, []


def temporal_embedding_metrics(
    normalized: np.ndarray,
    years: np.ndarray,
    *,
    year_min: int,
    year_max: int,
) -> tuple[dict[str, float], dict[str, int], list[str]]:
    years = np.asarray(years, dtype=int)
    warnings: list[str] = []
    window = resolve_metric_year_window(year_min, year_max)
    available_years = sorted(int(year) for year in np.unique(years))
    early_mask = (years >= window.early_year_min) & (
        years <= window.early_year_max
    )
    late_mask = (years >= window.late_year_min) & (years <= window.late_year_max)
    n_early = int(np.count_nonzero(early_mask))
    n_late = int(np.count_nonzero(late_mask))

    if n_early >= TEMPORAL_MIN_POINTS and n_late >= TEMPORAL_MIN_POINTS:
        early_centroid = np.mean(normalized[early_mask], axis=0)
        late_centroid = np.mean(normalized[late_mask], axis=0)
        drift = cosine_distance_between_centroids(early_centroid, late_centroid)
    else:
        drift = np.nan
        warnings.append(
            "embedding_centroid_drift_early_late unavailable because early or late "
            f"period has fewer than {TEMPORAL_MIN_POINTS} embeddings"
        )

    annual_centroids: list[tuple[int, np.ndarray]] = []
    global_centroid = np.mean(normalized, axis=0)
    annual_median_distance: list[tuple[int, float]] = []
    for year in range(year_min, year_max + 1):
        year_mask = years == year
        if np.count_nonzero(year_mask) < TEMPORAL_MIN_POINTS:
            continue
        annual_centroids.append((year, np.mean(normalized[year_mask], axis=0)))
        distances = cosine_distance_to_direction(normalized[year_mask], global_centroid)
        distances = distances[np.isfinite(distances)]
        if len(distances):
            annual_median_distance.append((year, float(np.median(distances))))

    if len(annual_centroids) >= 2:
        steps = [
            cosine_distance_between_centroids(previous, current)
            for (_, previous), (_, current) in zip(annual_centroids, annual_centroids[1:])
        ]
        steps = [step for step in steps if np.isfinite(step)]
        path = float(np.sum(steps)) if steps else np.nan
    else:
        path = np.nan
        warnings.append(
            "embedding_annual_centroid_path_length unavailable because fewer than two "
            f"years have at least {TEMPORAL_MIN_POINTS} embeddings"
        )

    if np.isfinite(drift) and np.isfinite(path) and path > EPS:
        directionality = float(drift / path)
    else:
        directionality = np.nan
        warnings.append(
            "embedding_directionality_ratio unavailable because drift or path is unavailable"
        )

    if len(annual_median_distance) >= 2:
        year_values = np.array([year for year, _ in annual_median_distance], dtype=float)
        distance_values = np.array(
            [distance for _, distance in annual_median_distance],
            dtype=float,
        )
        centered_years = year_values - year_values.mean()
        coefficients = np.polyfit(centered_years, distance_values, deg=1)
        slope = float(coefficients[0])
    else:
        slope = np.nan
        warnings.append(
            "embedding_radial_expansion_slope unavailable because fewer than two "
            f"years have at least {TEMPORAL_MIN_POINTS} embeddings"
        )

    if len(annual_median_distance) >= 3:
        year_values = np.array([year for year, _ in annual_median_distance], dtype=float)
        distance_values = np.array(
            [distance for _, distance in annual_median_distance],
            dtype=float,
        )
        centered_years = year_values - year_values.mean()
        coefficients = np.polyfit(centered_years, distance_values, deg=1)
        fitted = np.polyval(coefficients, centered_years)
        ss_residual = float(np.sum((distance_values - fitted) ** 2))
        ss_total = float(np.sum((distance_values - distance_values.mean()) ** 2))
        radial_r2 = float(1.0 - (ss_residual / ss_total)) if ss_total > EPS else np.nan
        if not np.isfinite(radial_r2):
            warnings.append(
                "embedding_radial_expansion_r2 unavailable because annual distances are constant"
            )
    else:
        radial_r2 = np.nan
        warnings.append(
            "embedding_radial_expansion_r2 unavailable because fewer than three "
            f"years have at least {TEMPORAL_MIN_POINTS} embeddings"
        )

    return (
        {
            "embedding_centroid_drift_early_late": drift,
            "embedding_annual_centroid_path_length": path,
            "embedding_directionality_ratio": directionality,
            "embedding_radial_expansion_slope": slope,
            "embedding_radial_expansion_r2": radial_r2,
        },
        {
            "n_years_available": int(len(available_years)),
            "early_year_min": int(window.early_year_min),
            "early_year_max": int(window.early_year_max),
            "late_year_min": int(window.late_year_min),
            "late_year_max": int(window.late_year_max),
            "n_early_points": n_early,
            "n_late_points": n_late,
            "n_annual_centroids": int(len(annual_centroids)),
        },
        warnings,
    )


def compute_embedding_metrics(
    embeddings: np.ndarray,
    years: np.ndarray,
    *,
    year_min: int,
    year_max: int,
    k_neighbors: int,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    validate_year_window(year_min, year_max)
    warnings: list[str] = []
    normalized, valid_mask = l2_normalize_rows(embeddings)
    valid_count = int(np.count_nonzero(valid_mask))
    if valid_count == 0:
        raise ValueError("no valid non-zero embedding rows are available")
    if valid_count < len(normalized):
        warnings.append(
            f"dropped {len(normalized) - valid_count} zero-norm or non-finite embedding rows"
        )
    normalized = normalized[valid_mask]
    years = np.asarray(years, dtype=int)[valid_mask]

    metrics: dict[str, Any] = {}
    controls: dict[str, Any] = {"n_valid_embedding_rows": valid_count}

    concentration, concentration_warnings = concentration_metrics(normalized)
    metrics.update(concentration)
    warnings.extend(concentration_warnings)

    knn, effective_k, knn_warnings = knn_metrics(normalized, k_neighbors=k_neighbors)
    metrics.update(knn)
    controls["effective_k_neighbors"] = int(effective_k)
    warnings.extend(knn_warnings)

    pca, pca_components, pca_warnings = pca_metrics(normalized)
    metrics.update(pca)
    controls["n_pca_components_used"] = int(pca_components)
    warnings.extend(pca_warnings)

    graph, graph_warnings = graph_metrics(normalized, k_neighbors=k_neighbors)
    metrics.update(graph)
    warnings.extend(graph_warnings)

    temporal, temporal_controls, temporal_warnings = temporal_embedding_metrics(
        normalized,
        years,
        year_min=year_min,
        year_max=year_max,
    )
    metrics.update(temporal)
    controls.update(temporal_controls)
    warnings.extend(temporal_warnings)

    for column in CORE_EMBEDDING_METRIC_COLUMNS:
        metrics.setdefault(column, np.nan)
    return metrics, controls, list(dict.fromkeys(warnings))


def compute_subfield_embedding_metric_row(
    subfield_index: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    n_available: int,
    year_min: int,
    year_max: int,
    analysis_index_path: str | Path,
    embedding_matrix_path: str | Path,
    k_neighbors: int,
    random_state: int,
    sampling_applied: bool,
    max_papers_per_subfield: int | None,
) -> dict[str, Any]:
    if len(subfield_index) != len(embeddings):
        raise ValueError(
            "subfield index row count does not match embedding row count: "
            f"{len(subfield_index)} != {len(embeddings)}"
        )
    if subfield_index.empty:
        raise ValueError("subfield index is empty")

    years = pd.to_numeric(subfield_index["publication_year"], errors="raise").to_numpy()
    metrics, controls, warnings = compute_embedding_metrics(
        embeddings,
        years,
        year_min=year_min,
        year_max=year_max,
        k_neighbors=k_neighbors,
    )
    row: dict[str, Any] = {
        "subfield_id": str(first_value(subfield_index, "subfield_id")),
        "subfield_display_name": str(first_value(subfield_index, "subfield_display_name")),
        "field_id": str(first_value(subfield_index, "field_id")),
        "field_display_name": str(first_value(subfield_index, "field_display_name")),
        "domain_id": str(first_value(subfield_index, "domain_id")),
        "domain_display_name": str(first_value(subfield_index, "domain_display_name")),
        "n_available": int(n_available),
        "n_used": int(len(subfield_index)),
        "year_min": int(year_min),
        "year_max": int(year_max),
        "early_year_min": int(controls.get("early_year_min", np.nan)),
        "early_year_max": int(controls.get("early_year_max", np.nan)),
        "late_year_min": int(controls.get("late_year_min", np.nan)),
        "late_year_max": int(controls.get("late_year_max", np.nan)),
        "metric_status": "completed_with_warnings" if warnings else "completed",
        "metric_error_message": "",
        "metric_warning_message": "; ".join(warnings),
        "embedding_matrix_path": str(embedding_matrix_path),
        "analysis_index_path": str(analysis_index_path),
        "k_neighbors": int(k_neighbors),
        "effective_k_neighbors": int(controls.get("effective_k_neighbors", 0)),
        "random_state": int(random_state),
        "sampling_applied": bool(sampling_applied),
        "max_papers_per_subfield": max_papers_per_subfield,
        **metrics,
        **controls,
    }
    labelled = add_subfield_label_columns(pd.DataFrame([row]))
    return embedding_metric_row_frame(labelled.to_dict("records")).iloc[0].to_dict()


def failed_embedding_metric_row(
    *,
    subfield_row: pd.Series | dict[str, Any],
    n_available: int,
    year_min: int,
    year_max: int,
    analysis_index_path: str | Path,
    embedding_matrix_path: str | Path,
    k_neighbors: int,
    random_state: int,
    sampling_applied: bool,
    max_papers_per_subfield: int | None,
    error_message: str,
) -> dict[str, Any]:
    getter = subfield_row.get
    try:
        window = resolve_metric_year_window(year_min, year_max)
        temporal_controls: dict[str, Any] = {
            "early_year_min": int(window.early_year_min),
            "early_year_max": int(window.early_year_max),
            "late_year_min": int(window.late_year_min),
            "late_year_max": int(window.late_year_max),
        }
    except ValueError:
        temporal_controls = {
            "early_year_min": np.nan,
            "early_year_max": np.nan,
            "late_year_min": np.nan,
            "late_year_max": np.nan,
        }
    row: dict[str, Any] = {
        "subfield_id": str(getter("subfield_id", "")),
        "subfield_display_name": str(getter("subfield_display_name", "")),
        "field_id": str(getter("field_id", "")),
        "field_display_name": str(getter("field_display_name", "")),
        "domain_id": str(getter("domain_id", "")),
        "domain_display_name": str(getter("domain_display_name", "")),
        "n_available": int(n_available),
        "n_used": 0,
        "year_min": int(year_min),
        "year_max": int(year_max),
        **temporal_controls,
        "metric_status": "failed",
        "metric_error_message": str(error_message),
        "metric_warning_message": "",
        "embedding_matrix_path": str(embedding_matrix_path),
        "analysis_index_path": str(analysis_index_path),
        "k_neighbors": int(k_neighbors),
        "effective_k_neighbors": 0,
        "random_state": int(random_state),
        "sampling_applied": bool(sampling_applied),
        "max_papers_per_subfield": max_papers_per_subfield,
    }
    row.update(nan_core_metrics())
    for column in DIAGNOSTIC_COLUMNS:
        row[column] = np.nan
    return embedding_metric_row_frame([row]).iloc[0].to_dict()


def embedding_metric_row_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if {"subfield_id", "subfield_display_name"}.issubset(frame.columns):
        frame = add_subfield_label_columns(frame)
    for column in OUTPUT_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[OUTPUT_COLUMNS]


def metric_dictionary_frame() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    definitions = {
        "embedding_centroid_norm": (
            "Semantic concentration",
            "Norm of the mean L2-normalized embedding vector.",
            "Stronger common semantic direction.",
        ),
        "embedding_distance_to_centroid_mean": (
            "Semantic dispersion",
            "Mean cosine distance from each paper to the normalized subfield centroid.",
            "Greater average semantic dispersion.",
        ),
        "embedding_distance_to_centroid_median": (
            "Semantic dispersion",
            "Median cosine distance from each paper to the normalized subfield centroid.",
            "Greater typical semantic dispersion.",
        ),
        "embedding_distance_to_centroid_std": (
            "Semantic dispersion",
            "Standard deviation of cosine distances to the normalized centroid.",
            "More heterogeneous semantic dispersion.",
        ),
        "embedding_distance_to_centroid_iqr": (
            "Semantic dispersion",
            "Interquartile range of cosine distances to the normalized centroid.",
            "Broader middle-range semantic dispersion.",
        ),
        "embedding_distance_to_centroid_p90": (
            "Semantic dispersion",
            "90th percentile cosine distance to the normalized centroid.",
            "More distant semantic periphery.",
        ),
        "embedding_tail_index_p90_median": (
            "Semantic dispersion",
            "P90 distance to centroid divided by median distance.",
            "Stronger semantic tail relative to the typical paper.",
        ),
        "embedding_knn_mean_distance": (
            "Local semantic density",
            "Mean within-subfield cosine distance to k nearest neighbours.",
            "Lower local semantic density.",
        ),
        "embedding_knn_median_distance": (
            "Local semantic density",
            "Median within-subfield cosine distance to k nearest neighbours.",
            "Lower typical local semantic density.",
        ),
        "embedding_knn_p90_distance": (
            "Local semantic density",
            "90th percentile within-subfield kNN cosine distance.",
            "Sparse local regions.",
        ),
        "embedding_knn_distance_cv": (
            "Local semantic density",
            "Coefficient of variation of within-subfield kNN cosine distances.",
            "More uneven local density.",
        ),
        "embedding_pca_first_component_share": (
            "Intrinsic dimensionality",
            "Share of variance explained by the first PCA component.",
            "More variance along one dominant axis.",
        ),
        "embedding_pca_top3_variance_share": (
            "Intrinsic dimensionality",
            "Share of variance explained by the top three PCA components.",
            "Lower dimensional semantic structure.",
        ),
        "embedding_pca_top5_variance_share": (
            "Intrinsic dimensionality",
            "Share of variance explained by the top five PCA components.",
            "Lower dimensional semantic structure.",
        ),
        "embedding_pca_dim_50": (
            "Intrinsic dimensionality",
            "Number of PCA components needed to explain at least 50 percent variance.",
            "Higher intrinsic dimensionality.",
        ),
        "embedding_pca_dim_80": (
            "Intrinsic dimensionality",
            "Number of PCA components needed to explain at least 80 percent variance.",
            "Higher intrinsic dimensionality.",
        ),
        "embedding_pca_participation_ratio": (
            "Intrinsic dimensionality",
            "One divided by the sum of squared PCA explained-variance shares.",
            "More evenly spread variance across dimensions.",
        ),
        "embedding_graph_connected_component_count": (
            "kNN graph structure",
            "Connected component count after symmetrizing the within-subfield kNN graph.",
            "More disconnected local semantic structure.",
        ),
        "embedding_graph_largest_component_share": (
            "kNN graph structure",
            "Share of embeddings in the largest symmetrized kNN graph component.",
            "More globally connected semantic structure.",
        ),
        "embedding_graph_edge_distance_median": (
            "kNN graph structure",
            "Median edge cosine distance in the symmetrized kNN graph.",
            "Longer typical graph edges.",
        ),
        "embedding_graph_edge_distance_p90": (
            "kNN graph structure",
            "90th percentile edge cosine distance in the symmetrized kNN graph.",
            "Longer sparse-bridge graph edges.",
        ),
        "embedding_centroid_drift_early_late": (
            "Temporal movement",
            "Cosine distance between mean embedding centroids for the first and "
            "last five years of the selected metric window.",
            "Greater net semantic movement.",
        ),
        "embedding_annual_centroid_path_length": (
            "Temporal movement",
            "Sum of cosine distances between consecutive available annual centroids.",
            "More total semantic movement.",
        ),
        "embedding_directionality_ratio": (
            "Temporal movement",
            "Early-late centroid drift divided by annual centroid path length.",
            "More directional semantic movement.",
        ),
        "embedding_radial_expansion_slope": (
            "Temporal movement",
            "Linear trend of annual median cosine distance to the full-period centroid.",
            "Semantic expansion away from the period centroid.",
        ),
    }
    for name in CORE_EMBEDDING_METRIC_COLUMNS:
        family, definition, higher = definitions[name]
        rows.append(
            {
                "metric_name": name,
                "family": family,
                "definition": definition,
                "interpretation": definition,
                "higher_means": higher,
                "computed_on": "L2-normalized SPECTER2 embeddings",
                "notes": "Core embedding-space structure metric.",
            }
        )

    for name in DIAGNOSTIC_COLUMNS:
        rows.append(
            {
                "metric_name": name,
                "family": "Quality/diagnostic",
                "definition": "Diagnostic column for embedding-space metric computation.",
                "interpretation": "Use for filtering, interpretation, and quality checks.",
                "higher_means": "Depends on the diagnostic.",
                "computed_on": "Embedding-space metric run",
                "notes": "Diagnostic only; not one of the 25 core embedding-space metrics.",
            }
        )

    control_definitions = {
        "early_year_min": "First year in the early temporal period.",
        "early_year_max": "Last year in the early temporal period.",
        "late_year_min": "First year in the late temporal period.",
        "late_year_max": "Last year in the late temporal period.",
    }
    for name in CONTROL_COLUMNS:
        rows.append(
            {
                "metric_name": name,
                "family": "Control",
                "definition": control_definitions.get(
                    name,
                    "Run metadata, subfield metadata, or output status column.",
                ),
                "interpretation": "Use for joining, filtering, reproducibility, or diagnostics.",
                "higher_means": "No direct morphology interpretation.",
                "computed_on": "Embedding-space metric run",
                "notes": "Control column, not a core metric.",
            }
        )

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
