from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.embedding_space_metrics import validate_embedding_index_columns
from src.per_subfield_umap_maps import filter_input_window, main_analysis_subfields
from src.storage import save_parquet
from src.subfield_labels import add_subfield_label_columns
from src.temporal_common import (
    EPS,
    TemporalWindow,
    cosine_distance,
    dataframe_to_records,
    default_or_computed_windows,
    display_path,
    ensure_outputs_do_not_exist,
    l2_normalize_vectors,
    utc_now_iso,
    vectors_to_columns,
    vector_columns,
    write_json,
    write_placeholder_figure,
    write_text,
)

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


VALID_CENTROID_STATUSES = {"completed", "completed_with_warnings"}
PROCESSED_TEMPORAL_DIR = Path("data/processed/temporal")

OUTPUT_FILENAMES = {
    "centroids_parquet": PROCESSED_TEMPORAL_DIR / "subfield_window_centroids.parquet",
    "path_metrics_parquet": PROCESSED_TEMPORAL_DIR / "subfield_centroid_path_metrics.parquet",
    "path_metrics_csv": PROCESSED_TEMPORAL_DIR / "subfield_centroid_path_metrics.csv",
}

ANALYSIS_FILENAMES = {
    "centroid_projections_parquet": "subfield_window_centroid_projections.parquet",
    "global_pca": "global_centroid_paths_pca.png",
    "global_pca_highlight": "global_centroid_paths_pca_highlight_top_dynamic.png",
    "global_pca_clean": "global_centroid_paths_pca_clean.png",
    "global_pca_top_dynamic_clean": "global_centroid_paths_pca_top_dynamic_clean.png",
    "top_dynamic_pca_small_multiples": "top_dynamic_subfield_pca_small_multiples.png",
    "largest_jump_window_distribution": "largest_jump_window_distribution.png",
    "largest_jump_window_distribution_csv": "largest_jump_window_distribution.csv",
    "domain_average_path_length": "domain_average_path_length.png",
    "domain_average_early_late_drift": "domain_average_early_late_drift.png",
    "field_average_path_length_top20": "field_average_path_length_top20.png",
    "field_average_early_late_drift_top20": "field_average_early_late_drift_top20.png",
    "top_dynamic_subfields_csv": "top_dynamic_subfields.csv",
    "domain_centroid_movement_summary_csv": "domain_centroid_movement_summary.csv",
    "field_centroid_movement_summary_csv": "field_centroid_movement_summary.csv",
    "domain_faceted_pca": "domain_faceted_centroid_paths_pca.png",
    "top_path_length": "top_centroid_path_length_subfields.png",
    "top_early_late_drift": "top_early_late_drift_subfields.png",
    "summary_md": "summary.md",
    "summary_json": "summary.json",
}

EXPERIMENTAL_UMAP_FILENAMES = {
    "global_umap_experimental": "experimental_global_centroid_paths_umap.png",
    "global_umap_top_dynamic_experimental": "experimental_global_centroid_paths_umap_top_dynamic.png",
}

LEGACY_UMAP_FILENAMES = [
    "global_centroid_paths_umap.png",
    "global_centroid_paths_umap_highlight_top_dynamic.png",
]


def output_paths(
    output_dir: str | Path,
    *,
    root: Path,
    include_umap_experimental: bool = False,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    paths = {key: root / value for key, value in OUTPUT_FILENAMES.items()}
    analysis_filenames = dict(ANALYSIS_FILENAMES)
    if include_umap_experimental:
        analysis_filenames.update(EXPERIMENTAL_UMAP_FILENAMES)
    paths.update({key: output_dir / value for key, value in analysis_filenames.items()})
    return paths


def ensure_centroid_path_outputs(
    output_dir: str | Path,
    *,
    overwrite: bool,
    root: Path,
    include_umap_experimental: bool = False,
) -> None:
    ensure_outputs_do_not_exist(
        output_paths(
            output_dir,
            root=root,
            include_umap_experimental=include_umap_experimental,
        ).values(),
        overwrite=overwrite,
        root=root,
        description="temporal centroid path outputs",
    )


def cleanup_legacy_umap_outputs(output_dir: str | Path, *, root: Path) -> list[str]:
    removed: list[str] = []
    for filename in LEGACY_UMAP_FILENAMES:
        path = Path(output_dir) / filename
        if path.exists() and path.is_file():
            path.unlink()
            removed.append(display_path(path, root=root))
    return removed


def select_attempted_subfields(
    subfields: pd.DataFrame,
    *,
    subfield_id: str | None,
    limit_subfields: int | None,
) -> pd.DataFrame:
    attempted = subfields.copy()
    if subfield_id is not None:
        attempted = attempted[
            attempted["subfield_id"].astype(str) == str(subfield_id)
        ].reset_index(drop=True)
        if attempted.empty:
            raise ValueError(f"No main-analysis subfield matched --subfield-id {subfield_id}")
    if limit_subfields is not None:
        if limit_subfields <= 0:
            raise ValueError("limit_subfields must be positive when provided")
        attempted = attempted.head(limit_subfields).reset_index(drop=True)
    return attempted


def _nan_vector(dim: int) -> np.ndarray:
    return np.full(dim, np.nan, dtype=np.float32)


def _base_centroid_record(
    *,
    subfield: pd.Series | dict[str, Any],
    window: TemporalWindow,
    n_available: int,
    n_used: int,
    status: str,
    error_message: str = "",
    warning_message: str = "",
    n_valid_embedding_rows: int = 0,
) -> dict[str, Any]:
    getter = subfield.get
    return {
        "subfield_id": str(getter("subfield_id", "")),
        "subfield_display_name": str(getter("subfield_display_name", "")),
        "field_id": str(getter("field_id", "")),
        "field_display_name": str(getter("field_display_name", "")),
        "domain_id": str(getter("domain_id", "")),
        "domain_display_name": str(getter("domain_display_name", "")),
        "window_start": int(window.window_start),
        "window_end": int(window.window_end),
        "window_index": int(window.window_index),
        "window_label": window.window_label,
        "window_mid_year": float(window.window_mid_year),
        "n_available": int(n_available),
        "n_used": int(n_used),
        "n_valid_embedding_rows": int(n_valid_embedding_rows),
        "centroid_status": status,
        "centroid_error_message": error_message,
        "centroid_warning_message": warning_message,
    }


def compute_centroid_vector(embeddings: np.ndarray) -> tuple[np.ndarray, int, list[str]]:
    normalized = l2_normalize_vectors(embeddings)
    norms = np.linalg.norm(normalized, axis=1)
    valid_mask = np.isfinite(norms) & (norms > 0)
    valid_count = int(np.count_nonzero(valid_mask))
    if valid_count == 0:
        raise ValueError("no valid non-zero embedding rows are available")
    warnings: list[str] = []
    if valid_count < len(normalized):
        warnings.append(
            f"dropped {len(normalized) - valid_count} zero-norm or non-finite embedding rows"
        )
    centroid = np.mean(normalized[valid_mask], axis=0).astype(np.float32)
    return centroid, valid_count, warnings


def compute_subfield_window_centroids(
    analysis_index: pd.DataFrame,
    matrix: np.ndarray,
    *,
    windows: list[TemporalWindow],
    attempted_subfields: pd.DataFrame,
    year_min: int,
    year_max: int,
    min_papers_per_window: int,
) -> pd.DataFrame:
    validate_embedding_index_columns(analysis_index)
    window_index = filter_input_window(
        analysis_index,
        year_min=year_min,
        year_max=year_max,
    )
    groups = {
        str(subfield_id): group.reset_index(drop=True)
        for subfield_id, group in window_index.groupby("subfield_id", sort=False)
    }
    records: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    dim = int(matrix.shape[1])

    for _, subfield in attempted_subfields.iterrows():
        subfield_key = str(subfield["subfield_id"])
        available_full = groups.get(subfield_key, window_index.head(0))
        for window in windows:
            mask = pd.to_numeric(
                available_full["publication_year"],
                errors="coerce",
            ).between(window.window_start, window.window_end, inclusive="both")
            available = available_full.loc[mask].copy().reset_index(drop=True)
            n_available = int(len(available))
            if n_available < min_papers_per_window:
                records.append(
                    _base_centroid_record(
                        subfield=subfield,
                        window=window,
                        n_available=n_available,
                        n_used=0,
                        status="skipped",
                        error_message=(
                            f"Only {n_available} papers available; "
                            f"min_papers_per_window is {min_papers_per_window}"
                        ),
                    )
                )
                vectors.append(_nan_vector(dim))
                continue
            try:
                ordered = available.sort_values(
                    ["publication_year", "work_id", "analysis_row_id"],
                    kind="mergesort",
                ).reset_index(drop=True)
                row_ids = ordered["analysis_row_id"].astype(int).to_numpy()
                embeddings = np.asarray(matrix[row_ids], dtype=np.float32)
                centroid, valid_count, warnings = compute_centroid_vector(embeddings)
                records.append(
                    _base_centroid_record(
                        subfield=subfield,
                        window=window,
                        n_available=n_available,
                        n_used=len(ordered),
                        status="completed_with_warnings" if warnings else "completed",
                        warning_message="; ".join(warnings),
                        n_valid_embedding_rows=valid_count,
                    )
                )
                vectors.append(centroid)
            except Exception as exc:
                records.append(
                    _base_centroid_record(
                        subfield=subfield,
                        window=window,
                        n_available=n_available,
                        n_used=0,
                        status="failed",
                        error_message=str(exc),
                    )
                )
                vectors.append(_nan_vector(dim))

    metadata = pd.DataFrame(records)
    vector_frame = vectors_to_columns(np.vstack(vectors)) if vectors else pd.DataFrame()
    result = pd.concat([metadata, vector_frame], axis=1)
    if result.empty:
        return result
    return add_subfield_label_columns(result).sort_values(
        ["subfield_id", "window_start"],
        kind="mergesort",
    ).reset_index(drop=True)


def completed_centroid_rows(centroids: pd.DataFrame) -> pd.DataFrame:
    return centroids.loc[
        centroids["centroid_status"].isin(VALID_CENTROID_STATUSES)
    ].copy()


def _unit_normalize_rows(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=float)
    if values.size == 0:
        return values
    norms = np.linalg.norm(values, axis=1)
    valid = np.isfinite(values).all(axis=1) & np.isfinite(norms) & (norms > EPS)
    normalized = np.full(values.shape, np.nan, dtype=float)
    normalized[valid] = values[valid] / norms[valid, None]
    return normalized


def _euclidean_distance(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        return np.nan
    return float(np.linalg.norm(first - second))


def _directionality_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= EPS:
        return np.nan
    value = float(numerator / denominator)
    if -1e-10 <= value <= 1.0 + 1e-10:
        return float(np.clip(value, 0.0, 1.0))
    return value


def compute_centroid_path_metrics(
    centroids: pd.DataFrame,
    *,
    windows: list[TemporalWindow],
) -> pd.DataFrame:
    if centroids.empty:
        return pd.DataFrame()
    dim_columns = vector_columns(centroids)
    if not dim_columns:
        raise ValueError("centroid table has no centroid_dim_* columns")

    first_label = windows[0].window_label
    last_label = windows[-1].window_label
    metadata_columns = [
        "subfield_id",
        "subfield_display_name",
        "subfield_label_unique",
        "subfield_label_short",
        "field_id",
        "field_display_name",
        "domain_id",
        "domain_display_name",
    ]
    rows: list[dict[str, Any]] = []
    for subfield_id, group in centroids.groupby("subfield_id", sort=False):
        metadata = group[metadata_columns].iloc[0].to_dict()
        completed = completed_centroid_rows(group).sort_values(
            "window_start",
            kind="mergesort",
        )
        vectors = completed[dim_columns].to_numpy(dtype=float)
        normalized_vectors = _unit_normalize_rows(vectors)
        labels = completed["window_label"].astype(str).tolist()
        steps: list[float] = []
        euclidean_steps: list[float] = []
        step_labels: list[str] = []
        for idx in range(1, len(completed)):
            step = cosine_distance(vectors[idx - 1], vectors[idx])
            euclidean_step = _euclidean_distance(
                normalized_vectors[idx - 1],
                normalized_vectors[idx],
            )
            steps.append(step)
            euclidean_steps.append(euclidean_step)
            step_labels.append(f"{labels[idx - 1]}_to_{labels[idx]}")
        finite_steps = [step for step in steps if np.isfinite(step)]
        total_path = float(np.sum(finite_steps)) if finite_steps else np.nan
        finite_euclidean_steps = [
            step for step in euclidean_steps if np.isfinite(step)
        ]
        total_path_euclidean = (
            float(np.sum(finite_euclidean_steps)) if finite_euclidean_steps else np.nan
        )

        lookup = {
            str(row.window_label): completed.iloc[position][dim_columns].to_numpy(dtype=float)
            for position, row in enumerate(completed.itertuples(index=False))
        }
        normalized_lookup = {
            str(row.window_label): normalized_vectors[position]
            for position, row in enumerate(completed.itertuples(index=False))
        }
        early_late = (
            cosine_distance(lookup[first_label], lookup[last_label])
            if first_label in lookup and last_label in lookup
            else np.nan
        )
        early_late_euclidean = (
            _euclidean_distance(normalized_lookup[first_label], normalized_lookup[last_label])
            if first_label in normalized_lookup and last_label in normalized_lookup
            else np.nan
        )
        directionality_cosine_diagnostic = _directionality_ratio(early_late, total_path)
        directionality_euclidean = _directionality_ratio(
            early_late_euclidean,
            total_path_euclidean,
        )
        if finite_steps:
            step_array = np.asarray(steps, dtype=float)
            max_idx = int(np.nanargmax(step_array))
            max_jump = float(step_array[max_idx])
            largest_jump_window = step_labels[max_idx]
        else:
            max_jump = np.nan
            largest_jump_window = ""
        rows.append(
            {
                **metadata,
                "n_windows_available": int(len(completed)),
                "total_path_length": total_path,
                "early_late_drift": early_late,
                "directionality_ratio_cosine_diagnostic": directionality_cosine_diagnostic,
                "max_single_step_jump": max_jump,
                "largest_jump_window": largest_jump_window,
                "initial_window": first_label,
                "final_window": last_label,
                "total_path_length_euclidean_normalized": total_path_euclidean,
                "early_late_drift_euclidean_normalized": early_late_euclidean,
                "directionality_ratio_euclidean": directionality_euclidean,
            }
        )
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["total_path_length", "subfield_id"],
        ascending=[False, True],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)


def project_centroids(
    centroids: pd.DataFrame,
    *,
    projections: list[str],
    random_state: int,
) -> pd.DataFrame:
    projected = centroids.copy()
    dim_columns = vector_columns(projected)
    completed = completed_centroid_rows(projected)
    if completed.empty or not dim_columns:
        for projection in projections:
            projected[f"{projection}_x"] = np.nan
            projected[f"{projection}_y"] = np.nan
        return projected
    vectors = completed[dim_columns].to_numpy(dtype=np.float32)
    finite = np.isfinite(vectors).all(axis=1)
    valid_index = completed.index[finite]
    valid_vectors = vectors[finite]

    for projection in projections:
        projected[f"{projection}_x"] = np.nan
        projected[f"{projection}_y"] = np.nan
        if len(valid_vectors) == 0:
            continue
        if projection == "pca":
            if len(valid_vectors) == 1:
                coords = np.zeros((1, 2), dtype=float)
            else:
                reducer = PCA(n_components=2, random_state=random_state)
                coords = reducer.fit_transform(valid_vectors)
        elif projection == "umap":
            if len(valid_vectors) < 3:
                coords = np.zeros((len(valid_vectors), 2), dtype=float)
            else:
                import umap

                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=min(15, max(2, len(valid_vectors) - 1)),
                    min_dist=0.05,
                    metric="cosine",
                    random_state=random_state,
                    low_memory=True,
                    verbose=False,
                )
                coords = reducer.fit_transform(valid_vectors)
        else:
            raise ValueError(f"Unsupported projection: {projection}")
        projected.loc[valid_index, f"{projection}_x"] = coords[:, 0]
        projected.loc[valid_index, f"{projection}_y"] = coords[:, 1]
    return projected


def _path_plot_limits(frame: pd.DataFrame, *, x_col: str, y_col: str) -> tuple[tuple[float, float], tuple[float, float]]:
    x = pd.to_numeric(frame[x_col], errors="coerce")
    y = pd.to_numeric(frame[y_col], errors="coerce")
    finite = x.notna() & y.notna()
    if not finite.any():
        return (-1.0, 1.0), (-1.0, 1.0)
    x_values = x.loc[finite].to_numpy(dtype=float)
    y_values = y.loc[finite].to_numpy(dtype=float)
    x_range = float(x_values.max() - x_values.min())
    y_range = float(y_values.max() - y_values.min())
    x_pad = x_range * 0.06 if x_range > 0 else 1.0
    y_pad = y_range * 0.06 if y_range > 0 else 1.0
    return (
        (float(x_values.min() - x_pad), float(x_values.max() + x_pad)),
        (float(y_values.min() - y_pad), float(y_values.max() + y_pad)),
    )


def _top_subfield_ids(
    path_metrics: pd.DataFrame,
    *,
    value_column: str,
    top_n: int,
) -> list[str]:
    if path_metrics.empty or value_column not in path_metrics:
        return []
    return (
        path_metrics.sort_values(
            [value_column, "subfield_id"],
            ascending=[False, True],
            kind="mergesort",
            na_position="last",
        )
        .head(top_n)["subfield_id"]
        .astype(str)
        .tolist()
    )


def _short_text(value: object, *, width: int = 52) -> str:
    return textwrap.shorten(str(value), width=width, placeholder="...")


def plot_global_paths(
    projections: pd.DataFrame,
    path_metrics: pd.DataFrame,
    output_path: str | Path,
    *,
    projection: str,
    highlight_top_n: int = 0,
    label_final_points: bool = False,
    experimental: bool = False,
) -> None:
    x_col = f"{projection}_x"
    y_col = f"{projection}_y"
    if projections.empty or x_col not in projections or projections[x_col].notna().sum() == 0:
        write_placeholder_figure(output_path, f"No {projection.upper()} centroid paths available")
        return
    highlighted_ids = set(
        _top_subfield_ids(path_metrics, value_column="total_path_length", top_n=highlight_top_n)
    )
    palette = [
        "#b45432",
        "#2f6f73",
        "#5b5aa8",
        "#a05c99",
        "#6f7d2c",
        "#c27c2c",
        "#4f78a8",
        "#8a5a44",
    ]
    color_by_id = {
        subfield_id: palette[idx % len(palette)]
        for idx, subfield_id in enumerate(sorted(highlighted_ids))
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 8), dpi=240)
    xlim, ylim = _path_plot_limits(projections, x_col=x_col, y_col=y_col)
    
    # Check if this is the top 10 PCA subfields highlight case
    is_top_10_pca_highlight = (
        projection == "pca"
        and len(highlighted_ids) == 10
        and label_final_points
    )

    if is_top_10_pca_highlight:
        # Pre-build top_ids to get their ranks (from 1 to 10)
        top_ids = _top_subfield_ids(path_metrics, value_column="total_path_length", top_n=10)
        rank_by_id = {str(subfield_id): rank for rank, subfield_id in enumerate(top_ids, start=1)}
        
        # Mappings of subfield_id to custom text label
        CUSTOM_LABELS = {
            "2611": "2611 Modeling and Simulation",
            "2302": "2302 Ecological Modeling",
            "3307": "3307 Human Factors",
            "2102": "2102 Energy Engineering",
            "3500": "3500 General Dentistry",
            "2725": "2725 Infectious Diseases",
            "1707": "1707 Computer Vision",
            "1711": "1711 Signal Processing",
            "1702": "1702 Artificial Intelligence",
            "1705": "1705 Computer Networks",
        }
        
        # Absolute coordinates for custom annotation placement
        CUSTOM_OFFSETS = {
            "2611": {"xytext": (-0.035, -0.015), "ha": "right", "va": "center"},
            "2302": {"xytext": (0.065, -0.020), "ha": "left", "va": "center"},
            "3307": {"xytext": (0.125, 0.068), "ha": "right", "va": "center"},
            "2102": {"xytext": (0.050, -0.178), "ha": "left", "va": "center"},
            "3500": {"xytext": (0.042, 0.086), "ha": "left", "va": "center"},
            "2725": {"xytext": (-0.075, 0.085), "ha": "left", "va": "center"},
            "1702": {"xytext": (0.072, -0.048), "ha": "left", "va": "center"},
            "1711": {"xytext": (0.068, -0.072), "ha": "left", "va": "center"},
            "1707": {"xytext": (0.068, -0.096), "ha": "left", "va": "center"},
            "1705": {"xytext": (0.045, -0.132), "ha": "left", "va": "center"},
        }

    for subfield_id, group in projections.groupby("subfield_id", sort=False):
        group = (
            group.sort_values("window_start", kind="mergesort")
            .dropna(subset=[x_col, y_col])
            .reset_index(drop=True)
        )
        if group.empty:
            continue
        is_highlighted = str(subfield_id) in highlighted_ids
        color = color_by_id.get(str(subfield_id), "#a9a9a9")
        alpha = 0.9 if is_highlighted else 0.28
        linewidth = 1.8 if is_highlighted else 0.62
        zorder = 3 if is_highlighted else 1
        ax.plot(
            group[x_col],
            group[y_col],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            zorder=zorder,
        )
        ax.scatter(
            group[x_col],
            group[y_col],
            s=14 if is_highlighted else 4,
            color=color,
            alpha=alpha,
            linewidths=0,
            zorder=zorder + 1,
        )
        first = group.iloc[0]
        last = group.iloc[-1]
        
        if is_highlighted or highlight_top_n == 0:
            if is_top_10_pca_highlight:
                # Custom starting circle: size 75, lw 1.6
                ax.scatter(
                    [first[x_col]],
                    [first[y_col]],
                    s=75,
                    marker="o",
                    facecolors="white",
                    edgecolors=color,
                    linewidths=1.6,
                    alpha=0.95,
                    zorder=zorder + 2,
                )
                # Custom ending square: size 100, solid color
                ax.scatter(
                    [last[x_col]],
                    [last[y_col]],
                    s=100,
                    marker="s",
                    color=color,
                    alpha=0.95,
                    zorder=zorder + 2,
                )
                # Centered white rank number
                rank = rank_by_id[str(subfield_id)]
                ax.text(
                    last[x_col],
                    last[y_col],
                    str(rank),
                    color="white",
                    fontsize=6,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    zorder=zorder + 3,
                )
            else:
                # Default behavior
                ax.scatter(
                    [first[x_col]],
                    [first[y_col]],
                    s=26 if is_highlighted else 12,
                    marker="o",
                    facecolors="white",
                    edgecolors=color,
                    linewidths=0.8 if is_highlighted else 0.45,
                    alpha=0.95,
                    zorder=zorder + 2,
                )
                ax.scatter(
                    [last[x_col]],
                    [last[y_col]],
                    s=32 if is_highlighted else 14,
                    marker="s",
                    color=color,
                    linewidths=0,
                    alpha=0.95,
                    zorder=zorder + 2,
                )
                
        if is_highlighted and label_final_points:
            if is_top_10_pca_highlight:
                label = CUSTOM_LABELS.get(str(subfield_id), str(subfield_id))
                offset_info = CUSTOM_OFFSETS.get(str(subfield_id), {"xytext": (0.01, 0.0), "ha": "left", "va": "center"})
                xytext = offset_info["xytext"]
                ha = offset_info["ha"]
                va = offset_info["va"]
                
                ax.annotate(
                    text=label,
                    xy=(last[x_col], last[y_col]),
                    xytext=xytext,
                    textcoords="data",
                    arrowprops=dict(
                        arrowstyle="-",
                        color=color,
                        alpha=0.55,
                        lw=0.7,
                        shrinkA=0,
                        shrinkB=4,
                    ),
                    fontsize=7,
                    fontweight="normal",
                    color=color,
                    ha=ha,
                    va=va,
                    zorder=zorder + 3,
                )
            else:
                label = last.get("subfield_label_short", last.get("subfield_display_name", subfield_id))
                ax.text(
                    last[x_col],
                    last[y_col],
                    _short_text(label, width=46),
                    fontsize=5.4,
                    color=color,
                    alpha=0.95,
                    ha="left",
                    va="center",
                    zorder=zorder + 3,
                )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(f"{projection.upper()} 1")
    ax.set_ylabel(f"{projection.upper()} 2")
    
    # Custom Legend (2000-2004 open circle, 2020-2024 filled square, other windows light grey dot)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor="white",
            markeredgecolor="#555555",
            markeredgewidth=1.2,
            markersize=7,
            label="2000-2004",
        ),
        Line2D(
            [0], [0],
            marker="s",
            color="w",
            markerfacecolor="#333333",
            markeredgecolor="none",
            markersize=7,
            label="2020-2024",
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor="#e0e0e0",
            markeredgecolor="none",
            markersize=4,
            label="other windows",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower left",
        fontsize=8,
        framealpha=0.9,
        facecolor="white",
        edgecolor="#dddddd",
    )

    if experimental:
        title = (
            f"Experimental {projection.upper()} projection of subfield centroids; "
            "appendix-only"
        )
    elif highlight_top_n > 0:
        title = (
            f"PCA projection of top {highlight_top_n} dynamic subfields; "
            "distances measured in embedding space"
        )
    else:
        title = (
            "PCA projection of subfield centroids; "
            "distances measured in embedding space"
        )
        
    # Conditionally omit title for thesis-ready (clean/fig_07) versions
    is_thesis_fig = any(k in str(output_path).lower() for k in ["clean", "fig_07"])
    if not is_thesis_fig:
        ax.set_title(title, fontsize=12)
        
    ax.grid(True, color="#dddddd", linewidth=0.4, alpha=0.55)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_top_dynamic_small_multiples(
    projections: pd.DataFrame,
    path_metrics: pd.DataFrame,
    output_path: str | Path,
    *,
    top_n: int = 12,
) -> None:
    if projections.empty or "pca_x" not in projections or projections["pca_x"].notna().sum() == 0:
        write_placeholder_figure(output_path, "No PCA centroid paths available")
        return
    top_ids = _top_subfield_ids(
        path_metrics,
        value_column="total_path_length",
        top_n=top_n,
    )
    if not top_ids:
        write_placeholder_figure(output_path, "No dynamic subfield ranking available")
        return
    subset = projections.loc[projections["subfield_id"].astype(str).isin(top_ids)].copy()
    subset = subset.dropna(subset=["pca_x", "pca_y"])
    if subset.empty:
        write_placeholder_figure(output_path, "No PCA coordinates for top dynamic subfields")
        return

    rank_by_id = {subfield_id: rank for rank, subfield_id in enumerate(top_ids, start=1)}
    n_panels = len(top_ids)
    n_cols = 4 if n_panels >= 8 else min(3, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 3.5 * n_rows),
        dpi=220,
        squeeze=False,
    )
    xlim, ylim = _path_plot_limits(projections, x_col="pca_x", y_col="pca_y")
    for subfield_id, ax in zip(top_ids, axes.ravel(), strict=False):
        group = (
            subset.loc[subset["subfield_id"].astype(str) == subfield_id]
            .sort_values("window_start", kind="mergesort")
            .reset_index(drop=True)
        )
        if group.empty:
            ax.axis("off")
            continue
        color = "#b45432"
        ax.plot(group["pca_x"], group["pca_y"], color=color, linewidth=1.5, alpha=0.92)
        ax.scatter(group["pca_x"], group["pca_y"], color=color, s=15, linewidths=0, alpha=0.92)
        first = group.iloc[0]
        last = group.iloc[-1]
        ax.scatter(
            [first["pca_x"]],
            [first["pca_y"]],
            marker="o",
            s=42,
            facecolors="white",
            edgecolors=color,
            linewidths=1.1,
            zorder=4,
        )
        ax.scatter(
            [last["pca_x"]],
            [last["pca_y"]],
            marker="s",
            s=46,
            color=color,
            linewidths=0,
            zorder=4,
        )
        label = group.iloc[-1].get("subfield_label_short", subfield_id)
        ax.set_title(
            f"{rank_by_id[subfield_id]}. {_short_text(label, width=44)}",
            fontsize=8.5,
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(True, color="#dddddd", linewidth=0.35, alpha=0.5)
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[n_panels:]:
        ax.axis("off")
    fig.suptitle(
        "Top Dynamic Subfield Centroid Paths in PCA Space",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(output_path)
    plt.close(fig)


def plot_domain_facets_pca(projections: pd.DataFrame, output_path: str | Path) -> None:
    if projections.empty or "pca_x" not in projections or projections["pca_x"].notna().sum() == 0:
        write_placeholder_figure(output_path, "No PCA centroid paths available")
        return
    domains = sorted(projections["domain_display_name"].dropna().astype(str).unique())
    n_cols = min(3, max(1, len(domains)))
    n_rows = int(np.ceil(len(domains) / n_cols))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 4.6 * n_rows),
        dpi=220,
        squeeze=False,
    )
    xlim, ylim = _path_plot_limits(projections, x_col="pca_x", y_col="pca_y")
    for ax, domain in zip(axes.ravel(), domains, strict=False):
        subset = projections.loc[projections["domain_display_name"].astype(str) == domain]
        for _, group in subset.groupby("subfield_id", sort=False):
            group = group.sort_values("window_start", kind="mergesort")
            ax.plot(group["pca_x"], group["pca_y"], color="#2f6f73", alpha=0.36, linewidth=0.8)
            ax.scatter(group["pca_x"], group["pca_y"], s=7, color="#2f6f73", alpha=0.42)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(domain, fontsize=10)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(domains) :]:
        ax.axis("off")
    fig.suptitle("PCA Centroid Paths Faceted By Domain", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path)
    plt.close(fig)


def plot_top_path_metric(
    path_metrics: pd.DataFrame,
    output_path: str | Path,
    *,
    value_column: str,
    title: str,
    top_n: int,
) -> None:
    if path_metrics.empty or value_column not in path_metrics:
        write_placeholder_figure(output_path, f"No {value_column} ranking available")
        return
    top = path_metrics.sort_values(
        [value_column, "subfield_id"],
        ascending=[False, True],
        kind="mergesort",
        na_position="last",
    ).head(top_n).iloc[::-1]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(top))), dpi=240)
    ax.barh(top["subfield_label_short"], top[value_column], color="#2f6f73")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Cosine distance in original embedding space")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def largest_jump_window_distribution(
    path_metrics: pd.DataFrame,
    *,
    windows: list[TemporalWindow],
) -> pd.DataFrame:
    expected = [
        f"{windows[idx - 1].window_label}_to_{windows[idx].window_label}"
        for idx in range(1, len(windows))
    ]
    if path_metrics.empty or "largest_jump_window" not in path_metrics:
        return pd.DataFrame(
            {
                "largest_jump_window": expected,
                "n_subfields": [0] * len(expected),
                "share_of_subfields": [np.nan] * len(expected),
            }
        )
    labels = (
        path_metrics["largest_jump_window"]
        .dropna()
        .astype(str)
        .loc[lambda values: values.str.len() > 0]
    )
    counts = labels.value_counts().to_dict()
    ordered_labels = expected + sorted(label for label in counts if label not in expected)
    total = int(labels.shape[0])
    rows = []
    for label in ordered_labels:
        count = int(counts.get(label, 0))
        rows.append(
            {
                "largest_jump_window": label,
                "n_subfields": count,
                "share_of_subfields": float(count / total) if total else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_largest_jump_window_distribution(
    distribution: pd.DataFrame,
    output_path: str | Path,
) -> None:
    if distribution.empty:
        write_placeholder_figure(output_path, "No largest-jump-window distribution available")
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_frame = distribution.iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.5 * len(plot_frame))), dpi=240)
    ax.barh(
        plot_frame["largest_jump_window"],
        plot_frame["n_subfields"],
        color="#4f78a8",
    )
    ax.set_title("Distribution of Largest Single-Step Centroid Jumps", fontsize=12)
    ax.set_xlabel("Number of subfields")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def centroid_movement_summary(path_metrics: pd.DataFrame, *, level: str) -> pd.DataFrame:
    id_col = f"{level}_id"
    display_col = f"{level}_display_name"
    columns = [
        id_col,
        display_col,
        "n_subfields",
        "mean_total_path_length",
        "mean_early_late_drift",
        "mean_directionality_ratio_euclidean",
        "median_total_path_length",
        "median_early_late_drift",
    ]
    if path_metrics.empty or id_col not in path_metrics or display_col not in path_metrics:
        return pd.DataFrame(columns=columns)
    summary = (
        path_metrics.groupby([id_col, display_col], dropna=False)
        .agg(
            n_subfields=("subfield_id", "nunique"),
            mean_total_path_length=("total_path_length", "mean"),
            mean_early_late_drift=("early_late_drift", "mean"),
            mean_directionality_ratio_euclidean=(
                "directionality_ratio_euclidean",
                "mean",
            ),
            median_total_path_length=("total_path_length", "median"),
            median_early_late_drift=("early_late_drift", "median"),
        )
        .reset_index()
    )
    return summary.sort_values(
        ["mean_total_path_length", id_col],
        ascending=[False, True],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)


def plot_group_average_metric(
    summary: pd.DataFrame,
    output_path: str | Path,
    *,
    label_column: str,
    value_column: str,
    title: str,
    top_n: int | None = None,
) -> None:
    if summary.empty or value_column not in summary:
        write_placeholder_figure(output_path, f"No {value_column} summary available")
        return
    plot_frame = summary.sort_values(
        [value_column, label_column],
        ascending=[False, True],
        kind="mergesort",
        na_position="last",
    )
    if top_n is not None:
        plot_frame = plot_frame.head(top_n)
    plot_frame = plot_frame.iloc[::-1].copy()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height = max(4.2, 0.36 * len(plot_frame) + 1.2)
    fig, ax = plt.subplots(figsize=(10, height), dpi=240)
    ax.barh(plot_frame[label_column].astype(str), plot_frame[value_column], color="#2f6f73")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Mean cosine distance in original embedding space")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def missing_or_failed_centroid_rows(centroids: pd.DataFrame) -> pd.DataFrame:
    if centroids.empty or "centroid_status" not in centroids:
        return pd.DataFrame()
    columns = [
        "subfield_id",
        "subfield_label_short",
        "window_label",
        "n_available",
        "n_used",
        "centroid_status",
        "centroid_error_message",
        "centroid_warning_message",
    ]
    available_columns = [column for column in columns if column in centroids]
    return centroids.loc[
        ~centroids["centroid_status"].isin(VALID_CENTROID_STATUSES),
        available_columns,
    ].reset_index(drop=True)


def markdown_summary(
    *,
    input_paths: dict[str, str],
    output_paths_payload: dict[str, str],
    n_centroid_rows: int,
    n_completed_rows: int,
    n_subfields: int,
    projections: list[str],
    missing_or_failed_rows: pd.DataFrame,
    top_path_length: pd.DataFrame,
    top_early_late_drift: pd.DataFrame,
    largest_jump_distribution: pd.DataFrame,
    domain_summary: pd.DataFrame,
    field_summary: pd.DataFrame,
    recommended_main_text_figures: list[str],
    appendix_only_figures: list[str],
) -> str:
    n_missing_or_failed = int(len(missing_or_failed_rows))
    lines = [
        "# Temporal Centroid Paths",
        "",
        "This analysis tracks how each subfield centroid moves across fixed five-year windows.",
        "UMAP is not part of the default/main analysis.",
        "",
        "## Inputs",
        "",
    ]
    for label, path in input_paths.items():
        lines.append(f"- {label}: `{path}`")
    lines.extend(
        [
            "",
            "## Methods",
            "",
            "- Centroids are computed from L2-normalized SPECTER2 paper embeddings in the original 768-dimensional space.",
            "- Step distances, total path length, and early-late drift are cosine distances in that original embedding space.",
            "- PCA is only a two-dimensional visualization of the 768D centroid paths; quantitative rankings do not use PCA distances.",
            "- The main directionality metric is early-late Euclidean drift divided by accumulated Euclidean path length after normalizing each centroid vector to unit norm.",
            "- The old cosine-distance directionality ratio is retained only as `directionality_ratio_cosine_diagnostic`.",
            "- Counts are quality-control metadata and are not interpreted as publication-volume trends.",
            "",
            "## Outputs",
            "",
        ]
    )
    for label, path in output_paths_payload.items():
        lines.append(f"- {label}: `{path}`")
    lines.extend(
        [
            "",
            "## Run Summary",
            "",
            f"- Subfields attempted: {n_subfields}",
            f"- Centroid rows: {n_centroid_rows}",
            f"- Completed centroid rows: {n_completed_rows}",
            f"- Missing/skipped/failed centroid rows: {n_missing_or_failed}",
            f"- Projections requested: {', '.join(projections)}",
            f"- Experimental UMAP included: {'yes' if 'umap' in projections else 'no'}",
            "",
            "## Recommended Main-Text Figures",
            "",
        ]
    )
    for path in recommended_main_text_figures:
        lines.append(f"- `{path}`")
    if appendix_only_figures:
        lines.extend(["", "## Appendix-Only Figures", ""])
        for path in appendix_only_figures:
            lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "## Top 10 By Accumulated Path Length",
            "",
        ]
    )
    if top_path_length.empty:
        lines.append("No centroid path ranking was available.")
    else:
        for row in top_path_length.head(10).itertuples(index=False):
            lines.append(
                f"- {row.subfield_label_short}: path={row.total_path_length:.3f}, "
                f"early-late drift={row.early_late_drift:.3f}"
            )
    lines.extend(["", "## Top 10 By Early-Late Drift", ""])
    if top_early_late_drift.empty:
        lines.append("No early-late drift ranking was available.")
    else:
        for row in top_early_late_drift.head(10).itertuples(index=False):
            lines.append(
                f"- {row.subfield_label_short}: drift={row.early_late_drift:.3f}, "
                f"path={row.total_path_length:.3f}"
            )
    lines.extend(["", "## Largest-Jump Windows", ""])
    if largest_jump_distribution.empty:
        lines.append("No largest-jump-window distribution was available.")
    else:
        for row in largest_jump_distribution.itertuples(index=False):
            lines.append(
                f"- {row.largest_jump_window}: {row.n_subfields} subfields"
            )
    lines.extend(["", "## Domain Summary", ""])
    if domain_summary.empty:
        lines.append("No domain movement summary was available.")
    else:
        for row in domain_summary.itertuples(index=False):
            lines.append(
                f"- {row.domain_display_name}: mean path={row.mean_total_path_length:.3f}, "
                f"mean drift={row.mean_early_late_drift:.3f}, n={row.n_subfields}"
            )
    lines.extend(["", "## Field Summary", ""])
    if field_summary.empty:
        lines.append("No field movement summary was available.")
    else:
        for row in field_summary.head(20).itertuples(index=False):
            lines.append(
                f"- {row.field_display_name}: mean path={row.mean_total_path_length:.3f}, "
                f"mean drift={row.mean_early_late_drift:.3f}, n={row.n_subfields}"
            )
    if not missing_or_failed_rows.empty:
        lines.extend(["", "## Missing Or Failed Centroid Rows", ""])
        for row in missing_or_failed_rows.head(20).itertuples(index=False):
            message = getattr(row, "centroid_error_message", "")
            lines.append(
                f"- {row.subfield_label_short}, {row.window_label}: "
                f"{row.centroid_status}; {message}"
            )
        if len(missing_or_failed_rows) > 20:
            lines.append(f"- ... {len(missing_or_failed_rows) - 20} additional rows")
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- A long path with a low Euclidean directionality ratio indicates movement with reversals or bends.",
            "- Projection geometry should not be read as exact distance; use the path metrics table for quantitative interpretation.",
            "",
        ]
    )
    return "\n".join(lines)


def write_centroid_path_outputs(
    *,
    centroids: pd.DataFrame,
    path_metrics: pd.DataFrame,
    projected: pd.DataFrame,
    windows: list[TemporalWindow],
    output_dir: str | Path,
    root: Path,
    input_paths: dict[str, str],
    projections: list[str],
    highlight_top_n: int,
    cleanup_legacy_umap: bool = False,
) -> dict[str, Any]:
    include_umap_experimental = "umap" in projections
    paths = output_paths(
        output_dir,
        root=root,
        include_umap_experimental=include_umap_experimental,
    )
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    removed_legacy_umap_outputs = (
        cleanup_legacy_umap_outputs(output_dir, root=root)
        if cleanup_legacy_umap and not include_umap_experimental
        else []
    )

    jump_distribution = largest_jump_window_distribution(path_metrics, windows=windows)
    domain_summary = centroid_movement_summary(path_metrics, level="domain")
    field_summary = centroid_movement_summary(path_metrics, level="field")
    missing_failed = missing_or_failed_centroid_rows(centroids)
    if not path_metrics.empty and {"total_path_length", "subfield_id"}.issubset(path_metrics):
        top_path_length = path_metrics.sort_values(
            ["total_path_length", "subfield_id"],
            ascending=[False, True],
            kind="mergesort",
            na_position="last",
        )
    else:
        top_path_length = path_metrics.copy()
    if not path_metrics.empty and {"early_late_drift", "subfield_id"}.issubset(path_metrics):
        top_early_late_drift = path_metrics.sort_values(
            ["early_late_drift", "subfield_id"],
            ascending=[False, True],
            kind="mergesort",
            na_position="last",
        )
    else:
        top_early_late_drift = path_metrics.copy()

    save_parquet(centroids, paths["centroids_parquet"])
    save_parquet(path_metrics, paths["path_metrics_parquet"])
    path_metrics.to_csv(paths["path_metrics_csv"], index=False)
    save_parquet(projected, paths["centroid_projections_parquet"])
    top_path_length.head(highlight_top_n).to_csv(paths["top_dynamic_subfields_csv"], index=False)
    jump_distribution.to_csv(paths["largest_jump_window_distribution_csv"], index=False)
    domain_summary.to_csv(paths["domain_centroid_movement_summary_csv"], index=False)
    field_summary.to_csv(paths["field_centroid_movement_summary_csv"], index=False)

    plot_global_paths(projected, path_metrics, paths["global_pca"], projection="pca")
    plot_global_paths(
        projected,
        path_metrics,
        paths["global_pca_highlight"],
        projection="pca",
        highlight_top_n=highlight_top_n,
        label_final_points=True,
    )
    plot_global_paths(projected, path_metrics, paths["global_pca_clean"], projection="pca")
    plot_global_paths(
        projected,
        path_metrics,
        paths["global_pca_top_dynamic_clean"],
        projection="pca",
        highlight_top_n=highlight_top_n,
        label_final_points=True,
    )
    plot_top_dynamic_small_multiples(
        projected,
        path_metrics,
        paths["top_dynamic_pca_small_multiples"],
        top_n=min(12, max(1, highlight_top_n)),
    )
    if "umap" in projections:
        plot_global_paths(
            projected,
            path_metrics,
            paths["global_umap_experimental"],
            projection="umap",
            experimental=True,
        )
        plot_global_paths(
            projected,
            path_metrics,
            paths["global_umap_top_dynamic_experimental"],
            projection="umap",
            highlight_top_n=highlight_top_n,
            label_final_points=True,
            experimental=True,
        )
    plot_domain_facets_pca(projected, paths["domain_faceted_pca"])
    plot_largest_jump_window_distribution(
        jump_distribution,
        paths["largest_jump_window_distribution"],
    )
    plot_group_average_metric(
        domain_summary,
        paths["domain_average_path_length"],
        label_column="domain_display_name",
        value_column="mean_total_path_length",
        title="Mean Accumulated Centroid Path By Domain",
    )
    plot_group_average_metric(
        domain_summary,
        paths["domain_average_early_late_drift"],
        label_column="domain_display_name",
        value_column="mean_early_late_drift",
        title="Mean Early-Late Centroid Drift By Domain",
    )
    plot_group_average_metric(
        field_summary,
        paths["field_average_path_length_top20"],
        label_column="field_display_name",
        value_column="mean_total_path_length",
        title="Mean Accumulated Centroid Path By Field",
        top_n=20,
    )
    plot_group_average_metric(
        field_summary,
        paths["field_average_early_late_drift_top20"],
        label_column="field_display_name",
        value_column="mean_early_late_drift",
        title="Mean Early-Late Centroid Drift By Field",
        top_n=20,
    )
    plot_top_path_metric(
        path_metrics,
        paths["top_path_length"],
        value_column="total_path_length",
        title="Longest accumulated centroid paths",
        top_n=highlight_top_n,
    )
    plot_top_path_metric(
        path_metrics,
        paths["top_early_late_drift"],
        value_column="early_late_drift",
        title="Largest early-late centroid drift",
        top_n=highlight_top_n,
    )

    output_paths_payload = {key: display_path(path, root=root) for key, path in paths.items()}
    n_completed = int(centroids["centroid_status"].isin(VALID_CENTROID_STATUSES).sum())
    recommended_main_text_figures = [
        output_paths_payload["global_pca_clean"],
        output_paths_payload["global_pca_top_dynamic_clean"],
        output_paths_payload["top_dynamic_pca_small_multiples"],
        output_paths_payload["top_path_length"],
        output_paths_payload["top_early_late_drift"],
        output_paths_payload["largest_jump_window_distribution"],
        output_paths_payload["domain_average_path_length"],
        output_paths_payload["domain_average_early_late_drift"],
        output_paths_payload["field_average_path_length_top20"],
        output_paths_payload["field_average_early_late_drift_top20"],
    ]
    appendix_only_figures = [output_paths_payload["domain_faceted_pca"]]
    if include_umap_experimental:
        appendix_only_figures.extend(
            [
                output_paths_payload["global_umap_experimental"],
                output_paths_payload["global_umap_top_dynamic_experimental"],
            ]
        )
    summary = {
        "created_at": utc_now_iso(),
        "input_paths": input_paths,
        "output_paths": output_paths_payload,
        "projections": projections,
        "main_projection": "pca",
        "umap_default_included": False,
        "experimental_umap_included": include_umap_experimental,
        "legacy_umap_outputs_removed": removed_legacy_umap_outputs,
        "n_centroid_rows": int(len(centroids)),
        "n_completed_centroid_rows": n_completed,
        "n_missing_or_failed_centroid_rows": int(len(missing_failed)),
        "n_failed_centroid_rows": int(
            (centroids["centroid_status"] == "failed").sum()
            if "centroid_status" in centroids
            else 0
        ),
        "n_skipped_centroid_rows": int(
            (centroids["centroid_status"] == "skipped").sum()
            if "centroid_status" in centroids
            else 0
        ),
        "n_subfields_attempted": int(centroids["subfield_id"].nunique())
        if "subfield_id" in centroids
        else 0,
        "top_dynamic_subfields": dataframe_to_records(top_path_length, limit=20),
        "top_total_path_length_subfields": dataframe_to_records(top_path_length, limit=10),
        "top_early_late_drift_subfields": dataframe_to_records(
            top_early_late_drift,
            limit=10,
        ),
        "largest_jump_window_distribution": dataframe_to_records(
            jump_distribution,
            limit=len(jump_distribution),
        ),
        "domain_centroid_movement_summary": dataframe_to_records(
            domain_summary,
            limit=len(domain_summary),
        ),
        "field_centroid_movement_summary": dataframe_to_records(
            field_summary,
            limit=len(field_summary),
        ),
        "missing_or_failed_centroid_rows": dataframe_to_records(
            missing_failed,
            limit=min(len(missing_failed), 50),
        ),
        "recommended_main_text_figures": recommended_main_text_figures,
        "appendix_only_figures": appendix_only_figures,
        "method_notes": [
            "UMAP is not part of the default/main analysis.",
            "Distances and rankings are measured in the original embedding space.",
            "PCA is only a visualization of 768D centroid paths.",
            (
                "directionality_ratio_euclidean = "
                "early_late_drift_euclidean_normalized / "
                "total_path_length_euclidean_normalized, where each centroid "
                "is first normalized to unit norm."
            ),
            (
                "directionality_ratio_cosine_diagnostic keeps the old "
                "cosine-distance ratio for diagnostics only."
            ),
        ],
    }
    write_json(paths["summary_json"], summary)
    write_text(
        paths["summary_md"],
        markdown_summary(
            input_paths=input_paths,
            output_paths_payload=output_paths_payload,
            n_centroid_rows=len(centroids),
            n_completed_rows=n_completed,
            n_subfields=summary["n_subfields_attempted"],
            projections=projections,
            missing_or_failed_rows=missing_failed,
            top_path_length=top_path_length,
            top_early_late_drift=top_early_late_drift,
            largest_jump_distribution=jump_distribution,
            domain_summary=domain_summary,
            field_summary=field_summary,
            recommended_main_text_figures=recommended_main_text_figures,
            appendix_only_figures=appendix_only_figures,
        ),
    )
    return summary


def build_centroid_path_outputs(
    analysis_index: pd.DataFrame,
    matrix: np.ndarray,
    *,
    year_min: int,
    year_max: int,
    window_size: int,
    min_papers_per_window: int,
    projection: list[str],
    highlight_top_n: int,
    random_state: int,
    subfield_id: str | None,
    limit_subfields: int | None,
    analysis_index_path: str,
    embedding_matrix_path: str,
    output_dir: str | Path,
    root: Path,
    cleanup_legacy_umap: bool = False,
) -> dict[str, Any]:
    windows = default_or_computed_windows(
        year_min=year_min,
        year_max=year_max,
        window_size=window_size,
    )
    subfields = main_analysis_subfields(analysis_index)
    attempted_subfields = select_attempted_subfields(
        subfields,
        subfield_id=subfield_id,
        limit_subfields=limit_subfields,
    )
    centroids = compute_subfield_window_centroids(
        analysis_index,
        matrix,
        windows=windows,
        attempted_subfields=attempted_subfields,
        year_min=year_min,
        year_max=year_max,
        min_papers_per_window=min_papers_per_window,
    )
    path_metrics = compute_centroid_path_metrics(centroids, windows=windows)
    projections = list(dict.fromkeys(["pca", *projection]))
    projected = project_centroids(
        centroids,
        projections=projections,
        random_state=random_state,
    )
    return write_centroid_path_outputs(
        centroids=centroids,
        path_metrics=path_metrics,
        projected=projected,
        windows=windows,
        output_dir=output_dir,
        root=root,
        input_paths={
            "analysis_index_path": analysis_index_path,
            "embedding_matrix_path": embedding_matrix_path,
        },
        projections=projections,
        highlight_top_n=highlight_top_n,
        cleanup_legacy_umap=cleanup_legacy_umap,
    )
