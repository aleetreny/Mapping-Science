from __future__ import annotations

import json
from pathlib import Path
from textwrap import wrap
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from src.embedding_space_metrics import (
    concentration_metrics,
    knn_metrics,
    l2_normalize_rows,
    pca_metrics,
    validate_embedding_index_columns,
)
from src.per_subfield_umap_maps import (
    filter_input_window,
    main_analysis_subfields,
)
from src.storage import save_parquet
from src.subfield_labels import add_subfield_label_columns
from src.temporal_common import (
    WINDOW_STRUCTURAL_METRICS,
    TemporalWindow,
    dataframe_to_records,
    default_or_computed_windows,
    direction_label,
    display_path,
    ensure_outputs_do_not_exist,
    linear_slope,
    robust_zscore_series,
    safe_spearman,
    standardization_parameters,
    utc_now_iso,
    write_json,
    write_placeholder_figure,
    write_text,
    zscore_series,
)

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


VALID_WINDOW_STATUSES = {"completed", "completed_with_warnings"}

PROCESSED_TEMPORAL_DIR = Path("data/processed/temporal")

OUTPUT_FILENAMES = {
    "window_metrics_parquet": PROCESSED_TEMPORAL_DIR
    / "subfield_window_embedding_metrics.parquet",
    "window_metrics_csv": PROCESSED_TEMPORAL_DIR
    / "subfield_window_embedding_metrics.csv",
    "changes_parquet": PROCESSED_TEMPORAL_DIR
    / "subfield_metric_temporal_changes.parquet",
    "changes_csv": PROCESSED_TEMPORAL_DIR
    / "subfield_metric_temporal_changes.csv",
    "overall_ranking_parquet": PROCESSED_TEMPORAL_DIR
    / "subfield_overall_temporal_change_ranking.parquet",
    "overall_ranking_csv": PROCESSED_TEMPORAL_DIR
    / "subfield_overall_temporal_change_ranking.csv",
    "field_trajectories_parquet": PROCESSED_TEMPORAL_DIR
    / "field_window_embedding_metric_trajectories.parquet",
    "domain_trajectories_parquet": PROCESSED_TEMPORAL_DIR
    / "domain_window_embedding_metric_trajectories.parquet",
    "overall_trajectories_parquet": PROCESSED_TEMPORAL_DIR
    / "overall_window_embedding_metric_trajectories.parquet",
}

ANALYSIS_FILENAMES = {
    "delta_heatmap": "temporal_metric_delta_heatmap.png",
    "slope_heatmap": "temporal_metric_slope_heatmap.png",
    "top_changed_overall": "top_changed_subfields_overall.png",
    "top_changed_by_metric": "top_changed_subfields_by_metric.png",
    "domain_trajectories": "domain_average_metric_trajectories.png",
    "domain_compactness_dimensionality_trajectories": (
        "domain_average_compactness_dimensionality_trajectories.png"
    ),
    "domain_heterogeneity_hubness_trajectories": (
        "domain_average_heterogeneity_hubness_trajectories.png"
    ),
    "field_trajectories": "field_average_metric_trajectories.png",
    "top_changes_by_metric_csv": "top_changes_by_metric.csv",
    "top_overall_changed_csv": "top_overall_changed_subfields.csv",
    "scaling_parameters_csv": "temporal_metric_scaling_parameters.csv",
    "window_sample_size_diagnostics_csv": "window_sample_size_diagnostics.csv",
    "subfield_window_sample_size_diagnostics_csv": (
        "subfield_window_sample_size_diagnostics.csv"
    ),
    "summary_md": "summary.md",
    "summary_json": "summary.json",
}

METRIC_DISPLAY_NAMES = {
    "embedding_distance_to_centroid_median": "Semantic dispersion (median)",
    "embedding_distance_to_centroid_iqr": "Dispersion heterogeneity (IQR)",
    "embedding_distance_to_centroid_p90": "Semantic periphery (p90)",
    "embedding_knn_median_distance": "Local semantic distance",
    "embedding_knn_distance_cv": "Local density heterogeneity",
    "embedding_knn_indegree_gini": "Semantic hubness (kNN Gini)",
    "embedding_pca_dim_80": "Intrinsic dimensionality (PCA dim 80)",
    "embedding_pca_spectral_entropy": "Spectral dimensionality entropy",
}

DOMAIN_COMPACTNESS_DIMENSIONALITY_METRICS = [
    "embedding_distance_to_centroid_median",
    "embedding_distance_to_centroid_p90",
    "embedding_knn_median_distance",
    "embedding_pca_dim_80",
    "embedding_pca_spectral_entropy",
]

DOMAIN_HETEROGENEITY_HUBNESS_METRICS = [
    "embedding_distance_to_centroid_iqr",
    "embedding_knn_distance_cv",
    "embedding_knn_indegree_gini",
]

TRAJECTORY_SCALING_NOTE = (
    "Values are z-scores computed across all subfield-window observations "
    "for each metric."
)


def output_paths(output_dir: str | Path, *, root: Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    paths = {
        key: root / value for key, value in OUTPUT_FILENAMES.items()
    }
    paths.update({key: output_dir / value for key, value in ANALYSIS_FILENAMES.items()})
    return paths


def existing_outputs(output_dir: str | Path, *, root: Path) -> list[Path]:
    return [path for path in output_paths(output_dir, root=root).values() if path.exists()]


def ensure_temporal_metric_outputs(
    output_dir: str | Path,
    *,
    overwrite: bool,
    root: Path,
) -> None:
    ensure_outputs_do_not_exist(
        output_paths(output_dir, root=root).values(),
        overwrite=overwrite,
        root=root,
        description="temporal embedding metric outputs",
    )


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


def _first_value(frame: pd.DataFrame, column: str, default: object = "") -> object:
    if column not in frame.columns or frame.empty:
        return default
    values = frame[column].dropna()
    if values.empty:
        return default
    return values.iloc[0]


def _nan_metrics() -> dict[str, float]:
    return {metric: np.nan for metric in WINDOW_STRUCTURAL_METRICS}


def _base_window_row(
    *,
    subfield: pd.Series | dict[str, Any],
    window: TemporalWindow,
    n_available: int,
    n_used: int,
    status: str,
    error_message: str = "",
    warning_message: str = "",
    k_neighbors: int,
    effective_k_neighbors: int = 0,
    n_valid_embedding_rows: int = 0,
    n_pca_components_used: int = 0,
    analysis_index_path: str = "",
    embedding_matrix_path: str = "",
) -> dict[str, Any]:
    getter = subfield.get
    row: dict[str, Any] = {
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
        "metric_status": status,
        "metric_error_message": error_message,
        "metric_warning_message": warning_message,
        "analysis_index_path": analysis_index_path,
        "embedding_matrix_path": embedding_matrix_path,
        "k_neighbors": int(k_neighbors),
        "effective_k_neighbors": int(effective_k_neighbors),
        "n_valid_embedding_rows": int(n_valid_embedding_rows),
        "n_pca_components_used": int(n_pca_components_used),
    }
    row.update(_nan_metrics())
    return row


def compute_window_structural_metrics(
    rows: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    k_neighbors: int,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    if len(rows) != len(embeddings):
        raise ValueError(
            "window index row count does not match embedding row count: "
            f"{len(rows)} != {len(embeddings)}"
        )
    normalized, valid_mask = l2_normalize_rows(embeddings)
    valid_count = int(np.count_nonzero(valid_mask))
    if valid_count == 0:
        raise ValueError("no valid non-zero embedding rows are available")
    warnings: list[str] = []
    if valid_count < len(normalized):
        warnings.append(
            f"dropped {len(normalized) - valid_count} zero-norm or non-finite embedding rows"
        )
    normalized = normalized[valid_mask]

    concentration, concentration_warnings = concentration_metrics(normalized)
    knn, effective_k, knn_warnings = knn_metrics(normalized, k_neighbors=k_neighbors)
    pca, n_pca_components, pca_warnings = pca_metrics(normalized)
    warnings.extend(concentration_warnings)
    warnings.extend(knn_warnings)
    warnings.extend(pca_warnings)

    metrics: dict[str, Any] = {}
    metrics.update(concentration)
    metrics.update(knn)
    metrics.update(pca)
    return (
        {metric: metrics.get(metric, np.nan) for metric in WINDOW_STRUCTURAL_METRICS},
        {
            "effective_k_neighbors": int(effective_k),
            "n_valid_embedding_rows": int(valid_count),
            "n_pca_components_used": int(n_pca_components),
        },
        list(dict.fromkeys(warnings)),
    )


def compute_subfield_window_metrics(
    analysis_index: pd.DataFrame,
    matrix: np.ndarray,
    *,
    windows: list[TemporalWindow],
    attempted_subfields: pd.DataFrame,
    year_min: int,
    year_max: int,
    min_papers_per_window: int,
    k_neighbors: int,
    analysis_index_path: str,
    embedding_matrix_path: str,
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
    rows: list[dict[str, Any]] = []
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
                rows.append(
                    _base_window_row(
                        subfield=subfield,
                        window=window,
                        n_available=n_available,
                        n_used=0,
                        status="skipped",
                        error_message=(
                            f"Only {n_available} papers available; "
                            f"min_papers_per_window is {min_papers_per_window}"
                        ),
                        k_neighbors=k_neighbors,
                        analysis_index_path=analysis_index_path,
                        embedding_matrix_path=embedding_matrix_path,
                    )
                )
                continue
            try:
                ordered = available.sort_values(
                    ["publication_year", "work_id", "analysis_row_id"],
                    kind="mergesort",
                ).reset_index(drop=True)
                row_ids = ordered["analysis_row_id"].astype(int).to_numpy()
                embeddings = np.asarray(matrix[row_ids], dtype=np.float32)
                metrics, controls, warnings = compute_window_structural_metrics(
                    ordered,
                    embeddings,
                    k_neighbors=k_neighbors,
                )
                row = _base_window_row(
                    subfield=subfield,
                    window=window,
                    n_available=n_available,
                    n_used=len(ordered),
                    status="completed_with_warnings" if warnings else "completed",
                    warning_message="; ".join(warnings),
                    k_neighbors=k_neighbors,
                    effective_k_neighbors=controls["effective_k_neighbors"],
                    n_valid_embedding_rows=controls["n_valid_embedding_rows"],
                    n_pca_components_used=controls["n_pca_components_used"],
                    analysis_index_path=analysis_index_path,
                    embedding_matrix_path=embedding_matrix_path,
                )
                row.update(metrics)
                rows.append(row)
            except Exception as exc:
                rows.append(
                    _base_window_row(
                        subfield=subfield,
                        window=window,
                        n_available=n_available,
                        n_used=0,
                        status="failed",
                        error_message=str(exc),
                        k_neighbors=k_neighbors,
                        analysis_index_path=analysis_index_path,
                        embedding_matrix_path=embedding_matrix_path,
                    )
                )
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame()
    return add_subfield_label_columns(result).sort_values(
        ["subfield_id", "window_start"],
        kind="mergesort",
    ).reset_index(drop=True)


def metric_long_frame(window_metrics: pd.DataFrame) -> pd.DataFrame:
    metadata_columns = [
        "subfield_id",
        "subfield_display_name",
        "subfield_label_unique",
        "subfield_label_short",
        "field_id",
        "field_display_name",
        "domain_id",
        "domain_display_name",
        "window_start",
        "window_end",
        "window_index",
        "window_label",
        "window_mid_year",
        "n_available",
        "n_used",
        "metric_status",
    ]
    available_metadata = [column for column in metadata_columns if column in window_metrics]
    available_metrics = [
        metric for metric in WINDOW_STRUCTURAL_METRICS if metric in window_metrics.columns
    ]
    if not available_metrics:
        return pd.DataFrame()
    long = window_metrics.melt(
        id_vars=available_metadata,
        value_vars=available_metrics,
        var_name="metric",
        value_name="metric_value",
    )
    long["metric_value"] = pd.to_numeric(long["metric_value"], errors="coerce")
    long["standardized_value"] = long.groupby("metric", group_keys=False)[
        "metric_value"
    ].transform(zscore_series)
    long["robust_standardized_value"] = long.groupby("metric", group_keys=False)[
        "metric_value"
    ].transform(robust_zscore_series)
    return long.sort_values(
        ["metric", "subfield_id", "window_start"],
        kind="mergesort",
    ).reset_index(drop=True)


def metric_scaling_parameters(long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric, group in long.groupby("metric", sort=True):
        row = {
            "metric": metric,
            "standardization_scope": "all_subfield_window_rows",
            "standardized_value_formula": "(metric_value - mean) / std",
            "robust_standardized_value_formula": "(metric_value - median) / iqr",
        }
        row.update(standardization_parameters(group["metric_value"]))
        rows.append(row)
    return pd.DataFrame(rows)


def _window_lookup(group: pd.DataFrame, window_label: str) -> pd.Series | None:
    selected = group.loc[group["window_label"].astype(str) == window_label]
    if selected.empty:
        return None
    return selected.iloc[0]


def temporal_change_frame(
    long: pd.DataFrame,
    *,
    windows: list[TemporalWindow],
    stable_threshold: float = 0.25,
) -> pd.DataFrame:
    if long.empty:
        return pd.DataFrame()
    initial_label = windows[0].window_label
    final_label = windows[-1].window_label
    rows: list[dict[str, Any]] = []
    group_columns = [
        "subfield_id",
        "subfield_display_name",
        "subfield_label_unique",
        "subfield_label_short",
        "field_id",
        "field_display_name",
        "domain_id",
        "domain_display_name",
        "metric",
    ]
    for keys, group in long.groupby(group_columns, sort=False, dropna=False):
        group = group.sort_values("window_start", kind="mergesort")
        initial = _window_lookup(group, initial_label)
        final = _window_lookup(group, final_label)
        initial_value = (
            float(initial["metric_value"]) if initial is not None else np.nan
        )
        final_value = float(final["metric_value"]) if final is not None else np.nan
        initial_z = (
            float(initial["standardized_value"]) if initial is not None else np.nan
        )
        final_z = float(final["standardized_value"]) if final is not None else np.nan
        initial_robust = (
            float(initial["robust_standardized_value"])
            if initial is not None
            else np.nan
        )
        final_robust = (
            float(final["robust_standardized_value"]) if final is not None else np.nan
        )
        values = group["metric_value"].to_numpy(dtype=float)
        z_values = group["standardized_value"].to_numpy(dtype=float)
        mid_years = group["window_mid_year"].to_numpy(dtype=float)
        window_order = group["window_index"].to_numpy(dtype=float)
        finite_values = np.isfinite(values)
        standardized_delta = final_z - initial_z
        rows.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "initial_window": initial_label,
                "final_window": final_label,
                "initial_value": initial_value,
                "final_value": final_value,
                "absolute_delta": final_value - initial_value,
                "standardized_delta": standardized_delta,
                "robust_standardized_delta": final_robust - initial_robust,
                "slope": linear_slope(mid_years, values),
                "standardized_slope": linear_slope(mid_years, z_values),
                "spearman_r": safe_spearman(window_order, values),
                "direction_label": direction_label(
                    standardized_delta,
                    stable_threshold=stable_threshold,
                ),
                "n_windows_available": int(np.count_nonzero(finite_values)),
            }
        )
    changes = pd.DataFrame(rows)
    return changes.sort_values(
        ["metric", "subfield_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def overall_change_ranking(changes: pd.DataFrame) -> pd.DataFrame:
    if changes.empty:
        return pd.DataFrame()
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
    metadata = changes[metadata_columns].drop_duplicates("subfield_id")
    delta_pivot = changes.pivot(
        index="subfield_id",
        columns="metric",
        values="standardized_delta",
    )
    slope_pivot = changes.pivot(
        index="subfield_id",
        columns="metric",
        values="standardized_slope",
    )
    row = pd.DataFrame(index=delta_pivot.index)
    row["overall_metric_change_l2"] = np.sqrt(np.nansum(delta_pivot.to_numpy() ** 2, axis=1))
    row["overall_metric_slope_l2"] = np.sqrt(np.nansum(slope_pivot.to_numpy() ** 2, axis=1))
    row["n_metrics_available_for_change"] = np.isfinite(delta_pivot.to_numpy()).sum(axis=1)
    row["n_metrics_available_for_slope"] = np.isfinite(slope_pivot.to_numpy()).sum(axis=1)
    ranking = metadata.merge(row.reset_index(), on="subfield_id", how="left")
    ranking = ranking.sort_values(
        ["overall_metric_change_l2", "subfield_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranking["overall_metric_change_rank"] = np.arange(1, len(ranking) + 1)
    return ranking[
        [
            *metadata_columns,
            "overall_metric_change_l2",
            "overall_metric_slope_l2",
            "overall_metric_change_rank",
            "n_metrics_available_for_change",
            "n_metrics_available_for_slope",
        ]
    ]


def aggregate_metric_trajectories(long: pd.DataFrame, *, level: str) -> pd.DataFrame:
    if long.empty:
        return pd.DataFrame()
    if level == "field":
        group_columns = ["field_id", "field_display_name"]
    elif level == "domain":
        group_columns = ["domain_id", "domain_display_name"]
    elif level == "overall":
        frame = long.copy()
        frame["overall_id"] = "all_subfields"
        frame["overall_display_name"] = "All subfields"
        group_columns = ["overall_id", "overall_display_name"]
        long = frame
    else:
        raise ValueError("level must be field, domain, or overall")

    aggregate_columns = [
        *group_columns,
        "window_start",
        "window_end",
        "window_index",
        "window_label",
        "window_mid_year",
        "metric",
    ]
    eligible = long.loc[long["metric_status"].isin(VALID_WINDOW_STATUSES)].copy()
    grouped = eligible.groupby(aggregate_columns, dropna=False, sort=True)
    result = grouped.agg(
        mean_value=("metric_value", "mean"),
        median_value=("metric_value", "median"),
        mean_standardized_value=("standardized_value", "mean"),
        median_standardized_value=("standardized_value", "median"),
        n_subfields=("subfield_id", "nunique"),
    ).reset_index()
    return result.sort_values(
        [*group_columns, "metric", "window_start"],
        kind="mergesort",
    ).reset_index(drop=True)


def top_changes_by_metric(changes: pd.DataFrame, *, top_n: int = 10) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if changes.empty:
        return pd.DataFrame()
    required_columns = [
        "metric",
        "rank_type",
        "rank",
        "subfield_id",
        "subfield_display_name",
        "field_display_name",
        "domain_display_name",
        "initial_window",
        "final_window",
        "initial_value",
        "final_value",
        "absolute_delta",
        "standardized_delta",
        "slope",
        "spearman_r",
        "n_windows_available",
    ]
    for metric, group in changes.groupby("metric", sort=True):
        finite = group.loc[group["standardized_delta"].notna()].copy()
        rank_specs = [
            (
                "highest_standardized_delta",
                ["standardized_delta", "subfield_id"],
                [False, True],
            ),
            (
                "lowest_standardized_delta",
                ["standardized_delta", "subfield_id"],
                [True, True],
            ),
        ]
        finite["abs_standardized_delta"] = finite["standardized_delta"].abs()
        for rank_type, sort_columns, ascending in rank_specs:
            selected = finite.sort_values(
                sort_columns,
                ascending=ascending,
                kind="mergesort",
            ).head(top_n).copy()
            selected["rank_type"] = rank_type
            selected["rank"] = np.arange(1, len(selected) + 1)
            rows.append(selected)
        selected = finite.sort_values(
            ["abs_standardized_delta", "subfield_id"],
            ascending=[False, True],
            kind="mergesort",
        ).head(top_n).copy()
        selected["rank_type"] = "largest_absolute_standardized_delta"
        selected["rank"] = np.arange(1, len(selected) + 1)
        rows.append(selected)
    if not rows:
        return pd.DataFrame(columns=required_columns)
    result = pd.concat(rows, ignore_index=True)
    return result[required_columns]


def readable_metric_label(metric: str, *, width: int = 17) -> str:
    label = METRIC_DISPLAY_NAMES.get(
        metric,
        metric.removeprefix("embedding_").replace("_", " "),
    )
    return "\n".join(wrap(label, width=width))


def _heatmap_values(changes: pd.DataFrame, value_column: str) -> pd.DataFrame:
    return changes.pivot(
        index="subfield_label_short",
        columns="metric",
        values=value_column,
    )[WINDOW_STRUCTURAL_METRICS]


def plot_temporal_change_heatmap(
    changes: pd.DataFrame,
    ranking: pd.DataFrame,
    output_path: str | Path,
    *,
    value_column: str,
    title: str,
    heatmap_top_n: int,
) -> None:
    if changes.empty or ranking.empty:
        write_placeholder_figure(output_path, "No temporal metric changes available")
        return
    sorted_subfields = ranking.sort_values(
        ["overall_metric_change_l2", "subfield_id"],
        ascending=[False, True],
        kind="mergesort",
    )["subfield_id"].astype(str)
    if heatmap_top_n > 0:
        sorted_subfields = sorted_subfields.head(heatmap_top_n)
    selected = changes.loc[changes["subfield_id"].astype(str).isin(set(sorted_subfields))]
    label_order = (
        ranking.loc[ranking["subfield_id"].astype(str).isin(set(sorted_subfields))]
        .sort_values(
            ["overall_metric_change_l2", "subfield_id"],
            ascending=[False, True],
            kind="mergesort",
        )["subfield_label_short"]
        .tolist()
    )
    pivot = _heatmap_values(selected, value_column).reindex(label_order)
    if pivot.empty:
        write_placeholder_figure(output_path, "No finite heatmap values available")
        return
    values = pivot.to_numpy(dtype=float)
    finite_abs = np.abs(values[np.isfinite(values)])
    vmax = float(np.nanpercentile(finite_abs, 98)) if finite_abs.size else 1.0
    vmax = max(vmax, 0.5)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height = max(6.0, min(20.0, 0.18 * len(pivot) + 2.8))
    fig, ax = plt.subplots(figsize=(11, height), dpi=240)
    image = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(
        [readable_metric_label(metric) for metric in pivot.columns],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.set_yticks(np.arange(len(pivot)))
    show_labels = len(pivot) <= 90
    ax.set_yticklabels(pivot.index if show_labels else [], fontsize=5 if len(pivot) > 60 else 7)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Subfield" if show_labels else f"Top {len(pivot)} subfields by overall change")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.015, label=value_column)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_top_changed_overall(
    ranking: pd.DataFrame,
    output_path: str | Path,
    *,
    top_n: int,
) -> None:
    if ranking.empty:
        write_placeholder_figure(output_path, "No overall temporal change ranking available")
        return
    top = ranking.head(top_n).iloc[::-1]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(top))), dpi=240)
    ax.barh(top["subfield_label_short"], top["overall_metric_change_l2"], color="#2f6f73")
    ax.set_xlabel("L2 norm of standardized metric deltas")
    ax.set_title("Subfields With Largest Overall Structural Change", fontsize=12)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_top_changed_by_metric(
    top_table: pd.DataFrame,
    output_path: str | Path,
    *,
    top_n_per_metric: int = 5,
) -> None:
    subset = top_table.loc[
        (top_table["rank_type"] == "largest_absolute_standardized_delta")
        & (top_table["rank"] <= top_n_per_metric)
    ].copy()
    if subset.empty:
        write_placeholder_figure(output_path, "No top metric changes available")
        return
    subset["label"] = [
        f"{readable_metric_label(metric, width=22).replace(chr(10), ' ')}: {name}"
        for metric, name in zip(subset["metric"], subset["subfield_display_name"], strict=False)
    ]
    subset = subset.sort_values(
        ["metric", "rank"],
        ascending=[True, False],
        kind="mergesort",
    )
    colors = np.where(subset["standardized_delta"].to_numpy(dtype=float) >= 0, "#b55a30", "#2f6f73")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, max(7, 0.24 * len(subset))), dpi=240)
    ax.barh(subset["label"], subset["standardized_delta"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Standardized delta, final window minus initial window")
    ax.set_title("Largest Absolute Standardized Temporal Changes By Metric", fontsize=12)
    ax.tick_params(axis="y", labelsize=6)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_domain_trajectories(
    trajectories: pd.DataFrame,
    output_path: str | Path,
    *,
    metrics: list[str] | None = None,
    title: str = "Domain-Average Standardized Structural Metric Trajectories",
    subtitle: str = TRAJECTORY_SCALING_NOTE,
    n_cols: int = 2,
) -> None:
    if trajectories.empty:
        write_placeholder_figure(output_path, "No domain average trajectories available")
        return
    requested_metrics = metrics or WINDOW_STRUCTURAL_METRICS
    metrics = [
        metric
        for metric in requested_metrics
        if metric in set(trajectories["metric"])
    ]
    if not metrics:
        write_placeholder_figure(output_path, "No requested domain trajectories available")
        return
    n_rows = int(np.ceil(len(metrics) / n_cols))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, max(7, 2.9 * n_rows)),
        dpi=240,
        squeeze=False,
    )
    for ax, metric in zip(axes.ravel(), metrics, strict=False):
        subset = trajectories.loc[trajectories["metric"] == metric]
        for domain, domain_group in subset.groupby("domain_display_name", sort=True):
            domain_group = domain_group.sort_values("window_start")
            ax.plot(
                domain_group["window_label"],
                domain_group["mean_standardized_value"],
                marker="o",
                linewidth=1.4,
                markersize=3,
                label=str(domain),
            )
        ax.axhline(0, color="#555555", linewidth=0.6, alpha=0.7)
        ax.set_title(readable_metric_label(metric, width=28), fontsize=9)
        ax.tick_params(axis="x", labelrotation=35, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_ylabel("Mean global z-score", fontsize=8)
    for ax in axes.ravel()[len(metrics) :]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=7)
    fig.suptitle(title, fontsize=13)
    if subtitle:
        fig.text(0.5, 0.945, subtitle, ha="center", va="top", fontsize=8)
        rect_top = 0.91
    else:
        rect_top = 0.96
    fig.tight_layout(rect=(0, 0.04, 1, rect_top))
    fig.savefig(output_path)
    plt.close(fig)


def plot_field_trajectory_heatmap(
    trajectories: pd.DataFrame,
    output_path: str | Path,
    *,
    top_n_fields: int = 30,
) -> None:
    if trajectories.empty:
        write_placeholder_figure(output_path, "No field average trajectories available")
        return
    first_label = trajectories["window_label"].sort_values(kind="mergesort").iloc[0]
    last_label = trajectories["window_label"].sort_values(kind="mergesort").iloc[-1]
    initial = trajectories.loc[trajectories["window_label"] == first_label]
    final = trajectories.loc[trajectories["window_label"] == last_label]
    merged = initial.merge(
        final,
        on=["field_id", "field_display_name", "metric"],
        suffixes=("_initial", "_final"),
    )
    merged["mean_z_delta"] = (
        merged["mean_standardized_value_final"]
        - merged["mean_standardized_value_initial"]
    )
    pivot = merged.pivot(
        index="field_display_name",
        columns="metric",
        values="mean_z_delta",
    )[WINDOW_STRUCTURAL_METRICS]
    if pivot.empty:
        write_placeholder_figure(output_path, "No field delta values available")
        return
    order = pivot.abs().mean(axis=1).sort_values(ascending=False).head(top_n_fields).index
    pivot = pivot.loc[order]
    values = pivot.to_numpy(dtype=float)
    vmax = max(float(np.nanpercentile(np.abs(values[np.isfinite(values)]), 98)), 0.5)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, max(6, 0.28 * len(pivot) + 2.5)), dpi=240)
    image = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(
        [readable_metric_label(metric) for metric in pivot.columns],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.set_yticks(np.arange(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_title("Field-Average Change In Standardized Structural Metrics", fontsize=12)
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.015, label="mean z-score delta")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def failed_or_skipped_window_rows(window_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "subfield_id",
        "subfield_display_name",
        "field_display_name",
        "domain_display_name",
        "window_label",
        "n_available",
        "n_used",
        "metric_status",
        "metric_error_message",
        "metric_warning_message",
    ]
    if window_metrics.empty:
        return pd.DataFrame(columns=columns)
    result = window_metrics.loc[
        ~window_metrics["metric_status"].isin(VALID_WINDOW_STATUSES),
        columns,
    ].copy()
    return result.sort_values(
        ["subfield_id", "window_label"],
        kind="mergesort",
    ).reset_index(drop=True)


def subfield_window_sample_size_diagnostics(window_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "subfield_id",
        "subfield_display_name",
        "field_display_name",
        "domain_display_name",
        "window",
        "n_available",
        "n_used",
        "metric_status",
        "metric_error_message",
        "metric_warning_message",
    ]
    if window_metrics.empty:
        return pd.DataFrame(columns=columns)
    result = window_metrics.copy()
    result["window"] = result["window_label"]
    return result[columns].sort_values(
        ["subfield_id", "window"],
        kind="mergesort",
    ).reset_index(drop=True)


def _quantile_or_nan(values: pd.Series, q: float) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.quantile(q))


def window_sample_size_diagnostics(window_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "window",
        "n_subfield_windows",
        "n_completed",
        "n_failed_or_skipped",
        "n_used_min",
        "n_used_p05",
        "n_used_median",
        "n_used_p95",
        "n_used_max",
        "n_available_min",
        "n_available_median",
        "n_available_max",
    ]
    if window_metrics.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for window_label, group in window_metrics.groupby("window_label", sort=False):
        completed = group.loc[group["metric_status"].isin(VALID_WINDOW_STATUSES)]
        rows.append(
            {
                "window": str(window_label),
                "n_subfield_windows": int(len(group)),
                "n_completed": int(len(completed)),
                "n_failed_or_skipped": int(len(group) - len(completed)),
                "n_used_min": int(completed["n_used"].min()) if len(completed) else np.nan,
                "n_used_p05": _quantile_or_nan(completed["n_used"], 0.05),
                "n_used_median": _quantile_or_nan(completed["n_used"], 0.50),
                "n_used_p95": _quantile_or_nan(completed["n_used"], 0.95),
                "n_used_max": int(completed["n_used"].max()) if len(completed) else np.nan,
                "n_available_min": int(group["n_available"].min()) if len(group) else np.nan,
                "n_available_median": _quantile_or_nan(group["n_available"], 0.50),
                "n_available_max": int(group["n_available"].max()) if len(group) else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def sample_size_balance_note(window_diagnostics: pd.DataFrame) -> str:
    if window_diagnostics.empty:
        return "No sample-size diagnostics were available."
    median_min = float(window_diagnostics["n_used_median"].min())
    median_max = float(window_diagnostics["n_used_median"].max())
    p05_min = float(window_diagnostics["n_used_p05"].min())
    p95_max = float(window_diagnostics["n_used_p95"].max())
    failed = int(window_diagnostics["n_failed_or_skipped"].sum())
    note = (
        "Completed subfield-window sample sizes are balanced: "
        f"window medians range from {median_min:.0f} to {median_max:.0f} papers, "
        f"with completed-row p05-p95 spanning {p05_min:.0f} to {p95_max:.0f}."
    )
    if failed:
        note += (
            f" There are {failed} failed/skipped subfield-window rows; inspect the "
            "per-subfield QA table before interpreting affected subfields."
        )
    return note


def interpretation_safety_note(
    *,
    expected_rows: int,
    completed_rows: int,
    failed_rows: pd.DataFrame,
) -> str:
    if expected_rows == 0:
        return "No expected rows were available, so the output is not interpretable."
    completion_share = completed_rows / expected_rows
    if failed_rows.empty:
        return (
            "All expected subfield-window rows completed; the Script 19 outputs "
            "are safe to interpret with the normal metric caveats."
        )
    if completion_share >= 0.99:
        affected = "; ".join(
            f"{row.subfield_id} {row.subfield_display_name} {row.window_label}"
            for row in failed_rows.itertuples(index=False)
        )
        return (
            "The outputs are safe to interpret overall because "
            f"{completion_share:.1%} of expected subfield-window rows completed. "
            "The affected subfield-window should be treated as unavailable in "
            f"subfield-level change interpretation: {affected}."
        )
    return (
        "Several subfield-window rows failed or were skipped, so interpretation "
        "should be restricted to completed rows and aggregates should be checked carefully."
    )


def build_qa_summary(
    *,
    window_metrics: pd.DataFrame,
    windows: list[TemporalWindow],
    window_diagnostics: pd.DataFrame,
    failed_rows: pd.DataFrame,
) -> dict[str, Any]:
    n_subfields = (
        int(window_metrics["subfield_id"].nunique())
        if "subfield_id" in window_metrics
        else 0
    )
    expected_rows = int(n_subfields * len(windows))
    completed_rows = int(window_metrics["metric_status"].isin(VALID_WINDOW_STATUSES).sum())
    return {
        "trajectory_scaling_method": (
            "metric-level z-scores across all subfield-window observations; "
            "domain and field trajectories average those standardized values."
        ),
        "expected_subfield_window_rows": expected_rows,
        "actual_subfield_window_rows": int(len(window_metrics)),
        "completed_subfield_window_rows": completed_rows,
        "failed_or_skipped_subfield_window_rows": int(len(failed_rows)),
        "failed_or_skipped_rows": dataframe_to_records(failed_rows, limit=20),
        "sample_size_balance_note": sample_size_balance_note(window_diagnostics),
        "interpretation_safety_note": interpretation_safety_note(
            expected_rows=expected_rows,
            completed_rows=completed_rows,
            failed_rows=failed_rows,
        ),
    }


def markdown_summary(
    *,
    input_paths: dict[str, str],
    output_paths_payload: dict[str, str],
    n_subfields: int,
    n_window_rows: int,
    n_completed_windows: int,
    windows: list[TemporalWindow],
    top_overall: pd.DataFrame,
    qa_summary: dict[str, Any],
    window_diagnostics: pd.DataFrame,
    failed_rows: pd.DataFrame,
) -> str:
    window_text = ", ".join(window.window_label for window in windows)
    lines = [
        "# Temporal Embedding Metric Evolution",
        "",
        "This analysis recomputes non-temporal embedding-space structural metrics "
        "inside fixed temporal windows, then summarizes how each subfield changes "
        "across the 2000-2024 period.",
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
            "- Temporal metrics are computed from SPECTER2 embeddings, not publication counts. Counts appear only as quality-control metadata.",
            f"- Windows: {window_text}. Five-year windows reduce annual noise and keep enough papers per subfield-window.",
            "- Metrics are the eight non-temporal structural metrics from the reduced interpretable embedding core.",
            "- Deltas compare the first and last window. Standardized deltas use z-scores across all subfield-window rows for each metric.",
            f"- Domain and field trajectory figures plot mean standardized values. {TRAJECTORY_SCALING_NOTE}",
            "- Slopes are linear trends across window mid-years. Spearman correlations use window order.",
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
            f"- Subfield-window rows: {n_window_rows}",
            f"- Completed subfield-window rows: {n_completed_windows}",
            "",
            "## QA Checks",
            "",
            f"- Scaling method in trajectory plots: {qa_summary['trajectory_scaling_method']}",
            f"- Expected subfield-window rows: {qa_summary['expected_subfield_window_rows']}",
            f"- Actual subfield-window rows: {qa_summary['actual_subfield_window_rows']}",
            f"- Completed subfield-window rows: {qa_summary['completed_subfield_window_rows']}",
            f"- Failed or skipped subfield-window rows: {qa_summary['failed_or_skipped_subfield_window_rows']}",
            f"- Sample-size balance: {qa_summary['sample_size_balance_note']}",
            f"- Interpretation status: {qa_summary['interpretation_safety_note']}",
            "",
            "Counts are used only as quality-control metadata. The analysis is based on semantic embedding-structure metrics, not publication-volume trends.",
            "",
            "### Failed Or Skipped Rows",
            "",
        ]
    )
    if failed_rows.empty:
        lines.append("No failed or skipped subfield-window rows were recorded.")
    else:
        for row in failed_rows.itertuples(index=False):
            warning = row.metric_error_message or row.metric_warning_message or ""
            lines.append(
                f"- `{row.subfield_id}` {row.subfield_display_name} "
                f"({row.field_display_name}, {row.domain_display_name}), "
                f"window `{row.window_label}`: n_available={row.n_available}, "
                f"n_used={row.n_used}, status={row.metric_status}, reason={warning}"
            )
    lines.extend(["", "### Window Sample-Size Diagnostics", ""])
    if window_diagnostics.empty:
        lines.append("No window sample-size diagnostics were available.")
    else:
        for row in window_diagnostics.itertuples(index=False):
            lines.append(
                f"- `{row.window}`: completed={row.n_completed}/"
                f"{row.n_subfield_windows}, failed_or_skipped={row.n_failed_or_skipped}, "
                f"n_used median={row.n_used_median:.0f}, "
                f"p05={row.n_used_p05:.0f}, p95={row.n_used_p95:.0f}"
            )
    lines.extend(
        [
            "",
            "## Most Changed Subfields",
            "",
        ]
    )
    if top_overall.empty:
        lines.append("No overall temporal change ranking was available.")
    else:
        for row in top_overall.head(10).itertuples(index=False):
            lines.append(
                f"- {row.overall_metric_change_rank}. "
                f"{row.subfield_label_short}: "
                f"{row.overall_metric_change_l2:.3f}"
            )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Subfields remain the main unit of interpretation; field and domain trajectories are broad summaries.",
            "- `n_used` and `n_available` are quality-control columns, not substantive publication-volume results.",
            "- Changes in these metrics describe internal semantic structure, not citation impact, publication growth, or prediction.",
            "",
        ]
    )
    return "\n".join(lines)


def write_temporal_metric_outputs(
    *,
    window_metrics: pd.DataFrame,
    changes: pd.DataFrame,
    ranking: pd.DataFrame,
    field_trajectories: pd.DataFrame,
    domain_trajectories: pd.DataFrame,
    overall_trajectories: pd.DataFrame,
    scaling_parameters: pd.DataFrame,
    output_dir: str | Path,
    root: Path,
    input_paths: dict[str, str],
    windows: list[TemporalWindow],
    heatmap_top_n: int = 80,
    top_n: int = 10,
    figure_top_n: int = 25,
) -> dict[str, Any]:
    paths = output_paths(output_dir, root=root)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(window_metrics, paths["window_metrics_parquet"])
    window_metrics.to_csv(paths["window_metrics_csv"], index=False)
    save_parquet(changes, paths["changes_parquet"])
    changes.to_csv(paths["changes_csv"], index=False)
    save_parquet(ranking, paths["overall_ranking_parquet"])
    ranking.to_csv(paths["overall_ranking_csv"], index=False)
    save_parquet(field_trajectories, paths["field_trajectories_parquet"])
    save_parquet(domain_trajectories, paths["domain_trajectories_parquet"])
    save_parquet(overall_trajectories, paths["overall_trajectories_parquet"])

    top_by_metric = top_changes_by_metric(changes, top_n=top_n)
    top_by_metric.to_csv(paths["top_changes_by_metric_csv"], index=False)
    ranking.head(max(figure_top_n, top_n)).to_csv(paths["top_overall_changed_csv"], index=False)
    scaling_parameters.to_csv(paths["scaling_parameters_csv"], index=False)
    window_diagnostics = window_sample_size_diagnostics(window_metrics)
    subfield_window_diagnostics = subfield_window_sample_size_diagnostics(window_metrics)
    failed_rows = failed_or_skipped_window_rows(window_metrics)
    window_diagnostics.to_csv(paths["window_sample_size_diagnostics_csv"], index=False)
    subfield_window_diagnostics.to_csv(
        paths["subfield_window_sample_size_diagnostics_csv"],
        index=False,
    )
    qa_summary = build_qa_summary(
        window_metrics=window_metrics,
        windows=windows,
        window_diagnostics=window_diagnostics,
        failed_rows=failed_rows,
    )

    plot_temporal_change_heatmap(
        changes,
        ranking,
        paths["delta_heatmap"],
        value_column="standardized_delta",
        title="Temporal Metric Change: Final Window Minus Initial Window",
        heatmap_top_n=heatmap_top_n,
    )
    plot_temporal_change_heatmap(
        changes,
        ranking,
        paths["slope_heatmap"],
        value_column="standardized_slope",
        title="Temporal Metric Slope Across Five-Year Windows",
        heatmap_top_n=heatmap_top_n,
    )
    plot_top_changed_overall(
        ranking,
        paths["top_changed_overall"],
        top_n=figure_top_n,
    )
    plot_top_changed_by_metric(top_by_metric, paths["top_changed_by_metric"])
    plot_domain_trajectories(domain_trajectories, paths["domain_trajectories"])
    plot_domain_trajectories(
        domain_trajectories,
        paths["domain_compactness_dimensionality_trajectories"],
        metrics=DOMAIN_COMPACTNESS_DIMENSIONALITY_METRICS,
        title="Domain-Average Standardized Compactness And Dimensionality Trajectories",
        subtitle=(
            "Semantic compactness, local distance, and intrinsic dimensionality. "
            + TRAJECTORY_SCALING_NOTE
        ),
    )
    plot_domain_trajectories(
        domain_trajectories,
        paths["domain_heterogeneity_hubness_trajectories"],
        metrics=DOMAIN_HETEROGENEITY_HUBNESS_METRICS,
        title="Domain-Average Standardized Heterogeneity And Hubness Trajectories",
        subtitle=(
            "Internal unevenness and hub concentration. "
            + TRAJECTORY_SCALING_NOTE
        ),
        n_cols=1,
    )
    plot_field_trajectory_heatmap(field_trajectories, paths["field_trajectories"])

    output_paths_payload = {
        key: display_path(path, root=root) for key, path in paths.items()
    }
    n_completed = int(window_metrics["metric_status"].isin(VALID_WINDOW_STATUSES).sum())
    summary = {
        "created_at": utc_now_iso(),
        "input_paths": input_paths,
        "output_paths": output_paths_payload,
        "windows": [
            {
                "window_start": window.window_start,
                "window_end": window.window_end,
                "window_label": window.window_label,
            }
            for window in windows
        ],
        "metrics": WINDOW_STRUCTURAL_METRICS,
        "n_subfields_attempted": int(window_metrics["subfield_id"].nunique())
        if "subfield_id" in window_metrics
        else 0,
        "n_window_rows": int(len(window_metrics)),
        "n_completed_window_rows": n_completed,
        "qa": qa_summary,
        "top_overall_changed_subfields": dataframe_to_records(ranking, limit=20),
        "top_changes_by_metric": dataframe_to_records(top_by_metric, limit=30),
    }
    write_json(paths["summary_json"], summary)
    write_text(
        paths["summary_md"],
        markdown_summary(
            input_paths=input_paths,
            output_paths_payload=output_paths_payload,
            n_subfields=summary["n_subfields_attempted"],
            n_window_rows=len(window_metrics),
            n_completed_windows=n_completed,
            windows=windows,
            top_overall=ranking,
            qa_summary=qa_summary,
            window_diagnostics=window_diagnostics,
            failed_rows=failed_rows,
        ),
    )
    return summary


def build_temporal_metric_evolution_outputs(
    analysis_index: pd.DataFrame,
    matrix: np.ndarray,
    *,
    year_min: int,
    year_max: int,
    window_size: int,
    min_papers_per_window: int,
    k_neighbors: int,
    subfield_id: str | None,
    limit_subfields: int | None,
    analysis_index_path: str,
    embedding_matrix_path: str,
    output_dir: str | Path,
    root: Path,
    heatmap_top_n: int = 80,
    top_n: int = 10,
    figure_top_n: int = 25,
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
    window_metrics = compute_subfield_window_metrics(
        analysis_index,
        matrix,
        windows=windows,
        attempted_subfields=attempted_subfields,
        year_min=year_min,
        year_max=year_max,
        min_papers_per_window=min_papers_per_window,
        k_neighbors=k_neighbors,
        analysis_index_path=analysis_index_path,
        embedding_matrix_path=embedding_matrix_path,
    )
    long = metric_long_frame(window_metrics)
    scaling = metric_scaling_parameters(long)
    changes = temporal_change_frame(long, windows=windows)
    ranking = overall_change_ranking(changes)
    field_trajectories = aggregate_metric_trajectories(long, level="field")
    domain_trajectories = aggregate_metric_trajectories(long, level="domain")
    overall_trajectories = aggregate_metric_trajectories(long, level="overall")
    return write_temporal_metric_outputs(
        window_metrics=window_metrics,
        changes=changes,
        ranking=ranking,
        field_trajectories=field_trajectories,
        domain_trajectories=domain_trajectories,
        overall_trajectories=overall_trajectories,
        scaling_parameters=scaling,
        output_dir=output_dir,
        root=root,
        input_paths={
            "analysis_index_path": analysis_index_path,
            "embedding_matrix_path": embedding_matrix_path,
        },
        windows=windows,
        heatmap_top_n=heatmap_top_n,
        top_n=top_n,
        figure_top_n=figure_top_n,
    )
