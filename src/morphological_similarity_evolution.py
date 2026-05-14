from __future__ import annotations

from pathlib import Path
from textwrap import wrap
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from src.storage import save_parquet
from src.temporal_common import (
    REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS,
    WINDOW_STRUCTURAL_METRICS,
    TemporalWindow,
    dataframe_to_records,
    default_or_computed_windows,
    display_path,
    ensure_outputs_do_not_exist,
    linear_slope,
    standardization_parameters,
    utc_now_iso,
    write_json,
    write_placeholder_figure,
    write_text,
)

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


PROCESSED_TEMPORAL_DIR = Path("data/processed/temporal")
PAIRWISE_OUTPUTS = {
    "static_pairs_parquet": PROCESSED_TEMPORAL_DIR
    / "morphological_pair_distances_static.parquet",
    "temporal_pairs_parquet": PROCESSED_TEMPORAL_DIR
    / "morphological_pair_distances_by_window.parquet",
    "temporal_changes_parquet": PROCESSED_TEMPORAL_DIR
    / "morphological_pair_distance_changes.parquet",
}

ANALYSIS_FILENAMES = {
    "top_converging_csv": "top_morphological_converging_pairs.csv",
    "top_diverging_csv": "top_morphological_diverging_pairs.csv",
    "scaling_parameters_csv": "scaling_parameters.csv",
    "summary_md": "summary.md",
    "summary_json": "summary.json",
}

STATIC_METRIC_SETS = {
    "static_full_reduced": REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS,
    "static_window_structural": WINDOW_STRUCTURAL_METRICS,
}
TEMPORAL_METRIC_SET = "temporal_window_structural"


def output_paths(output_dir: str | Path, *, root: Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    paths = {key: root / value for key, value in PAIRWISE_OUTPUTS.items()}
    paths.update({key: output_dir / value for key, value in ANALYSIS_FILENAMES.items()})
    paths["matrices_dir"] = output_dir / "matrices"
    return paths


def ensure_morphological_similarity_outputs(
    output_dir: str | Path,
    *,
    levels: list[str],
    overwrite: bool,
    root: Path,
) -> None:
    paths = output_paths(output_dir, root=root)
    figure_paths: list[Path] = []
    for level in levels:
        figure_paths.extend(
            [
                Path(output_dir) / f"{level}_static_morphological_distance_heatmap.png",
                Path(output_dir) / f"{level}_initial_morphological_distance_heatmap.png",
                Path(output_dir) / f"{level}_final_morphological_distance_heatmap.png",
                Path(output_dir) / f"{level}_morphological_distance_delta_heatmap.png",
                Path(output_dir) / f"{level}_top_morphological_converging_diverging_pairs.png",
            ]
        )
    ensure_outputs_do_not_exist(
        [
            paths["static_pairs_parquet"],
            paths["temporal_pairs_parquet"],
            paths["temporal_changes_parquet"],
            paths["top_converging_csv"],
            paths["top_diverging_csv"],
            paths["scaling_parameters_csv"],
            paths["summary_md"],
            paths["summary_json"],
            *figure_paths,
        ],
        overwrite=overwrite,
        root=root,
        description="morphological similarity outputs",
    )


def _entity_columns_for_level(level: str) -> tuple[str, str]:
    if level == "subfield":
        return "subfield_id", "subfield_display_name"
    if level == "field":
        return "field_id", "field_display_name"
    if level == "domain":
        return "domain_id", "domain_display_name"
    raise ValueError("level must be subfield, field, or domain")


def _valid_metric_rows(frame: pd.DataFrame, status_column: str) -> pd.DataFrame:
    if status_column not in frame.columns:
        return frame.copy()
    valid = {"completed", "completed_with_warnings"}
    return frame.loc[frame[status_column].isin(valid)].copy()


def robust_scale_metrics(
    frame: pd.DataFrame,
    *,
    metrics: list[str],
    metric_set: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [metric for metric in metrics if metric not in frame.columns]
    if missing:
        raise ValueError(
            f"Missing metrics for {metric_set}: {', '.join(missing)}"
        )
    scaled = frame.copy()
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        values = pd.to_numeric(frame[metric], errors="coerce")
        params = standardization_parameters(values)
        denominator = params["iqr"]
        fallback = ""
        if not np.isfinite(denominator) or denominator <= 1e-12:
            denominator = params["std"]
            fallback = "std"
        if not np.isfinite(denominator) or denominator <= 1e-12:
            scaled[metric] = np.nan
            denominator = np.nan
            fallback = "unavailable"
        else:
            scaled[metric] = (values - params["median"]) / denominator
        rows.append(
            {
                "metric_set": metric_set,
                "metric": metric,
                "scaler": "robust",
                "center": params["median"],
                "scale": denominator,
                "fallback_scale": fallback,
                **params,
            }
        )
    return scaled, pd.DataFrame(rows)


def profile_frame_for_level(
    scaled: pd.DataFrame,
    *,
    metrics: list[str],
    level: str,
    metric_set: str,
    temporal: bool,
) -> pd.DataFrame:
    id_column, name_column = _entity_columns_for_level(level)
    required = {id_column, name_column, *metrics}
    if temporal:
        required.update(
            {"window_label", "window_start", "window_end", "window_index", "window_mid_year"}
        )
    missing = required - set(scaled.columns)
    if missing:
        raise ValueError(f"Missing columns for {level} profiles: {', '.join(sorted(missing))}")
    group_columns = [id_column, name_column]
    if temporal:
        group_columns.extend(
            ["window_label", "window_start", "window_end", "window_index", "window_mid_year"]
        )
    rows: list[pd.DataFrame] = []
    aggregate = scaled.groupby(group_columns, sort=True, dropna=False)[metrics].mean().reset_index()
    aggregate.insert(0, "metric_set", metric_set)
    aggregate.insert(1, "level", level)
    aggregate["entity_id"] = aggregate[id_column].astype(str)
    aggregate["entity_display_name"] = aggregate[name_column].astype(str)
    if "subfield_id" in scaled.columns:
        counts = scaled.groupby(group_columns, sort=True, dropna=False)["subfield_id"].nunique().reset_index(
            name="n_source_subfields"
        )
        aggregate = aggregate.merge(counts, on=group_columns, how="left")
    else:
        aggregate["n_source_subfields"] = 1
    rows.append(aggregate)
    return pd.concat(rows, ignore_index=True)


def profile_distance(first: np.ndarray, second: np.ndarray, *, distance_metric: str) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    finite = np.isfinite(first) & np.isfinite(second)
    if distance_metric == "euclidean":
        if np.count_nonzero(finite) < 1:
            return np.nan
        return float(np.linalg.norm(first[finite] - second[finite]))
    if distance_metric == "correlation":
        if np.count_nonzero(finite) < 2:
            return np.nan
        left = first[finite]
        right = second[finite]
        left_centered = left - left.mean()
        right_centered = right - right.mean()
        denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
        if denominator <= 1e-12:
            return 0.0 if np.allclose(left, right) else np.nan
        correlation = float(np.dot(left_centered, right_centered) / denominator)
        return float(1.0 - np.clip(correlation, -1.0, 1.0))
    raise ValueError("distance_metric must be euclidean or correlation")


def pairwise_profile_distances(
    profiles: pd.DataFrame,
    *,
    metrics: list[str],
    distance_metrics: list[str],
    temporal: bool,
) -> pd.DataFrame:
    if profiles.empty:
        return pd.DataFrame()
    group_columns = ["metric_set", "level"]
    if temporal:
        group_columns.append("window_label")
    rows: list[dict[str, Any]] = []
    for keys, group in profiles.groupby(group_columns, sort=True, dropna=False):
        group = group.sort_values("entity_display_name", kind="mergesort").reset_index(drop=True)
        vectors = group[metrics].to_numpy(dtype=float)
        key_dict = dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,), strict=True))
        for left_idx in range(len(group)):
            for right_idx in range(left_idx + 1, len(group)):
                left = group.iloc[left_idx]
                right = group.iloc[right_idx]
                base = {
                    **key_dict,
                    "entity_id_a": str(left["entity_id"]),
                    "entity_display_name_a": str(left["entity_display_name"]),
                    "entity_id_b": str(right["entity_id"]),
                    "entity_display_name_b": str(right["entity_display_name"]),
                    "pair_id": f"{left['entity_id']}__{right['entity_id']}",
                    "n_metrics": len(metrics),
                }
                if temporal:
                    base.update(
                        {
                            "window_start": int(left["window_start"]),
                            "window_end": int(left["window_end"]),
                            "window_index": int(left["window_index"]),
                            "window_mid_year": float(left["window_mid_year"]),
                        }
                    )
                for distance_metric in distance_metrics:
                    rows.append(
                        {
                            **base,
                            "distance_metric": distance_metric,
                            "morphological_distance": profile_distance(
                                vectors[left_idx],
                                vectors[right_idx],
                                distance_metric=distance_metric,
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def temporal_distance_changes(
    pair_distances: pd.DataFrame,
    *,
    windows: list[TemporalWindow],
) -> pd.DataFrame:
    if pair_distances.empty:
        return pd.DataFrame()
    first_label = windows[0].window_label
    last_label = windows[-1].window_label
    group_columns = [
        "metric_set",
        "level",
        "distance_metric",
        "entity_id_a",
        "entity_display_name_a",
        "entity_id_b",
        "entity_display_name_b",
        "pair_id",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in pair_distances.groupby(group_columns, sort=True, dropna=False):
        group = group.sort_values("window_start", kind="mergesort")
        initial = group.loc[group["window_label"] == first_label]
        final = group.loc[group["window_label"] == last_label]
        initial_distance = (
            float(initial["morphological_distance"].iloc[0]) if len(initial) else np.nan
        )
        final_distance = (
            float(final["morphological_distance"].iloc[0]) if len(final) else np.nan
        )
        values = group["morphological_distance"].to_numpy(dtype=float)
        rows.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "initial_window": first_label,
                "final_window": last_label,
                "initial_morphological_distance": initial_distance,
                "final_morphological_distance": final_distance,
                "delta_morphological_distance": final_distance - initial_distance,
                "morphological_distance_slope": linear_slope(
                    group["window_mid_year"].to_numpy(dtype=float),
                    values,
                ),
                "n_windows_available": int(np.isfinite(values).sum()),
            }
        )
    return pd.DataFrame(rows)


def matrix_from_pairs(frame: pd.DataFrame, *, value_column: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    labels = sorted(
        set(frame["entity_display_name_a"].astype(str)).union(
            set(frame["entity_display_name_b"].astype(str))
        )
    )
    matrix = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)
    for label in labels:
        matrix.loc[label, label] = 0.0
    for row in frame.itertuples(index=False):
        matrix.loc[str(row.entity_display_name_a), str(row.entity_display_name_b)] = getattr(row, value_column)
        matrix.loc[str(row.entity_display_name_b), str(row.entity_display_name_a)] = getattr(row, value_column)
    return matrix


def top_morphological_pairs(
    changes: pd.DataFrame,
    *,
    top_n: int,
    direction: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for keys, group in changes.groupby(["level", "distance_metric"], sort=True, dropna=False):
        finite = group.loc[group["delta_morphological_distance"].notna()].copy()
        if direction == "converging":
            selected = finite.sort_values(
                ["delta_morphological_distance", "pair_id"],
                ascending=[True, True],
                kind="mergesort",
            ).head(top_n)
        elif direction == "diverging":
            selected = finite.sort_values(
                ["delta_morphological_distance", "pair_id"],
                ascending=[False, True],
                kind="mergesort",
            ).head(top_n)
        else:
            raise ValueError("direction must be converging or diverging")
        selected = selected.copy()
        selected["direction"] = direction
        selected["rank"] = np.arange(1, len(selected) + 1)
        frames.append(selected)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def save_morphological_matrices(
    static_pairs: pd.DataFrame,
    temporal_pairs: pd.DataFrame,
    changes: pd.DataFrame,
    *,
    output_dir: str | Path,
    levels: list[str],
    distance_metrics: list[str],
    default_distance: str,
) -> dict[str, str]:
    matrices_dir = Path(output_dir) / "matrices"
    matrices_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for level in levels:
        for distance_metric in distance_metrics:
            static_subset = static_pairs.loc[
                (static_pairs["level"] == level)
                & (static_pairs["distance_metric"] == distance_metric)
                & (static_pairs["metric_set"] == "static_full_reduced")
            ]
            if not static_subset.empty:
                matrix = matrix_from_pairs(static_subset, value_column="morphological_distance")
                explicit = matrices_dir / f"{level}_{distance_metric}_static_morphological_distance_matrix.csv"
                matrix.to_csv(explicit)
                written[f"{level}_{distance_metric}_static_matrix"] = str(explicit)
                if distance_metric == default_distance:
                    compat = matrices_dir / f"{level}_static_morphological_distance_matrix.csv"
                    matrix.to_csv(compat)
                    written[f"{level}_static_matrix"] = str(compat)

            temporal_subset = temporal_pairs.loc[
                (temporal_pairs["level"] == level)
                & (temporal_pairs["distance_metric"] == distance_metric)
                & (temporal_pairs["metric_set"] == TEMPORAL_METRIC_SET)
            ]
            for window_label, group in temporal_subset.groupby("window_label", sort=True):
                matrix = matrix_from_pairs(group, value_column="morphological_distance")
                explicit = matrices_dir / f"{level}_{distance_metric}_{window_label}_morphological_distance_matrix.csv"
                matrix.to_csv(explicit)
                written[f"{level}_{distance_metric}_{window_label}_matrix"] = str(explicit)
                if distance_metric == default_distance:
                    compat = matrices_dir / f"{level}_{window_label}_morphological_distance_matrix.csv"
                    matrix.to_csv(compat)
                    written[f"{level}_{window_label}_matrix"] = str(compat)

            change_subset = changes.loc[
                (changes["level"] == level)
                & (changes["distance_metric"] == distance_metric)
                & (changes["metric_set"] == TEMPORAL_METRIC_SET)
            ]
            if not change_subset.empty:
                delta = matrix_from_pairs(change_subset, value_column="delta_morphological_distance")
                slope = matrix_from_pairs(change_subset, value_column="morphological_distance_slope")
                explicit_delta = matrices_dir / f"{level}_{distance_metric}_morphological_distance_delta_matrix.csv"
                explicit_slope = matrices_dir / f"{level}_{distance_metric}_morphological_distance_slope_matrix.csv"
                delta.to_csv(explicit_delta)
                slope.to_csv(explicit_slope)
                written[f"{level}_{distance_metric}_delta_matrix"] = str(explicit_delta)
                written[f"{level}_{distance_metric}_slope_matrix"] = str(explicit_slope)
                if distance_metric == default_distance:
                    compat_delta = matrices_dir / f"{level}_morphological_distance_delta_matrix.csv"
                    compat_slope = matrices_dir / f"{level}_morphological_distance_slope_matrix.csv"
                    delta.to_csv(compat_delta)
                    slope.to_csv(compat_slope)
                    written[f"{level}_delta_matrix"] = str(compat_delta)
                    written[f"{level}_slope_matrix"] = str(compat_slope)
    return written


def _subset_matrix_for_plot(matrix: pd.DataFrame, *, max_labels: int = 70) -> pd.DataFrame:
    if len(matrix) <= max_labels:
        return matrix
    variability = matrix.abs().sum(axis=1, skipna=True).sort_values(ascending=False)
    labels = variability.head(max_labels).index
    return matrix.loc[labels, labels]


def plot_heatmap(matrix: pd.DataFrame, path: str | Path, *, title: str) -> None:
    if matrix.empty:
        write_placeholder_figure(path, title + "\nNo matrix values available")
        return
    plot_matrix = _subset_matrix_for_plot(matrix)
    values = plot_matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        write_placeholder_figure(path, title + "\nNo finite matrix values available")
        return
    if "delta" in title.lower():
        vmax = max(float(np.nanpercentile(np.abs(finite), 98)), 0.01)
        vmin = -vmax
        cmap = "coolwarm"
    else:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanpercentile(finite, 98))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        cmap = "viridis"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=230)
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    show_labels = len(plot_matrix) <= 45
    ax.set_xticks(np.arange(len(plot_matrix)))
    ax.set_yticks(np.arange(len(plot_matrix)))
    ax.set_xticklabels(plot_matrix.columns if show_labels else [], rotation=90, fontsize=5)
    ax.set_yticklabels(plot_matrix.index if show_labels else [], fontsize=5)
    ax.set_title(title, fontsize=12)
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_top_pairs(changes: pd.DataFrame, path: str | Path, *, level: str, distance_metric: str, top_n: int) -> None:
    subset = changes.loc[
        (changes["level"] == level)
        & (changes["distance_metric"] == distance_metric)
        & changes["delta_morphological_distance"].notna()
    ].copy()
    if subset.empty:
        write_placeholder_figure(path, f"No morphological changes available for {level}")
        return
    converging = subset.sort_values("delta_morphological_distance", kind="mergesort").head(top_n)
    diverging = subset.sort_values(
        "delta_morphological_distance",
        ascending=False,
        kind="mergesort",
    ).head(top_n)
    plot_frame = pd.concat([converging, diverging], ignore_index=True)
    plot_frame["label"] = [
        "\n".join(wrap(f"{a} / {b}", width=44))
        for a, b in zip(
            plot_frame["entity_display_name_a"],
            plot_frame["entity_display_name_b"],
            strict=False,
        )
    ]
    plot_frame = plot_frame.iloc[::-1]
    colors = np.where(plot_frame["delta_morphological_distance"].to_numpy(dtype=float) < 0, "#2f6f73", "#b45432")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, max(6, 0.42 * len(plot_frame))), dpi=230)
    ax.barh(plot_frame["label"], plot_frame["delta_morphological_distance"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title(f"Top Morphological Convergence And Divergence Pairs ({level})", fontsize=12)
    ax.set_xlabel("Delta morphological distance, final window minus initial window")
    ax.tick_params(axis="y", labelsize=6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_profiles_and_scaling(
    *,
    static_metrics: pd.DataFrame,
    temporal_metrics: pd.DataFrame,
    levels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    static_profiles: list[pd.DataFrame] = []
    temporal_profiles: list[pd.DataFrame] = []
    scaling_rows: list[pd.DataFrame] = []

    static_valid = _valid_metric_rows(static_metrics, "metric_status")
    for metric_set, metrics in STATIC_METRIC_SETS.items():
        available = [metric for metric in metrics if metric in static_valid.columns]
        if not available:
            continue
        scaled, params = robust_scale_metrics(
            static_valid,
            metrics=available,
            metric_set=metric_set,
        )
        params.insert(0, "analysis_scope", "static")
        scaling_rows.append(params)
        for level in levels:
            static_profiles.append(
                profile_frame_for_level(
                    scaled,
                    metrics=available,
                    level=level,
                    metric_set=metric_set,
                    temporal=False,
                )
            )

    temporal_valid = _valid_metric_rows(temporal_metrics, "metric_status")
    available_temporal = [
        metric for metric in WINDOW_STRUCTURAL_METRICS if metric in temporal_valid.columns
    ]
    if available_temporal:
        scaled_temporal, params = robust_scale_metrics(
            temporal_valid,
            metrics=available_temporal,
            metric_set=TEMPORAL_METRIC_SET,
        )
        params.insert(0, "analysis_scope", "temporal")
        scaling_rows.append(params)
        for level in levels:
            temporal_profiles.append(
                profile_frame_for_level(
                    scaled_temporal,
                    metrics=available_temporal,
                    level=level,
                    metric_set=TEMPORAL_METRIC_SET,
                    temporal=True,
                )
            )
    return (
        pd.concat(static_profiles, ignore_index=True) if static_profiles else pd.DataFrame(),
        pd.concat(temporal_profiles, ignore_index=True) if temporal_profiles else pd.DataFrame(),
        pd.concat(scaling_rows, ignore_index=True) if scaling_rows else pd.DataFrame(),
    )


def write_morphological_outputs(
    *,
    static_pairs: pd.DataFrame,
    temporal_pairs: pd.DataFrame,
    changes: pd.DataFrame,
    scaling_parameters: pd.DataFrame,
    output_dir: str | Path,
    root: Path,
    levels: list[str],
    distance_metrics: list[str],
    windows: list[TemporalWindow],
    top_n_pairs: int,
    input_paths: dict[str, str],
) -> dict[str, Any]:
    paths = output_paths(output_dir, root=root)
    for path in paths.values():
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(static_pairs, paths["static_pairs_parquet"])
    save_parquet(temporal_pairs, paths["temporal_pairs_parquet"])
    save_parquet(changes, paths["temporal_changes_parquet"])
    scaling_parameters.to_csv(paths["scaling_parameters_csv"], index=False)
    top_converging = top_morphological_pairs(
        changes,
        top_n=top_n_pairs,
        direction="converging",
    )
    top_diverging = top_morphological_pairs(
        changes,
        top_n=top_n_pairs,
        direction="diverging",
    )
    top_converging.to_csv(paths["top_converging_csv"], index=False)
    top_diverging.to_csv(paths["top_diverging_csv"], index=False)

    default_distance = distance_metrics[0]
    matrix_paths = save_morphological_matrices(
        static_pairs,
        temporal_pairs,
        changes,
        output_dir=output_dir,
        levels=levels,
        distance_metrics=distance_metrics,
        default_distance=default_distance,
    )
    first_label = windows[0].window_label
    last_label = windows[-1].window_label
    for level in levels:
        static_subset = static_pairs.loc[
            (static_pairs["level"] == level)
            & (static_pairs["distance_metric"] == default_distance)
            & (static_pairs["metric_set"] == "static_full_reduced")
        ]
        temporal_subset = temporal_pairs.loc[
            (temporal_pairs["level"] == level)
            & (temporal_pairs["distance_metric"] == default_distance)
            & (temporal_pairs["metric_set"] == TEMPORAL_METRIC_SET)
        ]
        change_subset = changes.loc[
            (changes["level"] == level)
            & (changes["distance_metric"] == default_distance)
            & (changes["metric_set"] == TEMPORAL_METRIC_SET)
        ]
        plot_heatmap(
            matrix_from_pairs(static_subset, value_column="morphological_distance"),
            Path(output_dir) / f"{level}_static_morphological_distance_heatmap.png",
            title=f"{level.title()} Static Morphological Distance",
        )
        plot_heatmap(
            matrix_from_pairs(
                temporal_subset.loc[temporal_subset["window_label"] == first_label],
                value_column="morphological_distance",
            ),
            Path(output_dir) / f"{level}_initial_morphological_distance_heatmap.png",
            title=f"{level.title()} Morphological Distance: {first_label}",
        )
        plot_heatmap(
            matrix_from_pairs(
                temporal_subset.loc[temporal_subset["window_label"] == last_label],
                value_column="morphological_distance",
            ),
            Path(output_dir) / f"{level}_final_morphological_distance_heatmap.png",
            title=f"{level.title()} Morphological Distance: {last_label}",
        )
        plot_heatmap(
            matrix_from_pairs(change_subset, value_column="delta_morphological_distance"),
            Path(output_dir) / f"{level}_morphological_distance_delta_heatmap.png",
            title=f"{level.title()} Morphological Distance Delta",
        )
        plot_top_pairs(
            changes,
            Path(output_dir) / f"{level}_top_morphological_converging_diverging_pairs.png",
            level=level,
            distance_metric=default_distance,
            top_n=min(top_n_pairs, 20),
        )

    output_paths_payload = {key: display_path(path, root=root) for key, path in paths.items()}
    output_paths_payload.update(
        {key: display_path(path, root=root) for key, path in matrix_paths.items()}
    )
    summary = {
        "created_at": utc_now_iso(),
        "input_paths": input_paths,
        "output_paths": output_paths_payload,
        "levels": levels,
        "distance_metrics": distance_metrics,
        "default_distance_for_figures": default_distance,
        "n_static_pair_rows": int(len(static_pairs)),
        "n_temporal_pair_rows": int(len(temporal_pairs)),
        "n_temporal_change_rows": int(len(changes)),
        "top_converging_pairs": dataframe_to_records(top_converging, limit=20),
        "top_diverging_pairs": dataframe_to_records(top_diverging, limit=20),
        "method_notes": [
            "Morphological distance compares robust-scaled structural metric profiles.",
            "It is distinct from semantic/content distance between centroid positions.",
        ],
    }
    write_json(paths["summary_json"], summary)
    write_text(paths["summary_md"], markdown_summary(summary))
    return summary


def markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Morphological Similarity Evolution",
        "",
        "This analysis compares disciplines by their internal semantic structure rather than by topic position.",
        "",
        "## Methods",
        "",
        "- Morphological distance is computed between robust-scaled metric profiles.",
        "- Temporal morphology uses the eight non-temporal structural metrics recomputed per five-year window.",
        "- Static morphology uses the 11-metric reduced interpretable embedding core when available, plus an 8-metric structural static sensitivity set.",
        "- Negative final-minus-initial distance means morphological convergence. Positive means morphological divergence.",
        "- This differs from script 22: semantic distance compares centroid positions; morphological distance compares structural profiles.",
        "",
        "## Inputs",
        "",
    ]
    for label, path in summary["input_paths"].items():
        lines.append(f"- {label}: `{path}`")
    lines.extend(["", "## Outputs", ""])
    for label, path in summary["output_paths"].items():
        lines.append(f"- {label}: `{path}`")
    lines.extend(
        [
            "",
            "## Run Summary",
            "",
            f"- Levels: {', '.join(summary['levels'])}",
            f"- Distance metrics: {', '.join(summary['distance_metrics'])}",
            f"- Static pair rows: {summary['n_static_pair_rows']}",
            f"- Temporal pair rows: {summary['n_temporal_pair_rows']}",
            f"- Temporal change rows: {summary['n_temporal_change_rows']}",
            "",
            "## Caveats",
            "",
            "- Field and domain morphology profiles are averages of robust-scaled subfield profiles.",
            "- Full subfield matrices are saved, but top-pair plots are usually more readable for thesis interpretation.",
            "",
        ]
    )
    return "\n".join(lines)


def build_morphological_similarity_outputs(
    static_metrics: pd.DataFrame,
    temporal_metrics: pd.DataFrame,
    *,
    levels: list[str],
    distance_metrics: list[str],
    output_dir: str | Path,
    root: Path,
    top_n_pairs: int,
    input_paths: dict[str, str],
    year_min: int = 2000,
    year_max: int = 2024,
    window_size: int = 5,
) -> dict[str, Any]:
    windows = default_or_computed_windows(
        year_min=year_min,
        year_max=year_max,
        window_size=window_size,
    )
    static_profiles, temporal_profiles, scaling = build_profiles_and_scaling(
        static_metrics=static_metrics,
        temporal_metrics=temporal_metrics,
        levels=levels,
    )
    static_pairs: list[pd.DataFrame] = []
    for metric_set, metrics in STATIC_METRIC_SETS.items():
        available = [metric for metric in metrics if metric in static_profiles.columns]
        if not available:
            continue
        profiles = static_profiles.loc[static_profiles["metric_set"] == metric_set]
        static_pairs.append(
            pairwise_profile_distances(
                profiles,
                metrics=available,
                distance_metrics=distance_metrics,
                temporal=False,
            )
        )
    temporal_available = [
        metric for metric in WINDOW_STRUCTURAL_METRICS if metric in temporal_profiles.columns
    ]
    temporal_pairs = (
        pairwise_profile_distances(
            temporal_profiles,
            metrics=temporal_available,
            distance_metrics=distance_metrics,
            temporal=True,
        )
        if temporal_available
        else pd.DataFrame()
    )
    static_pair_frame = (
        pd.concat(static_pairs, ignore_index=True) if static_pairs else pd.DataFrame()
    )
    changes = temporal_distance_changes(temporal_pairs, windows=windows)
    return write_morphological_outputs(
        static_pairs=static_pair_frame,
        temporal_pairs=temporal_pairs,
        changes=changes,
        scaling_parameters=scaling,
        output_dir=output_dir,
        root=root,
        levels=levels,
        distance_metrics=distance_metrics,
        windows=windows,
        top_n_pairs=top_n_pairs,
        input_paths=input_paths,
    )
