from __future__ import annotations

from pathlib import Path
from textwrap import shorten, wrap
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
    "entity_retention_diagnostics_csv": "entity_retention_diagnostics.csv",
    "dropped_entities_csv": "dropped_entities.csv",
    "top_pair_entity_frequency_csv": "top_pair_entity_frequency.csv",
    "top_pair_driver_diagnostics_csv": "top_pair_driver_diagnostics.csv",
    "summary_md": "summary.md",
    "summary_json": "summary.json",
}

STATIC_METRIC_SETS = {
    "static_full_reduced": REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS,
    "static_window_structural": WINDOW_STRUCTURAL_METRICS,
}
TEMPORAL_METRIC_SET = "temporal_window_structural"
DEFAULT_OVERALL_CHANGE_RANKING = (
    PROCESSED_TEMPORAL_DIR / "subfield_overall_temporal_change_ranking.parquet"
)
DEFAULT_METRIC_TEMPORAL_CHANGES = (
    PROCESSED_TEMPORAL_DIR / "subfield_metric_temporal_changes.parquet"
)


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
        for distance_metric in ["euclidean", "correlation"]:
            figure_paths.extend(
                [
                    Path(output_dir) / f"{level}_{distance_metric}_top_converging_diverging_pairs.png",
                    Path(output_dir) / f"{level}_{distance_metric}_static_closest_farthest_pairs.png",
                ]
            )
            if level in {"domain", "field"}:
                figure_paths.extend(
                    [
                        Path(output_dir) / f"{level}_{distance_metric}_delta_heatmap.png",
                        Path(output_dir) / f"{level}_{distance_metric}_initial_final_delta_triptych.png",
                    ]
                )
    figure_paths.extend(
        [
            Path(output_dir) / "domain_euclidean_vs_correlation_delta_heatmaps.png",
            Path(output_dir) / "subfield_top_pair_entity_frequency.png",
            Path(output_dir) / "field_top_pair_entity_frequency.png",
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
            paths["entity_retention_diagnostics_csv"],
            paths["dropped_entities_csv"],
            paths["top_pair_entity_frequency_csv"],
            paths["top_pair_driver_diagnostics_csv"],
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
    metadata_candidates = [
        "field_display_name",
        "domain_display_name",
        "field_id",
        "domain_id",
    ]
    metadata_columns = [
        column
        for column in metadata_candidates
        if column in scaled.columns and column not in group_columns
    ]
    if metadata_columns:
        metadata = (
            scaled.groupby(group_columns, sort=True, dropna=False)[metadata_columns]
            .first()
            .reset_index()
        )
        aggregate = aggregate.merge(metadata, on=group_columns, how="left")
    aggregate.insert(0, "metric_set", metric_set)
    aggregate.insert(1, "level", level)
    aggregate["entity_id"] = aggregate[id_column].astype(str)
    aggregate["entity_display_name"] = aggregate[name_column].astype(str)
    if level == "subfield":
        aggregate["entity_field"] = aggregate.get("field_display_name", pd.Series("", index=aggregate.index)).fillna("").astype(str)
        aggregate["entity_domain"] = aggregate.get("domain_display_name", pd.Series("", index=aggregate.index)).fillna("").astype(str)
    elif level == "field":
        aggregate["entity_field"] = aggregate["entity_display_name"]
        aggregate["entity_domain"] = aggregate.get("domain_display_name", pd.Series("", index=aggregate.index)).fillna("").astype(str)
    else:
        aggregate["entity_field"] = ""
        aggregate["entity_domain"] = aggregate["entity_display_name"]
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

    def profile_value(row: pd.Series, column: str) -> str:
        return str(row[column]) if column in row and pd.notna(row[column]) else ""

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
                    "entity_field_a": profile_value(left, "entity_field"),
                    "entity_domain_a": profile_value(left, "entity_domain"),
                    "entity_id_b": str(right["entity_id"]),
                    "entity_display_name_b": str(right["entity_display_name"]),
                    "entity_field_b": profile_value(right, "entity_field"),
                    "entity_domain_b": profile_value(right, "entity_domain"),
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
        "entity_field_a",
        "entity_domain_a",
        "entity_id_b",
        "entity_display_name_b",
        "entity_field_b",
        "entity_domain_b",
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
    if frame.empty or value_column not in frame:
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
            selected = finite.loc[finite["delta_morphological_distance"] < 0].sort_values(
                ["delta_morphological_distance", "pair_id"],
                ascending=[True, True],
                kind="mergesort",
            ).head(top_n)
        elif direction == "diverging":
            selected = finite.loc[finite["delta_morphological_distance"] > 0].sort_values(
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


def _distance_type_label(distance_metric: str) -> str:
    return "Euclidean" if distance_metric == "euclidean" else "Correlation"


def format_temporal_pair_table(pairs: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "level",
        "distance_type",
        "entity_a_id",
        "entity_a_name",
        "entity_a_field",
        "entity_a_domain",
        "entity_b_id",
        "entity_b_name",
        "entity_b_field",
        "entity_b_domain",
        "initial_distance",
        "final_distance",
        "delta_distance",
        "slope",
        "rank",
    ]
    optional = [
        "direction",
        "metric_set",
        "pair_id",
        "n_windows_available",
        "entity_a_overall_metric_change_l2",
        "entity_b_overall_metric_change_l2",
        "entity_a_top_changed_metrics",
        "entity_b_top_changed_metrics",
    ]
    if pairs.empty:
        return pd.DataFrame(columns=[*columns, *optional])
    renamed = pairs.rename(
        columns={
            "distance_metric": "distance_type",
            "entity_id_a": "entity_a_id",
            "entity_display_name_a": "entity_a_name",
            "entity_field_a": "entity_a_field",
            "entity_domain_a": "entity_a_domain",
            "entity_id_b": "entity_b_id",
            "entity_display_name_b": "entity_b_name",
            "entity_field_b": "entity_b_field",
            "entity_domain_b": "entity_b_domain",
            "initial_morphological_distance": "initial_distance",
            "final_morphological_distance": "final_distance",
            "delta_morphological_distance": "delta_distance",
            "morphological_distance_slope": "slope",
        }
    ).copy()
    for column in [*columns, *optional]:
        if column not in renamed:
            renamed[column] = np.nan if column not in {"direction", "metric_set", "pair_id"} else ""
    return renamed[[*columns, *[column for column in optional if column in renamed]]]


def format_static_pair_table(pairs: pd.DataFrame, *, rank_type: str) -> pd.DataFrame:
    columns = [
        "level",
        "distance_type",
        "entity_a_id",
        "entity_a_name",
        "entity_a_field",
        "entity_a_domain",
        "entity_b_id",
        "entity_b_name",
        "entity_b_field",
        "entity_b_domain",
        "morphological_distance",
        "rank",
        "rank_type",
        "metric_set",
        "pair_id",
    ]
    if pairs.empty:
        return pd.DataFrame(columns=columns)
    renamed = pairs.rename(
        columns={
            "distance_metric": "distance_type",
            "entity_id_a": "entity_a_id",
            "entity_display_name_a": "entity_a_name",
            "entity_field_a": "entity_a_field",
            "entity_domain_a": "entity_a_domain",
            "entity_id_b": "entity_b_id",
            "entity_display_name_b": "entity_b_name",
            "entity_field_b": "entity_b_field",
            "entity_domain_b": "entity_b_domain",
        }
    ).copy()
    renamed["rank_type"] = rank_type
    for column in columns:
        if column not in renamed:
            renamed[column] = ""
    return renamed[columns]


def _top_metric_strings(metric_changes: pd.DataFrame, *, level: str) -> pd.DataFrame:
    if metric_changes.empty:
        return pd.DataFrame(columns=["entity_id", "top_changed_metrics"])
    id_col, _name_col = _entity_columns_for_level(level)
    if id_col not in metric_changes or "metric" not in metric_changes:
        return pd.DataFrame(columns=["entity_id", "top_changed_metrics"])
    value_col = (
        "robust_standardized_delta"
        if "robust_standardized_delta" in metric_changes.columns
        else "standardized_delta"
        if "standardized_delta" in metric_changes.columns
        else "absolute_delta"
    )
    if value_col not in metric_changes:
        return pd.DataFrame(columns=["entity_id", "top_changed_metrics"])
    frame = metric_changes.copy()
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
    if level in {"field", "domain"}:
        frame = (
            frame.groupby([id_col, "metric"], dropna=False)[value_col]
            .median()
            .reset_index()
        )
    rows: list[dict[str, str]] = []
    for entity_id, group in frame.groupby(id_col, sort=True, dropna=False):
        ranked = group.assign(abs_delta=group[value_col].abs()).sort_values(
            ["abs_delta", "metric"],
            ascending=[False, True],
            kind="mergesort",
            na_position="last",
        )
        parts = [
            f"{row.metric}:{float(getattr(row, value_col)):.3g}"
            for row in ranked.head(3).itertuples(index=False)
            if pd.notna(getattr(row, value_col))
        ]
        rows.append(
            {
                "entity_id": str(entity_id),
                "top_changed_metrics": "; ".join(parts),
            }
        )
    return pd.DataFrame(rows)


def driver_metrics_by_level(
    overall_change: pd.DataFrame,
    metric_changes: pd.DataFrame,
    *,
    level: str,
) -> pd.DataFrame:
    id_col, name_col = _entity_columns_for_level(level)
    if overall_change.empty or id_col not in overall_change:
        return pd.DataFrame()
    if level == "subfield":
        base_columns = [
            column
            for column in [
                id_col,
                name_col,
                "field_display_name",
                "domain_display_name",
                "overall_metric_change_l2",
            ]
            if column in overall_change
        ]
        base = overall_change[base_columns].copy()
    else:
        metadata_columns = [id_col, name_col]
        aggregations: dict[str, tuple[str, str]] = {
            "overall_metric_change_l2": ("overall_metric_change_l2", "median")
        }
        base = overall_change.groupby(metadata_columns, dropna=False).agg(**aggregations).reset_index()
        if level == "field" and "domain_display_name" in overall_change:
            domains = overall_change.groupby(metadata_columns, dropna=False)["domain_display_name"].first().reset_index()
            base = base.merge(domains, on=metadata_columns, how="left")
    top_metrics = _top_metric_strings(metric_changes, level=level)
    base = base.rename(columns={id_col: "entity_id", name_col: "entity_name"})
    base["entity_id"] = base["entity_id"].astype(str)
    base = base.merge(top_metrics, on="entity_id", how="left")
    return base


def add_top_pair_driver_diagnostics(
    pairs: pd.DataFrame,
    *,
    root: Path,
) -> tuple[pd.DataFrame, bool, list[str]]:
    if pairs.empty:
        return pairs, False, []
    overall_path = root / DEFAULT_OVERALL_CHANGE_RANKING
    metric_changes_path = root / DEFAULT_METRIC_TEMPORAL_CHANGES
    if not overall_path.exists() or not metric_changes_path.exists():
        return pairs, False, [
            "Driver diagnostics skipped because script 19 temporal change outputs were not found."
        ]
    overall_change = pd.read_parquet(overall_path)
    metric_changes = pd.read_parquet(metric_changes_path)
    frames: list[pd.DataFrame] = []
    merged_any = False
    for level, group in pairs.groupby("level", sort=False):
        drivers = driver_metrics_by_level(overall_change, metric_changes, level=str(level))
        if drivers.empty:
            frames.append(group)
            continue
        left = drivers.rename(
            columns={
                "entity_id": "entity_id_a",
                "overall_metric_change_l2": "entity_a_overall_metric_change_l2",
                "top_changed_metrics": "entity_a_top_changed_metrics",
            }
        ).drop(columns=["entity_name", "field_display_name", "domain_display_name"], errors="ignore")
        right = drivers.rename(
            columns={
                "entity_id": "entity_id_b",
                "overall_metric_change_l2": "entity_b_overall_metric_change_l2",
                "top_changed_metrics": "entity_b_top_changed_metrics",
            }
        ).drop(columns=["entity_name", "field_display_name", "domain_display_name"], errors="ignore")
        merged = group.merge(left, on="entity_id_a", how="left").merge(right, on="entity_id_b", how="left")
        frames.append(merged)
        merged_any = True
    return pd.concat(frames, ignore_index=True), merged_any, []


def entity_frame_for_level(frame: pd.DataFrame, *, level: str) -> pd.DataFrame:
    id_col, name_col = _entity_columns_for_level(level)
    if frame.empty or id_col not in frame or name_col not in frame:
        return pd.DataFrame(
            columns=["level", "entity_id", "entity_display_name", "field_display_name", "domain_display_name"]
        )
    columns = [id_col, name_col]
    for column in ["field_display_name", "domain_display_name"]:
        if column in frame and column not in columns:
            columns.append(column)
    entities = frame[columns].drop_duplicates(subset=[id_col]).copy()
    entities = entities.rename(columns={id_col: "entity_id", name_col: "entity_display_name"})
    if "field_display_name" not in entities:
        entities["field_display_name"] = entities["entity_display_name"] if level == "field" else ""
    if "domain_display_name" not in entities:
        entities["domain_display_name"] = entities["entity_display_name"] if level == "domain" else ""
    if level == "domain":
        entities["field_display_name"] = ""
        entities["domain_display_name"] = entities["entity_display_name"]
    entities.insert(0, "level", level)
    entities["entity_id"] = entities["entity_id"].astype(str)
    return entities[
        ["level", "entity_id", "entity_display_name", "field_display_name", "domain_display_name"]
    ]


def entity_retention_diagnostics(
    temporal_metrics: pd.DataFrame,
    temporal_profiles: pd.DataFrame,
    *,
    levels: list[str],
    windows: list[TemporalWindow],
    metrics: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    first_label = windows[0].window_label
    last_label = windows[-1].window_label
    diagnostics: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []
    valid_temporal = _valid_metric_rows(temporal_metrics, "metric_status")
    for level in levels:
        input_entities = entity_frame_for_level(temporal_metrics, level=level)
        profiles = temporal_profiles.loc[
            (temporal_profiles.get("level") == level)
            & (temporal_profiles.get("metric_set") == TEMPORAL_METRIC_SET)
        ].copy()
        retained_ids: set[str] = set()
        profile_info: dict[str, dict[str, Any]] = {}
        for entity_id, group in profiles.groupby("entity_id", sort=True, dropna=False):
            windows_available = int(group["window_label"].nunique()) if "window_label" in group else 0
            endpoints = group.loc[group["window_label"].isin([first_label, last_label])]
            finite_by_endpoint = []
            missing_metrics: set[str] = set()
            for _, endpoint in endpoints.iterrows():
                finite_metrics = [
                    metric
                    for metric in metrics
                    if metric in endpoint.index and pd.notna(endpoint[metric])
                ]
                finite_by_endpoint.append(len(finite_metrics))
                missing_metrics.update(
                    metric
                    for metric in metrics
                    if metric in endpoint.index and pd.isna(endpoint[metric])
                )
            has_endpoints = set(endpoints["window_label"].astype(str)) == {first_label, last_label}
            enough_metrics = has_endpoints and finite_by_endpoint and min(finite_by_endpoint) >= 2
            if enough_metrics:
                retained_ids.add(str(entity_id))
            profile_info[str(entity_id)] = {
                "n_windows_available": windows_available,
                "has_endpoints": has_endpoints,
                "missing_metrics": "; ".join(sorted(missing_metrics)),
            }
        input_ids = set(input_entities["entity_id"].astype(str))
        dropped_ids = sorted(input_ids - retained_ids)
        if not dropped_ids:
            reason = "all input entities retained for initial/final temporal morphology profiles"
        else:
            reason = "dropped entities lack valid initial/final profiles with at least two finite structural metrics"
        diagnostics.append(
            {
                "level": level,
                "n_input_entities": int(len(input_ids)),
                "n_retained_entities": int(len(retained_ids)),
                "n_dropped_entities": int(len(dropped_ids)),
                "reason": reason,
            }
        )
        for entity_id in dropped_ids:
            entity = input_entities.loc[input_entities["entity_id"] == entity_id].head(1)
            entity_record = entity.iloc[0].to_dict() if not entity.empty else {}
            info = profile_info.get(entity_id, {})
            valid_rows = valid_temporal
            if level == "subfield":
                valid_rows = valid_rows.loc[valid_rows["subfield_id"].astype(str) == entity_id]
            elif level == "field":
                valid_rows = valid_rows.loc[valid_rows["field_id"].astype(str) == entity_id]
            elif level == "domain":
                valid_rows = valid_rows.loc[valid_rows["domain_id"].astype(str) == entity_id]
            missing_metrics = info.get("missing_metrics", "")
            if not missing_metrics and not valid_rows.empty:
                missing_metrics = "; ".join(
                    metric
                    for metric in metrics
                    if metric in valid_rows and valid_rows[metric].isna().all()
                )
            if entity_id not in profile_info:
                reason_dropped = "no valid temporal metric profile rows"
            elif not info.get("has_endpoints", False):
                reason_dropped = "missing initial or final temporal profile"
            else:
                reason_dropped = "fewer than two finite structural metrics in an endpoint profile"
            dropped_rows.append(
                {
                    "level": level,
                    "entity_id": entity_id,
                    "entity_display_name": entity_record.get("entity_display_name", ""),
                    "field_display_name": entity_record.get("field_display_name", ""),
                    "domain_display_name": entity_record.get("domain_display_name", ""),
                    "reason_dropped": reason_dropped,
                    "missing_metrics": missing_metrics,
                    "n_windows_available": int(info.get("n_windows_available", 0)),
                }
            )
    return pd.DataFrame(diagnostics), pd.DataFrame(
        dropped_rows,
        columns=[
            "level",
            "entity_id",
            "entity_display_name",
            "field_display_name",
            "domain_display_name",
            "reason_dropped",
            "missing_metrics",
            "n_windows_available",
        ],
    )


def top_pair_entity_frequency(top_pairs: pd.DataFrame) -> pd.DataFrame:
    if top_pairs.empty:
        return pd.DataFrame(
            columns=[
                "level",
                "distance_type",
                "entity_id",
                "entity_name",
                "field_display_name",
                "domain_display_name",
                "n_top_converging_pairs",
                "n_top_diverging_pairs",
                "n_top_pairs_total",
                "mean_delta_in_top_pairs",
                "median_abs_delta_in_top_pairs",
            ]
        )
    rows: list[dict[str, Any]] = []
    for row in top_pairs.itertuples(index=False):
        delta = float(row.delta_morphological_distance)
        direction = str(getattr(row, "direction", ""))
        for side in ["a", "b"]:
            rows.append(
                {
                    "level": row.level,
                    "distance_type": row.distance_metric,
                    "entity_id": getattr(row, f"entity_id_{side}"),
                    "entity_name": getattr(row, f"entity_display_name_{side}"),
                    "field_display_name": getattr(row, f"entity_field_{side}", ""),
                    "domain_display_name": getattr(row, f"entity_domain_{side}", ""),
                    "direction": direction,
                    "delta_distance": delta,
                }
            )
    long = pd.DataFrame(rows)
    grouped = (
        long.groupby(
            [
                "level",
                "distance_type",
                "entity_id",
                "entity_name",
                "field_display_name",
                "domain_display_name",
            ],
            dropna=False,
        )
        .agg(
            n_top_converging_pairs=("direction", lambda value: int((value == "converging").sum())),
            n_top_diverging_pairs=("direction", lambda value: int((value == "diverging").sum())),
            n_top_pairs_total=("direction", "size"),
            mean_delta_in_top_pairs=("delta_distance", "mean"),
            median_abs_delta_in_top_pairs=("delta_distance", lambda value: float(np.median(np.abs(value)))),
        )
        .reset_index()
    )
    return grouped.sort_values(
        ["n_top_pairs_total", "entity_name"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)


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


def _abbreviate_matrix_labels(matrix: pd.DataFrame, *, max_label_length: int = 55) -> pd.DataFrame:
    if matrix.empty:
        return matrix
    labels = [
        shorten(str(label), width=max_label_length, placeholder="...")
        for label in matrix.index
    ]
    output = matrix.copy()
    output.index = labels
    output.columns = labels
    return output


def plot_heatmap(
    matrix: pd.DataFrame,
    path: str | Path,
    *,
    title: str,
    diverging: bool | None = None,
    colorbar_label: str = "Morphological distance",
    max_labels: int = 70,
    max_label_length: int = 55,
) -> None:
    if matrix.empty:
        write_placeholder_figure(path, title + "\nNo matrix values available")
        return
    plot_matrix = _subset_matrix_for_plot(matrix, max_labels=max_labels)
    plot_matrix = _abbreviate_matrix_labels(plot_matrix, max_label_length=max_label_length)
    values = plot_matrix.to_numpy(dtype=float).copy()
    np.fill_diagonal(values, 0.0)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        write_placeholder_figure(path, title + "\nNo finite matrix values available")
        return
    use_diverging = ("delta" in title.lower() or "change" in title.lower()) if diverging is None else diverging
    if use_diverging:
        vmax = max(float(np.nanmax(np.abs(finite))), 0.01)
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
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(colorbar_label, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_triptych(
    initial_matrix: pd.DataFrame,
    final_matrix: pd.DataFrame,
    delta_matrix: pd.DataFrame,
    path: str | Path,
    *,
    title: str,
    max_label_length: int = 55,
) -> None:
    if initial_matrix.empty or final_matrix.empty or delta_matrix.empty:
        write_placeholder_figure(path, title + "\nTriptych matrix values unavailable")
        return
    labels = list(delta_matrix.index)
    panels = [
        _abbreviate_matrix_labels(initial_matrix.reindex(index=labels, columns=labels), max_label_length=max_label_length),
        _abbreviate_matrix_labels(final_matrix.reindex(index=labels, columns=labels), max_label_length=max_label_length),
        _abbreviate_matrix_labels(delta_matrix.reindex(index=labels, columns=labels), max_label_length=max_label_length),
    ]
    distance_values = np.concatenate(
        [panels[0].to_numpy(dtype=float).ravel(), panels[1].to_numpy(dtype=float).ravel()]
    )
    distance_values = distance_values[np.isfinite(distance_values)]
    if distance_values.size == 0:
        write_placeholder_figure(path, title + "\nNo finite initial/final values")
        return
    distance_vmin = float(np.nanmin(distance_values))
    distance_vmax = float(np.nanmax(distance_values))
    if np.isclose(distance_vmin, distance_vmax):
        distance_vmax = distance_vmin + 1.0
    delta_values = panels[2].to_numpy(dtype=float).copy()
    np.fill_diagonal(delta_values, 0.0)
    finite_delta = delta_values[np.isfinite(delta_values)]
    delta_vmax = max(float(np.nanmax(np.abs(finite_delta))), 0.01) if finite_delta.size else 0.01
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), dpi=230)
    subtitles = ["2000-2004", "2020-2024", "Change"]
    for idx, ax in enumerate(axes):
        values = panels[idx].to_numpy(dtype=float).copy()
        np.fill_diagonal(values, 0.0)
        if idx < 2:
            image = ax.imshow(values, aspect="auto", cmap="viridis", vmin=distance_vmin, vmax=distance_vmax)
            cbar_label = "Morphological distance"
        else:
            image = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-delta_vmax, vmax=delta_vmax)
            cbar_label = "Final minus initial morphological distance"
        show_labels = len(panels[idx]) <= 35
        ax.set_xticks(np.arange(len(panels[idx])))
        ax.set_yticks(np.arange(len(panels[idx])))
        ax.set_xticklabels(panels[idx].columns if show_labels else [], rotation=90, fontsize=5)
        ax.set_yticklabels(panels[idx].index if show_labels else [], fontsize=5)
        ax.set_title(subtitles[idx], fontsize=10)
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label(cbar_label, fontsize=7)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_delta_comparison(
    euclidean_matrix: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    path: str | Path,
    *,
    title: str,
) -> None:
    if euclidean_matrix.empty or correlation_matrix.empty:
        write_placeholder_figure(path, title + "\nComparison matrix values unavailable")
        return
    labels = list(euclidean_matrix.index)
    matrices = [
        _abbreviate_matrix_labels(euclidean_matrix.reindex(index=labels, columns=labels), max_label_length=55),
        _abbreviate_matrix_labels(correlation_matrix.reindex(index=labels, columns=labels), max_label_length=55),
    ]
    values_all = np.concatenate([matrix.to_numpy(dtype=float).ravel() for matrix in matrices])
    finite = values_all[np.isfinite(values_all)]
    vmax = max(float(np.nanmax(np.abs(finite))), 0.01) if finite.size else 0.01
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=230)
    for ax, matrix, subtitle in zip(axes, matrices, ["Euclidean", "Correlation"], strict=True):
        values = matrix.to_numpy(dtype=float).copy()
        np.fill_diagonal(values, 0.0)
        image = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(subtitle, fontsize=10)
        ax.set_xticks(np.arange(len(matrix)))
        ax.set_yticks(np.arange(len(matrix)))
        ax.set_xticklabels(matrix.columns, rotation=90, fontsize=6)
        ax.set_yticklabels(matrix.index, fontsize=6)
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("Final minus initial morphological distance", fontsize=7)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_top_pairs(
    changes: pd.DataFrame,
    path: str | Path,
    *,
    level: str,
    distance_metric: str,
    top_n: int,
    max_label_length: int = 60,
) -> None:
    subset = changes.loc[
        (changes["level"] == level)
        & (changes["distance_metric"] == distance_metric)
        & changes["delta_morphological_distance"].notna()
    ].copy()
    if subset.empty:
        write_placeholder_figure(path, f"No morphological changes available for {level}")
        return
    converging = subset.loc[subset["delta_morphological_distance"] < 0].sort_values(
        ["delta_morphological_distance", "pair_id"],
        ascending=[True, True],
        kind="mergesort",
    ).head(top_n)
    diverging = subset.loc[subset["delta_morphological_distance"] > 0].sort_values(
        ["delta_morphological_distance", "pair_id"],
        ascending=[False, True],
        kind="mergesort",
    ).head(top_n)
    plot_frame = pd.concat([converging, diverging], ignore_index=True)
    if plot_frame.empty:
        write_placeholder_figure(path, f"No signed morphological changes available for {level}")
        return
    plot_frame["label"] = [
        "\n".join(wrap(shorten(f"{a} <-> {b}", width=max_label_length, placeholder="..."), width=44))
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
    ax.set_title(
        f"{level.title()} {_distance_type_label(distance_metric)} Top Morphological Convergence And Divergence",
        fontsize=12,
    )
    ax.set_xlabel("Final minus initial morphological distance")
    ax.tick_params(axis="y", labelsize=6)
    ax.text(
        0.01,
        0.01,
        "Negative = convergence; positive = divergence.",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_static_pairs(
    closest: pd.DataFrame,
    farthest: pd.DataFrame,
    path: str | Path,
    *,
    level: str,
    distance_metric: str,
    max_label_length: int = 60,
) -> None:
    plot_frame = pd.concat([closest, farthest], ignore_index=True)
    if plot_frame.empty:
        write_placeholder_figure(path, f"No static morphological pairs available for {level}")
        return
    plot_frame["label"] = [
        "\n".join(wrap(shorten(f"{a} <-> {b}", width=max_label_length, placeholder="..."), width=44))
        for a, b in zip(
            plot_frame["entity_display_name_a"],
            plot_frame["entity_display_name_b"],
            strict=False,
        )
    ]
    plot_frame["kind"] = ["closest"] * len(closest) + ["farthest"] * len(farthest)
    plot_frame = plot_frame.iloc[::-1]
    colors = np.where(plot_frame["kind"] == "closest", "#2f6f73", "#7f4f9f")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, max(5, 0.42 * len(plot_frame))), dpi=230)
    ax.barh(plot_frame["label"], plot_frame["morphological_distance"], color=colors)
    ax.set_title(
        f"{level.title()} {_distance_type_label(distance_metric)} Static Closest And Farthest Morphological Pairs",
        fontsize=12,
    )
    ax.set_xlabel("Static morphological distance")
    ax.tick_params(axis="y", labelsize=6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_entity_frequency(frequency: pd.DataFrame, path: str | Path, *, level: str) -> None:
    subset = frequency.loc[frequency["level"] == level].copy()
    if subset.empty:
        write_placeholder_figure(path, f"No top-pair entity frequency available for {level}")
        return
    grouped = (
        subset.groupby(["entity_id", "entity_name"], dropna=False)[
            ["n_top_converging_pairs", "n_top_diverging_pairs", "n_top_pairs_total"]
        ]
        .sum()
        .reset_index()
        .sort_values(["n_top_pairs_total", "entity_name"], ascending=[False, True], kind="mergesort")
        .head(20)
        .iloc[::-1]
    )
    labels = [shorten(str(name), width=55, placeholder="...") for name in grouped["entity_name"]]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(grouped) + 1.5)), dpi=230)
    ax.barh(labels, grouped["n_top_converging_pairs"], color="#2f6f73", label="Converging")
    ax.barh(
        labels,
        grouped["n_top_diverging_pairs"],
        left=grouped["n_top_converging_pairs"],
        color="#b45432",
        label="Diverging",
    )
    ax.set_title(f"{level.title()} Entities Most Frequent In Extreme Morphological Pairs", fontsize=12)
    ax.set_xlabel("Appearances in top pairs")
    ax.legend(fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
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


def write_top_pair_tables(
    *,
    top_pairs: pd.DataFrame,
    output_dir: str | Path,
    root: Path,
    levels: list[str],
    distance_metrics: list[str],
) -> dict[str, str]:
    top_dir = Path(output_dir) / "top_pairs"
    top_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for level in levels:
        for distance_metric in distance_metrics:
            subset = top_pairs.loc[
                (top_pairs["level"] == level)
                & (top_pairs["distance_metric"] == distance_metric)
            ].copy()
            for direction in ["converging", "diverging"]:
                direction_subset = subset.loc[subset["direction"] == direction].copy()
                formatted = format_temporal_pair_table(direction_subset)
                path = top_dir / f"{level}_{distance_metric}_top_{direction}_pairs.csv"
                formatted.to_csv(path, index=False)
                written[f"{level}_{distance_metric}_top_{direction}_pairs"] = display_path(path, root=root)
    return written


def write_static_pair_summaries(
    *,
    static_pairs: pd.DataFrame,
    output_dir: str | Path,
    root: Path,
    levels: list[str],
    distance_metrics: list[str],
    top_n: int,
) -> tuple[dict[str, str], dict[str, list[dict[str, Any]]]]:
    static_dir = Path(output_dir) / "static_pairs"
    static_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    records: dict[str, list[dict[str, Any]]] = {}
    for level in levels:
        for distance_metric in distance_metrics:
            subset = static_pairs.loc[
                (static_pairs["level"] == level)
                & (static_pairs["distance_metric"] == distance_metric)
                & (static_pairs["metric_set"] == "static_full_reduced")
                & static_pairs["morphological_distance"].notna()
            ].copy()
            closest = subset.sort_values(
                ["morphological_distance", "pair_id"],
                ascending=[True, True],
                kind="mergesort",
            ).head(top_n).copy()
            farthest = subset.sort_values(
                ["morphological_distance", "pair_id"],
                ascending=[False, True],
                kind="mergesort",
            ).head(top_n).copy()
            closest["rank"] = np.arange(1, len(closest) + 1)
            farthest["rank"] = np.arange(1, len(farthest) + 1)
            closest_path = static_dir / f"{level}_{distance_metric}_closest_static_pairs.csv"
            farthest_path = static_dir / f"{level}_{distance_metric}_farthest_static_pairs.csv"
            format_static_pair_table(closest, rank_type="closest").to_csv(closest_path, index=False)
            format_static_pair_table(farthest, rank_type="farthest").to_csv(farthest_path, index=False)
            written[f"{level}_{distance_metric}_closest_static_pairs"] = display_path(closest_path, root=root)
            written[f"{level}_{distance_metric}_farthest_static_pairs"] = display_path(farthest_path, root=root)
            records[f"{level}_{distance_metric}_closest"] = dataframe_to_records(
                format_static_pair_table(closest, rank_type="closest"),
                limit=top_n,
            )
            records[f"{level}_{distance_metric}_farthest"] = dataframe_to_records(
                format_static_pair_table(farthest, rank_type="farthest"),
                limit=top_n,
            )
            if level == "field":
                plot_static_pairs(
                    closest,
                    farthest,
                    Path(output_dir) / f"{level}_{distance_metric}_static_closest_farthest_pairs.png",
                    level=level,
                    distance_metric=distance_metric,
                )
                written[f"{level}_{distance_metric}_static_closest_farthest_pairs_plot"] = display_path(
                    Path(output_dir) / f"{level}_{distance_metric}_static_closest_farthest_pairs.png",
                    root=root,
                )
    return written, records


def write_morphological_outputs(
    *,
    static_pairs: pd.DataFrame,
    temporal_pairs: pd.DataFrame,
    changes: pd.DataFrame,
    scaling_parameters: pd.DataFrame,
    retention_diagnostics: pd.DataFrame,
    dropped_entities: pd.DataFrame,
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
    retention_diagnostics.to_csv(paths["entity_retention_diagnostics_csv"], index=False)
    dropped_entities.to_csv(paths["dropped_entities_csv"], index=False)
    raw_top_converging = top_morphological_pairs(
        changes,
        top_n=top_n_pairs,
        direction="converging",
    )
    raw_top_diverging = top_morphological_pairs(
        changes,
        top_n=top_n_pairs,
        direction="diverging",
    )
    raw_top_pairs = pd.concat([raw_top_converging, raw_top_diverging], ignore_index=True)
    top_pairs, driver_diagnostics_merged, driver_warnings = add_top_pair_driver_diagnostics(
        raw_top_pairs,
        root=root,
    )
    top_converging = top_pairs.loc[top_pairs["direction"] == "converging"].copy()
    top_diverging = top_pairs.loc[top_pairs["direction"] == "diverging"].copy()
    format_temporal_pair_table(top_converging).to_csv(paths["top_converging_csv"], index=False)
    format_temporal_pair_table(top_diverging).to_csv(paths["top_diverging_csv"], index=False)
    format_temporal_pair_table(top_pairs).to_csv(paths["top_pair_driver_diagnostics_csv"], index=False)
    top_pair_paths = write_top_pair_tables(
        top_pairs=top_pairs,
        output_dir=output_dir,
        root=root,
        levels=levels,
        distance_metrics=distance_metrics,
    )
    frequency = top_pair_entity_frequency(top_pairs)
    frequency.to_csv(paths["top_pair_entity_frequency_csv"], index=False)
    plot_entity_frequency(
        frequency,
        Path(output_dir) / "subfield_top_pair_entity_frequency.png",
        level="subfield",
    )
    plot_entity_frequency(
        frequency,
        Path(output_dir) / "field_top_pair_entity_frequency.png",
        level="field",
    )

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
    static_pair_paths, static_pair_records = write_static_pair_summaries(
        static_pairs=static_pairs,
        output_dir=output_dir,
        root=root,
        levels=levels,
        distance_metrics=distance_metrics,
        top_n=top_n_pairs,
    )
    first_label = windows[0].window_label
    last_label = windows[-1].window_label
    figure_paths: dict[str, str] = {
        "subfield_top_pair_entity_frequency_plot": display_path(
            Path(output_dir) / "subfield_top_pair_entity_frequency.png",
            root=root,
        ),
        "field_top_pair_entity_frequency_plot": display_path(
            Path(output_dir) / "field_top_pair_entity_frequency.png",
            root=root,
        ),
    }
    for level in levels:
        for distance_metric in distance_metrics:
            static_subset = static_pairs.loc[
                (static_pairs["level"] == level)
                & (static_pairs["distance_metric"] == distance_metric)
                & (static_pairs["metric_set"] == "static_full_reduced")
            ]
            temporal_subset = temporal_pairs.loc[
                (temporal_pairs["level"] == level)
                & (temporal_pairs["distance_metric"] == distance_metric)
                & (temporal_pairs["metric_set"] == TEMPORAL_METRIC_SET)
            ]
            change_subset = changes.loc[
                (changes["level"] == level)
                & (changes["distance_metric"] == distance_metric)
                & (changes["metric_set"] == TEMPORAL_METRIC_SET)
            ]
            static_matrix = matrix_from_pairs(static_subset, value_column="morphological_distance")
            initial_matrix = matrix_from_pairs(
                temporal_subset.loc[temporal_subset["window_label"] == first_label],
                value_column="morphological_distance",
            )
            final_matrix = matrix_from_pairs(
                temporal_subset.loc[temporal_subset["window_label"] == last_label],
                value_column="morphological_distance",
            )
            delta_matrix = matrix_from_pairs(change_subset, value_column="delta_morphological_distance")
            if distance_metric == default_distance:
                plot_heatmap(
                    static_matrix,
                    Path(output_dir) / f"{level}_static_morphological_distance_heatmap.png",
                    title=f"{level.title()} Static Morphological Distance ({_distance_type_label(distance_metric)})",
                    colorbar_label="Morphological distance",
                )
                plot_heatmap(
                    initial_matrix,
                    Path(output_dir) / f"{level}_initial_morphological_distance_heatmap.png",
                    title=f"{level.title()} Morphological Distance: {first_label} ({_distance_type_label(distance_metric)})",
                    colorbar_label="Morphological distance",
                )
                plot_heatmap(
                    final_matrix,
                    Path(output_dir) / f"{level}_final_morphological_distance_heatmap.png",
                    title=f"{level.title()} Morphological Distance: {last_label} ({_distance_type_label(distance_metric)})",
                    colorbar_label="Morphological distance",
                )
                plot_heatmap(
                    delta_matrix,
                    Path(output_dir) / f"{level}_morphological_distance_delta_heatmap.png",
                    title=f"{level.title()} Morphological Distance Change ({_distance_type_label(distance_metric)})",
                    diverging=True,
                    colorbar_label="Final minus initial morphological distance",
                )
            if level in {"domain", "field"}:
                delta_path = Path(output_dir) / f"{level}_{distance_metric}_delta_heatmap.png"
                plot_heatmap(
                    delta_matrix,
                    delta_path,
                    title=(
                        f"{level.title()} {_distance_type_label(distance_metric)} "
                        "Morphological Distance Change"
                    ),
                    diverging=True,
                    colorbar_label="Final minus initial morphological distance",
                    max_label_length=55,
                )
                figure_paths[f"{level}_{distance_metric}_delta_heatmap"] = display_path(delta_path, root=root)
                triptych_path = Path(output_dir) / f"{level}_{distance_metric}_initial_final_delta_triptych.png"
                plot_triptych(
                    initial_matrix,
                    final_matrix,
                    delta_matrix,
                    triptych_path,
                    title=(
                        f"{level.title()} {_distance_type_label(distance_metric)} "
                        "Morphological Distance: Initial, Final, And Change"
                    ),
                )
                figure_paths[f"{level}_{distance_metric}_initial_final_delta_triptych"] = display_path(
                    triptych_path,
                    root=root,
                )
            plot_path = Path(output_dir) / f"{level}_{distance_metric}_top_converging_diverging_pairs.png"
            plot_top_pairs(
                changes,
                plot_path,
                level=level,
                distance_metric=distance_metric,
                top_n=top_n_pairs,
                max_label_length=65 if level != "subfield" else 70,
            )
            figure_paths[f"{level}_{distance_metric}_top_converging_diverging_pairs"] = display_path(
                plot_path,
                root=root,
            )
            if distance_metric == default_distance:
                legacy_plot = Path(output_dir) / f"{level}_top_morphological_converging_diverging_pairs.png"
                plot_top_pairs(
                    changes,
                    legacy_plot,
                    level=level,
                    distance_metric=distance_metric,
                    top_n=min(top_n_pairs, 20),
                )
                figure_paths[f"{level}_top_morphological_converging_diverging_pairs"] = display_path(
                    legacy_plot,
                    root=root,
                )
    if {"domain"}.issubset(set(levels)) and {"euclidean", "correlation"}.issubset(set(distance_metrics)):
        domain_euclidean = changes.loc[
            (changes["level"] == "domain")
            & (changes["distance_metric"] == "euclidean")
            & (changes["metric_set"] == TEMPORAL_METRIC_SET)
        ]
        domain_correlation = changes.loc[
            (changes["level"] == "domain")
            & (changes["distance_metric"] == "correlation")
            & (changes["metric_set"] == TEMPORAL_METRIC_SET)
        ]
        comparison_path = Path(output_dir) / "domain_euclidean_vs_correlation_delta_heatmaps.png"
        plot_delta_comparison(
            matrix_from_pairs(domain_euclidean, value_column="delta_morphological_distance"),
            matrix_from_pairs(domain_correlation, value_column="delta_morphological_distance"),
            comparison_path,
            title="Domain Morphological Distance Change: Euclidean vs Correlation",
        )
        figure_paths["domain_euclidean_vs_correlation_delta_heatmaps"] = display_path(
            comparison_path,
            root=root,
        )

    output_paths_payload = {key: display_path(path, root=root) for key, path in paths.items()}
    output_paths_payload.update(
        {key: display_path(path, root=root) for key, path in matrix_paths.items()}
    )
    output_paths_payload.update(top_pair_paths)
    output_paths_payload.update(static_pair_paths)
    output_paths_payload.update(figure_paths)
    retention_records = dataframe_to_records(retention_diagnostics, limit=len(retention_diagnostics))
    dropped_records = dataframe_to_records(dropped_entities, limit=min(len(dropped_entities), 50))
    warnings = driver_warnings.copy()
    if not dropped_entities.empty:
        warnings.append("Some entities were dropped from temporal morphology retention QA; inspect dropped_entities.csv.")
    summary = {
        "created_at": utc_now_iso(),
        "input_paths": input_paths,
        "output_paths": output_paths_payload,
        "levels": levels,
        "distance_metrics": distance_metrics,
        "default_distance_for_figures": default_distance,
        "scaler": "robust",
        "static_metric_sets": {key: [metric for metric in metrics] for key, metrics in STATIC_METRIC_SETS.items()},
        "temporal_metric_set": TEMPORAL_METRIC_SET,
        "temporal_metrics_used": [metric for metric in WINDOW_STRUCTURAL_METRICS],
        "n_static_pair_rows": int(len(static_pairs)),
        "n_temporal_pair_rows": int(len(temporal_pairs)),
        "n_temporal_change_rows": int(len(changes)),
        "entity_retention_diagnostics": retention_records,
        "dropped_entities": dropped_records,
        "driver_diagnostics_merged": driver_diagnostics_merged,
        "warnings": warnings,
        "top_converging_pairs": dataframe_to_records(format_temporal_pair_table(top_converging), limit=20),
        "top_diverging_pairs": dataframe_to_records(format_temporal_pair_table(top_diverging), limit=20),
        "top_pair_entity_frequency": dataframe_to_records(frequency, limit=30),
        "static_pair_results": static_pair_records,
        "method_notes": [
            "Morphological distance compares robust-scaled structural metric profiles.",
            "Euclidean distance compares magnitude differences in structural profiles.",
            "Correlation distance compares relative profile patterns.",
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
        "## Purpose",
        "",
        "This analysis compares disciplines by their internal semantic structure rather than by topic position.",
        "It is distinct from script 22: semantic distance compares centroid positions/content location, while morphological distance compares robust-scaled structural metric profiles.",
        "",
        "## Methods",
        "",
        "- Morphological distance is computed between robust-scaled metric profiles.",
        f"- Scaling method: `{summary.get('scaler', 'robust')}`.",
        "- Temporal morphology uses the eight non-temporal structural metrics recomputed per five-year window.",
        "- Static morphology uses the 11-metric reduced interpretable embedding core when available, plus an 8-metric structural static sensitivity set.",
        "- Negative final-minus-initial distance means morphological convergence. Positive means morphological divergence.",
        "- Field and domain profiles are averages of robust-scaled subfield profiles.",
        "",
        "## Interpretation of Distance Types",
        "",
        "- Euclidean morphological distance compares the magnitude of robust-scaled metric profiles.",
        "- Correlation morphological distance compares the relative pattern of metrics, ignoring much of the absolute magnitude.",
        "- The two may disagree; disagreement is substantively meaningful rather than automatically an error.",
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
            f"- Driver diagnostics merged: {'yes' if summary.get('driver_diagnostics_merged') else 'no'}",
            "",
            "## Entity Retention QA",
            "",
        ]
    )
    for row in summary.get("entity_retention_diagnostics", []):
        lines.append(
            "- {level}: input={input_n}, retained={retained}, dropped={dropped}. {reason}".format(
                level=row.get("level"),
                input_n=row.get("n_input_entities"),
                retained=row.get("n_retained_entities"),
                dropped=row.get("n_dropped_entities"),
                reason=row.get("reason"),
            )
        )
    if summary.get("dropped_entities"):
        lines.append("- Dropped entity details are written to `dropped_entities.csv`.")
    lines.extend(
        [
            "",
            "## Key Static Results",
            "",
            "Top static closest/farthest field pairs are listed below for the full-period reduced metric profile.",
            "",
            "### Field Euclidean Closest",
            "",
        ]
    )
    for row in summary.get("static_pair_results", {}).get("field_euclidean_closest", [])[:5]:
        lines.append(
            f"- {row.get('entity_a_name')} <-> {row.get('entity_b_name')}: {row.get('morphological_distance')}"
        )
    lines.extend(["", "### Field Euclidean Farthest", ""])
    for row in summary.get("static_pair_results", {}).get("field_euclidean_farthest", [])[:5]:
        lines.append(
            f"- {row.get('entity_a_name')} <-> {row.get('entity_b_name')}: {row.get('morphological_distance')}"
        )
    lines.extend(["", "### Field Correlation Closest", ""])
    for row in summary.get("static_pair_results", {}).get("field_correlation_closest", [])[:5]:
        lines.append(
            f"- {row.get('entity_a_name')} <-> {row.get('entity_b_name')}: {row.get('morphological_distance')}"
        )
    lines.extend(["", "### Field Correlation Farthest", ""])
    for row in summary.get("static_pair_results", {}).get("field_correlation_farthest", [])[:5]:
        lines.append(
            f"- {row.get('entity_a_name')} <-> {row.get('entity_b_name')}: {row.get('morphological_distance')}"
        )
    lines.extend(
        [
            "",
            "## Key Temporal Results",
            "",
            "The delta panel in the triptych figures is the key temporal result.",
            "",
            "### Field Euclidean Convergences",
            "",
        ]
    )
    top_converging = summary.get("top_converging_pairs", [])
    top_diverging = summary.get("top_diverging_pairs", [])
    for row in [
        item
        for item in top_converging
        if item.get("level") == "field" and item.get("distance_type") == "euclidean"
    ][:5]:
        lines.append(f"- {row.get('entity_a_name')} <-> {row.get('entity_b_name')}: {row.get('delta_distance')}")
    lines.extend(["", "### Field Euclidean Divergences", ""])
    for row in [
        item
        for item in top_diverging
        if item.get("level") == "field" and item.get("distance_type") == "euclidean"
    ][:5]:
        lines.append(f"- {row.get('entity_a_name')} <-> {row.get('entity_b_name')}: {row.get('delta_distance')}")
    lines.extend(["", "### Field Correlation Convergences", ""])
    for row in [
        item
        for item in top_converging
        if item.get("level") == "field" and item.get("distance_type") == "correlation"
    ][:5]:
        lines.append(f"- {row.get('entity_a_name')} <-> {row.get('entity_b_name')}: {row.get('delta_distance')}")
    lines.extend(["", "### Field Correlation Divergences", ""])
    for row in [
        item
        for item in top_diverging
        if item.get("level") == "field" and item.get("distance_type") == "correlation"
    ][:5]:
        lines.append(f"- {row.get('entity_a_name')} <-> {row.get('entity_b_name')}: {row.get('delta_distance')}")
    lines.extend(
        [
            "",
            "## Dominant Entities in Extreme Pairs",
            "",
        ]
    )
    frequency = summary.get("top_pair_entity_frequency", [])
    if frequency:
        for row in frequency[:10]:
            lines.append(
                "- {level}/{distance}: {name} appears in {n} top pairs ({conv} converging, {div} diverging).".format(
                    level=row.get("level"),
                    distance=row.get("distance_type"),
                    name=row.get("entity_name"),
                    n=row.get("n_top_pairs_total"),
                    conv=row.get("n_top_converging_pairs"),
                    div=row.get("n_top_diverging_pairs"),
                )
            )
    else:
        lines.append("No top-pair entity frequency diagnostics were available.")
    lines.append(
        "If a small number of entities dominate many extreme pairs, the pairwise results should be interpreted as evidence of those entities' unusual morphological evolution, not as many independent pairwise phenomena."
    )
    if summary.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in summary["warnings"]:
            lines.append(f"- {warning}")
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Field and domain morphology profiles are averages of robust-scaled subfield profiles.",
            "- Full subfield matrices are saved, but top-pair plots are usually more readable for thesis interpretation.",
            "- Publication counts are retained only as QA metadata and are not used as substantive morphology features.",
            "- Subfield extreme pairs may be outlier-driven; inspect top-pair entity frequency diagnostics.",
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
    retention_diagnostics, dropped_entities = entity_retention_diagnostics(
        temporal_metrics,
        temporal_profiles,
        levels=levels,
        windows=windows,
        metrics=temporal_available,
    )
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
        retention_diagnostics=retention_diagnostics,
        dropped_entities=dropped_entities,
        output_dir=output_dir,
        root=root,
        levels=levels,
        distance_metrics=distance_metrics,
        windows=windows,
        top_n_pairs=top_n_pairs,
        input_paths=input_paths,
    )
