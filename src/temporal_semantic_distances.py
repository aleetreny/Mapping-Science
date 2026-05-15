from __future__ import annotations

from pathlib import Path
from textwrap import shorten, wrap
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from src.storage import save_parquet
from src.temporal_common import (
    EPS,
    TemporalWindow,
    cosine_distance,
    dataframe_to_records,
    default_or_computed_windows,
    display_path,
    ensure_outputs_do_not_exist,
    linear_slope,
    utc_now_iso,
    vector_columns,
    write_json,
    write_placeholder_figure,
    write_text,
)

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


PROCESSED_TEMPORAL_DIR = Path("data/processed/temporal")
PAIRWISE_OUTPUTS = {
    "pair_distances_parquet": PROCESSED_TEMPORAL_DIR
    / "semantic_pair_distances_by_window.parquet",
    "pair_changes_parquet": PROCESSED_TEMPORAL_DIR
    / "semantic_pair_distance_changes.parquet",
}

ANALYSIS_FILENAMES = {
    "top_converging_csv": "top_semantic_converging_pairs.csv",
    "top_diverging_csv": "top_semantic_diverging_pairs.csv",
    "most_negative_delta_csv": "most_negative_delta_pairs.csv",
    "most_positive_delta_csv": "most_positive_delta_pairs.csv",
    "subfield_top_converging_csv": "subfield_top_semantic_converging_pairs.csv",
    "subfield_top_diverging_csv": "subfield_top_semantic_diverging_pairs.csv",
    "summary_md": "summary.md",
    "summary_json": "summary.json",
}

DEFAULT_PATH_METRICS = PROCESSED_TEMPORAL_DIR / "subfield_centroid_path_metrics.parquet"


def output_paths(output_dir: str | Path, *, root: Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    paths = {key: root / value for key, value in PAIRWISE_OUTPUTS.items()}
    paths.update({key: output_dir / value for key, value in ANALYSIS_FILENAMES.items()})
    paths["matrices_dir"] = output_dir / "matrices"
    return paths


def ensure_semantic_distance_outputs(
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
                Path(output_dir) / f"{level}_initial_distance_heatmap.png",
                Path(output_dir) / f"{level}_final_distance_heatmap.png",
                Path(output_dir) / f"{level}_distance_delta_heatmap.png",
                Path(output_dir) / f"{level}_top_converging_diverging_pairs.png",
                Path(output_dir) / f"{level}_top_semantic_converging_diverging_pairs_clean.png",
            ]
        )
    figure_paths.extend(
        [
            Path(output_dir) / "domain_distance_initial_final_delta_triptych.png",
            Path(output_dir) / "field_distance_initial_final_delta_triptych.png",
            Path(output_dir) / "subfield_distance_initial_final_delta_triptych_topN.png",
            Path(output_dir) / "subfield_semantic_convergence_divergence_network.png",
        ]
    )
    ensure_outputs_do_not_exist(
        [
            paths["pair_distances_parquet"],
            paths["pair_changes_parquet"],
            paths["top_converging_csv"],
            paths["top_diverging_csv"],
            paths["most_negative_delta_csv"],
            paths["most_positive_delta_csv"],
            paths["subfield_top_converging_csv"],
            paths["subfield_top_diverging_csv"],
            paths["summary_md"],
            paths["summary_json"],
            *figure_paths,
        ],
        overwrite=overwrite,
        root=root,
        description="temporal semantic distance outputs",
    )


def _entity_columns_for_level(level: str) -> tuple[str, str]:
    if level == "subfield":
        return "subfield_id", "subfield_display_name"
    if level == "field":
        return "field_id", "field_display_name"
    if level == "domain":
        return "domain_id", "domain_display_name"
    raise ValueError("level must be subfield, field, or domain")


def aggregate_centroids_for_level(centroids: pd.DataFrame, *, level: str) -> pd.DataFrame:
    dim_columns = vector_columns(centroids)
    if not dim_columns:
        raise ValueError("centroid table has no centroid_dim_* columns")
    status_col = "centroid_status" if "centroid_status" in centroids.columns else None
    frame = centroids.copy()
    if status_col is not None:
        frame = frame.loc[frame[status_col].isin({"completed", "completed_with_warnings"})]
    id_column, name_column = _entity_columns_for_level(level)
    required = {id_column, name_column, "window_label", "window_start", "window_end", "window_index", "n_used"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"centroid table missing columns for {level}: {', '.join(sorted(missing))}")
    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    group_columns = [
        id_column,
        name_column,
        "window_label",
        "window_start",
        "window_end",
        "window_index",
        "window_mid_year",
    ]
    for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
        weights = pd.to_numeric(group["n_used"], errors="coerce").fillna(0).to_numpy(dtype=float)
        matrix = group[dim_columns].to_numpy(dtype=float)
        finite = np.isfinite(matrix).all(axis=1) & (weights > 0)
        if not finite.any():
            continue
        centroid = np.average(matrix[finite], axis=0, weights=weights[finite])
        entity_id, entity_name, window_label, window_start, window_end, window_index, window_mid = keys
        rows.append(
            {
                "level": level,
                "entity_id": str(entity_id),
                "entity_display_name": str(entity_name),
                "window_label": str(window_label),
                "window_start": int(window_start),
                "window_end": int(window_end),
                "window_index": int(window_index),
                "window_mid_year": float(window_mid),
                "n_source_subfields": int(group["subfield_id"].nunique())
                if "subfield_id" in group
                else 1,
                "n_used": int(weights[finite].sum()),
            }
        )
        vectors.append(centroid.astype(np.float32))
    metadata = pd.DataFrame(rows)
    if metadata.empty:
        return metadata
    vector_frame = pd.DataFrame(
        np.vstack(vectors),
        columns=[f"centroid_dim_{idx:03d}" for idx in range(len(dim_columns))],
    )
    return pd.concat([metadata.reset_index(drop=True), vector_frame], axis=1)


def pairwise_semantic_distances(level_centroids: pd.DataFrame) -> pd.DataFrame:
    dim_columns = vector_columns(level_centroids)
    if level_centroids.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (level, window_label), group in level_centroids.groupby(
        ["level", "window_label"],
        sort=True,
        dropna=False,
    ):
        group = group.sort_values("entity_display_name", kind="mergesort").reset_index(drop=True)
        vectors = group[dim_columns].to_numpy(dtype=float)
        for left_idx in range(len(group)):
            for right_idx in range(left_idx + 1, len(group)):
                left = group.iloc[left_idx]
                right = group.iloc[right_idx]
                rows.append(
                    {
                        "level": level,
                        "window_label": str(window_label),
                        "window_start": int(left["window_start"]),
                        "window_end": int(left["window_end"]),
                        "window_index": int(left["window_index"]),
                        "window_mid_year": float(left["window_mid_year"]),
                        "entity_id_a": str(left["entity_id"]),
                        "entity_display_name_a": str(left["entity_display_name"]),
                        "entity_id_b": str(right["entity_id"]),
                        "entity_display_name_b": str(right["entity_display_name"]),
                        "pair_id": f"{left['entity_id']}__{right['entity_id']}",
                        "cosine_distance": cosine_distance(vectors[left_idx], vectors[right_idx]),
                    }
                )
    return pd.DataFrame(rows)


def semantic_distance_changes(
    pair_distances: pd.DataFrame,
    *,
    windows: list[TemporalWindow],
) -> pd.DataFrame:
    if pair_distances.empty:
        return pd.DataFrame()
    first_label = windows[0].window_label
    last_label = windows[-1].window_label
    rows: list[dict[str, Any]] = []
    group_columns = [
        "level",
        "entity_id_a",
        "entity_display_name_a",
        "entity_id_b",
        "entity_display_name_b",
        "pair_id",
    ]
    for keys, group in pair_distances.groupby(group_columns, sort=True, dropna=False):
        group = group.sort_values("window_start", kind="mergesort")
        initial = group.loc[group["window_label"] == first_label]
        final = group.loc[group["window_label"] == last_label]
        initial_distance = (
            float(initial["cosine_distance"].iloc[0]) if len(initial) else np.nan
        )
        final_distance = (
            float(final["cosine_distance"].iloc[0]) if len(final) else np.nan
        )
        values = group["cosine_distance"].to_numpy(dtype=float)
        rows.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "initial_window": first_label,
                "final_window": last_label,
                "initial_distance": initial_distance,
                "final_distance": final_distance,
                "delta_distance": final_distance - initial_distance,
                "distance_slope": linear_slope(
                    group["window_mid_year"].to_numpy(dtype=float),
                    values,
                ),
                "n_windows_available": int(np.isfinite(values).sum()),
            }
        )
    return pd.DataFrame(rows)


def matrix_from_pairs(
    frame: pd.DataFrame,
    *,
    value_column: str,
    label_a: str = "entity_display_name_a",
    label_b: str = "entity_display_name_b",
) -> pd.DataFrame:
    if frame.empty or label_a not in frame or label_b not in frame or value_column not in frame:
        return pd.DataFrame()
    labels = sorted(
        set(frame[label_a].astype(str)).union(set(frame[label_b].astype(str)))
    )
    matrix = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)
    for label in labels:
        matrix.loc[label, label] = 0.0
    for row in frame.itertuples(index=False):
        a = str(getattr(row, label_a))
        b = str(getattr(row, label_b))
        value = getattr(row, value_column)
        matrix.loc[a, b] = value
        matrix.loc[b, a] = value
    return matrix


def save_distance_matrices(
    pair_distances: pd.DataFrame,
    changes: pd.DataFrame,
    *,
    output_dir: str | Path,
    levels: list[str],
) -> dict[str, str]:
    matrices_dir = Path(output_dir) / "matrices"
    matrices_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for level in levels:
        level_pairs = pair_distances.loc[pair_distances["level"] == level]
        for window_label, group in level_pairs.groupby("window_label", sort=True):
            matrix = matrix_from_pairs(group, value_column="cosine_distance")
            path = matrices_dir / f"{level}_{window_label}_distance_matrix.csv"
            matrix.to_csv(path)
            written[f"{level}_{window_label}_distance_matrix"] = str(path)
        level_changes = changes.loc[changes["level"] == level]
        if not level_changes.empty:
            delta = matrix_from_pairs(level_changes, value_column="delta_distance")
            slope = matrix_from_pairs(level_changes, value_column="distance_slope")
            delta_path = matrices_dir / f"{level}_distance_delta_matrix.csv"
            slope_path = matrices_dir / f"{level}_distance_slope_matrix.csv"
            delta.to_csv(delta_path)
            slope.to_csv(slope_path)
            written[f"{level}_distance_delta_matrix"] = str(delta_path)
            written[f"{level}_distance_slope_matrix"] = str(slope_path)
    return written


def top_converging_diverging_pairs(
    changes: pd.DataFrame,
    *,
    top_n: int,
    direction: str,
) -> pd.DataFrame:
    if changes.empty:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for level, group in changes.groupby("level", sort=True):
        finite = group.loc[group["delta_distance"].notna()].copy()
        if direction == "converging":
            selected = finite.loc[finite["delta_distance"] < 0].sort_values(
                ["delta_distance", "pair_id"],
                ascending=[True, True],
                kind="mergesort",
            ).head(top_n)
        elif direction == "diverging":
            selected = finite.loc[finite["delta_distance"] > 0].sort_values(
                ["delta_distance", "pair_id"],
                ascending=[False, True],
                kind="mergesort",
            ).head(top_n)
        else:
            raise ValueError("direction must be converging or diverging")
        selected = selected.copy()
        selected["rank"] = np.arange(1, len(selected) + 1)
        selected["direction"] = direction
        frames.append(selected)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def most_extreme_delta_pairs(
    changes: pd.DataFrame,
    *,
    top_n: int,
    sign: str,
) -> pd.DataFrame:
    if changes.empty:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for level, group in changes.groupby("level", sort=True):
        finite = group.loc[group["delta_distance"].notna()].copy()
        if sign == "negative":
            selected = finite.sort_values(
                ["delta_distance", "pair_id"],
                ascending=[True, True],
                kind="mergesort",
            ).head(top_n)
        elif sign == "positive":
            selected = finite.sort_values(
                ["delta_distance", "pair_id"],
                ascending=[False, True],
                kind="mergesort",
            ).head(top_n)
        else:
            raise ValueError("sign must be negative or positive")
        selected = selected.copy()
        selected["rank"] = np.arange(1, len(selected) + 1)
        selected["rank_type"] = f"most_{sign}_delta"
        frames.append(selected)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def qa_checks_by_level(changes: pd.DataFrame) -> dict[str, dict[str, Any]]:
    checks: dict[str, dict[str, Any]] = {}
    if changes.empty:
        return checks
    for level, group in changes.groupby("level", sort=True):
        delta = pd.to_numeric(group["delta_distance"], errors="coerce")
        checks[str(level)] = {
            "n_pair_changes": int(len(group)),
            "n_converging_pairs": int((delta < 0).sum()),
            "n_diverging_pairs": int((delta > 0).sum()),
            "n_stable_zero_delta_pairs": int((delta == 0).sum()),
            "min_delta_distance": float(delta.min(skipna=True)) if delta.notna().any() else None,
            "max_delta_distance": float(delta.max(skipna=True)) if delta.notna().any() else None,
        }
    return checks


def _pair_label(frame: pd.DataFrame, *, max_length: int) -> pd.Series:
    return pd.Series(
        [
            shorten(f"{a} <-> {b}", width=max_length, placeholder="...")
            for a, b in zip(
                frame["entity_display_name_a"].astype(str),
                frame["entity_display_name_b"].astype(str),
                strict=False,
            )
        ],
        index=frame.index,
    )


def _subset_matrix_for_plot(matrix: pd.DataFrame, *, max_labels: int = 70) -> pd.DataFrame:
    if len(matrix) <= max_labels:
        return matrix
    variability = matrix.abs().sum(axis=1, skipna=True).sort_values(ascending=False)
    labels = variability.head(max_labels).index
    return matrix.loc[labels, labels]


def _abbreviate_index(matrix: pd.DataFrame, *, max_label_length: int) -> pd.DataFrame:
    if matrix.empty:
        return matrix
    labels = [
        shorten(str(label), width=max_label_length, placeholder="...")
        for label in matrix.index
    ]
    result = matrix.copy()
    result.index = labels
    result.columns = labels
    return result


def plot_heatmap(
    matrix: pd.DataFrame,
    path: str | Path,
    *,
    title: str,
    cmap: str = "viridis",
    top_n: int | None = None,
    max_label_length: int = 55,
) -> None:
    if matrix.empty:
        write_placeholder_figure(path, title + "\nNo matrix values available")
        return
    plot_matrix = _subset_matrix_for_plot(matrix, max_labels=top_n or 70)
    plot_matrix = _abbreviate_index(plot_matrix, max_label_length=max_label_length)
    values = plot_matrix.to_numpy(dtype=float).copy()
    np.fill_diagonal(values, 0.0)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        write_placeholder_figure(path, title + "\nNo finite matrix values available")
        return
    if "delta" in title.lower():
        vmax = max(float(np.nanmax(np.abs(finite))), 0.01)
        vmin = -vmax
        cmap = "coolwarm"
    else:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanpercentile(finite, 98))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
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


def plot_triptych(
    initial_matrix: pd.DataFrame,
    final_matrix: pd.DataFrame,
    delta_matrix: pd.DataFrame,
    path: str | Path,
    *,
    title: str,
    top_n: int | None = None,
    max_label_length: int = 55,
) -> None:
    if initial_matrix.empty or final_matrix.empty or delta_matrix.empty:
        write_placeholder_figure(path, title + "\nTriptych matrix values unavailable")
        return
    labels = list(delta_matrix.index)
    if top_n is not None and len(labels) > top_n:
        dynamic = delta_matrix.abs().sum(axis=1, skipna=True).sort_values(ascending=False)
        labels = dynamic.head(top_n).index.astype(str).tolist()
    panels = [
        initial_matrix.reindex(index=labels, columns=labels),
        final_matrix.reindex(index=labels, columns=labels),
        delta_matrix.reindex(index=labels, columns=labels),
    ]
    panels = [_abbreviate_index(panel, max_label_length=max_label_length) for panel in panels]
    initial_final_values = np.concatenate(
        [
            panels[0].to_numpy(dtype=float).ravel(),
            panels[1].to_numpy(dtype=float).ravel(),
        ]
    )
    initial_final_values = initial_final_values[np.isfinite(initial_final_values)]
    if initial_final_values.size == 0:
        write_placeholder_figure(path, title + "\nNo finite initial/final values")
        return
    distance_vmin = float(np.nanmin(initial_final_values))
    distance_vmax = float(np.nanmax(initial_final_values))
    if np.isclose(distance_vmin, distance_vmax):
        distance_vmax = distance_vmin + 1.0
    delta_values = panels[2].to_numpy(dtype=float).copy()
    np.fill_diagonal(delta_values, 0.0)
    delta_finite = delta_values[np.isfinite(delta_values)]
    delta_vmax = max(float(np.nanmax(np.abs(delta_finite))), 0.01) if delta_finite.size else 0.01
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), dpi=230)
    subtitles = ["Initial distance", "Final distance", "Delta distance"]
    cmaps = ["viridis", "viridis", "coolwarm"]
    for idx, ax in enumerate(axes):
        values = panels[idx].to_numpy(dtype=float).copy()
        np.fill_diagonal(values, 0.0)
        if idx < 2:
            image = ax.imshow(values, aspect="auto", cmap=cmaps[idx], vmin=distance_vmin, vmax=distance_vmax)
        else:
            image = ax.imshow(values, aspect="auto", cmap=cmaps[idx], vmin=-delta_vmax, vmax=delta_vmax)
        ax.set_title(subtitles[idx], fontsize=10)
        show_labels = len(panels[idx]) <= 35
        ax.set_xticks(np.arange(len(panels[idx])))
        ax.set_yticks(np.arange(len(panels[idx])))
        ax.set_xticklabels(panels[idx].columns if show_labels else [], rotation=90, fontsize=5)
        ax.set_yticklabels(panels[idx].index if show_labels else [], fontsize=5)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_top_pairs(
    changes: pd.DataFrame,
    path: str | Path,
    *,
    level: str,
    top_n: int,
    max_label_length: int = 55,
) -> None:
    subset = changes.loc[(changes["level"] == level) & changes["delta_distance"].notna()].copy()
    if subset.empty:
        write_placeholder_figure(path, f"No semantic distance changes available for {level}")
        return
    converging = subset.loc[subset["delta_distance"] < 0].sort_values(
        ["delta_distance", "pair_id"],
        ascending=[True, True],
        kind="mergesort",
    ).head(top_n)
    diverging = subset.loc[subset["delta_distance"] > 0].sort_values(
        ["delta_distance", "pair_id"],
        ascending=[False, True],
        kind="mergesort",
    ).head(top_n)
    plot_frame = pd.concat([converging, diverging], ignore_index=True)
    if plot_frame.empty:
        write_placeholder_figure(path, f"No non-zero signed semantic distance changes for {level}")
        return
    plot_frame["label"] = [
        "\n".join(wrap(label, width=44))
        for label in _pair_label(plot_frame, max_length=max_label_length)
    ]
    plot_frame = plot_frame.iloc[::-1]
    colors = np.where(plot_frame["delta_distance"].to_numpy(dtype=float) < 0, "#2f6f73", "#b45432")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, max(6, 0.42 * len(plot_frame))), dpi=230)
    ax.barh(plot_frame["label"], plot_frame["delta_distance"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title(f"Top Semantic Convergence And Divergence Pairs ({level})", fontsize=12)
    ax.set_xlabel("Delta cosine distance, final window minus initial window")
    ax.tick_params(axis="y", labelsize=6)
    ax.text(
        0.01,
        0.01,
        "Negative = semantic convergence. Positive = semantic divergence.",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_subfield_network(
    changes: pd.DataFrame,
    path: str | Path,
    *,
    top_n: int = 20,
    max_label_length: int = 35,
) -> str | None:
    subset = changes.loc[(changes["level"] == "subfield") & changes["delta_distance"].notna()].copy()
    if subset.empty:
        write_placeholder_figure(path, "No subfield semantic distance changes available")
        return "No subfield semantic distance changes available for network plot."
    converging = subset.loc[subset["delta_distance"] < 0].sort_values(
        ["delta_distance", "pair_id"],
        ascending=[True, True],
        kind="mergesort",
    ).head(top_n)
    diverging = subset.loc[subset["delta_distance"] > 0].sort_values(
        ["delta_distance", "pair_id"],
        ascending=[False, True],
        kind="mergesort",
    ).head(top_n)
    edges = pd.concat([converging, diverging], ignore_index=True)
    if edges.empty:
        write_placeholder_figure(path, "No signed subfield changes available for network plot")
        return "No signed subfield changes available for network plot."
    try:
        import networkx as nx
    except ImportError:
        return "networkx is unavailable; skipped subfield convergence/divergence network."

    graph = nx.Graph()
    for row in edges.itertuples(index=False):
        a = str(row.entity_display_name_a)
        b = str(row.entity_display_name_b)
        graph.add_edge(a, b, delta=float(row.delta_distance))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 10), dpi=230)
    pos = nx.spring_layout(graph, seed=42, weight=None)
    deltas = np.asarray([data["delta"] for _, _, data in graph.edges(data=True)], dtype=float)
    max_abs = max(float(np.nanmax(np.abs(deltas))), 0.01)
    widths = 0.8 + 4.0 * (np.abs(deltas) / max_abs)
    colors = ["#2f6f73" if delta < 0 else "#b45432" for delta in deltas]
    nx.draw_networkx_edges(graph, pos, ax=ax, width=widths, edge_color=colors, alpha=0.75)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=220, node_color="#f6f4ef", edgecolors="#333333", linewidths=0.5)
    labels = {
        node: shorten(str(node), width=max_label_length, placeholder="...")
        for node in graph.nodes
    }
    nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=6)
    ax.set_title("Extreme Subfield Semantic Convergence And Divergence", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return None


def movement_metrics_by_level(path_metrics: pd.DataFrame, *, level: str) -> pd.DataFrame:
    if path_metrics.empty:
        return pd.DataFrame()
    if level == "subfield":
        id_col = "subfield_id"
        name_col = "subfield_display_name"
        required = {id_col, name_col, "total_path_length", "early_late_drift", "n_windows_available"}
        if not required.issubset(path_metrics.columns):
            return pd.DataFrame()
        result = path_metrics[list(required)].copy()
    elif level in {"field", "domain"}:
        id_col = f"{level}_id"
        name_col = f"{level}_display_name"
        required = {id_col, name_col, "total_path_length", "early_late_drift", "n_windows_available"}
        if not required.issubset(path_metrics.columns):
            return pd.DataFrame()
        result = (
            path_metrics.groupby([id_col, name_col], dropna=False)
            .agg(
                total_path_length=("total_path_length", "median"),
                early_late_drift=("early_late_drift", "median"),
                n_windows_available=("n_windows_available", "median"),
            )
            .reset_index()
        )
    else:
        raise ValueError("level must be subfield, field, or domain")
    return result.rename(
        columns={
            id_col: "entity_id",
            name_col: "entity_display_name",
            "total_path_length": "path_length",
            "early_late_drift": "early_late_drift",
        }
    )


def _driver_note(path_a: float, path_b: float, *, median_path: float) -> str:
    if not np.isfinite(path_a) or not np.isfinite(path_b) or not np.isfinite(median_path):
        return "movement_unavailable"
    high_a = path_a > median_path
    high_b = path_b > median_path
    if high_a and high_b:
        return "both_move"
    if path_a >= 1.5 * max(path_b, EPS):
        return "mostly_entity_a"
    if path_b >= 1.5 * max(path_a, EPS):
        return "mostly_entity_b"
    return "small_or_balanced_movement"


def add_driver_diagnostics(
    pairs: pd.DataFrame,
    *,
    path_metrics: pd.DataFrame | None,
) -> tuple[pd.DataFrame, bool]:
    if pairs.empty:
        return pairs, bool(path_metrics is not None and not path_metrics.empty)
    if path_metrics is None or path_metrics.empty:
        return pairs, False
    frames: list[pd.DataFrame] = []
    merged_any = False
    for level, group in pairs.groupby("level", sort=False):
        movement = movement_metrics_by_level(path_metrics, level=str(level))
        if movement.empty:
            frames.append(group)
            continue
        median_path = float(pd.to_numeric(movement["path_length"], errors="coerce").median(skipna=True))
        left = movement.rename(
            columns={
                "entity_id": "entity_id_a",
                "path_length": "entity_a_path_length",
                "early_late_drift": "entity_a_early_late_drift",
                "n_windows_available": "entity_a_n_windows_available",
            }
        ).drop(columns=["entity_display_name"], errors="ignore")
        right = movement.rename(
            columns={
                "entity_id": "entity_id_b",
                "path_length": "entity_b_path_length",
                "early_late_drift": "entity_b_early_late_drift",
                "n_windows_available": "entity_b_n_windows_available",
            }
        ).drop(columns=["entity_display_name"], errors="ignore")
        merged = group.merge(left, on="entity_id_a", how="left").merge(right, on="entity_id_b", how="left")
        merged["driver_note"] = [
            _driver_note(a, b, median_path=median_path)
            for a, b in zip(
                pd.to_numeric(merged["entity_a_path_length"], errors="coerce"),
                pd.to_numeric(merged["entity_b_path_length"], errors="coerce"),
                strict=False,
            )
        ]
        frames.append(merged)
        merged_any = True
    return pd.concat(frames, ignore_index=True) if frames else pairs, merged_any


def write_semantic_distance_outputs(
    *,
    pair_distances: pd.DataFrame,
    changes: pd.DataFrame,
    output_dir: str | Path,
    root: Path,
    levels: list[str],
    windows: list[TemporalWindow],
    top_n_pairs: int,
    heatmap_top_n: int,
    max_label_length: int,
    aggregation_mode: str,
    input_paths: dict[str, str],
) -> dict[str, Any]:
    paths = output_paths(output_dir, root=root)
    for path in paths.values():
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(pair_distances, paths["pair_distances_parquet"])
    save_parquet(changes, paths["pair_changes_parquet"])
    path_metrics_path = root / DEFAULT_PATH_METRICS
    path_metrics = pd.read_parquet(path_metrics_path) if path_metrics_path.exists() else None
    driver_warning = "" if path_metrics is not None else (
        f"Driver diagnostics skipped because {display_path(path_metrics_path, root=root)} was not found."
    )

    raw_top_converging = top_converging_diverging_pairs(
        changes,
        top_n=top_n_pairs,
        direction="converging",
    )
    raw_top_diverging = top_converging_diverging_pairs(
        changes,
        top_n=top_n_pairs,
        direction="diverging",
    )
    top_converging, converging_drivers = add_driver_diagnostics(
        raw_top_converging,
        path_metrics=path_metrics,
    )
    top_diverging, diverging_drivers = add_driver_diagnostics(
        raw_top_diverging,
        path_metrics=path_metrics,
    )
    driver_diagnostics_merged = bool(converging_drivers or diverging_drivers)
    most_negative = most_extreme_delta_pairs(changes, top_n=top_n_pairs, sign="negative")
    most_positive = most_extreme_delta_pairs(changes, top_n=top_n_pairs, sign="positive")
    top_converging.to_csv(paths["top_converging_csv"], index=False)
    top_diverging.to_csv(paths["top_diverging_csv"], index=False)
    most_negative.to_csv(paths["most_negative_delta_csv"], index=False)
    most_positive.to_csv(paths["most_positive_delta_csv"], index=False)
    top_converging.loc[top_converging["level"] == "subfield"].to_csv(
        paths["subfield_top_converging_csv"],
        index=False,
    )
    top_diverging.loc[top_diverging["level"] == "subfield"].to_csv(
        paths["subfield_top_diverging_csv"],
        index=False,
    )
    matrix_paths = save_distance_matrices(
        pair_distances,
        changes,
        output_dir=output_dir,
        levels=levels,
    )

    first_label = windows[0].window_label
    last_label = windows[-1].window_label
    figure_paths: dict[str, Path] = {}
    for level in levels:
        level_pairs = pair_distances.loc[pair_distances["level"] == level]
        initial = level_pairs.loc[level_pairs["window_label"] == first_label]
        final = level_pairs.loc[level_pairs["window_label"] == last_label]
        level_changes = changes.loc[changes["level"] == level]
        initial_matrix = matrix_from_pairs(initial, value_column="cosine_distance")
        final_matrix = matrix_from_pairs(final, value_column="cosine_distance")
        delta_matrix = matrix_from_pairs(level_changes, value_column="delta_distance")
        plot_heatmap(
            initial_matrix,
            Path(output_dir) / f"{level}_initial_distance_heatmap.png",
            title=f"{level.title()} Semantic Distances: {first_label}",
            max_label_length=max_label_length,
        )
        plot_heatmap(
            final_matrix,
            Path(output_dir) / f"{level}_final_distance_heatmap.png",
            title=f"{level.title()} Semantic Distances: {last_label}",
            max_label_length=max_label_length,
        )
        plot_heatmap(
            delta_matrix,
            Path(output_dir) / f"{level}_distance_delta_heatmap.png",
            title=f"{level.title()} Semantic Distance Change, {last_label} minus {first_label}",
            top_n=heatmap_top_n if level == "subfield" else None,
            max_label_length=max_label_length if level != "subfield" else min(max_label_length, 35),
        )
        figure_paths[f"{level}_initial_distance_heatmap"] = Path(output_dir) / f"{level}_initial_distance_heatmap.png"
        figure_paths[f"{level}_final_distance_heatmap"] = Path(output_dir) / f"{level}_final_distance_heatmap.png"
        figure_paths[f"{level}_distance_delta_heatmap"] = Path(output_dir) / f"{level}_distance_delta_heatmap.png"
        plot_top_pairs(
            changes,
            Path(output_dir) / f"{level}_top_converging_diverging_pairs.png",
            level=level,
            top_n=min(top_n_pairs, 20),
            max_label_length=max_label_length,
        )
        plot_top_pairs(
            changes,
            Path(output_dir) / f"{level}_top_semantic_converging_diverging_pairs_clean.png",
            level=level,
            top_n=top_n_pairs,
            max_label_length=max_label_length,
        )
        figure_paths[f"{level}_top_semantic_converging_diverging_pairs_clean"] = (
            Path(output_dir) / f"{level}_top_semantic_converging_diverging_pairs_clean.png"
        )
        if level in {"domain", "field"}:
            triptych_path = Path(output_dir) / f"{level}_distance_initial_final_delta_triptych.png"
            plot_triptych(
                initial_matrix,
                final_matrix,
                delta_matrix,
                triptych_path,
                title=f"{level.title()} Semantic Distances: initial, final, and change",
                max_label_length=max_label_length,
            )
            figure_paths[f"{level}_distance_initial_final_delta_triptych"] = triptych_path
        elif level == "subfield":
            triptych_path = Path(output_dir) / "subfield_distance_initial_final_delta_triptych_topN.png"
            plot_triptych(
                initial_matrix,
                final_matrix,
                delta_matrix,
                triptych_path,
                title=f"Subfield Semantic Distances: initial, final, and change (top {heatmap_top_n})",
                top_n=heatmap_top_n,
                max_label_length=min(max_label_length, 35),
            )
            figure_paths["subfield_distance_initial_final_delta_triptych_topN"] = triptych_path

    network_warning = None
    if "subfield" in levels:
        network_path = Path(output_dir) / "subfield_semantic_convergence_divergence_network.png"
        network_warning = plot_subfield_network(
            changes,
            network_path,
            top_n=20,
            max_label_length=35,
        )
        if network_warning is None:
            figure_paths["subfield_semantic_convergence_divergence_network"] = network_path

    output_paths_payload = {key: display_path(path, root=root) for key, path in paths.items()}
    output_paths_payload.update(
        {key: display_path(path, root=root) for key, path in matrix_paths.items()}
    )
    output_paths_payload.update(
        {key: display_path(path, root=root) for key, path in figure_paths.items()}
    )
    qa_checks = qa_checks_by_level(changes)
    direct_paper_aggregation_run = aggregation_mode == "direct_paper_centroids" and False
    warnings = [
        warning
        for warning in [
            driver_warning,
            network_warning,
            (
                "direct_paper_centroids was requested, but direct paper-level aggregation "
                "is currently a TODO; weighted_subfield_centroids outputs were written."
                if aggregation_mode == "direct_paper_centroids"
                else ""
            ),
        ]
        if warning
    ]
    summary = {
        "created_at": utc_now_iso(),
        "input_paths": input_paths,
        "output_paths": output_paths_payload,
        "levels": levels,
        "aggregation_mode_requested": aggregation_mode,
        "aggregation_mode_used": "weighted_subfield_centroids",
        "direct_paper_level_aggregation_sensitivity_run": direct_paper_aggregation_run,
        "driver_diagnostics_merged": driver_diagnostics_merged,
        "driver_diagnostics_rule": (
            "Path movement uses total_path_length from script 20. Movement is high "
            "when above the level median; a 1.5x path-length ratio marks a mostly "
            "single-entity driver when both entities are not high."
        ),
        "warnings": warnings,
        "windows": [
            {"window_start": window.window_start, "window_end": window.window_end}
            for window in windows
        ],
        "n_pair_distance_rows": int(len(pair_distances)),
        "n_pair_change_rows": int(len(changes)),
        "qa_checks_by_level": qa_checks,
        "top_converging_pairs": dataframe_to_records(top_converging, limit=20),
        "top_diverging_pairs": dataframe_to_records(top_diverging, limit=20),
        "recommended_thesis_outputs": [
            output_paths_payload.get("domain_distance_delta_heatmap", ""),
            output_paths_payload.get("field_distance_delta_heatmap", ""),
            output_paths_payload.get("subfield_distance_delta_heatmap", ""),
            output_paths_payload.get("domain_distance_initial_final_delta_triptych", ""),
            output_paths_payload.get("field_distance_initial_final_delta_triptych", ""),
            output_paths_payload.get("top_converging_csv", ""),
            output_paths_payload.get("top_diverging_csv", ""),
        ],
        "method_notes": [
            "Semantic distance compares centroid positions in the original embedding space.",
            "It measures topic/content proximity, not internal morphological similarity.",
            "Negative delta_distance means convergence; positive means divergence.",
        ],
    }
    write_json(paths["summary_json"], summary)
    write_text(paths["summary_md"], markdown_summary(summary))
    return summary


def markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Temporal Semantic Distances",
        "",
        "This analysis compares where disciplines/subfields sit in semantic embedding space over time.",
        "",
        "## Interpretation",
        "",
        "- Semantic distance compares centroid positions in the original embedding space.",
        "- It measures topic/content proximity, not internal morphological similarity.",
        "- `delta_distance = final_distance - initial_distance`.",
        "- Negative delta means semantic convergence; positive delta means semantic divergence.",
        "",
        "## Methods",
        "",
        "- Distances are cosine distances between centroid vectors in the original normalized embedding space.",
        f"- Field/domain aggregation mode used: `{summary['aggregation_mode_used']}`.",
        f"- Aggregation mode requested: `{summary['aggregation_mode_requested']}`.",
        "- Full subfield matrices are saved even where heatmaps are too dense for easy reading.",
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
            f"- Pair-distance rows: {summary['n_pair_distance_rows']}",
            f"- Pair-change rows: {summary['n_pair_change_rows']}",
            f"- Driver diagnostics merged: {'yes' if summary['driver_diagnostics_merged'] else 'no'}",
            (
                "- Direct paper-level aggregation sensitivity run: "
                f"{'yes' if summary['direct_paper_level_aggregation_sensitivity_run'] else 'no'}"
            ),
            "",
            "## QA Checks",
            "",
        ]
    )
    for level, checks in summary.get("qa_checks_by_level", {}).items():
        lines.append(
            "- {level}: pairs={pairs}, converging={conv}, diverging={div}, "
            "stable_zero={stable}, min_delta={min_delta}, max_delta={max_delta}".format(
                level=level,
                pairs=checks["n_pair_changes"],
                conv=checks["n_converging_pairs"],
                div=checks["n_diverging_pairs"],
                stable=checks["n_stable_zero_delta_pairs"],
                min_delta=checks["min_delta_distance"],
                max_delta=checks["max_delta_distance"],
            )
        )
    lines.extend(
        [
            "",
            "## Main Results",
            "",
            "Top semantic convergences and divergences are written separately; the primary signed tables never mix opposite signs.",
            "",
            "### Top Convergences",
            "",
        ]
    )
    for row in summary.get("top_converging_pairs", [])[:10]:
        lines.append(
            "- {level}: {a} <-> {b}, delta={delta:.6f}".format(
                level=row.get("level"),
                a=row.get("entity_display_name_a"),
                b=row.get("entity_display_name_b"),
                delta=float(row.get("delta_distance") or 0.0),
            )
        )
    lines.extend(["", "### Top Divergences", ""])
    for row in summary.get("top_diverging_pairs", [])[:10]:
        lines.append(
            "- {level}: {a} <-> {b}, delta={delta:.6f}".format(
                level=row.get("level"),
                a=row.get("entity_display_name_a"),
                b=row.get("entity_display_name_b"),
                delta=float(row.get("delta_distance") or 0.0),
            )
        )
    domain_checks = summary.get("qa_checks_by_level", {}).get("domain", {})
    field_checks = summary.get("qa_checks_by_level", {}).get("field", {})
    lines.extend(
        [
            "",
            "### Domain-Level Change Summary",
            "",
            (
                "- Domain deltas range from "
                f"{domain_checks.get('min_delta_distance')} to {domain_checks.get('max_delta_distance')}."
                if domain_checks
                else "- Domain-level changes were not computed."
            ),
            "- Domain-level changes can be small; interpret tiny delta magnitudes cautiously.",
            "",
            "### Field-Level Notable Patterns",
            "",
            (
                "- Field deltas range from "
                f"{field_checks.get('min_delta_distance')} to {field_checks.get('max_delta_distance')}."
                if field_checks
                else "- Field-level changes were not computed."
            ),
            "",
            "## Recommended Thesis Outputs",
            "",
        ]
    )
    for path in summary.get("recommended_thesis_outputs", []):
        if path:
            lines.append(f"- `{path}`")
    if summary.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in summary["warnings"]:
            lines.append(f"- {warning}")
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Field/domain centroids are aggregate summaries and may hide subfield-level heterogeneity.",
            "- Weighted subfield-centroid aggregation is not identical to direct paper-level aggregation.",
            "- Full subfield matrices are saved but too dense for direct visual interpretation.",
            "- Semantic distance is separate from morphological distance, which compares internal metric profiles.",
            "",
        ]
    )
    return "\n".join(lines)


def build_semantic_distance_outputs(
    centroids: pd.DataFrame,
    *,
    levels: list[str],
    output_dir: str | Path,
    root: Path,
    top_n_pairs: int,
    input_paths: dict[str, str],
    heatmap_top_n: int = 40,
    max_label_length: int = 55,
    aggregation_mode: str = "weighted_subfield_centroids",
    year_min: int = 2000,
    year_max: int = 2024,
    window_size: int = 5,
) -> dict[str, Any]:
    windows = default_or_computed_windows(
        year_min=year_min,
        year_max=year_max,
        window_size=window_size,
    )
    level_frames = [
        aggregate_centroids_for_level(centroids, level=level)
        for level in levels
    ]
    all_level_centroids = pd.concat(level_frames, ignore_index=True)
    pair_distances = pairwise_semantic_distances(all_level_centroids)
    changes = semantic_distance_changes(pair_distances, windows=windows)
    return write_semantic_distance_outputs(
        pair_distances=pair_distances,
        changes=changes,
        output_dir=output_dir,
        root=root,
        levels=levels,
        windows=windows,
        top_n_pairs=top_n_pairs,
        heatmap_top_n=heatmap_top_n,
        max_label_length=max_label_length,
        aggregation_mode=aggregation_mode,
        input_paths=input_paths,
    )
