from __future__ import annotations

from pathlib import Path
from textwrap import wrap
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from src.storage import save_parquet
from src.temporal_common import (
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
    "summary_md": "summary.md",
    "summary_json": "summary.json",
}


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
            ]
        )
    ensure_outputs_do_not_exist(
        [
            paths["pair_distances_parquet"],
            paths["pair_changes_parquet"],
            paths["top_converging_csv"],
            paths["top_diverging_csv"],
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
            selected = finite.sort_values(
                ["delta_distance", "pair_id"],
                ascending=[True, True],
                kind="mergesort",
            ).head(top_n)
        elif direction == "diverging":
            selected = finite.sort_values(
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


def _subset_matrix_for_plot(matrix: pd.DataFrame, *, max_labels: int = 70) -> pd.DataFrame:
    if len(matrix) <= max_labels:
        return matrix
    variability = matrix.abs().sum(axis=1, skipna=True).sort_values(ascending=False)
    labels = variability.head(max_labels).index
    return matrix.loc[labels, labels]


def plot_heatmap(matrix: pd.DataFrame, path: str | Path, *, title: str, cmap: str = "viridis") -> None:
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


def plot_top_pairs(changes: pd.DataFrame, path: str | Path, *, level: str, top_n: int) -> None:
    subset = changes.loc[(changes["level"] == level) & changes["delta_distance"].notna()].copy()
    if subset.empty:
        write_placeholder_figure(path, f"No semantic distance changes available for {level}")
        return
    converging = subset.sort_values("delta_distance", kind="mergesort").head(top_n)
    diverging = subset.sort_values("delta_distance", ascending=False, kind="mergesort").head(top_n)
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
    colors = np.where(plot_frame["delta_distance"].to_numpy(dtype=float) < 0, "#2f6f73", "#b45432")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, max(6, 0.42 * len(plot_frame))), dpi=230)
    ax.barh(plot_frame["label"], plot_frame["delta_distance"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title(f"Top Semantic Convergence And Divergence Pairs ({level})", fontsize=12)
    ax.set_xlabel("Delta cosine distance, final window minus initial window")
    ax.tick_params(axis="y", labelsize=6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_semantic_distance_outputs(
    *,
    pair_distances: pd.DataFrame,
    changes: pd.DataFrame,
    output_dir: str | Path,
    root: Path,
    levels: list[str],
    windows: list[TemporalWindow],
    top_n_pairs: int,
    input_paths: dict[str, str],
) -> dict[str, Any]:
    paths = output_paths(output_dir, root=root)
    for path in paths.values():
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(pair_distances, paths["pair_distances_parquet"])
    save_parquet(changes, paths["pair_changes_parquet"])
    top_converging = top_converging_diverging_pairs(
        changes,
        top_n=top_n_pairs,
        direction="converging",
    )
    top_diverging = top_converging_diverging_pairs(
        changes,
        top_n=top_n_pairs,
        direction="diverging",
    )
    top_converging.to_csv(paths["top_converging_csv"], index=False)
    top_diverging.to_csv(paths["top_diverging_csv"], index=False)
    matrix_paths = save_distance_matrices(
        pair_distances,
        changes,
        output_dir=output_dir,
        levels=levels,
    )

    first_label = windows[0].window_label
    last_label = windows[-1].window_label
    for level in levels:
        level_pairs = pair_distances.loc[pair_distances["level"] == level]
        initial = level_pairs.loc[level_pairs["window_label"] == first_label]
        final = level_pairs.loc[level_pairs["window_label"] == last_label]
        level_changes = changes.loc[changes["level"] == level]
        plot_heatmap(
            matrix_from_pairs(initial, value_column="cosine_distance"),
            Path(output_dir) / f"{level}_initial_distance_heatmap.png",
            title=f"{level.title()} Semantic Distances: {first_label}",
        )
        plot_heatmap(
            matrix_from_pairs(final, value_column="cosine_distance"),
            Path(output_dir) / f"{level}_final_distance_heatmap.png",
            title=f"{level.title()} Semantic Distances: {last_label}",
        )
        plot_heatmap(
            matrix_from_pairs(level_changes, value_column="delta_distance"),
            Path(output_dir) / f"{level}_distance_delta_heatmap.png",
            title=f"{level.title()} Semantic Distance Delta",
        )
        plot_top_pairs(
            changes,
            Path(output_dir) / f"{level}_top_converging_diverging_pairs.png",
            level=level,
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
        "windows": [
            {"window_start": window.window_start, "window_end": window.window_end}
            for window in windows
        ],
        "n_pair_distance_rows": int(len(pair_distances)),
        "n_pair_change_rows": int(len(changes)),
        "top_converging_pairs": dataframe_to_records(top_converging, limit=20),
        "top_diverging_pairs": dataframe_to_records(top_diverging, limit=20),
        "method_notes": [
            "Semantic distance compares centroid positions in the original embedding space.",
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
        "## Methods",
        "",
        "- Distances are cosine distances between centroid vectors in the original normalized embedding space.",
        "- Negative final-minus-initial delta means semantic convergence. Positive delta means semantic divergence.",
        "- Field and domain centroids are weighted averages of subfield-window centroids using `n_used`; direct paper-level aggregation can be used as a future sensitivity check.",
        "- Full matrices are saved even where heatmaps are too dense for easy reading.",
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
            "",
            "## Caveat",
            "",
            "Semantic distance is about topic/content proximity. It is separate from morphological distance, which compares internal metric profiles.",
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
        input_paths=input_paths,
    )
