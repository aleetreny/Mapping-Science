from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from src.per_subfield_umap_maps import (
    coordinate_limits,
    plot_density_panel,
    safe_filename_part,
    stable_int_seed,
    validate_year_window,
)
from src.subfield_labels import add_subfield_label_columns
from src.umap_maps import UMAP_OUTPUT_COLUMNS

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


SUPPORTED_LEVELS = {"field", "domain"}

LEVEL_SPECS = {
    "field": {
        "id_column": "field_id",
        "name_column": "field_display_name",
        "output_dir": "outputs/maps/per_field_umap",
        "manifest_filename": "per_field_umap_manifest.parquet",
        "summary_filename": "per_field_umap_summary.json",
        "color_column": "subfield_display_name",
        "group_label": "field",
    },
    "domain": {
        "id_column": "domain_id",
        "name_column": "domain_display_name",
        "output_dir": "outputs/maps/per_domain_umap",
        "manifest_filename": "per_domain_umap_manifest.parquet",
        "summary_filename": "per_domain_umap_summary.json",
        "color_column": "field_display_name",
        "group_label": "domain",
    },
}

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
    "main_analysis_eligible_2500",
}

MANIFEST_COLUMNS = [
    "level",
    "group_id",
    "group_name",
    "n_available",
    "n_used",
    "year_min",
    "year_max",
    "status",
    "coordinate_path",
    "figure_path",
    "error_message",
    "umap_n_neighbors",
    "umap_min_dist",
    "umap_metric",
    "random_state",
    "max_papers_per_group",
    "sampling_applied",
    "color_by",
    "legend_included",
    "density_method",
]

VALID_STATUSES = {"completed", "skipped", "failed"}


@dataclass(frozen=True)
class CategoryManifestRow:
    level: str
    group_id: str
    group_name: str
    n_available: int
    n_used: int
    year_min: int
    year_max: int
    status: str
    coordinate_path: str
    figure_path: str
    error_message: str
    umap_n_neighbors: int
    umap_min_dist: float
    umap_metric: str
    random_state: int
    max_papers_per_group: int
    sampling_applied: bool
    color_by: str
    legend_included: bool
    density_method: str


def level_spec(level: str) -> dict[str, str]:
    normalized = str(level).strip().lower()
    if normalized not in SUPPORTED_LEVELS:
        raise ValueError(
            f"level must be one of {', '.join(sorted(SUPPORTED_LEVELS))}: {level}"
        )
    return LEVEL_SPECS[normalized]


def validate_category_index_columns(index: pd.DataFrame) -> None:
    missing = REQUIRED_INDEX_COLUMNS - set(index.columns)
    if missing:
        raise ValueError(
            "analysis_embedding_index.parquet is missing required columns: "
            f"{', '.join(sorted(missing))}"
        )


def filter_category_input_window(
    index: pd.DataFrame,
    *,
    year_min: int,
    year_max: int,
) -> pd.DataFrame:
    validate_category_index_columns(index)
    validate_year_window(year_min, year_max)

    years = pd.to_numeric(index["publication_year"], errors="coerce")
    main_mask = index["main_analysis_eligible_2500"].fillna(False).astype(bool)
    window_mask = years.between(year_min, year_max, inclusive="both")
    filtered = index.loc[main_mask & window_mask].copy()
    filtered["publication_year"] = years.loc[filtered.index].astype("int64")
    return (
        filtered.sort_values(
            [
                "domain_id",
                "field_id",
                "subfield_id",
                "publication_year",
                "work_id",
                "analysis_row_id",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def list_groups(index: pd.DataFrame, *, level: str) -> pd.DataFrame:
    validate_category_index_columns(index)
    spec = level_spec(level)
    id_column = spec["id_column"]
    name_column = spec["name_column"]
    groups = index[[id_column, name_column]].dropna(subset=[id_column]).copy()
    groups["_group_sort_key"] = groups[id_column].astype(str)
    groups = (
        groups.sort_values(["_group_sort_key", name_column], kind="mergesort")
        .drop_duplicates(id_column, keep="first")
        .drop(columns="_group_sort_key")
        .rename(columns={id_column: "group_id", name_column: "group_name"})
        .reset_index(drop=True)
    )
    return groups


def select_attempted_groups(
    groups: pd.DataFrame,
    *,
    group_id: str | None,
    limit_groups: int | None,
) -> pd.DataFrame:
    attempted = groups.copy()
    if group_id is not None:
        attempted = attempted[
            attempted["group_id"].astype(str) == str(group_id)
        ].reset_index(drop=True)
        if attempted.empty:
            raise ValueError(f"No group matched --group-id {group_id}")

    if limit_groups is not None:
        if limit_groups <= 0:
            raise ValueError("limit_groups must be positive when provided")
        attempted = attempted.head(limit_groups).reset_index(drop=True)
    return attempted


def make_group_lookup(index: pd.DataFrame, *, level: str) -> dict[str, pd.DataFrame]:
    spec = level_spec(level)
    id_column = spec["id_column"]
    return {
        str(group_id): group.reset_index(drop=True)
        for group_id, group in index.groupby(id_column, sort=False, dropna=True)
    }


def stable_sort_group_rows(rows: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [
        column
        for column in [
            "domain_id",
            "field_id",
            "subfield_id",
            "publication_year",
            "work_id",
            "analysis_row_id",
        ]
        if column in rows.columns
    ]
    return rows.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def sample_group_rows(
    rows: pd.DataFrame,
    *,
    max_papers: int,
    random_state: int,
    group_id: object,
) -> pd.DataFrame:
    if max_papers <= 0:
        raise ValueError("max_papers must be positive")
    ordered = stable_sort_group_rows(rows)
    if len(ordered) <= max_papers:
        return ordered

    rng = np.random.default_rng(stable_int_seed(random_state, group_id))
    selected_positions = np.sort(
        rng.choice(len(ordered), size=max_papers, replace=False)
    )
    return ordered.iloc[selected_positions].reset_index(drop=True)


def safe_group_stem(level: str, group_id: object, group_name: object) -> str:
    return (
        f"{safe_filename_part(level, max_length=30)}__"
        f"{safe_filename_part(group_id, max_length=50)}__"
        f"{safe_filename_part(group_name, max_length=120)}"
    )


def build_category_coordinate_frame(
    sampled_index: pd.DataFrame,
    coordinates: np.ndarray,
) -> pd.DataFrame:
    coordinates = np.asarray(coordinates)
    if coordinates.shape != (len(sampled_index), 2):
        raise ValueError(
            f"coordinates shape {coordinates.shape} does not match sampled rows "
            f"({len(sampled_index)}, 2)"
        )
    output = sampled_index.copy()
    output["umap_x"] = coordinates[:, 0].astype(np.float32)
    output["umap_y"] = coordinates[:, 1].astype(np.float32)
    output = add_subfield_label_columns(output)
    for column in UMAP_OUTPUT_COLUMNS:
        if column not in output.columns:
            output[column] = pd.NA
    extra_columns = [
        "subfield_label_unique",
        "subfield_label_short",
        "subfield_display_name_is_duplicated",
    ]
    return output[UMAP_OUTPUT_COLUMNS + extra_columns]


def category_manifest_row(
    *,
    level: str,
    group_id: object,
    group_name: object,
    n_available: int,
    n_used: int,
    year_min: int,
    year_max: int,
    status: str,
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
    random_state: int,
    max_papers_per_group: int,
    sampling_applied: bool,
    color_by: str,
    legend_included: bool = False,
    density_method: str = "",
    coordinate_path: str | Path | None = None,
    figure_path: str | Path | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    if status not in VALID_STATUSES:
        raise ValueError(
            f"status must be one of {', '.join(sorted(VALID_STATUSES))}: {status}"
        )
    row = CategoryManifestRow(
        level=str(level),
        group_id=str(group_id),
        group_name="" if pd.isna(group_name) else str(group_name),
        n_available=int(n_available),
        n_used=int(n_used),
        year_min=int(year_min),
        year_max=int(year_max),
        status=status,
        coordinate_path="" if coordinate_path is None else str(coordinate_path),
        figure_path="" if figure_path is None else str(figure_path),
        error_message="" if error_message is None else str(error_message),
        umap_n_neighbors=int(umap_n_neighbors),
        umap_min_dist=float(umap_min_dist),
        umap_metric=str(umap_metric),
        random_state=int(random_state),
        max_papers_per_group=int(max_papers_per_group),
        sampling_applied=bool(sampling_applied),
        color_by=str(color_by),
        legend_included=bool(legend_included),
        density_method=str(density_method),
    )
    return asdict(row)


def category_manifest_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    frame = pd.DataFrame(rows)
    for column in MANIFEST_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[MANIFEST_COLUMNS]


def color_codes(
    frame: pd.DataFrame,
    color_column: str,
    *,
    max_legend_categories: int,
) -> tuple[np.ndarray | str, list[str], bool]:
    if color_column not in frame.columns:
        return "#26547c", [], False
    values = frame[color_column].fillna("Unknown").astype(str)
    categories = pd.Categorical(values)
    labels = [str(label) for label in categories.categories]
    if len(labels) <= 1 or len(labels) > max_legend_categories:
        return "#26547c", [], False
    return categories.codes, labels, True


def plot_category_panels(
    coordinate_frame: pd.DataFrame,
    coordinates: np.ndarray,
    *,
    level: str,
    group_name: str,
    n_used: int,
    year_min: int,
    year_max: int,
    color_column: str,
    output_path: str | Path,
    dpi: int,
    max_legend_categories: int = 25,
) -> dict[str, Any]:
    coordinates = np.asarray(coordinates, dtype=float)
    xlim, ylim = coordinate_limits(coordinates)
    colors, labels, legend_included = color_codes(
        coordinate_frame,
        color_column,
        max_legend_categories=max_legend_categories,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)
    try:
        scatter = axes[0].scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=colors,
            cmap="tab20" if legend_included else None,
            s=2,
            alpha=0.45,
            linewidths=0,
            rasterized=True,
        )
        axes[0].set_xlim(xlim)
        axes[0].set_ylim(ylim)
        axes[0].set_xlabel("UMAP 1")
        axes[0].set_ylabel("UMAP 2")
        axes[0].set_title(
            f"A. Scatter - {level}: {group_name}\n"
            f"n={n_used:,}, years {year_min}-{year_max}",
            fontsize=10,
        )
        if legend_included:
            color_values = np.linspace(0, 1, max(len(labels), 1))
            handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=scatter.cmap(color_value),
                    markersize=4,
                    label=label,
                )
                for label, color_value in zip(labels, color_values)
            ]
            axes[0].legend(
                handles=handles,
                title=color_column,
                loc="best",
                fontsize=5,
                title_fontsize=6,
                frameon=False,
                markerscale=1.0,
            )

        density_artist, density_method = plot_density_panel(
            axes[1],
            coordinates,
            xlim=xlim,
            ylim=ylim,
        )
        axes[1].set_title("B. Density", fontsize=10)
        fig.colorbar(density_artist, ax=axes[1], fraction=0.046, pad=0.04)
        for ax in axes:
            ax.grid(False)
        fig.tight_layout()
        fig.savefig(output_path)
        return {
            "density_method": density_method,
            "legend_included": legend_included,
            "color_by": color_column if legend_included else "",
        }
    finally:
        plt.close(fig)
