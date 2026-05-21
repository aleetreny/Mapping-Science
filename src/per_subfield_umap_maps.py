from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from src.embeddings import MAIN_ANALYSIS_FLAG, normalize_eligibility_flags
from src.subfield_labels import add_subfield_label_columns
from src.umap_maps import UMAP_OUTPUT_COLUMNS

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


REQUIRED_INDEX_COLUMNS = {
    "analysis_row_id",
    "work_id",
    "subfield_id",
    "subfield_display_name",
    "publication_year",
}

MANIFEST_COLUMNS = [
    "subfield_id",
    "subfield_name",
    "field_id",
    "field_display_name",
    "domain_id",
    "domain_display_name",
    "subfield_label_unique",
    "subfield_label_short",
    "subfield_display_name_is_duplicated",
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
    "max_papers_per_subfield",
    "sampling_applied",
]

VALID_DENSITY_METHODS = {"auto", "kde", "smooth_hist"}
DEFAULT_DENSITY_METHOD = "smooth_hist"
DEFAULT_DENSITY_GRID_SIZE = 150
DEFAULT_DENSITY_SIGMA = 3.0
DEFAULT_DENSITY_VMAX_PERCENTILE = 99.0

VALID_STATUSES = {"completed", "skipped", "failed"}


def safe_filename_part(value: object, max_length: int = 100) -> str:
    if value is None or pd.isna(value):
        text = "unknown"
    else:
        text = str(value).strip()

    text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    if not text:
        text = "unknown"
    if len(text) > max_length:
        text = text[:max_length].rstrip("._-") or "unknown"
    return text


def safe_subfield_stem(subfield_id: object, subfield_name: object) -> str:
    return (
        f"{safe_filename_part(subfield_id, max_length=50)}"
        f"__{safe_filename_part(subfield_name, max_length=120)}"
    )


def validate_year_window(year_min: int, year_max: int) -> None:
    if year_min > year_max:
        raise ValueError(f"year_min ({year_min}) must be <= year_max ({year_max})")


def validate_index_columns(index: pd.DataFrame) -> None:
    missing = REQUIRED_INDEX_COLUMNS - set(index.columns)
    if missing:
        formatted = ", ".join(sorted(missing))
        raise ValueError(
            "analysis_embedding_index.parquet is missing required columns: "
            f"{formatted}. publication_year is required so per-subfield maps "
            "use the morphology input window instead of silently using all years."
        )
    try:
        normalize_eligibility_flags(index, require_robustness=False)
    except ValueError as exc:
        raise ValueError(
            "analysis_embedding_index.parquet eligibility flags could not be resolved: "
            f"{exc}"
        ) from exc


def stable_sort_work_rows(rows: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [
        column
        for column in ["publication_year", "work_id", "analysis_row_id"]
        if column in rows.columns
    ]
    if not sort_columns:
        return rows.copy()
    return rows.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def main_analysis_subfields(index: pd.DataFrame) -> pd.DataFrame:
    validate_index_columns(index)
    index = normalize_eligibility_flags(index, require_robustness=False)
    main_mask = index[MAIN_ANALYSIS_FLAG].fillna(False).astype(bool)
    columns = [
        "subfield_id",
        "subfield_display_name",
        "field_id",
        "field_display_name",
        "domain_id",
        "domain_display_name",
    ]
    main = index.loc[main_mask, columns].copy()
    main["_subfield_sort_key"] = main["subfield_id"].astype(str)
    subfields = (
        main.sort_values(["_subfield_sort_key", "subfield_display_name"], kind="mergesort")
        .drop_duplicates("subfield_id", keep="first")
        .drop(columns="_subfield_sort_key")
        .reset_index(drop=True)
    )
    return add_subfield_label_columns(subfields)


def filter_input_window(
    index: pd.DataFrame,
    *,
    year_min: int,
    year_max: int,
) -> pd.DataFrame:
    validate_index_columns(index)
    validate_year_window(year_min, year_max)

    index = normalize_eligibility_flags(index, require_robustness=False)
    years = pd.to_numeric(index["publication_year"], errors="coerce")
    main_mask = index[MAIN_ANALYSIS_FLAG].fillna(False).astype(bool)
    window_mask = years.between(year_min, year_max, inclusive="both")
    filtered = index.loc[main_mask & window_mask].copy()
    filtered["publication_year"] = years.loc[filtered.index].astype("int64")
    filtered["_subfield_sort_key"] = filtered["subfield_id"].astype(str)
    filtered = (
        filtered.sort_values(
            ["_subfield_sort_key", "publication_year", "work_id", "analysis_row_id"],
            kind="mergesort",
        )
        .drop(columns="_subfield_sort_key")
        .reset_index(drop=True)
    )
    return filtered


def stable_int_seed(random_state: int, key: object) -> int:
    payload = f"{random_state}:{key}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=4).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def sample_subfield_rows(
    rows: pd.DataFrame,
    *,
    max_papers: int,
    random_state: int,
    subfield_id: object,
) -> pd.DataFrame:
    if max_papers <= 0:
        raise ValueError("max_papers must be positive")

    ordered = stable_sort_work_rows(rows)
    if len(ordered) <= max_papers:
        return ordered

    rng = np.random.default_rng(stable_int_seed(random_state, subfield_id))
    selected_positions = np.sort(
        rng.choice(len(ordered), size=max_papers, replace=False)
    )
    return ordered.iloc[selected_positions].reset_index(drop=True)


def build_coordinate_frame(
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


@dataclass(frozen=True)
class ManifestRow:
    subfield_id: str
    subfield_name: str
    field_id: str
    field_display_name: str
    domain_id: str
    domain_display_name: str
    subfield_label_unique: str
    subfield_label_short: str
    subfield_display_name_is_duplicated: bool
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
    max_papers_per_subfield: int
    sampling_applied: bool


def build_manifest_row(
    *,
    subfield_id: object,
    subfield_name: object,
    n_available: int,
    n_used: int,
    year_min: int,
    year_max: int,
    status: str,
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
    random_state: int,
    max_papers_per_subfield: int,
    sampling_applied: bool,
    field_id: object = "",
    field_display_name: object = "",
    domain_id: object = "",
    domain_display_name: object = "",
    subfield_label_unique: object = "",
    subfield_label_short: object = "",
    subfield_display_name_is_duplicated: bool = False,
    coordinate_path: str | Path | None = None,
    figure_path: str | Path | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    if status not in VALID_STATUSES:
        raise ValueError(
            f"status must be one of {', '.join(sorted(VALID_STATUSES))}: {status}"
        )

    row = ManifestRow(
        subfield_id=str(subfield_id),
        subfield_name="" if pd.isna(subfield_name) else str(subfield_name),
        field_id="" if pd.isna(field_id) else str(field_id),
        field_display_name="" if pd.isna(field_display_name) else str(field_display_name),
        domain_id="" if pd.isna(domain_id) else str(domain_id),
        domain_display_name="" if pd.isna(domain_display_name) else str(domain_display_name),
        subfield_label_unique=""
        if pd.isna(subfield_label_unique)
        else str(subfield_label_unique),
        subfield_label_short="" if pd.isna(subfield_label_short) else str(subfield_label_short),
        subfield_display_name_is_duplicated=bool(subfield_display_name_is_duplicated),
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
        max_papers_per_subfield=int(max_papers_per_subfield),
        sampling_applied=bool(sampling_applied),
    )
    return asdict(row)


def manifest_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    frame = pd.DataFrame(rows)
    for column in MANIFEST_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[MANIFEST_COLUMNS]


def coordinate_limits(coordinates: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape (n_rows, 2)")

    finite = np.isfinite(coordinates).all(axis=1)
    if not finite.any():
        raise ValueError("coordinates contain no finite points")

    x = coordinates[finite, 0]
    y = coordinates[finite, 1]
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    x_range = x_max - x_min
    y_range = y_max - y_min
    x_pad = x_range * 0.05 if x_range > 0 else 1.0
    y_pad = y_range * 0.05 if y_range > 0 else 1.0
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def plot_continuous_density_panel(
    ax: plt.Axes,
    coordinates: np.ndarray,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    grid_size: int = DEFAULT_DENSITY_GRID_SIZE,
    sigma: float = DEFAULT_DENSITY_SIGMA,
    density_vmax_percentile: float = DEFAULT_DENSITY_VMAX_PERCENTILE,
) -> Any:
    if grid_size < 10:
        raise ValueError("density grid_size must be at least 10")
    if sigma <= 0:
        raise ValueError("density sigma must be positive")
    if not 0 < density_vmax_percentile <= 100:
        raise ValueError("density_vmax_percentile must be in (0, 100]")

    coordinates = np.asarray(coordinates, dtype=float)
    finite = np.isfinite(coordinates).all(axis=1)
    coordinates = coordinates[finite]
    if len(coordinates):
        hist, _, _ = np.histogram2d(
            coordinates[:, 0],
            coordinates[:, 1],
            bins=grid_size,
            range=[[xlim[0], xlim[1]], [ylim[0], ylim[1]]],
        )
    else:
        hist = np.zeros((grid_size, grid_size), dtype=float)

    from scipy.ndimage import gaussian_filter

    # Density panels are for visual interpretation only.
    # We use moderate Gaussian smoothing and percentile clipping so that density maps
    # show broad semantic mass rather than isolated maximum-density pixels.
    density = gaussian_filter(hist.T, sigma=sigma)
    density = np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
    density = np.clip(density, 0.0, None)
    vmax = density_plot_vmax(density, density_vmax_percentile)

    artist = ax.imshow(
        density,
        origin="lower",
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=vmax,
    )
    return artist


def density_plot_vmax(density: np.ndarray, percentile: float) -> float:
    if not 0 < percentile <= 100:
        raise ValueError("density_vmax_percentile must be in (0, 100]")

    positive = density[np.isfinite(density) & (density > 0)]
    if len(positive) == 0:
        return 1.0

    vmax = float(np.nanpercentile(positive, percentile))
    if np.isfinite(vmax) and vmax > 0:
        return vmax

    fallback = float(np.nanmax(positive))
    return fallback if np.isfinite(fallback) and fallback > 0 else 1.0


def plot_density_panel(
    ax: plt.Axes,
    coordinates: np.ndarray,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    density_method: str = DEFAULT_DENSITY_METHOD,
    kde_max_points: int = 5000,
    density_grid_size: int = DEFAULT_DENSITY_GRID_SIZE,
    density_sigma: float = DEFAULT_DENSITY_SIGMA,
    density_vmax_percentile: float = DEFAULT_DENSITY_VMAX_PERCENTILE,
) -> tuple[Any, str]:
    if density_method not in VALID_DENSITY_METHODS:
        raise ValueError(
            "density_method must be one of "
            f"{', '.join(sorted(VALID_DENSITY_METHODS))}: {density_method}"
        )
    if not 0 < density_vmax_percentile <= 100:
        raise ValueError("density_vmax_percentile must be in (0, 100]")
    coordinates = np.asarray(coordinates, dtype=float)
    finite = np.isfinite(coordinates).all(axis=1)
    coordinates = coordinates[finite]

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    use_kde = (
        density_method in {"auto", "kde"}
        and len(coordinates) >= 3
        and len(coordinates) <= kde_max_points
    )
    if use_kde:
        try:
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(coordinates.T)
            xs = np.linspace(xlim[0], xlim[1], 140)
            ys = np.linspace(ylim[0], ylim[1], 140)
            xx, yy = np.meshgrid(xs, ys)
            density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            density = np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
            density = np.clip(density, 0.0, None)
            vmax = density_plot_vmax(density, density_vmax_percentile)
            artist = ax.imshow(
                density,
                origin="lower",
                extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                aspect="auto",
                cmap="viridis",
                vmin=0,
                vmax=vmax,
            )
            return artist, "kde"
        except Exception:
            pass

    artist = plot_continuous_density_panel(
        ax,
        coordinates,
        xlim=xlim,
        ylim=ylim,
        grid_size=density_grid_size,
        sigma=density_sigma,
        density_vmax_percentile=density_vmax_percentile,
    )
    return artist, "smooth_hist"


def plot_subfield_panels(
    coordinates: np.ndarray,
    *,
    subfield_name: str,
    n_used: int,
    year_min: int,
    year_max: int,
    output_path: str | Path,
    dpi: int,
    density_method: str = DEFAULT_DENSITY_METHOD,
    density_grid_size: int = DEFAULT_DENSITY_GRID_SIZE,
    density_sigma: float = DEFAULT_DENSITY_SIGMA,
    density_vmax_percentile: float = DEFAULT_DENSITY_VMAX_PERCENTILE,
) -> str:
    coordinates = np.asarray(coordinates, dtype=float)
    xlim, ylim = coordinate_limits(coordinates)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)
    try:
        axes[0].scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=2,
            alpha=0.45,
            linewidths=0,
            color="#26547c",
            rasterized=True,
        )
        axes[0].set_xlim(xlim)
        axes[0].set_ylim(ylim)
        axes[0].set_xlabel("UMAP 1")
        axes[0].set_ylabel("UMAP 2")
        axes[0].set_title(
            f"A. Scatter - {subfield_name}\n"
            f"n={n_used:,}, years {year_min}-{year_max}",
            fontsize=10,
        )

        density_artist, density_method = plot_density_panel(
            axes[1],
            coordinates,
            xlim=xlim,
            ylim=ylim,
            density_method=density_method,
            density_grid_size=density_grid_size,
            density_sigma=density_sigma,
            density_vmax_percentile=density_vmax_percentile,
        )
        axes[1].set_title("B. Density", fontsize=10)
        fig.colorbar(density_artist, ax=axes[1], fraction=0.046, pad=0.04)

        for ax in axes:
            ax.grid(False)

        fig.tight_layout()
        fig.savefig(output_path)
        return density_method
    finally:
        plt.close(fig)
