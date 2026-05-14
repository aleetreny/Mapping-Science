from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from src.per_subfield_umap_maps import (
    build_coordinate_frame,
    filter_input_window,
    main_analysis_subfields,
    safe_subfield_stem,
    sample_subfield_rows,
)
from src.storage import load_parquet, save_parquet
from src.temporal_common import (
    TemporalWindow,
    dataframe_to_records,
    default_or_computed_windows,
    display_path,
    ensure_outputs_do_not_exist,
    utc_now_iso,
    write_json,
    write_text,
)

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


PROCESSED_MANIFEST_COLUMNS = [
    "subfield_id",
    "subfield_display_name",
    "field_id",
    "field_display_name",
    "domain_id",
    "domain_display_name",
    "subfield_label_unique",
    "subfield_label_short",
    "n_available",
    "n_used",
    "year_min",
    "year_max",
    "status",
    "coordinate_source",
    "coordinate_path",
    "static_panel_path",
    "gif_path",
    "frame_dir",
    "density_panel_path",
    "density_difference_panel_path",
    "density_gif_path",
    "n_windows_with_density",
    "n_windows_insufficient_points",
    "min_points_for_density",
    "density_grid_size",
    "density_sigma",
    "used_existing_umap_coordinates",
    "error_message",
    "warning_message",
]

ANALYSIS_FILENAMES = {
    "manifest_parquet": "manifest.parquet",
    "manifest_csv": "manifest.csv",
    "summary_md": "summary.md",
    "summary_json": "summary.json",
}


def output_paths(output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    return {key: output_dir / filename for key, filename in ANALYSIS_FILENAMES.items()}


def ensure_temporal_umap_outputs(
    output_dir: str | Path,
    *,
    overwrite: bool,
    root: Path,
) -> None:
    paths = output_paths(output_dir)
    ensure_outputs_do_not_exist(
        paths.values(),
        overwrite=overwrite,
        root=root,
        description="subfield temporal UMAP panel outputs",
    )


def select_subfields_for_panels(
    subfields: pd.DataFrame,
    *,
    subfield_id: str | None,
    limit_subfields: int | None,
    top_dynamic_from: str | Path | None,
    top_dynamic_n: int,
) -> pd.DataFrame:
    attempted = subfields.copy()
    if top_dynamic_n <= 0:
        raise ValueError("top_dynamic_n must be positive")
    if subfield_id is not None:
        selected = attempted[
            attempted["subfield_id"].astype(str) == str(subfield_id)
        ].reset_index(drop=True)
        if selected.empty:
            raise ValueError(f"No main-analysis subfield matched --subfield-id {subfield_id}")
        return selected
    if limit_subfields is not None:
        if limit_subfields <= 0:
            raise ValueError("limit_subfields must be positive when provided")
        return attempted.head(limit_subfields).reset_index(drop=True)
    if top_dynamic_from is not None and Path(top_dynamic_from).exists():
        dynamic = pd.read_parquet(top_dynamic_from)
        if "subfield_id" not in dynamic.columns:
            raise ValueError(f"top dynamic file has no subfield_id column: {top_dynamic_from}")
        sort_column = (
            "total_path_length"
            if "total_path_length" in dynamic.columns
            else "early_late_drift"
            if "early_late_drift" in dynamic.columns
            else None
        )
        if sort_column is not None:
            dynamic = dynamic.sort_values(
                [sort_column, "subfield_id"],
                ascending=[False, True],
                kind="mergesort",
            )
        dynamic_ids = dynamic.head(top_dynamic_n)["subfield_id"].astype(str).tolist()
        selected = attempted.loc[
            attempted["subfield_id"].astype(str).isin(dynamic_ids)
        ].copy()
        selected["_order"] = selected["subfield_id"].astype(str).map(
            {value: idx for idx, value in enumerate(dynamic_ids)}
        )
        return selected.sort_values("_order", kind="mergesort").drop(columns="_order").reset_index(drop=True)
    return attempted.head(top_dynamic_n).reset_index(drop=True)


def selection_summary(
    *,
    subfield_id: str | None,
    limit_subfields: int | None,
    top_dynamic_from: str | Path | None,
    top_dynamic_n: int,
) -> dict[str, Any]:
    if subfield_id is not None:
        return {
            "selection_source": "explicit_subfield_id",
            "selection_ranking_column": "",
            "top_n": 1,
        }
    if limit_subfields is not None:
        return {
            "selection_source": "main_analysis_head",
            "selection_ranking_column": "",
            "top_n": int(limit_subfields),
        }
    if top_dynamic_from is not None and Path(top_dynamic_from).exists():
        dynamic = pd.read_parquet(top_dynamic_from)
        ranking_column = (
            "total_path_length"
            if "total_path_length" in dynamic.columns
            else "early_late_drift"
            if "early_late_drift" in dynamic.columns
            else ""
        )
        return {
            "selection_source": "centroid_path_metrics",
            "selection_ranking_column": ranking_column,
            "top_n": int(top_dynamic_n),
        }
    return {
        "selection_source": "main_analysis_head",
        "selection_ranking_column": "",
        "top_n": int(top_dynamic_n),
    }


def _coordinate_limits(frame: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    x = pd.to_numeric(frame["umap_x"], errors="coerce")
    y = pd.to_numeric(frame["umap_y"], errors="coerce")
    finite = x.notna() & y.notna()
    if not finite.any():
        return (-1.0, 1.0), (-1.0, 1.0)
    x_values = x.loc[finite].to_numpy(dtype=float)
    y_values = y.loc[finite].to_numpy(dtype=float)
    x_range = float(x_values.max() - x_values.min())
    y_range = float(y_values.max() - y_values.min())
    x_pad = x_range * 0.05 if x_range > 0 else 1.0
    y_pad = y_range * 0.05 if y_range > 0 else 1.0
    return (
        (float(x_values.min() - x_pad), float(x_values.max() + x_pad)),
        (float(y_values.min() - y_pad), float(y_values.max() + y_pad)),
    )


def window_centroids_2d(frame: pd.DataFrame, windows: list[TemporalWindow]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    years = pd.to_numeric(frame["publication_year"], errors="coerce")
    for window in windows:
        mask = years.between(window.window_start, window.window_end, inclusive="both")
        subset = frame.loc[mask]
        rows.append(
            {
                "window_label": window.window_label,
                "window_start": window.window_start,
                "window_end": window.window_end,
                "window_index": window.window_index,
                "umap_x": float(subset["umap_x"].mean()) if len(subset) else np.nan,
                "umap_y": float(subset["umap_y"].mean()) if len(subset) else np.nan,
                "n_points": int(len(subset)),
            }
        )
    return pd.DataFrame(rows)


def plot_window_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    centroids: pd.DataFrame,
    *,
    window: TemporalWindow,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    title: str,
) -> None:
    years = pd.to_numeric(frame["publication_year"], errors="coerce")
    mask = years.between(window.window_start, window.window_end, inclusive="both")
    subset = frame.loc[mask]
    ax.scatter(
        frame["umap_x"],
        frame["umap_y"],
        s=2,
        linewidths=0,
        color="#b9b9b9",
        alpha=0.22,
        rasterized=True,
    )
    ax.scatter(
        subset["umap_x"],
        subset["umap_y"],
        s=3,
        linewidths=0,
        color="#2f6f73",
        alpha=0.62,
        rasterized=True,
    )
    path = centroids.loc[centroids["window_index"] <= window.window_index].dropna(
        subset=["umap_x", "umap_y"]
    )
    if len(path):
        ax.plot(path["umap_x"], path["umap_y"], color="#b45432", linewidth=1.2, alpha=0.9)
        ax.scatter(path["umap_x"], path["umap_y"], color="#b45432", s=16, linewidths=0, zorder=3)
    current = centroids.loc[centroids["window_label"] == window.window_label]
    if len(current) and current["umap_x"].notna().all():
        ax.scatter(
            current["umap_x"],
            current["umap_y"],
            s=46,
            marker="x",
            color="#222222",
            linewidths=1.2,
            zorder=4,
        )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_static_panel(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    subfield_label: str,
    windows: list[TemporalWindow],
    dpi: int,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xlim, ylim = _coordinate_limits(frame)
    centroids = window_centroids_2d(frame, windows)
    fig, axes = plt.subplots(1, len(windows), figsize=(3.0 * len(windows), 3.4), dpi=dpi)
    if len(windows) == 1:
        axes = np.array([axes])
    for ax, window in zip(axes.ravel(), windows, strict=False):
        plot_window_panel(
            ax,
            frame,
            centroids,
            window=window,
            xlim=xlim,
            ylim=ylim,
            title=window.window_label,
        )
    fig.suptitle(subfield_label, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path)
    plt.close(fig)


def relative_density_grid(
    coordinates: np.ndarray,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    grid_size: int,
    sigma: float,
) -> np.ndarray:
    if grid_size < 10:
        raise ValueError("density_grid_size must be at least 10")
    if sigma <= 0:
        raise ValueError("density_sigma must be positive")

    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.size == 0:
        return np.zeros((grid_size, grid_size), dtype=float)
    finite = np.isfinite(coordinates).all(axis=1)
    coordinates = coordinates[finite]
    if len(coordinates) == 0:
        return np.zeros((grid_size, grid_size), dtype=float)

    hist, _, _ = np.histogram2d(
        coordinates[:, 0],
        coordinates[:, 1],
        bins=grid_size,
        range=[[xlim[0], xlim[1]], [ylim[0], ylim[1]]],
    )
    from scipy.ndimage import gaussian_filter

    density = gaussian_filter(hist.T, sigma=sigma)
    density = np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
    total = float(density.sum())
    if total > 0:
        density = density / total
    return density


def window_density_grids(
    frame: pd.DataFrame,
    *,
    windows: list[TemporalWindow],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    grid_size: int,
    sigma: float,
    min_points_for_density: int,
) -> list[dict[str, Any]]:
    years = pd.to_numeric(frame["publication_year"], errors="coerce")
    results: list[dict[str, Any]] = []
    for window in windows:
        mask = years.between(window.window_start, window.window_end, inclusive="both")
        subset = frame.loc[mask]
        coordinates = subset[["umap_x", "umap_y"]].to_numpy(dtype=float)
        n_points = int(np.isfinite(coordinates).all(axis=1).sum()) if len(coordinates) else 0
        has_density = n_points >= min_points_for_density
        density = (
            relative_density_grid(
                coordinates,
                xlim=xlim,
                ylim=ylim,
                grid_size=grid_size,
                sigma=sigma,
            )
            if has_density
            else np.zeros((grid_size, grid_size), dtype=float)
        )
        results.append(
            {
                "window": window,
                "window_label": window.window_label,
                "n_points": n_points,
                "has_density": has_density,
                "density": density,
                "subset": subset,
            }
        )
    return results


def density_window_counts(density_results: list[dict[str, Any]]) -> tuple[int, int]:
    n_with_density = sum(1 for result in density_results if result["has_density"])
    return n_with_density, len(density_results) - n_with_density


def _overlay_centroid_path(
    ax: plt.Axes,
    centroids: pd.DataFrame,
    *,
    window: TemporalWindow,
    color: str = "#222222",
) -> None:
    path = centroids.loc[centroids["window_index"] <= window.window_index].dropna(
        subset=["umap_x", "umap_y"]
    )
    if len(path):
        ax.plot(path["umap_x"], path["umap_y"], color=color, linewidth=1.0, alpha=0.82)
        ax.scatter(
            path["umap_x"],
            path["umap_y"],
            color=color,
            s=12,
            linewidths=0,
            alpha=0.82,
            zorder=3,
        )
    current = centroids.loc[centroids["window_label"] == window.window_label]
    if len(current) and current["umap_x"].notna().all():
        ax.scatter(
            current["umap_x"],
            current["umap_y"],
            s=38,
            marker="x",
            color=color,
            linewidths=1.2,
            zorder=4,
        )


def _format_density_axes(
    ax: plt.Axes,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    title: str,
) -> None:
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_temporal_density_panel(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    subfield_label: str,
    windows: list[TemporalWindow],
    density_grid_size: int,
    density_sigma: float,
    min_points_for_density: int,
    density_alpha: float,
    dpi: int,
) -> tuple[int, int]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xlim, ylim = _coordinate_limits(frame)
    centroids = window_centroids_2d(frame, windows)
    density_results = window_density_grids(
        frame,
        windows=windows,
        xlim=xlim,
        ylim=ylim,
        grid_size=density_grid_size,
        sigma=density_sigma,
        min_points_for_density=min_points_for_density,
    )
    vmax = max(
        [float(result["density"].max()) for result in density_results if result["has_density"]]
        or [1.0]
    )

    fig, axes = plt.subplots(1, len(windows), figsize=(3.15 * len(windows), 3.75), dpi=dpi)
    if len(windows) == 1:
        axes = np.array([axes])
    artist = None
    for ax, result in zip(axes.ravel(), density_results, strict=False):
        window = result["window"]
        if result["has_density"]:
            artist = ax.imshow(
                result["density"],
                origin="lower",
                extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                aspect="auto",
                cmap="viridis",
                vmin=0.0,
                vmax=vmax,
                alpha=density_alpha,
            )
        else:
            subset = result["subset"]
            ax.scatter(
                subset["umap_x"],
                subset["umap_y"],
                s=5,
                linewidths=0,
                color="#2f6f73",
                alpha=0.45,
                rasterized=True,
            )
            ax.text(
                0.5,
                0.5,
                "insufficient points\nfor density",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                color="#333333",
            )
        _overlay_centroid_path(ax, centroids, window=window)
        _format_density_axes(ax, xlim=xlim, ylim=ylim, title=window.window_label)
    if artist is not None:
        colorbar = fig.colorbar(artist, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        colorbar.set_label("relative density within period", fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
    fig.suptitle(f"{subfield_label} - Relative Semantic Density by Period", fontsize=11)
    fig.text(
        0.5,
        0.025,
        "Density is normalized within each period; coordinates are fixed full-period UMAP coordinates.",
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.02, right=0.92, bottom=0.16, top=0.82, wspace=0.08)
    fig.savefig(output_path)
    plt.close(fig)
    return density_window_counts(density_results)


def plot_density_difference_panel(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    subfield_label: str,
    windows: list[TemporalWindow],
    density_grid_size: int,
    density_sigma: float,
    min_points_for_density: int,
    density_alpha: float,
    dpi: int,
) -> tuple[int, int]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xlim, ylim = _coordinate_limits(frame)
    centroids = window_centroids_2d(frame, windows)
    density_results = window_density_grids(
        frame,
        windows=windows,
        xlim=xlim,
        ylim=ylim,
        grid_size=density_grid_size,
        sigma=density_sigma,
        min_points_for_density=min_points_for_density,
    )
    if len(density_results) <= 1:
        raise ValueError("at least two temporal windows are required for density differences")
    baseline = density_results[0]
    later_results = density_results[1:]
    differences = [
        result["density"] - baseline["density"]
        for result in later_results
        if baseline["has_density"] and result["has_density"]
    ]
    vmax = max([float(np.abs(diff).max()) for diff in differences] or [1.0])
    if vmax <= 0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    n_cols = min(4, len(later_results))
    n_rows = int(np.ceil(len(later_results) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.5 * n_cols, 3.65 * n_rows),
        dpi=dpi,
        squeeze=False,
    )
    artist = None
    for ax, result in zip(axes.ravel(), later_results, strict=False):
        window = result["window"]
        title = f"{window.window_label} minus {baseline['window_label']}"
        if baseline["has_density"] and result["has_density"]:
            diff = result["density"] - baseline["density"]
            artist = ax.imshow(
                diff,
                origin="lower",
                extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                aspect="auto",
                cmap="RdBu_r",
                norm=norm,
                alpha=density_alpha,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "insufficient points\nfor comparison",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                color="#333333",
            )
        _overlay_centroid_path(ax, centroids, window=window, color="#111111")
        _format_density_axes(ax, xlim=xlim, ylim=ylim, title=title)
    for ax in axes.ravel()[len(later_results) :]:
        ax.axis("off")
    if artist is not None:
        colorbar = fig.colorbar(artist, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
        colorbar.set_label("change in relative density vs 2000-2004", fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
    fig.suptitle(
        f"{subfield_label} - Change in Relative Semantic Density vs {baseline['window_label']}",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.025,
        "Red = gain in relative density; blue = loss. Each period density is normalized before differencing.",
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.04, right=0.9, bottom=0.16, top=0.82, wspace=0.12, hspace=0.18)
    fig.savefig(output_path)
    plt.close(fig)
    return density_window_counts(density_results)


def plot_density_frame_pngs(
    frame: pd.DataFrame,
    frame_dir: str | Path,
    *,
    subfield_label: str,
    windows: list[TemporalWindow],
    density_grid_size: int,
    density_sigma: float,
    min_points_for_density: int,
    density_alpha: float,
    dpi: int,
) -> tuple[list[Path], tuple[int, int]]:
    frame_dir = Path(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    xlim, ylim = _coordinate_limits(frame)
    centroids = window_centroids_2d(frame, windows)
    density_results = window_density_grids(
        frame,
        windows=windows,
        xlim=xlim,
        ylim=ylim,
        grid_size=density_grid_size,
        sigma=density_sigma,
        min_points_for_density=min_points_for_density,
    )
    vmax = max(
        [float(result["density"].max()) for result in density_results if result["has_density"]]
        or [1.0]
    )
    paths: list[Path] = []
    for result in density_results:
        window = result["window"]
        path = frame_dir / f"{window.window_label}.png"
        fig, ax = plt.subplots(figsize=(5.2, 4.8), dpi=dpi)
        if result["has_density"]:
            artist = ax.imshow(
                result["density"],
                origin="lower",
                extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                aspect="auto",
                cmap="viridis",
                vmin=0.0,
                vmax=vmax,
                alpha=density_alpha,
            )
            colorbar = fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
            colorbar.set_label("relative density within period", fontsize=8)
            colorbar.ax.tick_params(labelsize=7)
        else:
            ax.text(
                0.5,
                0.5,
                "insufficient points for density",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
            )
        _overlay_centroid_path(ax, centroids, window=window)
        _format_density_axes(
            ax,
            xlim=xlim,
            ylim=ylim,
            title=f"{subfield_label} - {window.window_label}",
        )
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths, density_window_counts(density_results)


def plot_frame_pngs(
    frame: pd.DataFrame,
    frame_dir: str | Path,
    *,
    subfield_label: str,
    windows: list[TemporalWindow],
    dpi: int,
) -> list[Path]:
    frame_dir = Path(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    xlim, ylim = _coordinate_limits(frame)
    centroids = window_centroids_2d(frame, windows)
    paths: list[Path] = []
    for window in windows:
        path = frame_dir / f"{window.window_label}.png"
        fig, ax = plt.subplots(figsize=(5.2, 4.8), dpi=dpi)
        plot_window_panel(
            ax,
            frame,
            centroids,
            window=window,
            xlim=xlim,
            ylim=ylim,
            title=f"{subfield_label} - {window.window_label}",
        )
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def make_gif_from_frames(frame_paths: list[Path], gif_path: str | Path, *, fps: float) -> str:
    if fps <= 0:
        raise ValueError("fps must be positive")
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - depends on local optional package
        return f"GIF skipped because Pillow is unavailable: {exc}"

    if not frame_paths:
        return "GIF skipped because no frame PNGs were available"
    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in frame_paths]
    try:
        duration_ms = int(1000 / fps)
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
        )
    finally:
        for image in images:
            image.close()
    return ""


def load_existing_coordinates(
    coordinates_dir: str | Path,
    *,
    subfield_id: object,
    subfield_name: object,
) -> tuple[pd.DataFrame | None, Path]:
    stem = safe_subfield_stem(subfield_id, subfield_name)
    path = Path(coordinates_dir) / f"{stem}.parquet"
    if not path.exists():
        return None, path
    frame = load_parquet(path)
    required = {"analysis_row_id", "publication_year", "umap_x", "umap_y"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Coordinate file {path} is missing columns: {', '.join(sorted(missing))}"
        )
    return frame, path


def fit_fixed_umap_coordinates(
    subfield_rows: pd.DataFrame,
    matrix: np.ndarray,
    *,
    max_papers_per_subfield: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
    subfield_id: object,
) -> pd.DataFrame:
    if len(subfield_rows) == 0:
        raise ValueError("no rows available for subfield")
    sampled = sample_subfield_rows(
        subfield_rows,
        max_papers=max_papers_per_subfield,
        random_state=random_state,
        subfield_id=subfield_id,
    )
    row_ids = sampled["analysis_row_id"].astype(int).to_numpy()
    embeddings = np.asarray(matrix[row_ids], dtype=np.float32)
    effective_n_neighbors = min(n_neighbors, max(2, len(sampled) - 1))
    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=effective_n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        low_memory=True,
        verbose=False,
    )
    coordinates = reducer.fit_transform(embeddings)
    return build_coordinate_frame(sampled, coordinates)


def _manifest_row(
    *,
    subfield: pd.Series | dict[str, Any],
    n_available: int,
    n_used: int,
    year_min: int,
    year_max: int,
    status: str,
    coordinate_source: str = "",
    coordinate_path: str = "",
    static_panel_path: str = "",
    gif_path: str = "",
    frame_dir: str = "",
    density_panel_path: str = "",
    density_difference_panel_path: str = "",
    density_gif_path: str = "",
    n_windows_with_density: int = 0,
    n_windows_insufficient_points: int = 0,
    min_points_for_density: int = 0,
    density_grid_size: int = 0,
    density_sigma: float = 0.0,
    used_existing_umap_coordinates: bool = False,
    error_message: str = "",
    warning_message: str = "",
) -> dict[str, Any]:
    getter = subfield.get
    return {
        "subfield_id": str(getter("subfield_id", "")),
        "subfield_display_name": str(getter("subfield_display_name", "")),
        "field_id": str(getter("field_id", "")),
        "field_display_name": str(getter("field_display_name", "")),
        "domain_id": str(getter("domain_id", "")),
        "domain_display_name": str(getter("domain_display_name", "")),
        "subfield_label_unique": str(getter("subfield_label_unique", "")),
        "subfield_label_short": str(getter("subfield_label_short", "")),
        "n_available": int(n_available),
        "n_used": int(n_used),
        "year_min": int(year_min),
        "year_max": int(year_max),
        "status": status,
        "coordinate_source": coordinate_source,
        "coordinate_path": coordinate_path,
        "static_panel_path": static_panel_path,
        "gif_path": gif_path,
        "frame_dir": frame_dir,
        "density_panel_path": density_panel_path,
        "density_difference_panel_path": density_difference_panel_path,
        "density_gif_path": density_gif_path,
        "n_windows_with_density": int(n_windows_with_density),
        "n_windows_insufficient_points": int(n_windows_insufficient_points),
        "min_points_for_density": int(min_points_for_density),
        "density_grid_size": int(density_grid_size),
        "density_sigma": float(density_sigma),
        "used_existing_umap_coordinates": bool(used_existing_umap_coordinates),
        "error_message": error_message,
        "warning_message": warning_message,
    }


def manifest_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in PROCESSED_MANIFEST_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[PROCESSED_MANIFEST_COLUMNS]


def build_subfield_temporal_umap_panels(
    analysis_index: pd.DataFrame,
    matrix: np.ndarray,
    *,
    year_min: int,
    year_max: int,
    window_size: int,
    min_papers: int,
    max_papers_per_subfield: int,
    reuse_existing_coordinates: bool,
    coordinates_dir: str | Path,
    fit_if_missing: bool,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
    make_gif: bool,
    make_static_panel: bool,
    make_density_panels: bool,
    make_density_difference_panels: bool,
    make_density_gifs: bool,
    density_grid_size: int,
    density_sigma: float,
    min_points_for_density: int,
    density_alpha: float,
    fps: float,
    subfield_id: str | None,
    limit_subfields: int | None,
    top_dynamic_from: str | Path | None,
    top_dynamic_n: int,
    output_dir: str | Path,
    root: Path,
    overwrite: bool,
    dpi: int = 180,
) -> dict[str, Any]:
    windows = default_or_computed_windows(
        year_min=year_min,
        year_max=year_max,
        window_size=window_size,
    )
    output_dir = Path(output_dir)
    gifs_dir = output_dir / "gifs"
    static_dir = output_dir / "static_panels"
    frames_root = output_dir / "frames"
    density_dir = output_dir / "density_panels"
    density_difference_dir = output_dir / "density_difference_panels"
    density_gifs_dir = output_dir / "density_gifs"
    density_frames_root = output_dir / "density_frames"
    for directory in [
        gifs_dir,
        static_dir,
        frames_root,
        density_dir,
        density_difference_dir,
        density_gifs_dir,
        density_frames_root,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    subfields = main_analysis_subfields(analysis_index)
    attempted = select_subfields_for_panels(
        subfields,
        subfield_id=subfield_id,
        limit_subfields=limit_subfields,
        top_dynamic_from=top_dynamic_from,
        top_dynamic_n=top_dynamic_n,
    )
    selection = selection_summary(
        subfield_id=subfield_id,
        limit_subfields=limit_subfields,
        top_dynamic_from=top_dynamic_from,
        top_dynamic_n=top_dynamic_n,
    )
    window_index = filter_input_window(
        analysis_index,
        year_min=year_min,
        year_max=year_max,
    )
    groups = {
        str(subfield_id_value): group.reset_index(drop=True)
        for subfield_id_value, group in window_index.groupby("subfield_id", sort=False)
    }

    manifest_rows: list[dict[str, Any]] = []
    for _, subfield in attempted.iterrows():
        sid = str(subfield["subfield_id"])
        stem = safe_subfield_stem(sid, subfield["subfield_display_name"])
        static_path = static_dir / f"{stem}.png"
        gif_path = gifs_dir / f"{stem}.gif"
        frame_dir = frames_root / stem
        density_panel_path = density_dir / f"{stem}_density_panel.png"
        density_difference_panel_path = (
            density_difference_dir / f"{stem}_density_difference_panel.png"
        )
        density_gif_path = density_gifs_dir / f"{stem}_density.gif"
        density_frame_dir = density_frames_root / stem
        available = groups.get(sid, window_index.head(0))
        n_available = int(len(available))
        if n_available < min_papers:
            manifest_rows.append(
                _manifest_row(
                    subfield=subfield,
                    n_available=n_available,
                    n_used=0,
                    year_min=year_min,
                    year_max=year_max,
                    status="skipped",
                    min_points_for_density=min_points_for_density,
                    density_grid_size=density_grid_size,
                    density_sigma=density_sigma,
                    error_message=f"Only {n_available} papers available; min_papers is {min_papers}",
                )
            )
            continue
        try:
            requested_outputs = []
            if make_static_panel:
                requested_outputs.append(static_path)
            if make_gif:
                requested_outputs.append(gif_path)
            if make_density_panels:
                requested_outputs.append(density_panel_path)
            if make_density_difference_panels:
                requested_outputs.append(density_difference_panel_path)
            if make_density_gifs:
                requested_outputs.append(density_gif_path)
            existing_outputs = [path for path in requested_outputs if path.exists()]
            if existing_outputs and not overwrite:
                raise FileExistsError("panel or GIF output exists; rerun with --overwrite")
            coordinate_frame: pd.DataFrame | None = None
            coordinate_path = Path(coordinates_dir) / f"{stem}.parquet"
            coordinate_source = ""
            if reuse_existing_coordinates:
                coordinate_frame, coordinate_path = load_existing_coordinates(
                    coordinates_dir,
                    subfield_id=sid,
                    subfield_name=subfield["subfield_display_name"],
                )
                if coordinate_frame is not None:
                    coordinate_source = "reused_existing"
            if coordinate_frame is None:
                if not fit_if_missing:
                    raise FileNotFoundError(
                        f"Missing reusable coordinates: {coordinate_path}. "
                        "Use --fit-if-missing to fit fixed coordinates inside this script."
                    )
                coordinate_frame = fit_fixed_umap_coordinates(
                    available,
                    matrix,
                    max_papers_per_subfield=max_papers_per_subfield,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    metric=metric,
                    random_state=random_state,
                    subfield_id=sid,
                )
                coordinate_source = "fitted"
            years = pd.to_numeric(coordinate_frame["publication_year"], errors="coerce")
            coordinate_frame = coordinate_frame.loc[
                years.between(year_min, year_max, inclusive="both")
            ].copy()
            if len(coordinate_frame) < min_papers:
                raise ValueError(
                    f"Only {len(coordinate_frame)} coordinate rows available after year filtering; "
                    f"min_papers is {min_papers}"
                )

            if make_static_panel:
                plot_static_panel(
                    coordinate_frame,
                    static_path,
                    subfield_label=str(subfield["subfield_label_short"]),
                    windows=windows,
                    dpi=dpi,
                )
            warning_message = ""
            density_panel_value = ""
            density_difference_value = ""
            density_gif_value = ""
            n_windows_with_density = 0
            n_windows_insufficient_points = 0
            if make_density_panels:
                density_counts = plot_temporal_density_panel(
                    coordinate_frame,
                    density_panel_path,
                    subfield_label=str(subfield["subfield_label_short"]),
                    windows=windows,
                    density_grid_size=density_grid_size,
                    density_sigma=density_sigma,
                    min_points_for_density=min_points_for_density,
                    density_alpha=density_alpha,
                    dpi=dpi,
                )
                n_windows_with_density = max(n_windows_with_density, density_counts[0])
                n_windows_insufficient_points = max(
                    n_windows_insufficient_points,
                    density_counts[1],
                )
                density_panel_value = display_path(density_panel_path, root=root)
            if make_density_difference_panels:
                density_counts = plot_density_difference_panel(
                    coordinate_frame,
                    density_difference_panel_path,
                    subfield_label=str(subfield["subfield_label_short"]),
                    windows=windows,
                    density_grid_size=density_grid_size,
                    density_sigma=density_sigma,
                    min_points_for_density=min_points_for_density,
                    density_alpha=density_alpha,
                    dpi=dpi,
                )
                n_windows_with_density = max(n_windows_with_density, density_counts[0])
                n_windows_insufficient_points = max(
                    n_windows_insufficient_points,
                    density_counts[1],
                )
                density_difference_value = display_path(
                    density_difference_panel_path,
                    root=root,
                )
            frame_dir_value = ""
            gif_value = ""
            if make_gif:
                frame_paths = plot_frame_pngs(
                    coordinate_frame,
                    frame_dir,
                    subfield_label=str(subfield["subfield_label_short"]),
                    windows=windows,
                    dpi=dpi,
                )
                frame_dir_value = display_path(frame_dir, root=root)
                warning_message = make_gif_from_frames(frame_paths, gif_path, fps=fps)
                if not warning_message:
                    gif_value = display_path(gif_path, root=root)
            if make_density_gifs:
                density_frame_paths, density_counts = plot_density_frame_pngs(
                    coordinate_frame,
                    density_frame_dir,
                    subfield_label=str(subfield["subfield_label_short"]),
                    windows=windows,
                    density_grid_size=density_grid_size,
                    density_sigma=density_sigma,
                    min_points_for_density=min_points_for_density,
                    density_alpha=density_alpha,
                    dpi=dpi,
                )
                n_windows_with_density = max(n_windows_with_density, density_counts[0])
                n_windows_insufficient_points = max(
                    n_windows_insufficient_points,
                    density_counts[1],
                )
                density_warning = make_gif_from_frames(
                    density_frame_paths,
                    density_gif_path,
                    fps=fps,
                )
                if density_warning:
                    warning_message = "; ".join(
                        message
                        for message in [warning_message, density_warning]
                        if message
                    )
                else:
                    density_gif_value = display_path(density_gif_path, root=root)
            if n_windows_insufficient_points:
                density_warning = (
                    f"{n_windows_insufficient_points} temporal windows had fewer than "
                    f"{min_points_for_density} points for density"
                )
                warning_message = "; ".join(
                    message for message in [warning_message, density_warning] if message
                )
            manifest_rows.append(
                _manifest_row(
                    subfield=subfield,
                    n_available=n_available,
                    n_used=len(coordinate_frame),
                    year_min=year_min,
                    year_max=year_max,
                    status="completed",
                    coordinate_source=coordinate_source,
                    coordinate_path=display_path(coordinate_path, root=root)
                    if coordinate_path.exists()
                    else "",
                    static_panel_path=display_path(static_path, root=root)
                    if make_static_panel
                    else "",
                    gif_path=gif_value,
                    frame_dir=frame_dir_value,
                    density_panel_path=density_panel_value,
                    density_difference_panel_path=density_difference_value,
                    density_gif_path=density_gif_value,
                    n_windows_with_density=n_windows_with_density,
                    n_windows_insufficient_points=n_windows_insufficient_points,
                    min_points_for_density=min_points_for_density,
                    density_grid_size=density_grid_size,
                    density_sigma=density_sigma,
                    used_existing_umap_coordinates=coordinate_source == "reused_existing",
                    warning_message=warning_message,
                )
            )
        except Exception as exc:
            manifest_rows.append(
                _manifest_row(
                    subfield=subfield,
                    n_available=n_available,
                    n_used=0,
                    year_min=year_min,
                    year_max=year_max,
                    status="failed",
                    min_points_for_density=min_points_for_density,
                    density_grid_size=density_grid_size,
                    density_sigma=density_sigma,
                    error_message=str(exc),
                )
            )

    manifest = manifest_frame(manifest_rows)
    paths = output_paths(output_dir)
    save_parquet(manifest, paths["manifest_parquet"])
    manifest.to_csv(paths["manifest_csv"], index=False)
    status_counts = manifest["status"].value_counts().to_dict()
    output_paths_payload = {
        **{key: display_path(path, root=root) for key, path in paths.items()},
        "gifs_dir": display_path(gifs_dir, root=root),
        "static_panels_dir": display_path(static_dir, root=root),
        "frames_dir": display_path(frames_root, root=root),
        "density_panels_dir": display_path(density_dir, root=root),
        "density_difference_panels_dir": display_path(density_difference_dir, root=root),
        "density_gifs_dir": display_path(density_gifs_dir, root=root),
        "density_frames_dir": display_path(density_frames_root, root=root),
    }
    completed = manifest.loc[manifest["status"] == "completed"].copy()
    summary = {
        "created_at": utc_now_iso(),
        "output_paths": output_paths_payload,
        "coordinates_dir": display_path(coordinates_dir, root=root),
        "windows": [
            {"window_start": window.window_start, "window_end": window.window_end}
            for window in windows
        ],
        "n_subfields_attempted": int(len(manifest)),
        "n_completed": int(status_counts.get("completed", 0)),
        "n_skipped": int(status_counts.get("skipped", 0)),
        "n_failed": int(status_counts.get("failed", 0)),
        "make_gif": bool(make_gif),
        "make_static_panel": bool(make_static_panel),
        "make_density_panels": bool(make_density_panels),
        "make_density_difference_panels": bool(make_density_difference_panels),
        "make_density_gifs": bool(make_density_gifs),
        "density_method": "smoothed_histogram",
        "density_grid_size": int(density_grid_size),
        "density_sigma": float(density_sigma),
        "density_alpha": float(density_alpha),
        "min_points_for_density": int(min_points_for_density),
        "density_normalized_within_period": True,
        "density_difference_baseline_window": windows[0].window_label if windows else "",
        "density_difference_symmetric_zero_centered": True,
        "n_density_panels_completed": int(
            completed["density_panel_path"].fillna("").astype(str).ne("").sum()
        )
        if "density_panel_path" in completed
        else 0,
        "n_density_difference_panels_completed": int(
            completed["density_difference_panel_path"].fillna("").astype(str).ne("").sum()
        )
        if "density_difference_panel_path" in completed
        else 0,
        "n_density_gifs_completed": int(
            completed["density_gif_path"].fillna("").astype(str).ne("").sum()
        )
        if "density_gif_path" in completed
        else 0,
        "n_windows_with_density": int(completed["n_windows_with_density"].sum())
        if "n_windows_with_density" in completed
        else 0,
        "n_windows_insufficient_points": int(
            completed["n_windows_insufficient_points"].sum()
        )
        if "n_windows_insufficient_points" in completed
        else 0,
        "reuse_existing_coordinates": bool(reuse_existing_coordinates),
        "fit_if_missing": bool(fit_if_missing),
        "top_dynamic_from": "" if top_dynamic_from is None else str(top_dynamic_from),
        "top_dynamic_n": int(top_dynamic_n),
        "selection": selection,
        "method_notes": [
            (
                "These panels use one fixed full-period UMAP coordinate system per "
                "subfield; separate UMAPs are not fitted by period."
            ),
            (
                "Densities are estimated separately for each five-year window using "
                "a Gaussian-smoothed 2D histogram and normalized within each window."
            ),
            (
                "Difference-density panels subtract the 2000-2004 normalized density "
                "from later normalized densities using symmetric zero-centered color limits."
            ),
            (
                "These figures are visual summaries only; quantitative temporal "
                "movement is reported in embedding-space centroid and metric-evolution tables."
            ),
        ],
        "manifest_preview": dataframe_to_records(manifest, limit=20),
    }
    write_json(paths["summary_json"], summary)
    write_text(paths["summary_md"], markdown_summary(summary, manifest))
    return summary


def markdown_summary(summary: dict[str, Any], manifest: pd.DataFrame) -> str:
    lines = [
        "# Subfield Temporal UMAP Panels",
        "",
        "This output shows temporal windows on fixed within-subfield UMAP coordinates.",
        "",
        "## Methods",
        "",
        "- UMAP animations use fixed coordinates. A single UMAP is fitted per subfield across all selected years; frames only change which points are highlighted.",
        "- Existing full-period per-subfield UMAP coordinates are reused when available and valid.",
        "- The grey background is the full-period coordinate cloud; colored points are papers in the current five-year window.",
        "- Density panels use Gaussian-smoothed 2D histograms on the same fixed coordinates.",
        "- Densities are normalized within each five-year window, so density panels show redistribution of semantic mass rather than publication-volume changes.",
        "- Difference-density panels subtract the 2000-2004 normalized density from later normalized densities and use symmetric zero-centered color limits.",
        "- Counts are quality-control metadata, not publication-volume analysis.",
        "",
        "## Outputs",
        "",
    ]
    for label, path in summary["output_paths"].items():
        lines.append(f"- {label}: `{path}`")
    lines.extend(
        [
            "",
            "## Run Summary",
            "",
            f"- Subfields attempted: {summary['n_subfields_attempted']}",
            f"- Completed: {summary['n_completed']}",
            f"- Skipped: {summary['n_skipped']}",
            f"- Failed: {summary['n_failed']}",
            f"- Static panels requested: {summary['make_static_panel']}",
            f"- GIFs requested: {summary['make_gif']}",
            f"- Density panels requested: {summary['make_density_panels']}",
            f"- Difference-density panels requested: {summary['make_density_difference_panels']}",
            f"- Density GIFs requested: {summary['make_density_gifs']}",
            f"- Density panels completed: {summary['n_density_panels_completed']}",
            f"- Difference-density panels completed: {summary['n_density_difference_panels_completed']}",
            f"- Density GIFs completed: {summary['n_density_gifs_completed']}",
            f"- Density method: {summary['density_method']}",
            f"- Density grid size: {summary['density_grid_size']}",
            f"- Density sigma: {summary['density_sigma']}",
            f"- Minimum points for density: {summary['min_points_for_density']}",
            f"- Windows with density: {summary['n_windows_with_density']}",
            f"- Windows with insufficient density points: {summary['n_windows_insufficient_points']}",
            "",
            "## Selection",
            "",
            f"- Selection source: {summary['selection']['selection_source']}",
            f"- Ranking column: {summary['selection']['selection_ranking_column'] or 'not applicable'}",
            f"- Requested top N: {summary['selection']['top_n']}",
            f"- Top dynamic file: {summary['top_dynamic_from'] or 'not used'}",
            "",
            "## Density Method Note",
            "",
            (
                "These panels use one fixed full-period UMAP coordinate system per subfield. "
                "Densities are estimated separately for each five-year window and normalized "
                "within each window, so the panels show redistribution of semantic mass rather "
                "than publication-volume changes. Difference-density panels subtract the "
                "2000-2004 normalized density from later normalized densities. These figures "
                "are visual summaries only; quantitative temporal movement is reported in the "
                "embedding-space centroid and metric-evolution tables."
            ),
            "",
            "## Warnings And Failures",
            "",
        ]
    )
    warnings = manifest.loc[manifest["warning_message"].fillna("") != ""]
    failures = manifest.loc[manifest["status"] == "failed"]
    if warnings.empty and failures.empty:
        lines.append("No warnings or failures were recorded.")
    else:
        for row in warnings.head(20).itertuples(index=False):
            lines.append(f"- Warning for {row.subfield_label_short}: {row.warning_message}")
        for row in failures.head(20).itertuples(index=False):
            lines.append(f"- Failed {row.subfield_label_short}: {row.error_message}")
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- UMAP panels are visual summaries of within-subfield distributions; quantitative movement is in the centroid path tables.",
            "- Density and difference-density panels are interpretive aids, not quantitative evidence.",
            "- Reused coordinates may contain a deterministic sample if script 10 capped papers per subfield.",
            "",
        ]
    )
    return "\n".join(lines)
