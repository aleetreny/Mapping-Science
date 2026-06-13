from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "memory" / "figures" / "alternatives"
COORD_DIR = ROOT / "outputs" / "08_visualization" / "per_subfield_umap_smooth_density" / "coordinates"

CASES = [
    ("1211", "Philosophy", "Arts and Humanities", "#8b6bb5"),
    ("3500", "General Dentistry", "Dentistry", "#4c78a8"),
    ("1313", "Molecular Medicine", "Molecular Biology", "#2f7f5f"),
    ("1711", "Signal Processing", "Computer Science", "#d65f4b"),
]

WINDOWS = [
    ("2000-2004", 2000, 2004),
    ("2005-2009", 2005, 2009),
    ("2010-2014", 2010, 2014),
    ("2015-2019", 2015, 2019),
    ("2020-2024", 2020, 2024),
]

EARLY = WINDOWS[0]
LATE = WINDOWS[-1]
EARLY_COLOR = "#2b6cb0"
LATE_COLOR = "#d65f4b"
LOSS_COLOR = "#2b6cb0"
GAIN_COLOR = "#c23b4b"
PATH_COLOR = "#171717"


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.dpi": 140,
            "savefig.dpi": 260,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
        }
    )


def save(fig: plt.Figure, stem: str) -> Path:
    path = OUT / f"{stem}.png"
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def stem_for(sid: str, name: str) -> str:
    return f"{sid}__{name.replace(' ', '_')}"


def load_case(sid: str, name: str) -> pd.DataFrame:
    path = COORD_DIR / f"{stem_for(sid, name)}.parquet"
    frame = pd.read_parquet(path)
    frame["publication_year"] = pd.to_numeric(frame["publication_year"], errors="coerce")
    frame["umap_x"] = pd.to_numeric(frame["umap_x"], errors="coerce")
    frame["umap_y"] = pd.to_numeric(frame["umap_y"], errors="coerce")
    return frame.dropna(subset=["publication_year", "umap_x", "umap_y"]).copy()


def limits(frame: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    x = frame["umap_x"].to_numpy(dtype=float)
    y = frame["umap_y"].to_numpy(dtype=float)
    xr = float(np.nanmax(x) - np.nanmin(x))
    yr = float(np.nanmax(y) - np.nanmin(y))
    pad = max(xr, yr) * 0.055
    return (
        (float(np.nanmin(x) - pad), float(np.nanmax(x) + pad)),
        (float(np.nanmin(y) - pad), float(np.nanmax(y) + pad)),
    )


def density(
    frame: pd.DataFrame,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    grid_size: int = 260,
    sigma: float = 2.0,
) -> np.ndarray:
    coords = frame[["umap_x", "umap_y"]].to_numpy(dtype=float)
    if len(coords) == 0:
        return np.zeros((grid_size, grid_size), dtype=float)
    hist, _, _ = np.histogram2d(
        coords[:, 0],
        coords[:, 1],
        bins=grid_size,
        range=[[xlim[0], xlim[1]], [ylim[0], ylim[1]]],
    )
    grid = gaussian_filter(hist.T, sigma=sigma)
    total = float(grid.sum())
    if total > 0:
        grid = grid / total
    return grid


def subset_window(frame: pd.DataFrame, window: tuple[str, int, int]) -> pd.DataFrame:
    _, start, end = window
    return frame.loc[frame["publication_year"].between(start, end, inclusive="both")].copy()


def window_centroids(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, (label, start, end) in enumerate(WINDOWS):
        subset = subset_window(frame, (label, start, end))
        rows.append(
            {
                "label": label,
                "index": index,
                "x": float(subset["umap_x"].mean()) if len(subset) else np.nan,
                "y": float(subset["umap_y"].mean()) if len(subset) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def density_alpha_grid(grid: np.ndarray, *, gamma: float = 0.55) -> np.ndarray:
    max_value = float(np.nanmax(grid))
    if max_value <= 0:
        return grid
    return np.power(np.clip(grid / max_value, 0, 1), gamma)


def positive_threshold(grid: np.ndarray, percentile: float) -> float:
    values = grid[np.isfinite(grid) & (grid > 0)]
    if len(values) == 0:
        return 1.0
    return float(np.percentile(values, percentile))


def draw_background(ax: plt.Axes, full_density: np.ndarray, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    display = density_alpha_grid(full_density)
    ax.imshow(
        display,
        origin="lower",
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        cmap="Greys",
        vmin=0,
        vmax=1,
        alpha=0.82,
        zorder=1,
    )
    level = positive_threshold(full_density, 82)
    ax.contour(
        full_density,
        levels=[level],
        origin="lower",
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        colors=["#b5b5b5"],
        linewidths=0.8,
        alpha=0.75,
        zorder=2,
    )


def draw_density_contour(
    ax: plt.Axes,
    grid: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    color: str,
    label_level: float = 78,
    zorder: int = 5,
) -> None:
    lo = positive_threshold(grid, label_level)
    hi = positive_threshold(grid, 91)
    max_value = float(np.nanmax(grid))
    if max_value <= 0 or lo >= max_value:
        return
    ax.contourf(
        grid,
        levels=[lo, max_value],
        origin="lower",
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        colors=[color],
        alpha=0.22,
        zorder=zorder,
    )
    levels = [level for level in [lo, hi] if level < max_value]
    if levels:
        ax.contour(
            grid,
            levels=levels,
            origin="lower",
            extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
            colors=[color],
            linewidths=[1.2, 2.0][: len(levels)],
            alpha=0.95,
            zorder=zorder + 1,
        )


def draw_centroid_path(ax: plt.Axes, centroids: pd.DataFrame, *, labels: bool = False) -> None:
    path = centroids.dropna(subset=["x", "y"])
    if len(path) < 2:
        return
    ax.plot(path["x"], path["y"], color=PATH_COLOR, linewidth=1.25, alpha=0.88, zorder=8)
    ax.scatter(path["x"], path["y"], s=16, color=PATH_COLOR, linewidths=0, alpha=0.90, zorder=9)
    ax.add_patch(
        FancyArrowPatch(
            (path.iloc[-2]["x"], path.iloc[-2]["y"]),
            (path.iloc[-1]["x"], path.iloc[-1]["y"]),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color=PATH_COLOR,
            zorder=10,
        )
    )
    ax.scatter(path.iloc[0]["x"], path.iloc[0]["y"], s=52, marker="o", facecolor="#ffffff", edgecolor=EARLY_COLOR, linewidth=1.4, zorder=11)
    ax.scatter(path.iloc[-1]["x"], path.iloc[-1]["y"], s=62, marker="X", facecolor="#ffffff", edgecolor=LATE_COLOR, linewidth=1.4, zorder=11)
    if labels:
        for _, row in path.iterrows():
            ax.text(row["x"], row["y"], str(int(row["index"]) + 1), fontsize=6.8, color="#111111", ha="center", va="center", zorder=12)


def format_map(ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")


def draw_case_title(ax: plt.Axes, name: str, field: str, color: str) -> None:
    ax.text(0.02, 1.045, name, transform=ax.transAxes, ha="left", va="bottom", fontsize=12.5, weight="bold", color=color)
    ax.text(0.02, 1.002, field, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.7, color="#555555")


def option_06_contour_overlay() -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.0))
    fig.patch.set_facecolor("#fbfaf7")
    fig.suptitle("Selected subfield shape changes as early-late overlays", fontsize=17.0, weight="bold", y=0.982)
    fig.text(
        0.5,
        0.940,
        "Grey shows the full 2000-2024 field shape. Blue marks 2000-2004 density; red marks 2020-2024 density.",
        ha="center",
        fontsize=10.0,
        color="#4d4d4d",
    )

    for ax, (sid, name, field, color) in zip(axes.reshape(-1), CASES):
        frame = load_case(sid, name)
        xlim, ylim = limits(frame)
        full = density(frame, xlim=xlim, ylim=ylim, sigma=2.1)
        early = density(subset_window(frame, EARLY), xlim=xlim, ylim=ylim, sigma=2.1)
        late = density(subset_window(frame, LATE), xlim=xlim, ylim=ylim, sigma=2.1)
        draw_background(ax, full, xlim, ylim)
        draw_density_contour(ax, early, xlim, ylim, color=EARLY_COLOR, label_level=76)
        draw_density_contour(ax, late, xlim, ylim, color=LATE_COLOR, label_level=76)
        draw_centroid_path(ax, window_centroids(frame))
        format_map(ax, xlim, ylim)
        draw_case_title(ax, name, field, color)

    handles = [
        Patch(facecolor="#bdbdbd", edgecolor="#8c8c8c", alpha=0.55, label="Full-period field shape"),
        Line2D([0], [0], color=EARLY_COLOR, linewidth=2.2, label="2000-2004 high-density region"),
        Line2D([0], [0], color=LATE_COLOR, linewidth=2.2, label="2020-2024 high-density region"),
        Line2D([0], [0], color=PATH_COLOR, marker="o", linewidth=1.3, markersize=4.5, label="Window centroid path"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.8, bbox_to_anchor=(0.5, 0.030))
    fig.tight_layout(rect=[0.025, 0.075, 0.985, 0.900], h_pad=2.4, w_pad=2.0)
    return save(fig, "fig_07_7_option_06_umap_early_late_overlay")


def option_07_before_after_pairs() -> Path:
    fig, axes = plt.subplots(4, 2, figsize=(10.8, 13.2))
    fig.patch.set_facecolor("#ffffff")
    fig.suptitle("The same field shape, two temporal windows", fontsize=17.0, weight="bold", y=0.992)
    fig.text(
        0.5,
        0.970,
        "Each row fixes the UMAP coordinates for one subfield; the grey background keeps the whole field visible.",
        ha="center",
        fontsize=9.8,
        color="#4d4d4d",
    )
    axes[0, 0].set_title("2000-2004", fontsize=11.5, weight="bold", pad=16)
    axes[0, 1].set_title("2020-2024", fontsize=11.5, weight="bold", pad=16)

    for row, (sid, name, field, color) in enumerate(CASES):
        frame = load_case(sid, name)
        xlim, ylim = limits(frame)
        full = density(frame, xlim=xlim, ylim=ylim, sigma=2.1)
        early = density(subset_window(frame, EARLY), xlim=xlim, ylim=ylim, sigma=2.1)
        late = density(subset_window(frame, LATE), xlim=xlim, ylim=ylim, sigma=2.1)
        centroids = window_centroids(frame)
        for col, (grid, c) in enumerate([(early, EARLY_COLOR), (late, LATE_COLOR)]):
            ax = axes[row, col]
            draw_background(ax, full, xlim, ylim)
            draw_density_contour(ax, grid, xlim, ylim, color=c, label_level=73)
            draw_centroid_path(ax, centroids)
            format_map(ax, xlim, ylim)
            if col == 0:
                ax.text(-0.05, 0.5, f"{name}\n{field}", transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=9.8, weight="bold", color=color)

    handles = [
        Patch(facecolor="#bdbdbd", edgecolor="#8c8c8c", alpha=0.55, label="Full-period field shape"),
        Line2D([0], [0], color=EARLY_COLOR, linewidth=2.2, label="Early density"),
        Line2D([0], [0], color=LATE_COLOR, linewidth=2.2, label="Late density"),
        Line2D([0], [0], color=PATH_COLOR, marker="o", linewidth=1.3, markersize=4.5, label="Centroid path"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.8, bbox_to_anchor=(0.5, 0.020))
    fig.tight_layout(rect=[0.075, 0.055, 0.985, 0.940], h_pad=1.4, w_pad=1.2)
    return save(fig, "fig_07_7_option_07_umap_before_after_pairs")


def option_08_change_overlay() -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.0))
    fig.patch.set_facecolor("#fbfaf7")
    fig.suptitle("Where each subfield gains and loses relative density", fontsize=17.0, weight="bold", y=0.982)
    fig.text(
        0.5,
        0.940,
        "Grey keeps the full field visible. Red shows 2020-2024 density gain relative to 2000-2004; blue shows loss.",
        ha="center",
        fontsize=10.0,
        color="#4d4d4d",
    )

    for ax, (sid, name, field, color) in zip(axes.reshape(-1), CASES):
        frame = load_case(sid, name)
        xlim, ylim = limits(frame)
        full = density(frame, xlim=xlim, ylim=ylim, sigma=2.1)
        early = density(subset_window(frame, EARLY), xlim=xlim, ylim=ylim, sigma=2.1)
        late = density(subset_window(frame, LATE), xlim=xlim, ylim=ylim, sigma=2.1)
        diff = late - early
        draw_background(ax, full, xlim, ylim)
        magnitude = np.abs(diff)
        threshold = positive_threshold(magnitude, 90)
        gain = np.ma.masked_where(diff <= threshold, diff)
        loss = np.ma.masked_where(diff >= -threshold, -diff)
        vmax = max(float(np.nanmax(magnitude)), threshold * 1.01)
        ax.imshow(
            gain,
            origin="lower",
            extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
            cmap=matplotlib.colors.LinearSegmentedColormap.from_list("gain", ["#ffffff00", GAIN_COLOR]),
            vmin=threshold,
            vmax=vmax,
            alpha=0.78,
            zorder=5,
        )
        ax.imshow(
            loss,
            origin="lower",
            extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
            cmap=matplotlib.colors.LinearSegmentedColormap.from_list("loss", ["#ffffff00", LOSS_COLOR]),
            vmin=threshold,
            vmax=vmax,
            alpha=0.78,
            zorder=5,
        )
        draw_centroid_path(ax, window_centroids(frame))
        format_map(ax, xlim, ylim)
        draw_case_title(ax, name, field, color)

    handles = [
        Patch(facecolor="#bdbdbd", edgecolor="#8c8c8c", alpha=0.55, label="Full-period field shape"),
        Patch(facecolor=GAIN_COLOR, alpha=0.78, label="Relative density gain"),
        Patch(facecolor=LOSS_COLOR, alpha=0.78, label="Relative density loss"),
        Line2D([0], [0], color=PATH_COLOR, marker="o", linewidth=1.3, markersize=4.5, label="Window centroid path"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.8, bbox_to_anchor=(0.5, 0.030))
    fig.tight_layout(rect=[0.025, 0.075, 0.985, 0.900], h_pad=2.4, w_pad=2.0)
    return save(fig, "fig_07_7_option_08_umap_gain_loss_overlay")


def transparent_cmap(name: str, color: str) -> LinearSegmentedColormap:
    cmap = LinearSegmentedColormap.from_list(name, [(1, 1, 1, 0), color])
    cmap.set_bad((1, 1, 1, 0))
    return cmap


def draw_density_wash(
    ax: plt.Axes,
    grid: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    color: str,
    name: str,
    threshold_percentile: float = 70,
    alpha: float = 0.62,
    zorder: int = 5,
) -> None:
    threshold = positive_threshold(grid, threshold_percentile)
    max_value = float(np.nanmax(grid))
    if max_value <= 0 or threshold >= max_value:
        return
    masked = np.ma.masked_less(grid, threshold)
    ax.imshow(
        masked,
        origin="lower",
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        cmap=transparent_cmap(name, color),
        vmin=threshold,
        vmax=max_value,
        alpha=alpha,
        zorder=zorder,
    )


def option_09_clean_density_overlay() -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.0))
    fig.patch.set_facecolor("#fbfaf7")
    fig.suptitle("Early and late semantic mass on the full field shape", fontsize=17.0, weight="bold", y=0.982)
    fig.text(
        0.5,
        0.940,
        "Grey gives the full subfield map. Blue and red highlight only the denser early and late regions.",
        ha="center",
        fontsize=10.0,
        color="#4d4d4d",
    )

    for ax, (sid, name, field, color) in zip(axes.reshape(-1), CASES):
        frame = load_case(sid, name)
        xlim, ylim = limits(frame)
        full = density(frame, xlim=xlim, ylim=ylim, sigma=2.1)
        early = density(subset_window(frame, EARLY), xlim=xlim, ylim=ylim, sigma=2.1)
        late = density(subset_window(frame, LATE), xlim=xlim, ylim=ylim, sigma=2.1)
        draw_background(ax, full, xlim, ylim)
        draw_density_wash(ax, early, xlim, ylim, color=EARLY_COLOR, name=f"{sid}_early", threshold_percentile=70, alpha=0.58)
        draw_density_wash(ax, late, xlim, ylim, color=LATE_COLOR, name=f"{sid}_late", threshold_percentile=70, alpha=0.58)
        draw_centroid_path(ax, window_centroids(frame))
        format_map(ax, xlim, ylim)
        draw_case_title(ax, name, field, color)

    handles = [
        Patch(facecolor="#bdbdbd", edgecolor="#8c8c8c", alpha=0.55, label="Full-period field shape"),
        Patch(facecolor=EARLY_COLOR, alpha=0.58, label="2000-2004 dense region"),
        Patch(facecolor=LATE_COLOR, alpha=0.58, label="2020-2024 dense region"),
        Line2D([0], [0], color=PATH_COLOR, marker="o", linewidth=1.3, markersize=4.5, label="Window centroid path"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.8, bbox_to_anchor=(0.5, 0.030))
    fig.tight_layout(rect=[0.025, 0.075, 0.985, 0.900], h_pad=2.4, w_pad=2.0)
    return save(fig, "fig_07_7_option_09_umap_clean_density_overlay")


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    for candidate in ["arialbd.ttf" if bold else "arial.ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_contact(paths: list[Path]) -> Path:
    labels = [
        "06 early-late overlay",
        "07 before/after pairs",
        "08 gain/loss overlay",
        "09 clean density overlay",
    ]
    thumbs = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((980, 680), Image.Resampling.LANCZOS)
        thumbs.append(image.copy())
    cell_w, cell_h = 1050, 780
    canvas = Image.new("RGB", (cell_w * 2, cell_h * 2), "#f5f5f2")
    draw = ImageDraw.Draw(canvas)
    for idx, (image, label) in enumerate(zip(thumbs, labels)):
        col = idx % 2
        row = idx // 2
        x0 = col * cell_w
        y0 = row * cell_h
        draw.text((x0 + 34, y0 + 22), label, font=_font(24), fill="#222222")
        canvas.paste(image, (x0 + (cell_w - image.width) // 2, y0 + 70 + (cell_h - 90 - image.height) // 2))
    out = OUT / "fig_07_7_map_redesign_options_contact.png"
    canvas.save(out)
    return out


def main() -> None:
    setup()
    paths = [
        option_06_contour_overlay(),
        option_07_before_after_pairs(),
        option_08_change_overlay(),
        option_09_clean_density_overlay(),
    ]
    contact = build_contact(paths)
    print("Generated:")
    for path in paths + [contact]:
        print(f"- {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
