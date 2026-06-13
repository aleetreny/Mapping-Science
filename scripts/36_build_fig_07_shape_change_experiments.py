from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, gaussian_filter1d, label


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

EARLY = ("2000-2004", 2000, 2004)
LATE = ("2020-2024", 2020, 2024)
EARLY_COLOR = "#2b6cb0"
LATE_COLOR = "#d65f4b"
FIELD_COLOR = "#9a9a9a"
PATH_COLOR = "#151515"


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
    for column in ["publication_year", "umap_x", "umap_y"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["publication_year", "umap_x", "umap_y"]).copy()


def subset_window(frame: pd.DataFrame, window: tuple[str, int, int]) -> pd.DataFrame:
    _, start, end = window
    return frame.loc[frame["publication_year"].between(start, end, inclusive="both")].copy()


def limits(frame: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    x = frame["umap_x"].to_numpy(dtype=float)
    y = frame["umap_y"].to_numpy(dtype=float)
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    xmid = (xmin + xmax) / 2
    ymid = (ymin + ymax) / 2
    side = max(xmax - xmin, ymax - ymin) * 1.12
    half = side / 2
    return (
        (xmid - half, xmid + half),
        (ymid - half, ymid + half),
    )


def density_grid(
    frame: pd.DataFrame,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    grid_size: int = 280,
    sigma: float = 2.3,
) -> np.ndarray:
    coords = frame[["umap_x", "umap_y"]].to_numpy(dtype=float)
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
    return np.nan_to_num(grid)


def top_mass_threshold(grid: np.ndarray, mass: float) -> float:
    values = np.sort(grid.ravel())[::-1]
    total = float(values.sum())
    if total <= 0:
        return 1.0
    cumulative = np.cumsum(values)
    index = int(np.searchsorted(cumulative, total * mass, side="left"))
    index = min(max(index, 0), len(values) - 1)
    return float(values[index])


def alpha_density(grid: np.ndarray, gamma: float = 0.58) -> np.ndarray:
    vmax = float(np.nanmax(grid))
    if vmax <= 0:
        return grid
    return np.power(np.clip(grid / vmax, 0, 1), gamma)


def draw_field(ax: plt.Axes, full: np.ndarray, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.imshow(
        alpha_density(full),
        origin="lower",
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        cmap="Greys",
        vmin=0,
        vmax=1,
        alpha=0.76,
        zorder=1,
    )


def draw_mass_silhouette(
    ax: plt.Axes,
    grid: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    color: str,
    mass: float,
    linewidth: float = 2.4,
    alpha: float = 0.24,
) -> None:
    threshold = top_mass_threshold(grid, mass)
    vmax = float(np.nanmax(grid))
    if vmax <= 0 or threshold >= vmax:
        return
    ax.contourf(
        grid,
        levels=[threshold, vmax],
        origin="lower",
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        colors=[color],
        alpha=alpha,
        zorder=4,
    )
    ax.contour(
        grid,
        levels=[threshold],
        origin="lower",
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        colors=[color],
        linewidths=linewidth,
        alpha=0.96,
        zorder=5,
    )


def window_centroids(frame: pd.DataFrame) -> pd.DataFrame:
    windows = [
        ("2000-2004", 2000, 2004),
        ("2005-2009", 2005, 2009),
        ("2010-2014", 2010, 2014),
        ("2015-2019", 2015, 2019),
        ("2020-2024", 2020, 2024),
    ]
    rows = []
    for idx, (label_text, start, end) in enumerate(windows):
        subset = subset_window(frame, (label_text, start, end))
        rows.append(
            {
                "label": label_text,
                "index": idx,
                "x": float(subset["umap_x"].mean()) if len(subset) else np.nan,
                "y": float(subset["umap_y"].mean()) if len(subset) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def draw_centroid_path(ax: plt.Axes, centroids: pd.DataFrame) -> None:
    path = centroids.dropna(subset=["x", "y"])
    if len(path) < 2:
        return
    ax.plot(path["x"], path["y"], color=PATH_COLOR, linewidth=1.25, alpha=0.88, zorder=9)
    ax.scatter(path["x"], path["y"], s=18, color=PATH_COLOR, linewidths=0, alpha=0.92, zorder=10)
    ax.add_patch(
        FancyArrowPatch(
            (path.iloc[-2]["x"], path.iloc[-2]["y"]),
            (path.iloc[-1]["x"], path.iloc[-1]["y"]),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color=PATH_COLOR,
            zorder=11,
        )
    )
    ax.scatter(path.iloc[0]["x"], path.iloc[0]["y"], s=58, marker="o", facecolor="#ffffff", edgecolor=EARLY_COLOR, linewidth=1.5, zorder=12)
    ax.scatter(path.iloc[-1]["x"], path.iloc[-1]["y"], s=70, marker="X", facecolor="#ffffff", edgecolor=LATE_COLOR, linewidth=1.5, zorder=12)


def format_map(ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    ax.set_facecolor("#ffffff")


def title_case(ax: plt.Axes, name: str, field: str, color: str) -> None:
    ax.text(0.02, 1.045, name, transform=ax.transAxes, fontsize=12.5, weight="bold", color=color, ha="left", va="bottom")
    ax.text(0.02, 1.005, field, transform=ax.transAxes, fontsize=8.8, color="#555555", ha="left", va="bottom")


def radial_signature(frame: pd.DataFrame, *, n_bins: int = 48) -> tuple[np.ndarray, np.ndarray]:
    coords = frame[["umap_x", "umap_y"]].to_numpy(dtype=float)
    center = coords.mean(axis=0)
    shifted = coords - center
    angles = np.arctan2(shifted[:, 1], shifted[:, 0])
    radii = np.sqrt(np.sum(shifted**2, axis=1))
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    values = np.zeros(n_bins, dtype=float)
    for idx in range(n_bins):
        mask = (angles >= bins[idx]) & (angles < bins[idx + 1])
        values[idx] = float(np.percentile(radii[mask], 88)) if np.any(mask) else 0.0
    values = gaussian_filter1d(values, sigma=1.4, mode="wrap")
    theta = (bins[:-1] + bins[1:]) / 2
    theta = np.r_[theta, theta[0]]
    values = np.r_[values, values[0]]
    return theta, values


def option_13_equal_mass_silhouettes() -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.0))
    fig.patch.set_facecolor("#fbfaf7")
    fig.suptitle("Equal-mass semantic silhouettes", fontsize=17.0, weight="bold", y=0.982)
    fig.text(
        0.5,
        0.940,
        "Each colored contour encloses the densest 60% of papers in the first or last window, over the full field shape.",
        ha="center",
        fontsize=10.0,
        color="#4d4d4d",
    )
    for ax, (sid, name, field, color) in zip(axes.reshape(-1), CASES):
        frame = load_case(sid, name)
        xlim, ylim = limits(frame)
        full = density_grid(frame, xlim=xlim, ylim=ylim)
        early = density_grid(subset_window(frame, EARLY), xlim=xlim, ylim=ylim)
        late = density_grid(subset_window(frame, LATE), xlim=xlim, ylim=ylim)
        draw_field(ax, full, xlim, ylim)
        draw_mass_silhouette(ax, early, xlim, ylim, color=EARLY_COLOR, mass=0.60)
        draw_mass_silhouette(ax, late, xlim, ylim, color=LATE_COLOR, mass=0.60)
        draw_centroid_path(ax, window_centroids(frame))
        format_map(ax, xlim, ylim)
        title_case(ax, name, field, color)
    handles = [
        Patch(facecolor=FIELD_COLOR, alpha=0.35, label="Full field shape"),
        Line2D([0], [0], color=EARLY_COLOR, linewidth=2.4, label="2000-2004 densest 60%"),
        Line2D([0], [0], color=LATE_COLOR, linewidth=2.4, label="2020-2024 densest 60%"),
        Line2D([0], [0], color=PATH_COLOR, marker="o", linewidth=1.3, markersize=4.5, label="Centroid path"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.8, bbox_to_anchor=(0.5, 0.030))
    fig.tight_layout(rect=[0.025, 0.075, 0.985, 0.900], h_pad=2.4, w_pad=2.0)
    return save(fig, "fig_07_7_option_13_equal_mass_silhouettes")


def option_14_radial_shape_fingerprints() -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 9.2), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#ffffff")
    fig.suptitle("Radial shape fingerprints", fontsize=17.0, weight="bold", y=0.985)
    fig.text(
        0.5,
        0.945,
        "Each curve summarizes the directional footprint around that period's own centroid; larger lobes mean farther semantic reach.",
        ha="center",
        fontsize=9.8,
        color="#4d4d4d",
    )
    for ax, (sid, name, field, color) in zip(axes.reshape(-1), CASES):
        frame = load_case(sid, name)
        early = subset_window(frame, EARLY)
        late = subset_window(frame, LATE)
        theta_e, values_e = radial_signature(early)
        theta_l, values_l = radial_signature(late)
        scale = max(float(values_e.max()), float(values_l.max()), 1.0)
        values_e = values_e / scale
        values_l = values_l / scale
        ax.plot(theta_e, values_e, color=EARLY_COLOR, linewidth=2.2)
        ax.fill(theta_e, values_e, color=EARLY_COLOR, alpha=0.18)
        ax.plot(theta_l, values_l, color=LATE_COLOR, linewidth=2.2)
        ax.fill(theta_l, values_l, color=LATE_COLOR, alpha=0.18)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.grid(color="#dddddd", linewidth=0.65)
        ax.spines["polar"].set_color("#cfcfcf")
        ax.set_title(f"{name}\n{field}", fontsize=11.0, weight="bold", color=color, pad=14)
    handles = [
        Line2D([0], [0], color=EARLY_COLOR, linewidth=2.2, label="2000-2004 footprint"),
        Line2D([0], [0], color=LATE_COLOR, linewidth=2.2, label="2020-2024 footprint"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9.0, bbox_to_anchor=(0.5, 0.030))
    fig.tight_layout(rect=[0.04, 0.070, 0.98, 0.900], h_pad=2.0)
    return save(fig, "fig_07_7_option_14_radial_shape_fingerprints")


def component_centers(
    grid: np.ndarray,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    mass: float = 0.55,
    max_components: int = 5,
) -> list[tuple[float, float, float]]:
    threshold = top_mass_threshold(grid, mass)
    mask = grid >= threshold
    labels, n_labels = label(mask)
    centers: list[tuple[float, float, float]] = []
    height, width = grid.shape
    xs = np.linspace(xlim[0], xlim[1], width)
    ys = np.linspace(ylim[0], ylim[1], height)
    xx, yy = np.meshgrid(xs, ys)
    for component in range(1, n_labels + 1):
        component_mask = labels == component
        weights = grid[component_mask]
        total = float(weights.sum())
        if total <= 0:
            continue
        cx = float(np.average(xx[component_mask], weights=weights))
        cy = float(np.average(yy[component_mask], weights=weights))
        centers.append((cx, cy, total))
    centers.sort(key=lambda item: item[2], reverse=True)
    return centers[:max_components]


def option_15_semantic_island_flow() -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.6))
    fig.patch.set_facecolor("#ffffff")
    for ax, (sid, name, field, color) in zip(axes.reshape(-1), CASES):
        frame = load_case(sid, name)
        xlim, ylim = limits(frame)
        full = density_grid(frame, xlim=xlim, ylim=ylim)
        early = density_grid(subset_window(frame, EARLY), xlim=xlim, ylim=ylim)
        late = density_grid(subset_window(frame, LATE), xlim=xlim, ylim=ylim)
        draw_field(ax, full, xlim, ylim)
        draw_mass_silhouette(ax, early, xlim, ylim, color=EARLY_COLOR, mass=0.55, alpha=0.16, linewidth=1.8)
        draw_mass_silhouette(ax, late, xlim, ylim, color=LATE_COLOR, mass=0.55, alpha=0.16, linewidth=1.8)
        early_centers = component_centers(early, xlim=xlim, ylim=ylim, mass=0.55, max_components=5)
        late_centers = component_centers(late, xlim=xlim, ylim=ylim, mass=0.55, max_components=5)
        for ex, ey, mass_value in early_centers:
            if not late_centers:
                continue
            lx, ly, _ = min(late_centers, key=lambda item: (item[0] - ex) ** 2 + (item[1] - ey) ** 2)
            linewidth = 0.9 + 9.0 * mass_value
            ax.add_patch(
                FancyArrowPatch(
                    (ex, ey),
                    (lx, ly),
                    arrowstyle="-|>",
                    mutation_scale=11,
                    linewidth=linewidth,
                    color="#222222",
                    alpha=0.56,
                    zorder=8,
                )
            )
        ax.scatter([item[0] for item in early_centers], [item[1] for item in early_centers], s=42, facecolor="#ffffff", edgecolor=EARLY_COLOR, linewidth=1.5, zorder=9)
        ax.scatter([item[0] for item in late_centers], [item[1] for item in late_centers], s=48, marker="X", facecolor="#ffffff", edgecolor=LATE_COLOR, linewidth=1.5, zorder=9)
        format_map(ax, xlim, ylim)
        title_case(ax, name, field, color)
    handles = [
        Patch(facecolor=FIELD_COLOR, alpha=0.35, label="Full field shape"),
        Line2D([0], [0], color=EARLY_COLOR, linewidth=2.0, label="Early density island"),
        Line2D([0], [0], color=LATE_COLOR, linewidth=2.0, label="Late density island"),
        Line2D([0], [0], color="#222222", linewidth=1.8, label="Nearest-island flow"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.8, bbox_to_anchor=(0.5, 0.030))
    fig.tight_layout(rect=[0.025, 0.080, 0.985, 0.965], h_pad=2.2, w_pad=2.0)
    return save(fig, "fig_07_7_option_15_semantic_island_flow")


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    for candidate in ["arialbd.ttf" if bold else "arial.ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_contact(paths: list[Path]) -> Path:
    labels = [
        "13 equal-mass silhouettes",
        "14 radial fingerprints",
        "15 semantic island flow",
    ]
    thumbs = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((980, 680), Image.Resampling.LANCZOS)
        thumbs.append(image.copy())
    cell_w, cell_h = 1050, 780
    canvas = Image.new("RGB", (cell_w * 2, cell_h * 2), "#f5f5f2")
    draw = ImageDraw.Draw(canvas)
    for idx, (image, label_text) in enumerate(zip(thumbs, labels)):
        col = idx % 2
        row = idx // 2
        x0 = col * cell_w
        y0 = row * cell_h
        draw.text((x0 + 34, y0 + 22), label_text, font=_font(24), fill="#222222")
        canvas.paste(image, (x0 + (cell_w - image.width) // 2, y0 + 70 + (cell_h - 90 - image.height) // 2))
    out = OUT / "fig_07_7_shape_change_experiments_contact.png"
    canvas.save(out)
    return out


def main() -> None:
    setup()
    paths = [
        option_13_equal_mass_silhouettes(),
        option_14_radial_shape_fingerprints(),
        option_15_semantic_island_flow(),
    ]
    contact = build_contact(paths)
    print("Generated:")
    for path in paths + [contact]:
        print(f"- {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
