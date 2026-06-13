from __future__ import annotations

from pathlib import Path
from textwrap import shorten

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "memory" / "figures" / "alternatives"
MAP_SOURCE = ROOT / "outputs" / "17_fig_07_visual_options" / "exact_temporal_umap_panels"
METRICS_PATH = ROOT / "data" / "processed" / "temporal" / "subfield_window_embedding_metrics.csv"

CASES = [
    ("1211", "Philosophy", "Arts and Humanities", "#8b6bb5"),
    ("3500", "General Dentistry", "Dentistry", "#4c78a8"),
    ("1313", "Molecular Medicine", "Molecular Biology", "#2f7f5f"),
    ("1711", "Signal Processing", "Computer Science", "#d65f4b"),
]

METRICS = [
    ("embedding_distance_to_centroid_iqr", "Dispersion"),
    ("embedding_knn_median_distance", "Local distance"),
    ("embedding_pca_dim_80", "PCA D80"),
    ("embedding_knn_distance_cv", "kNN CV"),
]

WINDOWS = ["2000-2004", "2005-2009", "2010-2014", "2015-2019", "2020-2024"]


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.dpi": 140,
            "savefig.dpi": 260,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, stem: str) -> Path:
    path = OUT / f"{stem}.png"
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def load_metric_zscores() -> pd.DataFrame:
    frame = pd.read_csv(METRICS_PATH, dtype={"subfield_id": str})
    frame = frame.loc[frame["metric_status"].eq("completed")].copy()
    for column, _ in METRICS:
        values = pd.to_numeric(frame[column], errors="coerce")
        frame[f"{column}_z"] = (values - values.mean()) / values.std(ddof=0)
    return frame


def selected_metric_frame() -> pd.DataFrame:
    frame = load_metric_zscores()
    selected = frame.loc[frame["subfield_id"].isin([case[0] for case in CASES])].copy()
    selected["_case_order"] = selected["subfield_id"].map({case[0]: i for i, case in enumerate(CASES)})
    selected["window_label"] = pd.Categorical(selected["window_label"], categories=WINDOWS, ordered=True)
    return selected.sort_values(["_case_order", "window_label"], kind="mergesort")


def option_01_metric_delta_bars(frame: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.3, 7.8), sharex=True)
    fig.patch.set_facecolor("#fbfaf7")
    fig.suptitle("First-to-last morphology shifts", fontsize=17.0, weight="bold", y=0.985)
    fig.text(
        0.5,
        0.940,
        "Bars show standardized 2020-2024 minus 2000-2004 change for the same four examples.",
        ha="center",
        fontsize=10.0,
        color="#444444",
    )
    for ax, (sid, name, field, color) in zip(axes.reshape(-1), CASES):
        case = frame.loc[frame["subfield_id"].eq(sid)].copy()
        initial = case.loc[case["window_label"].astype(str).eq("2000-2004")].iloc[0]
        final = case.loc[case["window_label"].astype(str).eq("2020-2024")].iloc[0]
        labels = [label for _, label in METRICS]
        deltas = np.array([final[f"{column}_z"] - initial[f"{column}_z"] for column, _ in METRICS], dtype=float)
        y = np.arange(len(labels))
        colors = ["#b84a43" if value > 0 else "#3b6e8a" for value in deltas]
        ax.barh(y, deltas, color=colors, alpha=0.90)
        ax.axvline(0, color="#2d2d2d", linewidth=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9.0)
        ax.invert_yaxis()
        ax.grid(axis="x", color="#e6e2da", linewidth=0.7)
        ax.set_title(f"{name}\n{field}", fontsize=12.0, weight="bold", color=color, pad=8)
        for yi, value in zip(y, deltas):
            x = value + (0.06 if value >= 0 else -0.06)
            ha = "left" if value >= 0 else "right"
            ax.text(x, yi, f"{value:+.1f}", ha=ha, va="center", fontsize=8.6, color="#222222")
    fig.text(0.5, 0.035, "Blue = lower final relative position; red = higher final relative position.", ha="center", fontsize=8.2, color="#555555")
    fig.tight_layout(rect=[0.04, 0.065, 0.99, 0.900])
    return save(fig, "fig_07_7_option_01_first_last_metric_bars")


def option_02_metric_heatmaps(frame: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.0))
    fig.patch.set_facecolor("#ffffff")
    fig.suptitle("Windowed metric profiles for the selected examples", fontsize=17.0, weight="bold", y=0.985)
    fig.text(
        0.5,
        0.940,
        "Each small heatmap keeps the five temporal windows, but removes the visual noise of four overlaid lines.",
        ha="center",
        fontsize=10.0,
        color="#444444",
    )
    matrix_values: list[np.ndarray] = []
    for sid, *_ in CASES:
        case = frame.loc[frame["subfield_id"].eq(sid)].sort_values("window_label")
        matrix_values.append(np.array([[row[f"{column}_z"] for row in case.to_dict("records")] for column, _ in METRICS], dtype=float))
    limit = float(np.nanmax(np.abs(np.concatenate([m.ravel() for m in matrix_values]))))
    limit = min(max(limit, 2.5), 6.0)

    for ax, matrix, (sid, name, field, color) in zip(axes.reshape(-1), matrix_values, CASES):
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_title(f"{name}\n{field}", fontsize=12.0, weight="bold", color=color, pad=8)
        ax.set_xticks(np.arange(len(WINDOWS)))
        ax.set_xticklabels(WINDOWS, rotation=25, ha="right", fontsize=8.3)
        ax.set_yticks(np.arange(len(METRICS)))
        ax.set_yticklabels([label for _, label in METRICS], fontsize=8.8)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                text_color = "#ffffff" if abs(value) > limit * 0.55 else "#202020"
                ax.text(col, row, f"{value:+.1f}", ha="center", va="center", fontsize=7.7, color=text_color)
    fig.subplots_adjust(left=0.075, right=0.875, bottom=0.090, top=0.865, wspace=0.34, hspace=0.48)
    cbar_ax = fig.add_axes([0.900, 0.190, 0.018, 0.600])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("metric-level z-score", fontsize=8.5)
    return save(fig, "fig_07_7_option_02_metric_heatmaps")


def _open_panel(path: Path, target_width: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    ratio = target_width / image.width
    return image.resize((target_width, int(image.height * ratio)), Image.Resampling.LANCZOS)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = ["arialbd.ttf" if bold else "arial.ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_centered(draw: ImageDraw.ImageDraw, x0: int, x1: int, y: int, text: str, font: ImageFont.ImageFont, fill: str) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    draw.text((x0 + (x1 - x0 - (box[2] - box[0])) // 2, y), text, font=font, fill=fill)


def option_03_map_density_rows() -> Path:
    panels = []
    for sid, name, *_ in CASES:
        path = MAP_SOURCE / "density_panels" / f"{sid}__{name.replace(' ', '_')}_density_panel.png"
        panels.append(_open_panel(path, 2500))
    pad = 28
    title_h = 120
    width = max(panel.width for panel in panels) + pad * 2
    height = title_h + sum(panel.height for panel in panels) + pad * (len(panels) + 1)
    canvas = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    _draw_centered(draw, 0, width, 24, "Option 03 - temporal density maps across five windows", _font(34, True), "#111111")
    _draw_centered(draw, 0, width, 72, "Fixed within-subfield UMAP coordinates; density is normalized inside each period.", _font(20), "#555555")
    y = title_h
    for panel in panels:
        x = (width - panel.width) // 2
        canvas.paste(panel, (x, y))
        y += panel.height + pad
    out = OUT / "fig_07_7_option_03_umap_density_windows.png"
    canvas.save(out)
    return out


def option_04_map_difference_rows() -> Path:
    panels = []
    for sid, name, *_ in CASES:
        path = MAP_SOURCE / "density_difference_panels" / f"{sid}__{name.replace(' ', '_')}_density_difference_panel.png"
        panels.append(_open_panel(path, 2500))
    pad = 28
    title_h = 120
    width = max(panel.width for panel in panels) + pad * 2
    height = title_h + sum(panel.height for panel in panels) + pad * (len(panels) + 1)
    canvas = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    _draw_centered(draw, 0, width, 24, "Option 04 - density change maps versus 2000-2004", _font(34, True), "#111111")
    _draw_centered(draw, 0, width, 72, "Red marks gain in relative density; blue marks loss. Each row is one selected subfield.", _font(20), "#555555")
    y = title_h
    for panel in panels:
        x = (width - panel.width) // 2
        canvas.paste(panel, (x, y))
        y += panel.height + pad
    out = OUT / "fig_07_7_option_04_umap_density_difference_rows.png"
    canvas.save(out)
    return out


def _crop_late_difference(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    w, h = image.size
    box = (int(w * 0.725), int(h * 0.195), int(w * 0.858), int(h * 0.820))
    return image.crop(box)


def option_05_late_difference_grid() -> Path:
    crops = []
    for sid, name, field, color in CASES:
        path = MAP_SOURCE / "density_difference_panels" / f"{sid}__{name.replace(' ', '_')}_density_difference_panel.png"
        crop = _crop_late_difference(path)
        crop.thumbnail((710, 390), Image.Resampling.LANCZOS)
        crops.append((crop.copy(), name, field, color))

    pad = 38
    label_h = 70
    title_h = 130
    cell_w = 820
    cell_h = 520
    width = cell_w * 2 + pad * 3
    height = title_h + cell_h * 2 + pad * 3 + 38
    canvas = Image.new("RGB", (width, height), "#fbfaf7")
    draw = ImageDraw.Draw(canvas)
    _draw_centered(draw, 0, width, 28, "Option 05 - final-window density change only", _font(34, True), "#111111")
    _draw_centered(draw, 0, width, 76, "A compact map version: 2020-2024 minus 2000-2004 for each example.", _font(20), "#555555")

    for idx, (crop, name, field, color) in enumerate(crops):
        col = idx % 2
        row = idx // 2
        x0 = pad + col * (cell_w + pad)
        y0 = title_h + pad + row * (cell_h + pad)
        draw.rounded_rectangle((x0, y0, x0 + cell_w, y0 + cell_h), radius=12, fill="#ffffff", outline="#d2cec6", width=2)
        draw.text((x0 + 24, y0 + 18), name, font=_font(24, True), fill=color)
        draw.text((x0 + 24, y0 + 48), field, font=_font(18), fill="#555555")
        x = x0 + (cell_w - crop.width) // 2
        y = y0 + label_h + (cell_h - label_h - crop.height) // 2
        canvas.paste(crop, (x, y))
    _draw_centered(draw, 0, width, height - 42, "Red = relative density gain; blue = relative density loss. Coordinates are fixed within each subfield.", _font(18), "#555555")
    out = OUT / "fig_07_7_option_05_umap_late_difference_grid.png"
    canvas.save(out)
    return out


def build_contact_sheet(paths: list[Path]) -> Path:
    labels = [
        "01 metric bars",
        "02 metric heatmaps",
        "03 density windows",
        "04 density differences",
        "05 final difference grid",
    ]
    thumbs = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((860, 520), Image.Resampling.LANCZOS)
        thumbs.append(image.copy())
    cols = 2
    rows = 3
    pad = 32
    label_h = 34
    cell_w = 940
    cell_h = 610
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), "#f5f5f2")
    draw = ImageDraw.Draw(canvas)
    for idx, (thumb, label) in enumerate(zip(thumbs, labels)):
        col = idx % cols
        row = idx // cols
        x0 = col * cell_w
        y0 = row * cell_h
        draw.text((x0 + pad, y0 + 18), label, font=_font(22), fill="#222222")
        x = x0 + (cell_w - thumb.width) // 2
        y = y0 + pad + label_h
        canvas.paste(thumb, (x, y))
    out = OUT / "fig_07_7_visual_options_contact.png"
    canvas.save(out)
    return out


def main() -> None:
    setup()
    frame = selected_metric_frame()
    paths = [
        option_01_metric_delta_bars(frame),
        option_02_metric_heatmaps(frame),
        option_03_map_density_rows(),
        option_04_map_difference_rows(),
        option_05_late_difference_grid(),
    ]
    contact = build_contact_sheet(paths)
    print("Generated:")
    for path in paths + [contact]:
        print(f"- {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
