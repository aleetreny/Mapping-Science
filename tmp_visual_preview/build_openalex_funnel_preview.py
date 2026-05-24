from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tmp_visual_preview"
API_COUNTS_PATH = OUT_DIR / "openalex_funnel_api_counts.json"


COLORS = {
    "ink": "#111111",
    "muted": "#64748B",
    "grid": "#DDDDDD",
    "blue": "#4C8BAD",
    "blue_dark": "#2F6F95",
    "blue_light": "#DCEBF3",
    "rust": "#C95847",
    "rust_dark": "#A94535",
    "rust_light": "#F4E3DF",
    "green": "#2F7D57",
    "green_light": "#E2F0E9",
    "line": "#9AA3AD",
}


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.titlesize": 11.0,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "figure.dpi": 180,
        "savefig.dpi": 320,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def fmt_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:,}"


def fmt_ratio(value: float) -> str:
    if value >= 0.1:
        return f"{value * 100:.0f}%"
    return f"{value * 100:.1f}%"


def read_api_counts() -> dict[str, int]:
    rows = json.loads(API_COUNTS_PATH.read_text(encoding="utf-8"))
    return {row["label"]: int(row["count"]) for row in rows}


def add_band(ax, x0, x1, y, text, face, edge):
    band_h = 0.035
    band = FancyBboxPatch(
        (x0, y),
        x1 - x0,
        band_h,
        boxstyle="round,pad=0.010,rounding_size=0.020",
        linewidth=0.8,
        edgecolor=edge,
        facecolor=face,
        zorder=1,
    )
    ax.add_patch(band)
    ax.text(
        (x0 + x1) / 2,
        y + band_h / 2,
        text,
        ha="center",
        va="center",
        color=edge,
        fontsize=7.5,
        fontweight="bold",
        zorder=2,
    )


def main() -> None:
    api = read_api_counts()
    steps = [
        {
            "label": "OpenAlex works",
            "count": api["OpenAlex works"],
            "note": "current global Works index",
            "source": "api",
        },
        {
            "label": "2000-2024",
            "count": api["Published 2000-2024"],
            "note": "publication-date window",
            "source": "api",
        },
        {
            "label": "article or preprint",
            "count": api["Article or preprint"],
            "note": "document-type filter",
            "source": "api",
        },
        {
            "label": "English records",
            "count": api["English records"],
            "note": "language filter",
            "source": "api",
        },
        {
            "label": "abstract, not retracted",
            "count": api["Has abstract, not retracted"],
            "note": "broad API text pool",
            "source": "api",
        },
        {
            "label": "planned sample",
            "count": 2_431_303,
            "note": "252 subfields; <=400/year",
            "source": "snapshot",
        },
        {
            "label": "validated corpus",
            "count": 2_378_036,
            "note": "local text and metadata checks",
            "source": "snapshot",
        },
        {
            "label": "analysis subset",
            "count": 2_344_927,
            "note": "row-aligned SPECTER2 matrix",
            "source": "snapshot",
        },
    ]

    log_counts = [math.log10(step["count"]) for step in steps]
    lo, hi = min(log_counts), max(log_counts)
    widths = [0.22 + 0.70 * (v - lo) / (hi - lo) for v in log_counts]

    fig, ax = plt.subplots(figsize=(7.35, 4.55))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.02,
        0.965,
        "OpenAlex scope to analysis corpus",
        ha="left",
        va="center",
        color=COLORS["ink"],
        fontsize=11.0,
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.925,
        "Each row keeps the filters above it; bar length is log-scaled so the sampled corpus remains visible.",
        ha="left",
        va="center",
        color=COLORS["muted"],
        fontsize=7.7,
    )

    add_band(
        ax,
        0.02,
        0.96,
        0.855,
        "OpenAlex API live counts, queried 24 May 2026",
        COLORS["blue_light"],
        COLORS["blue_dark"],
    )
    add_band(
        ax,
        0.02,
        0.96,
        0.330,
        "Frozen TFM pipeline snapshot",
        COLORS["rust_light"],
        COLORS["rust_dark"],
    )

    y_positions = [0.795, 0.705, 0.615, 0.525, 0.435, 0.270, 0.180, 0.090]
    label_x = 0.04
    bar_x = 0.35
    bar_max_w = 0.39
    count_x = 0.79
    ratio_x = 0.95
    bar_h = 0.030

    for i, step in enumerate(steps):
        edge = COLORS["blue_dark"] if step["source"] == "api" else COLORS["rust_dark"]
        face = COLORS["blue_light"] if step["source"] == "api" else COLORS["rust_light"]
        y = y_positions[i]
        ax.text(
            label_x,
            y + 0.010,
            step["label"],
            ha="left",
            va="center",
            color=COLORS["ink"],
            fontsize=8.3,
            fontweight="bold",
        )
        ax.text(
            label_x,
            y - 0.020,
            step["note"],
            ha="left",
            va="center",
            color=COLORS["muted"],
            fontsize=6.8,
        )
        bg = FancyBboxPatch(
            (bar_x, y - bar_h / 2),
            bar_max_w,
            bar_h,
            boxstyle="round,pad=0.002,rounding_size=0.010",
            linewidth=0,
            facecolor="#EEF1F4",
            zorder=1,
        )
        ax.add_patch(bg)
        fg = FancyBboxPatch(
            (bar_x, y - bar_h / 2),
            widths[i] * bar_max_w,
            bar_h,
            boxstyle="round,pad=0.002,rounding_size=0.010",
            linewidth=0,
            facecolor=face,
            zorder=2,
        )
        ax.add_patch(fg)
        ax.plot(
            [bar_x, bar_x + widths[i] * bar_max_w],
            [y, y],
            color=edge,
            linewidth=1.1,
            solid_capstyle="round",
            zorder=3,
        )
        ax.text(
            count_x,
            y,
            fmt_count(step["count"]),
            ha="right",
            va="center",
            color=edge,
            fontsize=9.2,
            fontweight="bold",
        )
        if i > 0:
            ratio = step["count"] / steps[i - 1]["count"]
            ax.text(
                ratio_x,
                y,
                fmt_ratio(ratio),
                ha="right",
                va="center",
                color=COLORS["muted"],
                fontsize=7.0,
            )
        else:
            ax.text(
                ratio_x,
                y,
                "retained",
                ha="right",
                va="center",
                color=COLORS["muted"],
                fontsize=6.8,
            )

    ax.text(0.79, 0.832, "works", ha="right", va="center", color=COLORS["muted"], fontsize=6.9)
    ax.text(0.95, 0.832, "from previous", ha="right", va="center", color=COLORS["muted"], fontsize=6.9)

    ax.plot([0.02, 0.96], [0.385, 0.385], color="#D8DDE3", lw=0.8)

    metric_text = "241 analysis subfields -> 11 morphology metrics"
    metric_box = FancyBboxPatch(
        (0.35, 0.018),
        0.39,
        0.038,
        boxstyle="round,pad=0.010,rounding_size=0.012",
        linewidth=0.8,
        edgecolor=COLORS["green"],
        facecolor=COLORS["green_light"],
        zorder=2,
    )
    ax.add_patch(metric_box)
    ax.text(
        0.545,
        0.037,
        metric_text,
        ha="center",
        va="center",
        color=COLORS["green"],
        fontsize=7.1,
        fontweight="bold",
    )

    note = (
        "API stages are broad, reproducible OpenAlex filters. Title/abstract token thresholds, "
        "paratext exclusion, balanced sampling, validation, and embedding alignment are local "
        "pipeline stages in dataset version 2000_2024_400py."
    )
    ax.text(
        0.02,
        -0.025,
        "Note. " + textwrap.fill(note, width=135),
        ha="left",
        va="top",
        color=COLORS["muted"],
        fontsize=6.9,
    )

    fig.savefig(OUT_DIR / "openalex_corpus_funnel_preview.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / "openalex_corpus_funnel_preview.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
