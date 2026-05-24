from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tmp_visual_preview"
API_COUNTS_PATH = OUT_DIR / "openalex_funnel_api_counts.json"


COLORS = {
    "ink": "#111111",
    "muted": "#64748B",
    "grid": "#DDDDDD",
    "blue": "#4C8BAD",
    "blue_dark": "#2F6F95",
    "blue_fill": "#EFF6FA",
    "blue_band": "#DCEBF3",
    "rust": "#C95847",
    "rust_dark": "#A94535",
    "rust_fill": "#FCF3F0",
    "rust_band": "#F4E3DF",
    "green": "#2F7D57",
    "green_fill": "#E2F0E9",
    "connector_api": "#D9DCEB",
    "connector_bridge": "#E3E6EA",
    "connector_snapshot": "#F4DCD8",
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
    if value >= 10_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    return f"{value:,}"


def fmt_percent(value: float) -> str:
    if value >= 0.10:
        return f"{value * 100:.0f}%"
    return f"{value * 100:.2f}%"


def read_api_counts() -> dict[str, int]:
    rows = json.loads(API_COUNTS_PATH.read_text(encoding="utf-8"))
    return {row["label"]: int(row["count"]) for row in rows}


def draw_header(ax, x0, x1, y, text, face, edge):
    header = FancyBboxPatch(
        (x0, y),
        x1 - x0,
        0.052,
        boxstyle="round,pad=0.008,rounding_size=0.010",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.0,
    )
    ax.add_patch(header)
    ax.text(
        (x0 + x1) / 2,
        y + 0.026,
        text,
        ha="center",
        va="center",
        color=edge,
        fontsize=8.4,
        fontweight="bold",
    )


def draw_connector(ax, left, right, y_mid, h0, h1, color):
    x0, w0 = left
    x1, _ = right
    verts = [
        (x0 + w0, y_mid + h0 / 2),
        (x1, y_mid + h1 / 2),
        (x1, y_mid - h1 / 2),
        (x0 + w0, y_mid - h0 / 2),
    ]
    ax.add_patch(Polygon(verts, closed=True, facecolor=color, edgecolor="none", alpha=0.86, zorder=1))


def draw_step(ax, x, w, y_mid, h, step):
    is_api = step["source"] == "api"
    edge = COLORS["blue_dark"] if is_api else COLORS["rust"]
    face = COLORS["blue_fill"] if is_api else COLORS["rust_fill"]
    count_color = COLORS["blue_dark"] if is_api else COLORS["rust_dark"]

    ax.text(
        x + w / 2,
        0.710,
        step["label"],
        ha="center",
        va="bottom",
        color=COLORS["ink"],
        fontsize=7.9,
        fontweight="bold",
    )
    ax.text(
        x + w / 2,
        0.686,
        step["description"],
        ha="center",
        va="top",
        color=COLORS["muted"],
        fontsize=6.7,
    )

    rect = Rectangle(
        (x, y_mid - h / 2),
        w,
        h,
        facecolor=face,
        edgecolor=edge,
        linewidth=1.25,
        zorder=3,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y_mid - min(h * 0.25, 0.065),
        fmt_count(step["count"]),
        ha="center",
        va="center",
        color=count_color,
        fontsize=8.4,
        fontweight="bold",
        zorder=4,
    )
    ax.text(
        x + w / 2,
        0.115,
        step["pct_label"],
        ha="center",
        va="center",
        color="#475569",
        fontsize=7.8,
        fontweight="bold",
    )


def main() -> None:
    api = read_api_counts()
    api_pool = api["Has abstract, not retracted"]

    raw_steps = [
        ("OpenAlex\nworks", "current global\nWorks index", api["OpenAlex works"], "api"),
        ("2000-2024", "publication-date window", api["Published 2000-2024"], "api"),
        ("article or\npreprint", "document-type filter", api["Article or preprint"], "api"),
        ("English\nrecords", "language filter", api["English records"], "api"),
        ("abstract,\nnot retracted", "broad API\ntext pool", api_pool, "api"),
        ("planned\nsample", "252 subfields;\n<=400/year", 2_431_303, "snapshot"),
        ("validated\ncorpus", "local text\nand metadata", 2_378_036, "snapshot"),
        ("analysis\nsubset", "row-aligned\nSPECTER2", 2_344_927, "snapshot"),
    ]

    steps = []
    for idx, (label, description, count, source) in enumerate(raw_steps):
        if idx == 0:
            pct = "100%"
        elif idx <= 4:
            pct = fmt_percent(count / raw_steps[idx - 1][2])
        else:
            pct = fmt_percent(count / api_pool)
        steps.append(
            {
                "label": label,
                "description": description,
                "count": count,
                "source": source,
                "pct_label": pct,
            }
        )

    # Figurative heights: the goal is to show the narrowing pipeline clearly,
    # not to encode the full several-orders-of-magnitude reduction literally.
    heights = [0.48, 0.41, 0.35, 0.30, 0.25, 0.19, 0.145, 0.110]

    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.955,
        "OpenAlex scope to analysis corpus",
        ha="center",
        va="center",
        color=COLORS["ink"],
        fontsize=10.8,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.918,
        "Filter stages from the live OpenAlex works index to the frozen embedding-analysis corpus",
        ha="center",
        va="center",
        color=COLORS["muted"],
        fontsize=7.8,
    )

    xs = [0.035, 0.160, 0.285, 0.410, 0.535, 0.675, 0.800, 0.915]
    widths = [0.078, 0.078, 0.078, 0.078, 0.078, 0.072, 0.072, 0.072]
    y_mid = 0.390

    draw_header(
        ax,
        xs[0] - 0.010,
        xs[4] + widths[4] + 0.012,
        0.825,
        "OpenAlex API live counts, queried 25 May 2026",
        COLORS["blue_band"],
        COLORS["blue_dark"],
    )
    draw_header(
        ax,
        xs[5] - 0.010,
        0.985,
        0.825,
        "Frozen TFM pipeline snapshot",
        COLORS["rust_band"],
        COLORS["rust_dark"],
    )

    for idx in range(len(steps) - 1):
        if idx < 4:
            color = COLORS["connector_api"]
        elif idx == 4:
            color = COLORS["connector_bridge"]
        else:
            color = COLORS["connector_snapshot"]
        draw_connector(
            ax,
            (xs[idx], widths[idx]),
            (xs[idx + 1], widths[idx + 1]),
            y_mid,
            heights[idx],
            heights[idx + 1],
            color,
        )

    for idx, step in enumerate(steps):
        draw_step(ax, xs[idx], widths[idx], y_mid, heights[idx], step)

    fig.savefig(OUT_DIR / "openalex_pipeline_flow_tfm_preview.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / "openalex_pipeline_flow_tfm_preview.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
