from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "memory" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


COLORS = {
    "ink": "#111111",
    "muted": "#64748B",
    "grid": "#DDDDDD",
    "blue": "#4C8BAD",
    "blue_light": "#7FB3D5",
    "rust": "#C95847",
    "grey": "#7D8793",
    "light_grey": "#C6CCD2",
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
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def style_axis(ax, xlabel=None, ylabel=None):
    ax.set_facecolor("white")
    ax.grid(True, color=COLORS["grid"], linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#222222")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(length=3.0, width=0.8, colors=COLORS["ink"])
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=3)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=3)


def save_figure(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_citation_semantic_contrast():
    rng = np.random.default_rng(8)
    fig, (ax_graph, ax_space) = plt.subplots(
        1, 2, figsize=(7.35, 3.0), constrained_layout=True
    )

    # Panel A: link-defined graph.
    center_a = np.array([-1.05, 0.35])
    center_b = np.array([1.05, -0.35])
    pts_a = center_a + rng.normal(scale=[0.32, 0.28], size=(12, 2))
    pts_b = center_b + rng.normal(scale=[0.32, 0.28], size=(12, 2))
    pts = np.vstack([pts_a, pts_b])

    edge_pairs = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            same = (i < 12 and j < 12) or (i >= 12 and j >= 12)
            distance = np.linalg.norm(pts[i] - pts[j])
            p = 0.54 * np.exp(-distance) if same else 0.055 * np.exp(-0.35 * distance)
            if rng.random() < p:
                edge_pairs.append((i, j))

    for i, j in edge_pairs:
        ax_graph.plot(
            [pts[i, 0], pts[j, 0]],
            [pts[i, 1], pts[j, 1]],
            color="#8D99A6",
            lw=0.9,
            alpha=0.55,
            zorder=1,
        )

    ax_graph.scatter(
        pts_a[:, 0],
        pts_a[:, 1],
        s=35,
        color=COLORS["blue"],
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    ax_graph.scatter(
        pts_b[:, 0],
        pts_b[:, 1],
        s=35,
        color=COLORS["rust"],
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    ax_graph.set_xlim(-2.0, 2.0)
    ax_graph.set_ylim(-1.55, 1.55)
    ax_graph.set_title("A. Citation graph", color=COLORS["ink"], pad=5)
    ax_graph.text(
        0.5,
        0.03,
        "Observed links define proximity",
        transform=ax_graph.transAxes,
        ha="center",
        va="bottom",
        color=COLORS["muted"],
        fontsize=8.0,
    )
    style_axis(ax_graph, "layout coordinate 1", "layout coordinate 2")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            label="community A",
            markerfacecolor=COLORS["blue"],
            markeredgecolor="white",
            markersize=6,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            label="community B",
            markerfacecolor=COLORS["rust"],
            markeredgecolor="white",
            markersize=6,
        ),
    ]
    ax_graph.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#D0D0D0",
        fontsize=7.8,
    )

    # Panel B: semantic geometry as a continuous density field.
    n = 190
    sem_a = rng.normal(loc=[-1.35, 0.75], scale=[0.62, 0.48], size=(n, 2))
    sem_b = rng.normal(loc=[1.35, -0.85], scale=[0.72, 0.52], size=(n, 2))
    sem_bridge = rng.normal(loc=[0.15, -0.05], scale=[0.9, 0.6], size=(75, 2))
    sem = np.vstack([sem_a, sem_b, sem_bridge])

    xx = np.linspace(-3.1, 3.1, 180)
    yy = np.linspace(-2.45, 2.35, 150)
    X, Y = np.meshgrid(xx, yy)

    def gaussian_density(mu, sigma, weight):
        sx, sy = sigma
        return weight * np.exp(-0.5 * (((X - mu[0]) / sx) ** 2 + ((Y - mu[1]) / sy) ** 2))

    Z = (
        gaussian_density((-1.35, 0.75), (0.75, 0.58), 1.0)
        + gaussian_density((1.35, -0.85), (0.86, 0.62), 0.94)
        + gaussian_density((0.15, -0.05), (1.15, 0.82), 0.34)
    )

    levels = np.quantile(Z, [0.50, 0.66, 0.78, 0.88, 0.95])
    fill_levels = np.r_[Z.min(), levels, Z.max()]
    ax_space.contourf(X, Y, Z, levels=fill_levels, cmap="Blues", alpha=0.13)
    ax_space.contour(X, Y, Z, levels=levels, colors=COLORS["blue_light"], linewidths=0.8)
    ax_space.scatter(
        sem[:, 0],
        sem[:, 1],
        s=10,
        facecolor=COLORS["grey"],
        edgecolor="white",
        linewidth=0.25,
        alpha=0.34,
        zorder=3,
    )
    ax_space.text(
        -1.33,
        0.82,
        "semantic core A",
        color=COLORS["blue"],
        weight="bold",
        ha="center",
        fontsize=8.0,
    )
    ax_space.text(
        1.36,
        -0.92,
        "semantic core B",
        color=COLORS["rust"],
        weight="bold",
        ha="center",
        fontsize=8.0,
    )
    ax_space.text(
        0.5,
        0.03,
        "Distances and densities define proximity",
        transform=ax_space.transAxes,
        ha="center",
        va="bottom",
        color=COLORS["muted"],
        fontsize=8.0,
    )
    ax_space.set_xlim(-3.1, 3.1)
    ax_space.set_ylim(-2.45, 2.35)
    ax_space.set_title("B. Semantic embedding space", color=COLORS["ink"], pad=5)
    style_axis(ax_space, "SPECTER2 coordinate 1", "SPECTER2 coordinate 2")
    save_figure(fig, "fig_02_citation_semantic_contrast")


def make_temporal_modes_schematic():
    rng = np.random.default_rng(24)
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 3.2))
    fig.subplots_adjust(left=0.03, right=0.995, top=0.86, bottom=0.22, wspace=0.13)

    def setup_panel(ax, title):
        ax.set_facecolor("white")
        ax.set_xlim(-3.0, 3.0)
        ax.set_ylim(-3.0, 3.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(1)
        ax.set_title(title, color=COLORS["ink"], pad=5, fontsize=9.5)
        ax.grid(True, color=COLORS["grid"], linewidth=0.8, alpha=0.75)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color("#D0D0D0")
            side.set_linewidth(0.8)

    def scatter_cloud(ax, pts, color, alpha, size=15):
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=size,
            color=color,
            alpha=alpha,
            edgecolor="white",
            linewidth=0.25,
            zorder=3,
        )

    # A. Drift.
    ax = axes[0]
    setup_panel(ax, "A. Centroid drift")
    early = rng.normal(loc=[-1.25, -0.65], scale=[0.45, 0.35], size=(45, 2))
    late = rng.normal(loc=[1.25, 0.65], scale=[0.45, 0.35], size=(45, 2))
    scatter_cloud(ax, early, COLORS["grey"], 0.42)
    scatter_cloud(ax, late, COLORS["rust"], 0.78)
    ax.annotate(
        "",
        xy=(1.25, 0.65),
        xytext=(-1.25, -0.65),
        arrowprops=dict(arrowstyle="->", color=COLORS["rust"], lw=2.0, shrinkA=5, shrinkB=5),
        zorder=4,
    )
    ax.scatter([-1.25], [-0.65], s=38, color=COLORS["ink"], edgecolor="white", linewidth=0.6, zorder=5)
    ax.scatter([1.25], [0.65], s=46, color=COLORS["rust"], edgecolor="white", linewidth=0.6, zorder=5)
    ax.text(
        0.5,
        0.04,
        "center moves, spread is similar",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color=COLORS["muted"],
        fontsize=7.2,
    )

    # B. Expansion.
    ax = axes[1]
    setup_panel(ax, "B. Radial expansion")
    early = rng.normal(loc=[0, 0], scale=[0.38, 0.34], size=(45, 2))
    late = rng.normal(loc=[0, 0], scale=[1.25, 1.02], size=(65, 2))
    late = np.clip(late, [-2.55, -2.20], [2.55, 2.20])
    scatter_cloud(ax, late, COLORS["rust"], 0.70)
    scatter_cloud(ax, early, COLORS["grey"], 0.50)
    ax.add_patch(Circle((0, 0), 0.68, fill=False, ec=COLORS["grey"], lw=1.1, ls=(0, (4, 2))))
    ax.add_patch(Circle((0, 0), 1.82, fill=False, ec=COLORS["rust"], lw=1.2, ls=(0, (4, 2))))
    ax.text(
        0.5,
        0.04,
        "papers occupy a wider radius",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color=COLORS["muted"],
        fontsize=7.2,
    )

    # C. Contraction.
    ax = axes[2]
    setup_panel(ax, "C. Densification")
    early = rng.normal(loc=[0, 0], scale=[1.22, 0.96], size=(65, 2))
    late = rng.normal(loc=[0.18, 0.02], scale=[0.40, 0.34], size=(55, 2))
    scatter_cloud(ax, early, COLORS["grey"], 0.36)
    scatter_cloud(ax, late, COLORS["rust"], 0.78)
    ax.add_patch(Circle((0, 0), 1.82, fill=False, ec=COLORS["muted"], lw=1.1, ls=(0, (4, 2))))
    ax.add_patch(Circle((0.18, 0.02), 0.68, fill=False, ec=COLORS["rust"], lw=1.2, ls=(0, (4, 2))))
    ax.text(
        0.5,
        0.04,
        "papers pack into a tighter region",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color=COLORS["muted"],
        fontsize=7.2,
    )

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["grey"], alpha=0.6, markersize=5, label="early window"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["rust"], alpha=0.8, markersize=5, label="late window"),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=2,
        frameon=False,
        fontsize=8.0,
    )
    save_figure(fig, "fig_07_temporal_modes_schematic")


if __name__ == "__main__":
    make_citation_semantic_contrast()
    make_temporal_modes_schematic()
    print(f"Wrote explanatory figures to {OUT_DIR}")
