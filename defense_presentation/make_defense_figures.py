"""Generate the custom slide figures for the defense presentation.

Reads the pipeline outputs already stored in the repository and produces
vector PDFs in defense_presentation/assets/. Run from the repo root:

    .venv/Scripts/python.exe defense_presentation/make_defense_figures.py
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "defense_presentation" / "assets"

UC3M_NAVY = "#000066"
DOMAIN_COLORS = {
    "Life Sciences": "#2f8961",
    "Social Sciences": "#8f7bbf",
    "Physical Sciences": "#c96a32",
    "Health Sciences": "#4d7fb5",
}
DOMAIN_ORDER = ["Life Sciences", "Social Sciences", "Physical Sciences", "Health Sciences"]

METRICS = {
    "embedding_distance_to_centroid_median": "spread",
    "embedding_distance_to_centroid_iqr": "spread unevenness",
    "embedding_distance_to_centroid_p90": "outer spread",
    "embedding_knn_median_distance": "local distance",
    "embedding_knn_distance_cv": "local unevenness",
    "embedding_knn_indegree_gini": "hub concentration",
    "embedding_pca_dim_80": "dimensions",
    "embedding_pca_spectral_entropy": "direction evenness",
}
METRIC_HIGHLIGHT_COLORS = {
    "embedding_distance_to_centroid_median": "#00337f",
    "embedding_knn_median_distance": "#c96a32",
    "embedding_knn_distance_cv": "#8f7bbf",
    "embedding_knn_indegree_gini": "#a63d40",
    "embedding_pca_dim_80": "#2f8961",
}

mpl.rcParams.update(
    {
        "font.family": "Segoe UI",
        "font.size": 11,
        "axes.edgecolor": "#8a93a1",
        "axes.linewidth": 0.8,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#5a6472",
        "ytick.color": "#5a6472",
        "text.color": "#1c232d",
        "axes.labelcolor": "#3c4552",
        "figure.facecolor": "white",
        "savefig.bbox": "tight",
    }
)


def save(fig, name):
    fig.savefig(ASSETS / f"{name}.pdf")
    fig.savefig(ASSETS / f"{name}.png", dpi=150)
    plt.close(fig)
    print(f"saved {name}")


def fig_morphospace():
    """241 subfields in the spread / local-distance plane, colored by domain."""
    import numpy as np

    df = pd.read_csv(ROOT / "outputs/11_active_8metric_rebuild/subfield_structural_profiles_scaled.csv")
    x = "embedding_distance_to_centroid_median"
    y = "embedding_knn_median_distance"

    fig, ax = plt.subplots(figsize=(9.4, 4.55))
    for dom in DOMAIN_ORDER:
        sub = df[df["domain_display_name"] == dom]
        ax.scatter(sub[x], sub[y], s=17, color=DOMAIN_COLORS[dom], alpha=0.40,
                   edgecolors="none", zorder=2)

    # Soft covariance ellipse per domain (1.6 sigma) as its "shape signature".
    for dom in DOMAIN_ORDER:
        sub = df[df["domain_display_name"] == dom]
        pts = sub[[x, y]].to_numpy()
        mean = pts.mean(axis=0)
        cov = np.cov(pts.T)
        evals, evecs = np.linalg.eigh(cov)
        order = evals.argsort()[::-1]
        evals, evecs = evals[order], evecs[:, order]
        angle = float(np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0])))
        w, h = 2 * 1.6 * np.sqrt(evals)
        ell = mpl.patches.Ellipse(mean, w, h, angle=angle,
                                  facecolor=DOMAIN_COLORS[dom], alpha=0.10,
                                  edgecolor=DOMAIN_COLORS[dom], lw=1.4, zorder=3)
        ell.set_edgecolor(mpl.colors.to_rgba(DOMAIN_COLORS[dom], 0.85))
        ax.add_patch(ell)
        ax.scatter([mean[0]], [mean[1]], s=190, marker="D", color=DOMAIN_COLORS[dom],
                   edgecolors="white", linewidths=1.7, zorder=4)
        offsets = {
            "Life Sciences": (34, -6),
            "Social Sciences": (-26, 22),
            "Physical Sciences": (30, 12),
            "Health Sciences": (-34, -24),
        }
        dx, dy = offsets[dom]
        ha = "left" if dx > 0 else "right"
        va = "bottom" if dy > 0 else "top"
        ax.annotate(dom, (mean[0], mean[1]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=12, fontweight="bold", color=DOMAIN_COLORS[dom],
                    ha=ha, va=va, zorder=5,
                    path_effects=[mpl.patheffects.withStroke(linewidth=3.2, foreground="white")])

    ax.text(0.015, 0.03, "compact, tightly packed", transform=ax.transAxes,
            fontsize=10.5, color="#8a93a1", style="italic", ha="left")
    ax.text(0.985, 0.96, "broad, loosely packed", transform=ax.transAxes,
            fontsize=10.5, color="#8a93a1", style="italic", ha="right", va="top")

    ax.set_xlabel("Spread of papers around the field centre  →", fontsize=12)
    ax.set_ylabel("Distance between\nneighbouring papers  →", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["bottom"].set_color("#ccd3db")
    ax.spines["left"].set_color("#ccd3db")
    ax.margins(x=0.05, y=0.07)
    save(fig, "def_morphospace")


SHORT_SUBFIELD = {
    "Applied Microbiology and Biotechnology": "Applied Microbiology & Biotech.",
    "Nuclear and High Energy Physics": "Nuclear & High-Energy Physics",
    "Molecular Biology": "Molecular Biology",
    "Artificial Intelligence": "Artificial Intelligence",
    "General Materials Science": "General Materials Science",
    "Business and International Management": "Business & Intl. Management",
    "Classics": "Classics",
}

DOT_COLOR = "#5f6b7a"


def _bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("left", "bottom", "top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_aspect("equal")


def _extreme_strip(ax, df, metric, color, lowtag, hightag, rng):
    """All 241 subfields on one metric, extremes named."""
    vals = df[metric].to_numpy()
    jitter = rng.uniform(-0.16, 0.16, len(vals))
    ax.scatter(vals, jitter, s=16, color="#b3bdc9", alpha=0.75, edgecolors="none", zorder=2)
    lo = df.loc[df[metric].idxmin()]
    hi = df.loc[df[metric].idxmax()]
    for row, tag, ha in ((lo, lowtag, "left"), (hi, hightag, "right")):
        v = row[metric]
        name = SHORT_SUBFIELD.get(row["subfield_display_name"], row["subfield_display_name"])
        ax.scatter([v], [0], s=130, color=color, edgecolors="white", linewidths=1.5, zorder=4)
        anchor = v - 0.08 if ha == "left" else v + 0.08
        ax.text(anchor, -0.62, f"{tag}: {name}", fontsize=10.5, color=color,
                fontweight="bold", ha=ha, va="center", zorder=5)
    ax.text(0.5, 0.97, "the real 241 subfields on this measure", transform=ax.transAxes,
            fontsize=10, color="#8a93a1", style="italic", ha="center", va="top")
    ax.set_xlim(vals.min() - 0.28, vals.max() + 0.28)
    ax.set_ylim(-1.0, 0.62)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("left", "bottom", "top", "right"):
        ax.spines[side].set_visible(False)


def _family_figure(name, metric, lowtag, hightag, low_title, high_title,
                   draw_low, draw_high):
    import numpy as np

    rng = np.random.default_rng(11)
    df = pd.read_csv(ROOT / "outputs/11_active_8metric_rebuild/subfield_structural_profiles_scaled.csv")
    color = METRIC_HIGHLIGHT_COLORS.get(metric, UC3M_NAVY)

    fig = plt.figure(figsize=(10.4, 4.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.9, 1.5], hspace=0.16, wspace=0.08)
    ax_lo = fig.add_subplot(gs[0, 0])
    ax_hi = fig.add_subplot(gs[0, 1])
    ax_strip = fig.add_subplot(gs[1, :])

    draw_low(ax_lo, np.random.default_rng(3), color)
    draw_high(ax_hi, np.random.default_rng(5), color)
    for ax, title in ((ax_lo, low_title), (ax_hi, high_title)):
        _bare(ax)
        ax.set_title(title, fontsize=13.5, color="#1c232d", pad=8)

    _extreme_strip(ax_strip, df, metric, color, lowtag, hightag, rng)
    save(fig, name)


def fig_metric_families():
    """Four figures, one per metric family: concept illustration + real extremes."""
    import numpy as np

    def spread_low(ax, rng, color):
        pts = rng.normal(0, 0.45, (85, 2))
        ax.scatter(pts[:, 0], pts[:, 1], s=16, color=DOT_COLOR, alpha=0.75, edgecolors="none")
        r = float(np.median(np.linalg.norm(pts, axis=1)))
        ax.add_patch(mpl.patches.Circle((0, 0), r, fill=False, color=color,
                                        lw=1.8, linestyle=(0, (5, 3))))
        ax.scatter([0], [0], s=150, marker="D", color=color, edgecolors="white",
                   linewidths=1.5, zorder=4)
        ax.set_xlim(-3.1, 3.1)
        ax.set_ylim(-2.4, 2.4)

    def spread_high(ax, rng, color):
        pts = rng.normal(0, 1.15, (85, 2))
        ax.scatter(pts[:, 0], pts[:, 1], s=16, color=DOT_COLOR, alpha=0.75, edgecolors="none")
        r = float(np.median(np.linalg.norm(pts, axis=1)))
        ax.add_patch(mpl.patches.Circle((0, 0), r, fill=False, color=color,
                                        lw=1.8, linestyle=(0, (5, 3))))
        ax.scatter([0], [0], s=150, marker="D", color=color, edgecolors="white",
                   linewidths=1.5, zorder=4)
        ax.set_xlim(-3.1, 3.1)
        ax.set_ylim(-2.4, 2.4)

    _family_figure("def_family_spread", "embedding_distance_to_centroid_median",
                   "most compact", "most spread out",
                   "compact: papers stay near the centre", "spread out: papers reach far",
                   spread_low, spread_high)

    def _nn_links(ax, pts, color):
        for i in range(len(pts)):
            d = np.linalg.norm(pts - pts[i], axis=1)
            d[i] = np.inf
            j = int(np.argmin(d))
            ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
                    color=color, lw=1.0, alpha=0.85, zorder=1)

    def packing_low(ax, rng, color):
        centers = rng.uniform(-2.3, 2.3, (14, 2)) * np.array([1.15, 0.75])
        pts = np.vstack([c + rng.normal(0, 0.10, (4, 2)) for c in centers])
        _nn_links(ax, pts, color)
        ax.scatter(pts[:, 0], pts[:, 1], s=16, color=DOT_COLOR, alpha=0.8,
                   edgecolors="none", zorder=2)
        ax.set_xlim(-3.1, 3.1)
        ax.set_ylim(-2.4, 2.4)

    def packing_high(ax, rng, color):
        pts = rng.uniform(-1, 1, (52, 2)) * np.array([2.75, 2.0])
        _nn_links(ax, pts, color)
        ax.scatter(pts[:, 0], pts[:, 1], s=16, color=DOT_COLOR, alpha=0.8,
                   edgecolors="none", zorder=2)
        ax.set_xlim(-3.1, 3.1)
        ax.set_ylim(-2.4, 2.4)

    _family_figure("def_family_packing", "embedding_knn_median_distance",
                   "tightest neighbourhoods", "loosest neighbourhoods",
                   "dense: each paper has neighbours right beside it",
                   "sparse: the nearest neighbour is far away",
                   packing_low, packing_high)

    def hubs_low(ax, rng, color):
        pts = rng.uniform(-1, 1, (46, 2)) * np.array([2.75, 2.0])
        _nn_links(ax, pts, "#c2cad4")
        ax.scatter(pts[:, 0], pts[:, 1], s=16, color=DOT_COLOR, alpha=0.8,
                   edgecolors="none", zorder=2)
        ax.set_xlim(-3.1, 3.1)
        ax.set_ylim(-2.4, 2.4)

    def hubs_high(ax, rng, color):
        hubs = np.array([[-1.25, 0.25], [1.35, -0.35]])
        pts = rng.uniform(-1, 1, (44, 2)) * np.array([2.75, 2.0])
        for p in pts:
            h = hubs[int(np.argmin(np.linalg.norm(hubs - p, axis=1)))]
            ax.plot([p[0], h[0]], [p[1], h[1]], color=color, lw=0.9, alpha=0.55, zorder=1)
        ax.scatter(pts[:, 0], pts[:, 1], s=16, color=DOT_COLOR, alpha=0.8,
                   edgecolors="none", zorder=2)
        ax.scatter(hubs[:, 0], hubs[:, 1], s=210, color=color, edgecolors="white",
                   linewidths=1.6, zorder=4)
        ax.set_xlim(-3.1, 3.1)
        ax.set_ylim(-2.4, 2.4)

    _family_figure("def_family_hubs", "embedding_knn_indegree_gini",
                   "no dominant papers", "a few papers dominate",
                   "balanced: attention is spread evenly",
                   "hub-driven: many links point to the same few papers",
                   hubs_low, hubs_high)

    def dims_low(ax, rng, color):
        t = rng.normal(0, 1.35, 80)
        direction = np.array([np.cos(0.35), np.sin(0.35)])
        pts = np.outer(t, direction) + rng.normal(0, 0.10, (80, 2))
        ax.scatter(pts[:, 0], pts[:, 1], s=16, color=DOT_COLOR, alpha=0.8, edgecolors="none")
        ax.annotate("", xy=(2.45 * direction[0], 2.45 * direction[1]),
                    xytext=(-2.45 * direction[0], -2.45 * direction[1]),
                    arrowprops=dict(arrowstyle="<->", color=color, lw=2.2))
        ax.set_xlim(-3.1, 3.1)
        ax.set_ylim(-2.4, 2.4)

    def dims_high(ax, rng, color):
        pts = rng.normal(0, 0.95, (85, 2))
        ax.scatter(pts[:, 0], pts[:, 1], s=16, color=DOT_COLOR, alpha=0.8, edgecolors="none")
        for ang in np.linspace(0, np.pi, 4, endpoint=False):
            d = np.array([np.cos(ang), np.sin(ang)])
            ax.annotate("", xy=(1.9 * d[0], 1.9 * d[1]), xytext=(-1.9 * d[0], -1.9 * d[1]),
                        arrowprops=dict(arrowstyle="<->", color=color, lw=1.7, alpha=0.85))
        ax.set_xlim(-3.1, 3.1)
        ax.set_ylim(-2.4, 2.4)

    _family_figure("def_family_dims", "embedding_pca_dim_80",
                   "fewest directions", "most directions",
                   "simple: one direction explains almost everything",
                   "complex: variation needs many directions",
                   dims_low, dims_high)


def _shape_glyph(ax, rng, color, z_spread, z_pack, z_hub, z_dims,
                 with_arrows=False, n_clusters=15, pts_per_cluster=7):
    """Draw a stylized paper cloud whose look is driven by real profile values:
    radius = spread, clumpiness = local packing, red stars = hubs,
    elongation/arrows = dimensionality."""
    import numpy as np

    R = float(np.clip(1.0 + 0.42 * z_spread, 0.70, 1.55))
    ratio = float(np.clip(0.88 + 0.30 * z_dims, 0.34, 1.0))
    sigma = float(np.clip(0.16 + 0.10 * z_pack, 0.075, 0.30)) * R
    theta = 0.42
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    ang = rng.uniform(0, 2 * np.pi, n_clusters)
    rad = np.sqrt(rng.uniform(0.05, 1.0, n_clusters))
    centers = np.column_stack([rad * np.cos(ang) * R, rad * np.sin(ang) * R * ratio]) @ rot.T
    pts = np.vstack([c + rng.normal(0, sigma, (pts_per_cluster, 2)) for c in centers])

    if z_hub > 0.15:
        n_hubs = 2 if z_hub < 0.5 else 3
        hub_idx = rng.choice(len(pts), n_hubs, replace=False)
        hubs = pts[hub_idx] * 0.55
        for p in pts[rng.choice(len(pts), int(len(pts) * 0.45), replace=False)]:
            h = hubs[int(np.argmin(np.linalg.norm(hubs - p, axis=1)))]
            ax.plot([p[0], h[0]], [p[1], h[1]], color="#a63d40", lw=0.6, alpha=0.28, zorder=1)
        ax.scatter(hubs[:, 0], hubs[:, 1], s=55 + 55 * R, color="#a63d40",
                   edgecolors="white", linewidths=1.3, zorder=4)

    ax.scatter(pts[:, 0], pts[:, 1], s=15, color=color, alpha=0.75, edgecolors="none", zorder=2)

    r_med = float(np.median(np.linalg.norm(pts, axis=1)))
    ax.add_patch(mpl.patches.Circle((0, 0), r_med, fill=False, color=color,
                                    lw=1.5, linestyle=(0, (5, 3)), alpha=0.85, zorder=3))

    if with_arrows:
        main = rot @ np.array([1.0, 0.0])
        if ratio < 0.6:
            ends = [(main, R * 1.35)]
        else:
            ends = [(rot @ np.array([np.cos(a), np.sin(a)]), R * 1.12 * ratio)
                    for a in np.linspace(0, np.pi, 4, endpoint=False)]
        for d, ln in ends:
            ax.annotate("", xy=(ln * d[0], ln * d[1]), xytext=(-ln * d[0], -ln * d[1]),
                        arrowprops=dict(arrowstyle="<->", color="#2f8961", lw=2.0, alpha=0.9))

    _bare(ax)
    ax.set_xlim(-2.35, 2.35)
    ax.set_ylim(-1.95, 1.95)


def fig_domain_glyphs():
    """The four domains as shape pictograms driven by their real profiles."""
    import numpy as np

    df = pd.read_csv(ROOT / "outputs/11_active_8metric_rebuild/domain_structural_profiles_scaled.csv")
    captions = {
        "Physical Sciences": "broad and loosely packed",
        "Health Sciences": "tight, uneven, hub-leaning",
        "Life Sciences": "close to the average profile",
        "Social Sciences": "more compact, no hub papers",
    }
    order = ["Physical Sciences", "Life Sciences", "Social Sciences", "Health Sciences"]

    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.55))
    for ax, dom in zip(axes, order):
        row = df[df["domain_display_name"] == dom].iloc[0]
        # Amplify the standardized profile so the tendencies are visible at a glance.
        _shape_glyph(ax, np.random.default_rng(8), DOMAIN_COLORS[dom],
                     3.0 * row["embedding_distance_to_centroid_median"],
                     2.2 * row["embedding_knn_median_distance"],
                     1.6 * row["embedding_knn_indegree_gini"],
                     2.4 * row["embedding_pca_dim_80"],
                     n_clusters=19, pts_per_cluster=6)
        ax.set_title(dom, fontsize=13, color=DOMAIN_COLORS[dom], pad=8)
        ax.text(0.5, -0.045, captions[dom], transform=ax.transAxes, fontsize=10.5,
                color="#5a6472", ha="center", va="top")
    fig.text(0.5, -0.035, "stylized clouds, drawn to the same scale from each domain's real average profile",
             fontsize=10, color="#8a93a1", style="italic", ha="center")
    fig.tight_layout()
    save(fig, "def_domain_glyphs")


def fig_opposite_glyphs():
    """Computer Science and Arts & Humanities as full shape pictograms."""
    import numpy as np

    df = pd.read_csv(ROOT / "outputs/11_active_8metric_rebuild/field_structural_profiles_scaled.csv")
    picks = [
        ("Computer Science", "a Physical Science", DOMAIN_COLORS["Physical Sciences"],
         "broad · hub-free · one dominant direction"),
        ("Arts and Humanities", "a Social Science", DOMAIN_COLORS["Social Sciences"],
         "compact · hub-driven · many directions"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))
    for ax, (field, tag, color, caption) in zip(axes, picks):
        row = df[df["field_display_name"] == field].iloc[0]
        _shape_glyph(ax, np.random.default_rng(15), color,
                     1.5 * row["embedding_distance_to_centroid_median"],
                     1.6 * row["embedding_knn_median_distance"],
                     1.1 * row["embedding_knn_indegree_gini"],
                     1.6 * row["embedding_pca_dim_80"],
                     with_arrows=True, n_clusters=17, pts_per_cluster=8)
        ax.set_title(f"{field}  ({tag})", fontsize=13.5, color="#1c232d", pad=9)
        ax.text(0.5, -0.05, caption, transform=ax.transAxes, fontsize=11.5,
                color=color, fontweight="bold", ha="center", va="top")
    fig.text(0.5, -0.045, "same drawing rules, each field's own real eight-metric profile (full fingerprints in backup)",
             fontsize=10, color="#8a93a1", style="italic", ha="center")
    fig.tight_layout(w_pad=4.0)
    save(fig, "def_opposite_glyphs")


def fig_domain_panels():
    """Small-multiple morphospace: each panel highlights one domain's real subfields."""
    df = pd.read_csv(ROOT / "outputs/11_active_8metric_rebuild/subfield_structural_profiles_scaled.csv")
    x = "embedding_distance_to_centroid_median"
    y = "embedding_knn_median_distance"
    captions = {
        "Physical Sciences": "broad and loosely packed",
        "Life Sciences": "close to the average profile",
        "Social Sciences": "more compact",
        "Health Sciences": "tight and dense",
    }
    order = ["Physical Sciences", "Life Sciences", "Social Sciences", "Health Sciences"]

    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.35), sharex=True, sharey=True)
    for ax, dom in zip(axes, order):
        others = df[df["domain_display_name"] != dom]
        mine = df[df["domain_display_name"] == dom]
        ax.scatter(others[x], others[y], s=10, color="#d9dee5", alpha=0.9,
                   edgecolors="none", zorder=1)
        ax.scatter(mine[x], mine[y], s=24, color=DOMAIN_COLORS[dom], alpha=0.85,
                   edgecolors="white", linewidths=0.3, zorder=2)
        ax.scatter([mine[x].mean()], [mine[y].mean()], s=180, marker="D",
                   color=DOMAIN_COLORS[dom], edgecolors="white", linewidths=1.6, zorder=3)
        ax.set_title(dom, fontsize=12.5, color=DOMAIN_COLORS[dom], pad=7)
        ax.text(0.5, -0.075, captions[dom], transform=ax.transAxes, fontsize=10.5,
                color="#5a6472", ha="center", va="top")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("left", "bottom", "top", "right"):
            ax.spines[side].set_color("#dde2e8")
    axes[0].set_ylabel("distance between\nneighbours  →", fontsize=10)
    fig.text(0.5, -0.06, "each dot is one real subfield  ·  horizontal: spread around the centre  ·  vertical: distance between neighbouring papers",
             fontsize=10, color="#8a93a1", style="italic", ha="center")
    fig.tight_layout()
    save(fig, "def_domain_panels")


def fig_mirror_profiles():
    """The real eight-metric profiles of Computer Science and Arts & Humanities."""
    df = pd.read_csv(ROOT / "outputs/11_active_8metric_rebuild/field_structural_profiles_scaled.csv")
    metric_order = [
        ("embedding_distance_to_centroid_median", "spread"),
        ("embedding_distance_to_centroid_iqr", "spread\nunevenness"),
        ("embedding_distance_to_centroid_p90", "outer\nspread"),
        ("embedding_knn_median_distance", "local\ndistance"),
        ("embedding_knn_distance_cv", "local\nunevenness"),
        ("embedding_knn_indegree_gini", "hub\nconcentration"),
        ("embedding_pca_dim_80", "dimensions"),
        ("embedding_pca_spectral_entropy", "direction\nevenness"),
    ]
    bands = [(-0.5, 2.5, "#00337f"), (2.5, 4.5, "#c96a32"),
             (4.5, 5.5, "#a63d40"), (5.5, 7.5, "#2f8961")]
    band_names = [(1.0, "SPREAD"), (3.5, "LOCAL PACKING"), (5.0, "HUBS"), (6.5, "DIMENSIONALITY")]
    fields = [("Computer Science", DOMAIN_COLORS["Physical Sciences"]),
              ("Arts and Humanities", DOMAIN_COLORS["Social Sciences"])]

    fig, ax = plt.subplots(figsize=(10.6, 4.15))
    for x0, x1, c in bands:
        ax.axvspan(x0, x1, color=c, alpha=0.05, zorder=0)
    for xb, name in band_names:
        ax.text(xb, 1.42, name, fontsize=8.5, color="#8a93a1", ha="center",
                va="center", fontweight="bold")
    ax.axhline(0, color="#5a6472", linewidth=1.0, zorder=1)
    ax.text(-0.42, 0.06, "median field", fontsize=9, color="#8a93a1",
            style="italic", ha="left", va="bottom")

    xs = range(len(metric_order))
    for field, color in fields:
        row = df[df["field_display_name"] == field].iloc[0]
        vals = [row[m] for m, _ in metric_order]
        ax.plot(xs, vals, color=color, lw=2.6, marker="o", markersize=6.5,
                markerfacecolor=color, markeredgecolor="white", markeredgewidth=1.1,
                zorder=3)
        ax.annotate(field, (7, vals[-1]), xytext=(12, 0), textcoords="offset points",
                    fontsize=11.5, color=color, fontweight="bold", va="center",
                    path_effects=[mpl.patheffects.withStroke(linewidth=3, foreground="white")])

    ax.set_xticks(list(xs))
    ax.set_xticklabels([lab for _, lab in metric_order], fontsize=9.5, color="#3c4552")
    ax.set_xlim(-0.5, 9.4)
    ax.set_ylim(-1.75, 1.6)
    ax.set_yticks([-1, 0, 1])
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_ylabel("standard units", fontsize=9.5)
    ax.grid(True, axis="y", color="#eef1f5", linewidth=0.6, zorder=0)
    fig.tight_layout()
    save(fig, "def_mirror_profiles")


def fig_opt_extreme_shares():
    """Domain ownership of the 25 most extreme subfields at each end of each measure."""
    df = pd.read_csv(ROOT / "outputs/11_active_8metric_rebuild/subfield_structural_profiles_scaled.csv")
    rows = [
        ("embedding_knn_median_distance", "Local packing", "tightest 25", "loosest 25"),
        ("embedding_distance_to_centroid_median", "Spread", "most compact 25", "most spread 25"),
        ("embedding_knn_indegree_gini", "Hubs", "least hub-driven 25", "most hub-driven 25"),
        ("embedding_pca_dim_80", "Dimensionality", "fewest directions 25", "most directions 25"),
    ]
    fig, ax = plt.subplots(figsize=(10.6, 4.15))
    for r, (metric, family, lowtag, hightag) in enumerate(rows):
        y = -r * 1.25
        lo = df.nsmallest(25, metric)["domain_display_name"].value_counts()
        hi = df.nlargest(25, metric)["domain_display_name"].value_counts()
        for counts, sign in ((lo, -1), (hi, 1)):
            xpos = 5.0
            for dom in sorted(counts.index, key=lambda d: -counts[d]):
                n = counts[dom]
                ax.barh(y, sign * n, left=sign * xpos, height=0.72,
                        color=DOMAIN_COLORS[dom], zorder=2)
                if n >= 4:
                    ax.text(sign * (xpos + n / 2), y, str(n), ha="center", va="center",
                            fontsize=9.5, color="white", fontweight="bold", zorder=3)
                xpos += n
        ax.text(0, y, family, ha="center", va="center", fontsize=11.5,
                fontweight="bold", color="#1c232d")
        ax.text(-5.3, y - 0.62, f"← {lowtag}", ha="right", va="center",
                fontsize=9, color="#8a93a1", style="italic")
        ax.text(5.3, y - 0.62, f"{hightag} →", ha="left", va="center",
                fontsize=9, color="#8a93a1", style="italic")
    ax.set_xlim(-31.0, 31.0)
    ax.set_ylim(-4.55, 0.8)
    ax.axis("off")
    handles = [mpl.patches.Patch(color=DOMAIN_COLORS[d], label=d) for d in DOMAIN_ORDER]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.10),
              ncol=4, frameon=False, fontsize=10, handletextpad=0.3, columnspacing=1.2)
    save(fig, "def_opt_extreme_shares")


def fig_opt_ridges():
    """Full subfield distributions by domain for the two most telling measures."""
    import numpy as np
    from scipy.stats import gaussian_kde

    df = pd.read_csv(ROOT / "outputs/11_active_8metric_rebuild/subfield_structural_profiles_scaled.csv")
    panels = [
        ("embedding_knn_median_distance", "Distance between neighbouring papers"),
        ("embedding_distance_to_centroid_median", "Spread around the field centre"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
    for ax, (metric, title) in zip(axes, panels):
        grid = np.linspace(df[metric].min() - 0.5, df[metric].max() + 0.5, 240)
        for k, dom in enumerate(DOMAIN_ORDER[::-1]):
            vals = df.loc[df["domain_display_name"] == dom, metric].to_numpy()
            dens = gaussian_kde(vals, bw_method=0.45)(grid)
            base = k * 0.55
            ax.fill_between(grid, base, base + dens / dens.max() * 1.05,
                            color=DOMAIN_COLORS[dom], alpha=0.60, zorder=2 + k, lw=0)
            ax.plot(grid, base + dens / dens.max() * 1.05, color=DOMAIN_COLORS[dom],
                    lw=1.3, zorder=2 + k)
            med = float(np.median(vals))
            ax.plot([med, med], [base, base + 0.28], color="white", lw=2.0, zorder=3 + k)
            ax.text(grid[0], base + 0.14, dom, fontsize=10, color=DOMAIN_COLORS[dom],
                    fontweight="bold", ha="left", va="center", zorder=6)
        ax.set_title(title, fontsize=12, pad=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("left", "top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#ccd3db")
        ax.text(0.0, -0.055, "← tighter / more compact", transform=ax.transAxes,
                fontsize=9.5, color="#8a93a1", style="italic", ha="left")
        ax.text(1.0, -0.055, "looser / broader →", transform=ax.transAxes,
                fontsize=9.5, color="#8a93a1", style="italic", ha="right")
    fig.text(0.5, -0.045, "each curve: the full distribution of that domain's real subfields · white tick: domain median",
             fontsize=10, color="#8a93a1", style="italic", ha="center")
    fig.tight_layout(w_pad=2.6)
    save(fig, "def_opt_ridges")


def fig_opt_leaderboard():
    """Named leaderboard: the loosest and tightest packed subfields, domain-colored."""
    df = pd.read_csv(ROOT / "outputs/11_active_8metric_rebuild/subfield_structural_profiles_scaled.csv")
    metric = "embedding_knn_median_distance"
    loose = df.nlargest(12, metric)
    tight = df.nsmallest(12, metric)

    def shorten(n):
        return n if len(n) <= 30 else n[:28] + "…"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.35))
    for ax, sub, title, sign in ((ax1, loose, "the 12 loosest-packed subfields", 1),
                                 (ax2, tight, "the 12 most tightly packed", -1)):
        vals = (sign * sub[metric]).tolist()
        colors = [DOMAIN_COLORS[d] for d in sub["domain_display_name"]]
        ax.barh(range(len(sub)), vals, color=colors, height=0.70, zorder=2)
        for y, (name, v) in enumerate(zip(sub["subfield_display_name"], vals)):
            ax.text(-0.045, y, shorten(name), ha="right", va="center", fontsize=9.2,
                    color="#1c232d")
        ax.invert_yaxis()
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlim(0, max(vals) * 1.06)
        for side in ("left", "top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#ccd3db")
        ax.set_title(title, fontsize=12.5, pad=9)
    handles = [mpl.patches.Patch(color=DOMAIN_COLORS[d], label=d) for d in DOMAIN_ORDER]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.045),
               ncol=4, frameon=False, fontsize=10, handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(w_pad=11.0)
    save(fig, "def_opt_leaderboard")


def fig_cs_collapse():
    """Computer Science's spectral collapse versus the average field."""
    win = pd.read_csv(ROOT / "outputs/11_active_8metric_rebuild/subfield_window_structural_profiles_scaled.csv")
    panels = [
        ("embedding_pca_dim_80", "Dimensions (PCA $D_{80}$)"),
        ("embedding_pca_spectral_entropy", "Direction evenness (PCA entropy)"),
    ]
    cs_color = DOMAIN_COLORS["Physical Sciences"]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.5), sharey=True)
    for ax, (metric, title) in zip(axes, panels):
        overall = win.groupby("window_index")[metric].mean()
        cs = win[win["field_display_name"] == "Computer Science"].groupby("window_index")[metric].mean()
        xs = sorted(overall.index)
        ax.plot(xs, [overall[x] for x in xs], color="#aab2bd", lw=2.0, marker="o",
                markersize=4.5, markerfacecolor="#aab2bd", markeredgecolor="white",
                zorder=2)
        ax.plot(xs, [cs[x] for x in xs], color=cs_color, lw=2.8, marker="o",
                markersize=6, markerfacecolor=cs_color, markeredgecolor="white",
                markeredgewidth=1.0, zorder=3)
        ax.annotate("average field", (xs[-1], overall[xs[-1]]), xytext=(8, 0),
                    textcoords="offset points", fontsize=10, color="#8a93a1",
                    fontweight="bold", va="center")
        ax.annotate("Computer Science", (xs[-1], cs[xs[-1]]), xytext=(8, 0),
                    textcoords="offset points", fontsize=10.5, color=cs_color,
                    fontweight="bold", va="center")
        ax.set_title(title, fontsize=12, pad=9)
        ax.set_xticks([0, 4])
        ax.set_xticklabels(["2000–04", "2020–24"], fontsize=9)
        ax.set_xlim(-0.3, 6.8)
        ax.tick_params(axis="y", labelsize=8.5)
        ax.grid(True, axis="y", color="#eef1f5", linewidth=0.6, zorder=0)
    axes[0].set_ylabel("standard units", fontsize=9.5)
    fig.tight_layout(w_pad=2.0)
    save(fig, "def_cs_collapse")


def fig_isolates():
    """Each field's distance to its nearest morphological neighbour."""
    df = pd.read_csv(ROOT / "outputs/11_active_8metric_rebuild/field_nearest_morphological_neighbors.csv")
    df = df.sort_values("nearest_distance", ascending=True).reset_index(drop=True)
    highlight = {"Pharmacology, Toxicology and Pharmaceutics", "Computer Science"}

    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    for y, row in df.iterrows():
        isolated = row["field"] in highlight
        color = "#c0563a" if isolated else "#9fb4cc"
        name = SHORT_FIELD.get(row["field"], row["field"])
        ax.barh(y, row["nearest_distance"], color=color, height=0.68, zorder=2)
        ax.text(-0.03, y, name, ha="right", va="center", fontsize=9.0,
                fontweight="bold" if isolated else "normal", color="#1c232d")
        if isolated:
            ax.text(row["nearest_distance"] + 0.04, y,
                    f"{row['nearest_distance']:.2f}  (nearest: {SHORT_FIELD.get(row['nearest_field'], row['nearest_field'])})",
                    ha="left", va="center", fontsize=9.2, fontweight="bold", color="#c0563a")
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_xlim(0, 2.75)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Distance to the most similar field (shape profile)", fontsize=11)
    ax.tick_params(labelsize=8.5)
    ax.grid(True, axis="x", color="#eef1f5", linewidth=0.6, zorder=0)
    fig.tight_layout()
    save(fig, "def_isolates")


def fig_alike_pairs():
    """The most similar and most different field pairs as linked domain-colored capsules."""
    from itertools import combinations
    import numpy as np

    df = pd.read_csv(ROOT / "outputs/11_active_8metric_rebuild/field_structural_profiles_scaled.csv")
    metrics = [c for c in df.columns if c.startswith("embedding_")]
    X = df[metrics].to_numpy()
    names = df["field_display_name"].tolist()
    doms = df["domain_display_name"].tolist()
    pairs = [(i, j, float(np.linalg.norm(X[i] - X[j])))
             for i, j in combinations(range(len(df)), 2)]
    pairs.sort(key=lambda t: t[2])
    top = pairs[:5]
    worst = pairs[-2:][::-1]

    def short(n):
        return SHORT_FIELD.get(n, n)

    def capsule(x, y, text, color, ha):
        return plt.gca().text(x, y, text, fontsize=10.5, fontweight="bold", color="white",
                              ha=ha, va="center", zorder=4,
                              bbox=dict(boxstyle="round,pad=0.42", facecolor=color,
                                        edgecolor="white", linewidth=1.2))

    def badge(x, y, value, color):
        plt.gca().text(x, y, value, fontsize=9.5, fontweight="bold", color=color,
                       ha="center", va="center", zorder=5,
                       bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                                 edgecolor="#d9dee5", linewidth=0.9))

    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(-8.45, 1.5)
    ax.axis("off")

    ax.text(0.15, 1.0, "MOST SIMILAR SHAPES", fontsize=11, color="#2f8961",
            fontweight="bold", va="center")
    for k, (i, j, d) in enumerate(top):
        y = -k * 1.14
        ax.plot([4.35, 5.65], [y, y], color="#aab2bd", lw=1.4, zorder=1)
        capsule(4.35, y, short(names[i]), DOMAIN_COLORS[doms[i]], "right")
        capsule(5.65, y, short(names[j]), DOMAIN_COLORS[doms[j]], "left")
        badge(5.0, y, f"{d:.2f}", "#2f8961")

    ax.text(0.15, -5.65, "MOST DIFFERENT SHAPES", fontsize=11, color="#c0563a",
            fontweight="bold", va="center")
    d_max = worst[0][2]
    for k, (i, j, d) in enumerate(worst):
        y = -6.55 - k * 1.2
        half = 4.65 * d / d_max
        ax.plot([5.0 - half, 5.0 + half], [y, y], color="#aab2bd", lw=1.4,
                linestyle=(0, (5, 4)), zorder=1)
        capsule(5.0 - half, y, short(names[i]), DOMAIN_COLORS[doms[i]], "left")
        capsule(5.0 + half, y, short(names[j]), DOMAIN_COLORS[doms[j]], "right")
        badge(5.0, y, f"{d:.2f}", "#c0563a")
    save(fig, "def_alike_pairs")


def fig_fingerprints():
    """Full eight-metric fingerprints of two structurally opposite fields."""
    df = pd.read_csv(ROOT / "outputs/11_active_8metric_rebuild/field_structural_profiles_scaled.csv")
    metric_order = [
        ("embedding_distance_to_centroid_median", "spread", "#00337f"),
        ("embedding_distance_to_centroid_iqr", "spread unevenness", "#00337f"),
        ("embedding_distance_to_centroid_p90", "outer spread", "#00337f"),
        ("embedding_knn_median_distance", "local distance", "#c96a32"),
        ("embedding_knn_distance_cv", "local unevenness", "#c96a32"),
        ("embedding_knn_indegree_gini", "hub concentration", "#a63d40"),
        ("embedding_pca_dim_80", "dimensions", "#2f8961"),
        ("embedding_pca_spectral_entropy", "direction evenness", "#2f8961"),
    ]
    fields = [("Computer Science", "a Physical Science"),
              ("Arts and Humanities", "a Social Science")]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3), sharex=True)
    for ax, (field, domain_tag) in zip(axes, fields):
        row = df[df["field_display_name"] == field].iloc[0]
        ys = range(len(metric_order))
        vals = [row[m] for m, _, _ in metric_order]
        colors = [c for _, _, c in metric_order]
        ax.barh(ys, vals, color=colors, height=0.62, zorder=2)
        for y, (v, (_, label, c)) in enumerate(zip(vals, metric_order)):
            anchor, ha = (-0.06, "right") if v >= 0 else (0.06, "left")
            ax.text(anchor, y, label, ha=ha, va="center", fontsize=10, color="#1c232d")
        ax.axvline(0, color="#5a6472", linewidth=1.0, zorder=3)
        ax.invert_yaxis()
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_title(f"{field}\n({domain_tag})", fontsize=12.5, pad=9)
        ax.set_xlim(-1.75, 1.35)
        ax.tick_params(labelsize=8.5)
        ax.grid(True, axis="x", color="#eef1f5", linewidth=0.6, zorder=0)
        ax.text(0.02, -0.115, "← below the median field", transform=ax.transAxes,
                fontsize=9.5, color="#8a93a1", style="italic", ha="left")
        ax.text(0.98, -0.115, "above →", transform=ax.transAxes,
                fontsize=9.5, color="#8a93a1", style="italic", ha="right")
    fig.tight_layout(w_pad=3.2)
    save(fig, "def_fingerprints")


def fig_convergence():
    """Top diverging and converging field pairs as diverging bars."""
    df = pd.read_csv(ROOT / "outputs/11_active_8metric_rebuild/field_top_convergence_divergence_pairs.csv")
    df = df[df["level"] == "field"]
    div = df[df["direction"] == "Diverging"].nsmallest(6, "rank")
    con = df[df["direction"] == "Converging"].nsmallest(6, "rank")

    def short(name):
        table = dict(SHORT_FIELD)
        table.update({
            "Computer Science": "Computer Science",
            "Immunology and Microbiology": "Immunology & Microbiology",
            "Business, Management and Accounting": "Business & Management",
            "Materials Science": "Materials Science",
            "Mathematics": "Mathematics",
        })
        return table.get(name, name)

    green, red = "#2f8961", "#c0563a"
    fig, ax = plt.subplots(figsize=(10.6, 4.9))
    labels_done = []
    for i, (_, row) in enumerate(div.iterrows()):
        yv = i
        ax.barh(yv, row["delta_distance"], color=red, height=0.62, zorder=2)
        labels_done.append((yv, row, red, True))
    for i, (_, row) in enumerate(con.iterrows()):
        yv = len(div) + 0.85 + i
        ax.barh(yv, row["delta_distance"], color=green, height=0.62, zorder=2)
        labels_done.append((yv, row, green, False))

    for yv, row, color, diverging in labels_done:
        pair = f"{short(row['entity_a_name'])}  ·  {short(row['entity_b_name'])}"
        involves_cs = "Computer Science" in (row["entity_a_name"], row["entity_b_name"])
        weight = "bold" if involves_cs else "normal"
        delta = row["delta_distance"]
        if diverging:
            ax.text(-0.05, yv, pair, ha="right", va="center", fontsize=9.6,
                    fontweight=weight, color="#1c232d")
            ax.text(delta + 0.05, yv, f"+{delta:.2f}", ha="left", va="center",
                    fontsize=9.4, fontweight="bold", color=red)
        else:
            ax.text(0.05, yv, pair, ha="left", va="center", fontsize=9.6,
                    fontweight=weight, color="#1c232d")
            ax.text(delta - 0.05, yv, f"{delta:.2f}", ha="right", va="center",
                    fontsize=9.4, fontweight="bold", color=green)

    ax.axvline(0, color="#5a6472", linewidth=1.0, zorder=3)
    ax.text(1.35, -1.15, "grew further apart →", fontsize=11.5, color=red,
            fontweight="bold", ha="center")
    ax.text(-0.85, -1.15, "← moved closer", fontsize=11.5, color=green,
            fontweight="bold", ha="center")
    ax.invert_yaxis()
    ax.set_ylim(len(div) + 0.85 + len(con) - 0.3, -1.7)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-1.75, 2.6)
    ax.set_xlabel("Change in shape difference between the two fields (2000–04 → 2020–24)",
                  fontsize=11)
    ax.tick_params(labelsize=8.5)
    ax.grid(True, axis="x", color="#eef1f5", linewidth=0.6, zorder=0)
    fig.tight_layout()
    save(fig, "def_convergence")


def fig_field_spread():
    """26 fields ranked by spread, colored by domain."""
    df = pd.read_csv(ROOT / "outputs/11_active_8metric_rebuild/field_structural_profiles_scaled.csv")
    x = "embedding_distance_to_centroid_median"
    df = df.sort_values(x, ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    colors = [DOMAIN_COLORS[d] for d in df["domain_display_name"]]
    bars = ax.barh(range(len(df)), df[x], color=colors, height=0.72, zorder=2)

    emphasized = {"Physics and Astronomy", "Computer Science", "Energy", "Mathematics",
                  "Nursing", "Dentistry"}
    for i, (name, val) in enumerate(zip(df["field_display_name"], df[x])):
        weight = "bold" if name in emphasized else "normal"
        ax.text(-0.03 if val >= 0 else 0.03, i, name, ha="right" if val >= 0 else "left",
                va="center", fontsize=9.3, fontweight=weight, color="#1c232d", zorder=3)

    ax.axvline(0, color="#8a93a1", linewidth=0.9, zorder=1)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(df[x].min() - 0.85, df[x].max() + 0.15)
    ax.set_xlabel("Spread of papers around the field centre (relative to the median field)",
                  fontsize=11.5)
    ax.text(0.0, -0.115, "← more compact", transform=ax.transAxes, fontsize=10,
            color="#8a93a1", style="italic", ha="left")
    ax.text(1.0, -0.115, "more spread out →", transform=ax.transAxes, fontsize=10,
            color="#8a93a1", style="italic", ha="right")
    ax.tick_params(labelsize=8.5)
    ax.grid(True, axis="x", color="#e7ebf0", linewidth=0.6, zorder=0)

    handles = [mpl.patches.Patch(color=DOMAIN_COLORS[d], label=d) for d in DOMAIN_ORDER]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9.3,
              handlelength=1.1, handleheight=1.0, borderaxespad=0.2)
    save(fig, "def_field_spread")


def fig_packing_curves():
    """Raw all-subfield means per window for the three headline metrics."""
    df = pd.read_parquet(ROOT / "data/processed/temporal/overall_window_embedding_metric_trajectories.parquet")
    panels = [
        ("embedding_distance_to_centroid_median", "Distance to the field centre", "−12.6%", "{:.4f}"),
        ("embedding_knn_median_distance", "Distance between neighbouring papers", "−18.1%", "{:.4f}"),
        ("embedding_pca_dim_80", "Directions for 80% of the variation", "−12.5%", "{:.1f}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.15))
    for ax, (metric, title, delta, fmt) in zip(axes, panels):
        sub = df[df["metric"] == metric].sort_values("window_index")
        xs = range(len(sub))
        ys = sub["mean_value"].to_numpy()
        ax.plot(xs, ys, color=UC3M_NAVY, linewidth=2.4, marker="o", markersize=5.5,
                markerfacecolor=UC3M_NAVY, markeredgecolor="white", markeredgewidth=1.0,
                zorder=3)
        ax.set_title(title, fontsize=11, pad=10)
        ax.text(0.97, 0.88, delta, transform=ax.transAxes, fontsize=17, fontweight="bold",
                color="#2f8961", ha="right", va="top")
        ax.annotate(fmt.format(ys[0]), (0, ys[0]), xytext=(4, 9), textcoords="offset points",
                    fontsize=9, color="#5a6472", ha="left")
        ax.annotate(fmt.format(ys[-1]), (len(sub) - 1, ys[-1]), xytext=(0, -14),
                    textcoords="offset points", fontsize=9, color="#5a6472", ha="right")
        ax.set_xticks(list(xs))
        ax.set_xticklabels(sub["window_label"].str.replace("-", "–"), fontsize=8, rotation=30)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ymin, ymax = ys.min(), ys.max()
        pad = (ymax - ymin) * 0.35
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.grid(True, axis="x", color="#eef1f5", linewidth=0.6, zorder=0)
    fig.tight_layout(w_pad=2.4)
    save(fig, "def_packing_curves")


SHORT_FIELD = {
    "Agricultural and Biological Sciences": "Agricultural & Biological Sci.",
    "Environmental Science": "Environmental Science",
    "Immunology and Microbiology": "Immunology & Microbiology",
    "Decision Sciences": "Decision Sciences",
    "Neuroscience": "Neuroscience",
    "Biochemistry, Genetics and Molecular Biology": "Biochemistry & Genetics",
    "Psychology": "Psychology",
    "Pharmacology, Toxicology and Pharmaceutics": "Pharmacology & Toxicology",
    "Economics, Econometrics and Finance": "Economics & Finance",
    "Computer Science": "Computer Science",
    "Arts and Humanities": "Arts and Humanities",
    "Dentistry": "Dentistry",
    "Nursing": "Nursing",
    "Energy": "Energy",
    "Engineering": "Engineering",
}


def fig_field_similarity():
    """All 325 field pairs by shape difference, extremes annotated."""
    from itertools import combinations
    import numpy as np

    rng = np.random.default_rng(7)
    df = pd.read_csv(ROOT / "outputs/11_active_8metric_rebuild/field_structural_profiles_scaled.csv")
    metrics = [c for c in df.columns if c.startswith("embedding_")]
    X = df[metrics].to_numpy()
    names = df["field_display_name"].tolist()
    doms = df["domain_display_name"].tolist()
    pairs = [(names[i], names[j], doms[i] == doms[j], float(np.linalg.norm(X[i] - X[j])))
             for i, j in combinations(range(len(df)), 2)]
    pairs.sort(key=lambda t: t[3])

    green, red = "#2f8961", "#c0563a"
    fig, ax = plt.subplots(figsize=(10.8, 3.5))
    ys = rng.uniform(-1, 1, len(pairs))
    for (a, b, same, v), y in zip(pairs, ys):
        color = "#00337f" if same else "#b3bdc9"
        ax.scatter([v], [y], s=30, color=color, alpha=0.75,
                   edgecolors="white", linewidths=0.4, zorder=2)

    med = np.median([p[3] for p in pairs])
    ax.axvline(med, color="#8a93a1", linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.text(med, 1.62, f"typical pair: {med:.2f}", ha="center", fontsize=11,
            color="#5a6472", style="italic")

    lo, hi = pairs[0], pairs[-1]
    ax.scatter([lo[3]], [0], s=170, color=green, edgecolors="white", linewidths=1.6, zorder=4)
    ax.annotate(f"most alike ({lo[3]:.2f})\n{SHORT_FIELD.get(lo[0], lo[0])}\nand {SHORT_FIELD.get(lo[1], lo[1])}",
                (lo[3], 0), xytext=(lo[3] - 0.05, -2.6), fontsize=11, color=green,
                fontweight="bold", ha="left", va="top",
                arrowprops=dict(arrowstyle="-", color=green, lw=0.9))
    ax.scatter([hi[3]], [0], s=170, color=red, edgecolors="white", linewidths=1.6, zorder=4)
    ax.annotate(f"most different ({hi[3]:.2f})\n{SHORT_FIELD.get(hi[0], hi[0])}\nand {SHORT_FIELD.get(hi[1], hi[1])}",
                (hi[3], 0), xytext=(hi[3] + 0.05, -2.6), fontsize=11, color=red,
                fontweight="bold", ha="right", va="top",
                arrowprops=dict(arrowstyle="-", color=red, lw=0.9))

    handles = [
        mpl.lines.Line2D([], [], marker="o", linestyle="", color="#00337f", label="same domain"),
        mpl.lines.Line2D([], [], marker="o", linestyle="", color="#b3bdc9", label="different domains"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=11,
              borderaxespad=0.1, handletextpad=0.2)

    ax.set_ylim(-5.4, 2.0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0.2, 4.3)
    ax.set_xlabel("Difference in shape between the two fields of the pair", fontsize=12.5)
    ax.tick_params(labelsize=9.5)
    ax.grid(True, axis="x", color="#eef1f5", linewidth=0.6, zorder=0)
    fig.tight_layout()
    save(fig, "def_field_similarity")


def fig_drift_movers():
    """Top subfields by early--late centroid drift, single column."""
    df = pd.read_csv(ROOT / "outputs/11_active_8metric_rebuild/centroid_drift_largest_subfields.csv")
    df = df.sort_values("embedding_centroid_drift_early_late", ascending=False).head(10)
    names = df["subfield_display_name"].tolist()
    vals = df["embedding_centroid_drift_early_late"].tolist()
    median_drift = 0.0045

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ys = range(len(df))
    colors = ["#c0563a" if i == 0 else "#d99277" for i in ys]
    ax.barh(ys, vals, color=colors, height=0.66, zorder=2)
    for y, (name, v) in enumerate(zip(names, vals)):
        display = name if len(name) <= 34 else name[:32] + "…"
        weight = "bold" if y == 0 else "normal"
        ax.text(-0.0006, y, display, va="center", ha="right", fontsize=9.3,
                fontweight=weight, color="#1c232d")
        ax.text(v + 0.0006, y, f"{v:.4f}", va="center", ha="left", fontsize=8.8,
                fontweight=weight, color="#8a5a4a")
    ax.axvline(median_drift, color="#5a6472", linewidth=1.0, linestyle=(0, (4, 3)), zorder=3)
    ax.text(median_drift + 0.0005, -1.0, "typical field: 0.0045", fontsize=9,
            color="#5a6472", style="italic", ha="left", va="center")
    ax.invert_yaxis()
    ax.set_ylim(len(df) - 0.3, -1.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 0.043)
    ax.set_xlabel("How far the subfield's centre moved (2000–04 → 2020–24)", fontsize=10.5)
    ax.tick_params(labelsize=8.5)
    ax.grid(True, axis="x", color="#eef1f5", linewidth=0.6, zorder=0)
    fig.tight_layout()
    save(fig, "def_drift_movers")


def fig_trajectory_patterns():
    """Mean standardized trajectories of the eight metrics for the four dynamic groups."""
    win = pd.read_csv(ROOT / "outputs/11_active_8metric_rebuild/subfield_window_structural_profiles_scaled.csv")
    lab = pd.read_csv(ROOT / "outputs/09_morphological_typologies/temporal_trajectory_clustering/trajectory_cluster_assignments.csv")
    win = win.merge(lab[["subfield_id", "trajectory_cluster_label"]], on="subfield_id", how="inner")

    groups = [
        ("D1", "Packing tighter", 93,
         ["embedding_distance_to_centroid_median", "embedding_knn_distance_cv"]),
        ("D2", "Spreading out", 58,
         ["embedding_distance_to_centroid_median", "embedding_knn_indegree_gini"]),
        ("D3", "Loosening locally", 54,
         ["embedding_knn_median_distance", "embedding_knn_distance_cv"]),
        ("D4", "Gaining dimensions", 35,
         ["embedding_pca_dim_80", "embedding_knn_indegree_gini"]),
    ]

    # Groups are defined by how a subfield moves compared with the rest of
    # science, so trajectories are shown as differences from the all-subfield
    # mean in each window ("the average field" = horizontal zero line).
    overall = win.groupby("window_index")[list(METRICS)].mean()
    means = (
        win.groupby(["trajectory_cluster_label", "window_index"])[list(METRICS)]
        .mean()
        .reset_index()
    )
    for metric in METRICS:
        means[metric] = means[metric] - means["window_index"].map(overall[metric])
    ymin = means[list(METRICS)].to_numpy().min() - 0.25
    ymax = means[list(METRICS)].to_numpy().max() + 0.25

    fig, axes = plt.subplots(1, 4, figsize=(11.6, 3.35), sharey=True)
    for ax, (gid, title, n, highlights) in zip(axes, groups):
        sub = means[means["trajectory_cluster_label"] == gid].sort_values("window_index")
        xs = sub["window_index"].to_numpy()
        for metric in METRICS:
            if metric in highlights:
                continue
            ax.plot(xs, sub[metric], color="#ccd3db", linewidth=1.1, zorder=1)
        for metric in highlights:
            color = METRIC_HIGHLIGHT_COLORS[metric]
            ax.plot(xs, sub[metric], color=color, linewidth=2.6, marker="o",
                    markersize=4, markerfacecolor=color, markeredgecolor="white",
                    markeredgewidth=0.8, zorder=3)
            ax.annotate(METRICS[metric], (xs[-1], sub[metric].iloc[-1]),
                        xytext=(5, 0), textcoords="offset points", fontsize=8.8,
                        color=color, fontweight="bold", va="center",
                        path_effects=[mpl.patheffects.withStroke(linewidth=2.5, foreground="white")])
        ax.axhline(0, color="#aab2bd", linewidth=0.9, zorder=0)
        ax.set_title(f"{title}\n({n} subfields)", fontsize=10.6, pad=8)
        ax.set_xticks([0, 4])
        ax.set_xticklabels(["2000–04", "2020–24"], fontsize=8.5)
        ax.set_xlim(-0.25, 6.6)
        ax.set_ylim(ymin, ymax)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, axis="y", color="#eef1f5", linewidth=0.6, zorder=0)
    axes[0].set_ylabel("vs. the average field", fontsize=9.5)
    axes[0].annotate("average field", (0.1, 0), xytext=(0, 5), textcoords="offset points",
                     fontsize=8, color="#8a93a1", style="italic", va="bottom")
    fig.tight_layout(w_pad=1.6)
    save(fig, "def_trajectory_patterns")


if __name__ == "__main__":
    import matplotlib.patheffects  # noqa: F401  (registered as mpl.patheffects)

    # Figures used by the current deck. Alternative designs kept above
    # (glyphs, morphospace, domain panels, extreme shares, fingerprints,
    # field spread) can be regenerated by calling their functions.
    fig_metric_families()
    fig_mirror_profiles()
    fig_opt_ridges()
    fig_opt_leaderboard()
    fig_alike_pairs()
    fig_packing_curves()
    fig_trajectory_patterns()
    fig_field_similarity()
    fig_drift_movers()
    fig_convergence()
    fig_cs_collapse()
    fig_isolates()
