from __future__ import annotations

import sys
from math import cos, pi, sin
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.morphological_typologies import DOMAIN_COLORS


OUTPUT_DIR = ROOT / "outputs" / "13_selected_visual_variants"
DOMAIN_ORDER = ["Life Sciences", "Social Sciences", "Physical Sciences", "Health Sciences"]
CHANGE_ORDER = ["compacting + more uneven", "compacting + smoothing", "broadening + more uneven", "broadening + smoothing"]
CHANGE_COLORS = {
    "compacting + more uneven": "#2f7f5f",
    "compacting + smoothing": "#8b6bb5",
    "broadening + more uneven": "#ba6a36",
    "broadening + smoothing": "#4c78a8",
}
FIELD_LABELS = {
    "Agricultural and Biological Sciences": "Agric. & Bio.",
    "Arts and Humanities": "Arts & Hum.",
    "Biochemistry, Genetics and Molecular Biology": "Biochem. & Gen.",
    "Business, Management and Accounting": "Business & Mgmt.",
    "Chemical Engineering": "Chem. Eng.",
    "Chemistry": "Chemistry",
    "Computer Science": "Computer Sci.",
    "Decision Sciences": "Decision Sci.",
    "Dentistry": "Dentistry",
    "Earth and Planetary Sciences": "Earth & Planet.",
    "Economics, Econometrics and Finance": "Econ. & Finance",
    "Energy": "Energy",
    "Engineering": "Engineering",
    "Environmental Science": "Environ. Sci.",
    "Health Professions": "Health Prof.",
    "Immunology and Microbiology": "Immunol. & Microbio.",
    "Materials Science": "Materials Sci.",
    "Mathematics": "Mathematics",
    "Medicine": "Medicine",
    "Neuroscience": "Neuroscience",
    "Nursing": "Nursing",
    "Pharmacology, Toxicology and Pharmaceutics": "Pharm. & Tox.",
    "Physics and Astronomy": "Physics & Astron.",
    "Psychology": "Psychology",
    "Social Sciences": "Social Sci.",
    "Veterinary": "Veterinary",
}
SPREAD_METRICS = [
    "embedding_distance_to_centroid_median",
    "embedding_distance_to_centroid_p90",
    "embedding_knn_median_distance",
    "embedding_pca_dim_80",
    "embedding_pca_spectral_entropy",
]
CONCENTRATION_METRICS = [
    "embedding_distance_to_centroid_iqr",
    "embedding_knn_distance_cv",
    "embedding_knn_indegree_gini",
]


def setup() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "figure.dpi": 130,
            "savefig.dpi": 260,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def short_field(name: str) -> str:
    return FIELD_LABELS.get(name, name)


def domain_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=DOMAIN_COLORS[d], markeredgecolor="white", markersize=8, label=d)
        for d in DOMAIN_ORDER
    ]


def change_handles() -> list[Line2D]:
    return [Line2D([0], [0], color=CHANGE_COLORS[c], linewidth=7, label=c) for c in CHANGE_ORDER]


def field_domain_frame() -> pd.DataFrame:
    return pd.read_csv(ROOT / "outputs" / "09_morphological_typologies" / "subfield_typology_assignments.csv")[
        ["field_display_name", "domain_display_name"]
    ].drop_duplicates()


def nearest_with_domains() -> pd.DataFrame:
    nearest = pd.read_csv(ROOT / "outputs" / "10_global_map_candidates" / "field_nearest_morphological_neighbors.csv")
    domains = field_domain_frame()
    nearest = nearest.merge(domains, left_on="field", right_on="field_display_name", how="left").rename(
        columns={"domain_display_name": "source_domain"}
    )
    nearest = nearest.merge(domains, left_on="nearest_field", right_on="field_display_name", how="left").rename(
        columns={"domain_display_name": "target_domain"}
    )
    return nearest.drop(columns=["field_display_name_x", "field_display_name_y"], errors="ignore")


def attractor_targets(nearest: pd.DataFrame, min_links: int = 2) -> list[str]:
    counts = nearest["nearest_field"].value_counts()
    return counts[counts >= min_links].index.tolist()


def subfield_change_frame() -> pd.DataFrame:
    changes = pd.read_csv(ROOT / "data" / "processed" / "temporal" / "subfield_metric_temporal_changes.csv")
    pivot = changes.pivot_table(
        index=["subfield_id", "subfield_display_name", "field_display_name", "domain_display_name"],
        columns="metric",
        values="robust_standardized_delta",
        aggfunc="first",
    ).reset_index()
    pivot["spread_delta"] = pivot[SPREAD_METRICS].mean(axis=1)
    pivot["concentration_delta"] = pivot[CONCENTRATION_METRICS].mean(axis=1)
    pivot["change_magnitude"] = np.hypot(pivot["spread_delta"], pivot["concentration_delta"])
    pivot["change_type"] = [
        ("broadening" if x >= 0 else "compacting") + (" + more uneven" if y >= 0 else " + smoothing")
        for x, y in zip(pivot["spread_delta"], pivot["concentration_delta"])
    ]
    return pivot.dropna(subset=["spread_delta", "concentration_delta"])


def change_counts() -> pd.DataFrame:
    frame = subfield_change_frame()
    return (
        frame.groupby(["domain_display_name", "change_type"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=DOMAIN_ORDER, columns=CHANGE_ORDER, fill_value=0)
    )


def draw_ribbon(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float, width: float, color: str, alpha: float) -> None:
    ctrl = 0.34
    top0, bot0 = y0 + width / 2, y0 - width / 2
    top1, bot1 = y1 + width / 2, y1 - width / 2
    verts = [
        (x0, top0),
        (x0 + ctrl, top0),
        (x1 - ctrl, top1),
        (x1, top1),
        (x1, bot1),
        (x1 - ctrl, bot1),
        (x0 + ctrl, bot0),
        (x0, bot0),
        (x0, top0),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha, zorder=1))


def stacked_ranges(counts: pd.DataFrame, order: list, total: float, gap: float, *, axis: int = 0) -> dict:
    if axis == 0:
        heights = {key: counts.loc[key].sum() / total for key in order}
    else:
        heights = {key: counts[key].sum() / total for key in order}
    scale = (1.0 - gap * (len(order) - 1)) / sum(v for v in heights.values() if v > 0)
    y = 1.0
    ranges = {}
    for key in order:
        h = heights[key] * scale if heights[key] > 0 else 0.018
        ranges[key] = (y - h, y)
        y -= h + gap
    return ranges


def figure_08v1_attractor_focus_stars() -> None:
    nearest = nearest_with_domains()
    targets = attractor_targets(nearest)[:6]
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.9))
    axes = axes.reshape(-1)
    for ax, target in zip(axes, targets):
        inc = nearest[nearest["nearest_field"] == target].sort_values("distance")
        target_domain = inc["target_domain"].iloc[0]
        ax.set_aspect("equal")
        ax.axis("off")
        ax.scatter(0, 0, s=620, color=DOMAIN_COLORS[target_domain], edgecolor="white", linewidth=1.2, zorder=4)
        ax.text(0, 0, short_field(target), ha="center", va="center", color="white", fontsize=8.2, weight="bold", zorder=5)
        angles = np.linspace(pi / 2, pi / 2 + 2 * pi, len(inc), endpoint=False)
        for angle, row in zip(angles, inc.itertuples(index=False)):
            x, y = 0.86 * cos(angle), 0.86 * sin(angle)
            cross = row.source_domain != row.target_domain
            ax.add_patch(
                FancyArrowPatch(
                    (x, y),
                    (0, 0),
                    arrowstyle="-|>",
                    mutation_scale=13,
                    color="#222222" if cross else "#8c8c8c",
                    linewidth=1.25 if cross else 0.9,
                    alpha=0.62 if cross else 0.45,
                    shrinkA=11,
                    shrinkB=25,
                    zorder=2,
                )
            )
            ax.scatter(x, y, s=185, color=DOMAIN_COLORS[row.source_domain], edgecolor="white", linewidth=0.8, zorder=3)
            ax.text(1.12 * x, 1.12 * y, short_field(row.field), ha="center", va="center", fontsize=7.1)
        ax.set_title(f"{len(inc)} incoming fields", fontsize=9, color=DOMAIN_COLORS[target_domain])
        ax.set_xlim(-1.38, 1.38)
        ax.set_ylim(-1.22, 1.22)
    fig.suptitle("Attractor focus stars", fontsize=16, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fig_08u_attractor_variant_01_focus_stars")


def figure_08v2_attractor_distance_ladder() -> None:
    nearest = nearest_with_domains()
    targets = attractor_targets(nearest)
    order = sorted(targets, key=lambda t: (-len(nearest[nearest["nearest_field"] == t]), nearest[nearest["nearest_field"] == t]["distance"].mean()))
    y_pos = {target: i for i, target in enumerate(order)}
    fig, ax = plt.subplots(figsize=(11.4, 7.4))
    for target in order:
        inc = nearest[nearest["nearest_field"] == target].sort_values("distance")
        y = y_pos[target]
        target_domain = inc["target_domain"].iloc[0]
        ax.hlines(y, inc["distance"].min(), inc["distance"].max(), color="#d5d5d5", linewidth=1.1, zorder=1)
        ax.scatter(
            [inc["distance"].mean()],
            [y],
            s=120 + 65 * len(inc),
            color=DOMAIN_COLORS[target_domain],
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        for j, row in enumerate(inc.itertuples(index=False)):
            jitter = (j - (len(inc) - 1) / 2) * 0.08
            ax.scatter(row.distance, y + jitter, s=58, color=DOMAIN_COLORS[row.source_domain], edgecolor="white", linewidth=0.7, zorder=4)
            ax.text(row.distance + 0.025, y + jitter, short_field(row.field), fontsize=7.1, va="center", ha="left")
        ax.text(0.70, y, f"{short_field(target)} ({len(inc)})", ha="right", va="center", fontsize=8.2, color=DOMAIN_COLORS[target_domain], weight="bold")
    ax.set_yticks([])
    ax.set_xlabel("Source-to-attractor morphological distance")
    ax.set_title("Attractor distance ladder", fontsize=16, pad=16)
    ax.grid(axis="x", color="#eeeeee", linewidth=0.55)
    ax.set_xlim(0.70, max(1.55, nearest[nearest["nearest_field"].isin(order)]["distance"].max() + 0.25))
    ax.set_ylim(len(order) - 0.35, -0.65)
    ax.legend(handles=domain_handles(), loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, fontsize=8.3)
    save(fig, "fig_08u_attractor_variant_02_distance_ladder")


def figure_08v3_domain_to_attractor_ribbons() -> None:
    nearest = nearest_with_domains()
    targets = attractor_targets(nearest)
    frame = nearest[nearest["nearest_field"].isin(targets)]
    counts = frame.groupby(["source_domain", "nearest_field"]).size().unstack(fill_value=0).reindex(index=DOMAIN_ORDER, columns=targets, fill_value=0)
    targets = sorted(targets, key=lambda t: (-counts[t].sum(), t))
    counts = counts[targets]
    total = float(counts.to_numpy().sum())
    left_ranges = stacked_ranges(counts, DOMAIN_ORDER, total, 0.045, axis=0)
    right_ranges = stacked_ranges(counts, targets, total, 0.025, axis=1)
    left_offsets = {key: left_ranges[key][1] for key in DOMAIN_ORDER}
    right_offsets = {key: right_ranges[key][1] for key in targets}
    fig, ax = plt.subplots(figsize=(12.2, 8.0))
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.04, 1.11)
    ax.axis("off")
    for domain in DOMAIN_ORDER:
        for target in targets:
            n = int(counts.loc[domain, target])
            if n == 0:
                continue
            h = max(n / total * 0.86, 0.006)
            y0 = left_offsets[domain] - h / 2
            y1 = right_offsets[target] - h / 2
            draw_ribbon(ax, 0.19, y0, 0.81, y1, h, DOMAIN_COLORS[domain], 0.48)
            left_offsets[domain] -= h
            right_offsets[target] -= h
    for domain in DOMAIN_ORDER:
        bottom, top = left_ranges[domain]
        color = DOMAIN_COLORS[domain]
        ax.add_patch(Rectangle((0.03, bottom), 0.12, top - bottom, facecolor=color, edgecolor="white", linewidth=1.0, alpha=0.95, zorder=3))
        ax.text(0.01, (bottom + top) / 2, f"{domain}\n{int(counts.loc[domain].sum())}", ha="right", va="center", fontsize=8.3, color=color, weight="bold")
    target_domains = dict(zip(field_domain_frame()["field_display_name"], field_domain_frame()["domain_display_name"]))
    for target in targets:
        bottom, top = right_ranges[target]
        color = DOMAIN_COLORS[target_domains[target]]
        ax.add_patch(Rectangle((0.85, bottom), 0.12, top - bottom, facecolor=color, edgecolor="white", linewidth=1.0, alpha=0.95, zorder=3))
        ax.text(0.99, (bottom + top) / 2, f"{short_field(target)}\n{int(counts[target].sum())}", ha="left", va="center", fontsize=7.7, color=color, weight="bold")
    ax.text(0.50, 1.075, "Source domains flowing into attractor fields", ha="center", fontsize=15.5, weight="bold")
    ax.text(0.50, 1.035, "Only fields receiving two or more nearest-neighbor links are shown", ha="center", fontsize=9.0, color="#444444")
    save(fig, "fig_08u_attractor_variant_03_domain_to_attractor_ribbons")


def figure_08v4_attractor_bead_matrix() -> None:
    nearest = nearest_with_domains()
    targets = attractor_targets(nearest)
    targets = sorted(targets, key=lambda t: (-len(nearest[nearest["nearest_field"] == t]), t))
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    ax.set_xlim(-0.6, len(DOMAIN_ORDER) - 0.15)
    ax.set_ylim(-0.65, len(targets) - 0.25)
    ax.invert_yaxis()
    ax.set_xticks(range(len(DOMAIN_ORDER)), [d.replace(" ", "\n") for d in DOMAIN_ORDER], fontsize=8.8)
    ax.set_yticks(range(len(targets)), [short_field(t) for t in targets], fontsize=8.6)
    for x in range(len(DOMAIN_ORDER)):
        ax.axvline(x, color="#eeeeee", linewidth=0.8, zorder=0)
    for y, target in enumerate(targets):
        inc = nearest[nearest["nearest_field"] == target]
        target_domain = inc["target_domain"].iloc[0]
        ax.axhline(y, color="#f3f3f3", linewidth=0.8, zorder=0)
        for x, domain in enumerate(DOMAIN_ORDER):
            sub = inc[inc["source_domain"] == domain]
            if sub.empty:
                ax.scatter(x, y, s=18, color="#e5e5e5", zorder=1)
                continue
            ax.scatter(
                x,
                y,
                s=90 + 95 * len(sub),
                color=DOMAIN_COLORS[target_domain],
                edgecolor=DOMAIN_COLORS[domain],
                linewidth=2.2,
                alpha=0.92,
                zorder=3,
            )
            labels = ", ".join(short_field(v) for v in sub["field"].tolist())
            ax.text(x + 0.13, y, labels, fontsize=6.8, ha="left", va="center", color="#222222")
    ax.set_title("Attractor source-domain bead matrix", fontsize=15.5, pad=16)
    ax.set_xlabel("Source domain")
    ax.set_ylabel("Attractor field")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "fig_08u_attractor_variant_04_bead_matrix")


def figure_07v1_clean_braids() -> None:
    counts = change_counts()
    total = float(counts.to_numpy().sum())
    left_ranges = stacked_ranges(counts, DOMAIN_ORDER, total, 0.045, axis=0)
    right_ranges = stacked_ranges(counts, CHANGE_ORDER, total, 0.045, axis=1)
    left_offsets = {key: left_ranges[key][1] for key in DOMAIN_ORDER}
    right_offsets = {key: right_ranges[key][1] for key in CHANGE_ORDER}
    fig, ax = plt.subplots(figsize=(11.7, 7.8))
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.04, 1.12)
    ax.axis("off")
    for domain in DOMAIN_ORDER:
        for change_type in CHANGE_ORDER:
            n = int(counts.loc[domain, change_type])
            if n == 0:
                continue
            h = max(n / total * 0.86, 0.004)
            y0 = left_offsets[domain] - h / 2
            y1 = right_offsets[change_type] - h / 2
            draw_ribbon(ax, 0.18, y0, 0.82, y1, h, DOMAIN_COLORS[domain], 0.46)
            left_offsets[domain] -= h
            right_offsets[change_type] -= h
    for domain in DOMAIN_ORDER:
        bottom, top = left_ranges[domain]
        color = DOMAIN_COLORS[domain]
        ax.add_patch(Rectangle((0.03, bottom), 0.12, top - bottom, facecolor=color, edgecolor="white", linewidth=1.0, alpha=0.95, zorder=3))
        ax.text(0.01, (bottom + top) / 2, f"{domain}\n{int(counts.loc[domain].sum())}", ha="right", va="center", fontsize=8.2, color=color, weight="bold")
    for change_type in CHANGE_ORDER:
        bottom, top = right_ranges[change_type]
        color = CHANGE_COLORS[change_type]
        n = int(counts[change_type].sum())
        alpha = 0.95 if n > 0 else 0.18
        ax.add_patch(Rectangle((0.85, bottom), 0.12, max(top - bottom, 0.018), facecolor=color, edgecolor="white", linewidth=1.0, alpha=alpha, zorder=3))
        ax.text(0.99, (bottom + top) / 2, f"{change_type}\n{n}", ha="left", va="center", fontsize=8.1, color=color, weight="bold", alpha=1.0 if n > 0 else 0.48)
    ax.text(0.50, 1.095, "Subfield temporal change braids", ha="center", fontsize=15.5, weight="bold")
    ax.text(0.50, 1.055, "All observed subfields compact; the split is whether local structure becomes more uneven or smoother", ha="center", fontsize=9.0, color="#444444")
    save(fig, "fig_07u_domain_variant_01_clean_braids")


def figure_07v2_domain_split_cards() -> None:
    counts = change_counts()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.8), sharex=True)
    axes = axes.reshape(-1)
    for ax, domain in zip(axes, DOMAIN_ORDER):
        row = counts.loc[domain]
        total = row.sum()
        left = 0.0
        for change_type in CHANGE_ORDER:
            n = int(row[change_type])
            if n == 0:
                continue
            share = n / total
            ax.barh([0], [share], left=left, color=CHANGE_COLORS[change_type], height=0.44)
            ax.text(left + share / 2, 0, f"{n}\n{share:.0%}", ha="center", va="center", fontsize=9.0, color="white", weight="bold")
            left += share
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.7, 0.7)
        ax.set_yticks([])
        ax.set_title(f"{domain} (n={int(total)})", color=DOMAIN_COLORS[domain], fontsize=12.5, weight="bold")
        ax.grid(axis="x", color="#eeeeee", linewidth=0.55)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.suptitle("Domain split cards: two observed temporal destinations", fontsize=15.5, y=0.98)
    fig.legend(handles=change_handles()[:2], loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=True, fontsize=8.5)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    save(fig, "fig_07u_domain_variant_02_split_cards")


def figure_07v3_domain_change_swarm() -> None:
    frame = subfield_change_frame()
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.8), sharex=True, sharey=True)
    axes = axes.reshape(-1)
    xlim = (frame["spread_delta"].quantile(0.01) - 0.15, frame["spread_delta"].quantile(0.99) + 0.15)
    ylim = (frame["concentration_delta"].quantile(0.01) - 0.15, frame["concentration_delta"].quantile(0.99) + 0.15)
    for ax, domain in zip(axes, DOMAIN_ORDER):
        sub = frame[frame["domain_display_name"] == domain]
        ax.axhline(0, color="#d0d0d0", linewidth=0.9)
        ax.axvline(0, color="#d0d0d0", linewidth=0.9)
        for change_type in CHANGE_ORDER:
            ss = sub[sub["change_type"] == change_type]
            if ss.empty:
                continue
            ax.scatter(ss["spread_delta"], ss["concentration_delta"], s=25, color=CHANGE_COLORS[change_type], alpha=0.58, edgecolor="white", linewidth=0.25)
        top = sub.nlargest(3, "change_magnitude")
        for row in top.itertuples(index=False):
            label = row.subfield_display_name
            if len(label) > 23:
                label = label[:21] + "..."
            ax.text(row.spread_delta + 0.04, row.concentration_delta + 0.04, label, fontsize=6.7)
        ax.set_title(domain, color=DOMAIN_COLORS[domain], fontsize=12.0, weight="bold")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(color="#eeeeee", linewidth=0.55)
    fig.suptitle("Domain change swarms", fontsize=15.5, y=0.99)
    fig.text(0.5, 0.02, "Spread/sparsity delta", ha="center", fontsize=10)
    fig.text(0.02, 0.5, "Unevenness/hubness delta", va="center", rotation="vertical", fontsize=10)
    fig.legend(handles=change_handles()[:2], loc="lower center", bbox_to_anchor=(0.5, -0.035), ncol=2, frameon=True, fontsize=8.2)
    fig.tight_layout(rect=[0.04, 0.06, 1, 0.95])
    save(fig, "fig_07u_domain_variant_03_change_swarms")


def figure_07v4_radial_domain_split() -> None:
    counts = change_counts()
    fig, ax = plt.subplots(figsize=(9.8, 9.8), subplot_kw={"projection": "polar"})
    ax.set_theta_direction(-1)
    ax.set_theta_offset(pi / 2)
    ax.set_ylim(0, 1.28)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color="#eeeeee", linewidth=0.55)
    ax.spines["polar"].set_visible(False)
    width = 2 * pi / len(DOMAIN_ORDER) * 0.72
    for idx, domain in enumerate(DOMAIN_ORDER):
        theta = idx * 2 * pi / len(DOMAIN_ORDER)
        total = counts.loc[domain].sum()
        start = 0.38
        for change_type in CHANGE_ORDER:
            n = counts.loc[domain, change_type]
            if n == 0:
                continue
            length = 0.72 * n / total
            ax.bar(theta, length, bottom=start, width=width, color=CHANGE_COLORS[change_type], alpha=0.86, edgecolor="white", linewidth=0.9)
            ax.text(theta, start + length / 2, str(int(n)), ha="center", va="center", fontsize=9.0, color="white", weight="bold")
            start += length
        ax.scatter(theta, 0.25, s=360, color=DOMAIN_COLORS[domain], edgecolor="white", linewidth=1.0, zorder=4)
        ax.text(theta, 1.18, domain.replace(" ", "\n"), color=DOMAIN_COLORS[domain], ha="center", va="center", fontsize=10, weight="bold")
    ax.text(0, 0, "temporal\nsplit", ha="center", va="center", fontsize=14, weight="bold", color="#333333")
    ax.set_title("Radial domain split", fontsize=15.5, pad=22)
    ax.legend(handles=change_handles()[:2], loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=True, fontsize=8.2)
    save(fig, "fig_07u_domain_variant_04_radial_split")


def build_gallery(stems: list[str], gallery_name: str) -> None:
    ncols = 2
    nrows = int(np.ceil(len(stems) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.0, nrows * 5.2))
    axes = np.array(axes).reshape(-1)
    for ax, stem in zip(axes, stems):
        img = mpimg.imread(OUTPUT_DIR / f"{stem}.png")
        ax.imshow(img)
        ax.set_title(stem.replace("_", " "), fontsize=10)
        ax.axis("off")
    for ax in axes[len(stems) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / gallery_name, bbox_inches="tight", dpi=180)
    plt.close(fig)


def write_report() -> None:
    text = """# Selected Visual Variants

Generated from existing TFM outputs only. No thesis chapter was modified.

## 08u Attractor Variants

1. `fig_08u_attractor_variant_01_focus_stars`
   - Small-multiple ego maps for the six strongest attractor fields.
   - Best if we want a memorable figure that still shows the actual incoming fields.

2. `fig_08u_attractor_variant_02_distance_ladder`
   - Attractor fields as rows; incoming source fields are placed by morphology distance.
   - Best if we want a precise, readable analytical version.

3. `fig_08u_attractor_variant_03_domain_to_attractor_ribbons`
   - Source domains flow into fields that receive two or more incoming nearest-neighbor links.
   - Best if we want the attractor story as a taxonomy-bending flow.

4. `fig_08u_attractor_variant_04_bead_matrix`
   - Matrix of attractor fields by source domain, using beads and source labels.
   - Best as a compact diagnostic; less visually dramatic.

## 07u Domain Variants

1. `fig_07u_domain_variant_01_clean_braids`
   - Clean alluvial version of the domain-to-change-archetype idea.
   - Best overall version if we keep this idea.

2. `fig_07u_domain_variant_02_split_cards`
   - Four domain cards showing the split between compacting + uneven and compacting + smoothing.
   - Best if we want maximum clarity and low visual risk.

3. `fig_07u_domain_variant_03_change_swarms`
   - Four domain scatter swarms showing subfield-level distributions in change space.
   - Best if we want to emphasize within-domain heterogeneity.

4. `fig_07u_domain_variant_04_radial_split`
   - Circular domain split, with segment length encoding the two observed change destinations.
   - Best if we want something more visual and compact than bars.

## Initial Preference

- For 08u: try variant 03 first, then variant 01.
- For 07u: variant 01 is the strongest, variant 03 is the most analytically rich, and variant 04 is the most visually distinctive.
"""
    (OUTPUT_DIR / "selected_visual_variants_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    setup()
    attractor_stems = [
        "fig_08u_attractor_variant_01_focus_stars",
        "fig_08u_attractor_variant_02_distance_ladder",
        "fig_08u_attractor_variant_03_domain_to_attractor_ribbons",
        "fig_08u_attractor_variant_04_bead_matrix",
    ]
    domain_stems = [
        "fig_07u_domain_variant_01_clean_braids",
        "fig_07u_domain_variant_02_split_cards",
        "fig_07u_domain_variant_03_change_swarms",
        "fig_07u_domain_variant_04_radial_split",
    ]
    figure_08v1_attractor_focus_stars()
    figure_08v2_attractor_distance_ladder()
    figure_08v3_domain_to_attractor_ribbons()
    figure_08v4_attractor_bead_matrix()
    figure_07v1_clean_braids()
    figure_07v2_domain_split_cards()
    figure_07v3_domain_change_swarm()
    figure_07v4_radial_domain_split()
    build_gallery(attractor_stems, "attractor_variants_gallery.png")
    build_gallery(domain_stems, "domain_change_variants_gallery.png")
    build_gallery([*attractor_stems, *domain_stems], "selected_visual_variants_gallery.png")
    write_report()


if __name__ == "__main__":
    main()
