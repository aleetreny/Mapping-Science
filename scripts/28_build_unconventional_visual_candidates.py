from __future__ import annotations

import sys
from math import cos, sin, pi
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
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.morphological_typologies import DOMAIN_COLORS, TYPOLOGY_COLORS, TYPOLOGY_ORDER
from src.temporal_common import REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS


OUTPUT_DIR = ROOT / "outputs" / "12_unconventional_visual_candidates"

DOMAIN_ORDER = ["Life Sciences", "Social Sciences", "Physical Sciences", "Health Sciences"]
TYPOLOGY_LABELS = dict(TYPOLOGY_ORDER)
DYNAMIC_COLORS = {1: "#4c78a8", 2: "#c47a2c", 3: "#6b8e23", 4: "#8e5ea2"}
DYNAMIC_LABELS = {
    1: "Compacting locally uneven",
    2: "Broadening hub-diffusing",
    3: "Smoothing sparse-neighborhood",
    4: "Dimensionalizing hub-concentrating",
}
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
SPECTRAL_METRICS = ["embedding_pca_dim_80", "embedding_pca_spectral_entropy"]
TEMPORAL_METRICS = [
    "embedding_centroid_drift_early_late",
    "embedding_radial_expansion_slope",
    "embedding_recent_novelty_score",
]
STATIC_FAMILIES = {
    "Spread / sparsity": SPREAD_METRICS,
    "Local unevenness": CONCENTRATION_METRICS,
    "Spectral structure": SPECTRAL_METRICS,
    "Temporal movement": TEMPORAL_METRICS,
}


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


def zscore(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    values = frame[metrics].astype(float)
    std = values.std(axis=0, ddof=0).replace(0, np.nan)
    return (values - values.mean(axis=0)) / std


def short_field(name: str) -> str:
    return FIELD_LABELS.get(name, name)


def domain_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=DOMAIN_COLORS[d], markeredgecolor="white", markersize=8, label=d)
        for d in DOMAIN_ORDER
    ]


def static_profiles(level: str) -> pd.DataFrame:
    static = pd.read_csv(ROOT / "outputs" / "05_static_comparison" / "static_discipline_profiles.csv")
    frame = static[static["level"] == level].copy()
    scaled = zscore(frame, REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS)
    for metric in REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS:
        frame[f"z_{metric}"] = scaled[metric].to_numpy()
    for family, metrics in STATIC_FAMILIES.items():
        frame[family] = scaled[metrics].mean(axis=1)
    frame["spread_sparse_score"] = scaled[SPREAD_METRICS].mean(axis=1)
    frame["local_concentration_score"] = scaled[CONCENTRATION_METRICS].mean(axis=1)
    return frame


def field_domains() -> pd.DataFrame:
    return pd.read_csv(ROOT / "outputs" / "09_morphological_typologies" / "subfield_typology_assignments.csv")[
        ["field_display_name", "domain_display_name"]
    ].drop_duplicates()


def field_change_scores() -> pd.DataFrame:
    raw = pd.read_parquet(ROOT / "data" / "processed" / "temporal" / "field_window_embedding_metric_trajectories.parquet")
    rows = []
    for (field, window), group in raw.groupby(["field_display_name", "window_index"], sort=False):
        values = group.set_index("metric")["mean_standardized_value"]
        rows.append(
            {
                "field_display_name": field,
                "window_index": int(window),
                "spread_score": float(values.reindex(SPREAD_METRICS).mean()),
                "concentration_score": float(values.reindex(CONCENTRATION_METRICS).mean()),
            }
        )
    scores = pd.DataFrame(rows)
    wide = scores.pivot_table(index="field_display_name", columns="window_index", values=["spread_score", "concentration_score"], aggfunc="first")
    out = pd.DataFrame(
        {
            "field_display_name": wide.index,
            "spread_delta": wide[("spread_score", 4)] - wide[("spread_score", 0)],
            "concentration_delta": wide[("concentration_score", 4)] - wide[("concentration_score", 0)],
        }
    ).reset_index(drop=True)
    out["change_magnitude"] = np.hypot(out["spread_delta"], out["concentration_delta"])
    out["change_type"] = [
        ("broadening" if x >= 0 else "compacting") + (" + more uneven" if y >= 0 else " + smoothing")
        for x, y in zip(out["spread_delta"], out["concentration_delta"])
    ]
    return out.merge(field_domains(), on="field_display_name", how="left")


def subfield_change_scores() -> pd.DataFrame:
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


def stacked_ranges(counts: pd.DataFrame, order: list, total: float, gap: float) -> dict:
    heights = {key: counts.loc[key].sum() / total for key in order}
    scale = (1.0 - gap * (len(order) - 1)) / sum(heights.values())
    heights = {key: heights[key] * scale for key in order}
    y = 1.0
    ranges = {}
    for key in order:
        h = heights[key]
        ranges[key] = (y - h, y)
        y -= h + gap
    return ranges


def figure_06u_domain_morphology_crests() -> None:
    frame = static_profiles("domain")
    families = list(STATIC_FAMILIES)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.6), sharex=True, sharey=True)
    axes = axes.reshape(-1)
    for ax, (_, row) in zip(axes, frame.iterrows()):
        domain = row["entity_display_name"]
        color = DOMAIN_COLORS[domain]
        vals = [float(row[family]) for family in families]
        y = np.arange(len(families))
        ax.axvline(0, color="#d0d0d0", linewidth=1)
        ax.hlines(y, 0, vals, color=color, linewidth=5, alpha=0.58)
        ax.scatter(vals, y, s=210, color=color, edgecolor="white", linewidth=1.2, zorder=3)
        ax.scatter([0] * len(y), y, s=24, color="#dddddd", zorder=2)
        ax.set_yticks(y, families)
        ax.invert_yaxis()
        ax.set_title(f"{domain} crest", color=color, fontsize=13, weight="bold")
        ax.grid(axis="x", color="#eeeeee", linewidth=0.6)
        ax.set_xlim(-1.65, 1.65)
    fig.suptitle("Domain morphology crests", fontsize=16, y=0.985)
    fig.text(0.5, 0.025, "Mean standardized family score (left = below-average, right = above-average)", ha="center", fontsize=10)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    save(fig, "fig_06u_domain_morphology_crests_candidate")


def figure_06u_field_morphology_islands() -> None:
    frame = static_profiles("field").merge(field_domains(), left_on="entity_display_name", right_on="field_display_name", how="left")
    scaled = zscore(frame, REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS)
    coords = PCA(n_components=2, random_state=42).fit_transform(scaled)
    frame["x"], frame["y"] = coords[:, 0], coords[:, 1]
    frame["radius"] = np.hypot(frame["x"], frame["y"])

    fig, ax = plt.subplots(figsize=(10.8, 8.4))
    xy = np.vstack([frame["x"], frame["y"]])
    kde = gaussian_kde(xy)
    xx, yy = np.mgrid[frame["x"].min() - 0.5 : frame["x"].max() + 0.5 : 150j, frame["y"].min() - 0.5 : frame["y"].max() + 0.5 : 150j]
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    levels = np.linspace(zz.min(), zz.max() * 0.9, 8)
    ax.contourf(xx, yy, zz, levels=levels, cmap="Greys", alpha=0.22)
    ax.contour(xx, yy, zz, levels=levels[2:], colors="#bdbdbd", linewidths=0.55, alpha=0.65)
    for domain in DOMAIN_ORDER:
        sub = frame[frame["domain_display_name"] == domain]
        ax.scatter(sub["x"], sub["y"], s=80 + sub["n_subfields"] * 14, color=DOMAIN_COLORS[domain], edgecolor="white", linewidth=0.9, alpha=0.92, label=domain)
    for row in frame.nlargest(12, "radius").itertuples(index=False):
        ax.text(row.x + 0.07, row.y + 0.06, short_field(row.entity_display_name), fontsize=7.8, color="#222222")
    ax.set_title("Field morphology islands", fontsize=16, pad=16)
    ax.set_xlabel("Profile PCA 1")
    ax.set_ylabel("Profile PCA 2")
    ax.legend(handles=domain_handles(), loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, fontsize=8.5)
    ax.grid(color="#f0f0f0", linewidth=0.45)
    save(fig, "fig_06u_field_morphology_islands_candidate")


def figure_06u_field_extreme_glyph_ring() -> None:
    frame = static_profiles("field").merge(field_domains(), left_on="entity_display_name", right_on="field_display_name", how="left")
    frame["profile_energy"] = np.sqrt(sum(frame[family] ** 2 for family in STATIC_FAMILIES))
    frame = frame.sort_values(["domain_display_name", "profile_energy"], ascending=[True, False]).reset_index(drop=True)
    n = len(frame)
    angles = np.linspace(pi / 2, pi / 2 + 2 * pi, n, endpoint=False)
    fig, ax = plt.subplots(figsize=(10.0, 10.0), subplot_kw={"projection": "polar"})
    ax.set_theta_direction(-1)
    ax.set_theta_offset(pi / 2)
    ax.set_ylim(0, 1.45)
    ax.grid(color="#eeeeee", linewidth=0.55)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.spines["polar"].set_visible(False)
    family_colors = ["#222222", "#7a7a7a", "#b7b7b7", "#d55e00"]
    for angle, (_, row) in zip(angles, frame.iterrows()):
        color = DOMAIN_COLORS[row["domain_display_name"]]
        base = 0.72
        ax.scatter(angle, base, s=58 + row["n_subfields"] * 12, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        vals = [float(row[family]) for family in STATIC_FAMILIES]
        for idx, val in enumerate(vals):
            theta = angle + (idx - 1.5) * 0.018
            length = np.clip(abs(val), 0, 1.8) * 0.18
            direction = 1 if val >= 0 else -1
            r0 = base
            r1 = base + direction * length
            ax.plot([theta, theta], [r0, r1], color=family_colors[idx], linewidth=1.7, alpha=0.75, zorder=2)
        if row["profile_energy"] >= frame["profile_energy"].quantile(0.72):
            ha = "left" if cos(angle) > 0 else "right"
            ax.text(angle, 1.06, short_field(row["entity_display_name"]), fontsize=7.3, ha=ha, va="center")
    for domain in DOMAIN_ORDER:
        idxs = frame.index[frame["domain_display_name"] == domain].tolist()
        if not idxs:
            continue
        mid = float(np.mean(angles[idxs]))
        ax.text(mid, 1.34, domain.replace(" ", "\n"), color=DOMAIN_COLORS[domain], ha="center", va="center", fontsize=9.5, weight="bold")
    ax.set_title("Field morphology glyph ring", fontsize=16, pad=22)
    save(fig, "fig_06u_field_extreme_glyph_ring_candidate")


def figure_07u_field_delta_rose() -> None:
    frame = field_change_scores().sort_values(["domain_display_name", "change_type", "change_magnitude"], ascending=[True, True, False]).reset_index(drop=True)
    n = len(frame)
    angles = np.linspace(0, 2 * pi, n, endpoint=False)
    max_mag = frame["change_magnitude"].max()
    fig, ax = plt.subplots(figsize=(10.2, 10.2), subplot_kw={"projection": "polar"})
    ax.set_theta_direction(-1)
    ax.set_theta_offset(pi / 2)
    ax.set_ylim(0, 1.25)
    ax.grid(color="#eeeeee", linewidth=0.55)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.spines["polar"].set_visible(False)
    for angle, row in zip(angles, frame.itertuples(index=False)):
        r0 = 0.48
        r1 = 0.48 + 0.55 * row.change_magnitude / max_mag
        ax.bar(angle, r1 - r0, bottom=r0, width=2 * pi / n * 0.78, color=CHANGE_COLORS[row.change_type], alpha=0.78, edgecolor="white", linewidth=0.45)
        ax.scatter(angle, 0.42, s=18, color=DOMAIN_COLORS[row.domain_display_name], alpha=0.95)
    for row, angle in zip(frame.nlargest(9, "change_magnitude").itertuples(index=False), angles[frame.nlargest(9, "change_magnitude").index]):
        rotation = np.rad2deg(pi / 2 - angle)
        ha = "left" if cos(angle) >= 0 else "right"
        ax.text(angle, 1.10, short_field(row.field_display_name), fontsize=7.4, rotation=rotation, rotation_mode="anchor", ha=ha, va="center")
    handles = [Line2D([0], [0], color=color, linewidth=7, label=label) for label, color in CHANGE_COLORS.items()]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=True, fontsize=8.3)
    ax.set_title("Field temporal delta rose", fontsize=16, pad=24)
    save(fig, "fig_07u_field_delta_rose_candidate")


def figure_07u_subfield_change_constellation() -> None:
    frame = subfield_change_scores()
    metrics = SPREAD_METRICS + CONCENTRATION_METRICS
    coords = PCA(n_components=2, random_state=42).fit_transform(frame[metrics].fillna(0))
    frame["x"], frame["y"] = coords[:, 0], coords[:, 1]
    fig, ax = plt.subplots(figsize=(10.8, 8.0))
    for domain in DOMAIN_ORDER:
        sub = frame[frame["domain_display_name"] == domain]
        ax.scatter(sub["x"], sub["y"], s=24, color=DOMAIN_COLORS[domain], alpha=0.36, edgecolor="none")
    top = frame.nlargest(14, "change_magnitude")
    ax.scatter(top["x"], top["y"], s=85, color=[DOMAIN_COLORS[d] for d in top["domain_display_name"]], edgecolor="white", linewidth=0.9, zorder=3)
    for row in top.itertuples(index=False):
        label = str(row.subfield_display_name)
        if len(label) > 24:
            label = label[:22] + "..."
        ax.text(row.x + 0.06, row.y + 0.05, label, fontsize=7.2, color="#222222")
    ax.set_title("Subfield change constellation", fontsize=16, pad=16)
    ax.set_xlabel("Change PCA 1")
    ax.set_ylabel("Change PCA 2")
    ax.legend(handles=domain_handles(), loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, fontsize=8.5)
    ax.grid(color="#eeeeee", linewidth=0.55)
    save(fig, "fig_07u_subfield_change_constellation_candidate")


def figure_07u_domain_change_braids() -> None:
    frame = subfield_change_scores()
    counts = (
        frame.groupby(["domain_display_name", "change_type"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=DOMAIN_ORDER, columns=list(CHANGE_COLORS), fill_value=0)
    )
    total = float(counts.to_numpy().sum())
    left_ranges = stacked_ranges(counts, DOMAIN_ORDER, total, 0.045)
    right_ranges = stacked_ranges(counts.T, list(CHANGE_COLORS), total, 0.045)
    left_offsets = {key: left_ranges[key][1] for key in DOMAIN_ORDER}
    right_offsets = {key: right_ranges[key][1] for key in CHANGE_COLORS}
    fig, ax = plt.subplots(figsize=(11.4, 7.6))
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.04, 1.10)
    ax.axis("off")
    for domain in DOMAIN_ORDER:
        for change_type in CHANGE_COLORS:
            n = int(counts.loc[domain, change_type])
            if n == 0:
                continue
            h = max(n / total * 0.86, 0.003)
            y0 = left_offsets[domain] - h / 2
            y1 = right_offsets[change_type] - h / 2
            draw_ribbon(ax, 0.18, y0, 0.82, y1, h, DOMAIN_COLORS[domain], 0.46)
            left_offsets[domain] -= h
            right_offsets[change_type] -= h
    for domain in DOMAIN_ORDER:
        bottom, top = left_ranges[domain]
        color = DOMAIN_COLORS[domain]
        ax.add_patch(Rectangle((0.03, bottom), 0.12, top - bottom, facecolor=color, edgecolor="white", linewidth=1, alpha=0.95, zorder=3))
        ax.text(0.01, (bottom + top) / 2, f"{domain}\n{int(counts.loc[domain].sum())}", ha="right", va="center", fontsize=8.3, color=color, weight="bold")
    for change_type in CHANGE_COLORS:
        bottom, top = right_ranges[change_type]
        color = CHANGE_COLORS[change_type]
        ax.add_patch(Rectangle((0.85, bottom), 0.12, top - bottom, facecolor=color, edgecolor="white", linewidth=1, alpha=0.95, zorder=3))
        ax.text(0.99, (bottom + top) / 2, f"{change_type}\n{int(counts[change_type].sum())}", ha="left", va="center", fontsize=8.1, color=color, weight="bold")
    ax.text(0.50, 1.075, "Subfield temporal change braids", ha="center", fontsize=16, weight="bold")
    ax.text(0.50, 1.035, "Domains flow into change archetypes, not one shared temporal story", ha="center", fontsize=9.2, color="#444444")
    save(fig, "fig_07u_domain_change_braids_candidate")


def figure_08u_domain_breach_chords() -> None:
    nearest = pd.read_csv(ROOT / "outputs" / "10_global_map_candidates" / "field_nearest_morphological_neighbors.csv")
    fd = field_domains()
    nearest = nearest.merge(fd, left_on="field", right_on="field_display_name", how="left").rename(columns={"domain_display_name": "source_domain"})
    nearest = nearest.merge(fd, left_on="nearest_field", right_on="field_display_name", how="left").rename(columns={"domain_display_name": "target_domain"})
    cross = nearest[nearest["source_domain"] != nearest["target_domain"]]
    counts = cross.groupby(["source_domain", "target_domain"]).size().reset_index(name="n")
    angles = dict(zip(DOMAIN_ORDER, np.linspace(pi / 2, pi / 2 + 2 * pi, len(DOMAIN_ORDER), endpoint=False)))
    pos = {d: (cos(a), sin(a)) for d, a in angles.items()}
    fig, ax = plt.subplots(figsize=(9.6, 8.6))
    ax.set_aspect("equal")
    ax.axis("off")
    for domain in DOMAIN_ORDER:
        x, y = pos[domain]
        ax.scatter(x, y, s=950, color=DOMAIN_COLORS[domain], edgecolor="white", linewidth=1.5, zorder=4)
        ax.text(1.22 * x, 1.22 * y, domain.replace(" ", "\n"), color=DOMAIN_COLORS[domain], ha="center", va="center", fontsize=10.5, weight="bold")
    for row in counts.itertuples(index=False):
        a = pos[row.source_domain]
        b = pos[row.target_domain]
        rad = 0.26 if row.source_domain < row.target_domain else -0.26
        ax.add_patch(
            FancyArrowPatch(
                a,
                b,
                arrowstyle="-|>",
                mutation_scale=17,
                linewidth=1.8 + row.n * 1.45,
                color=DOMAIN_COLORS[row.source_domain],
                alpha=0.46,
                connectionstyle=f"arc3,rad={rad}",
                shrinkA=28,
                shrinkB=28,
                zorder=2,
            )
        )
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        lx, ly = mx * 0.82, my * 0.82
        if abs(lx) < 0.22 and abs(ly) < 0.18:
            lx += 0.34
            ly += 0.08
        ax.text(
            lx,
            ly,
            str(row.n),
            fontsize=9,
            color="#222222",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
            zorder=5,
        )
    ax.text(
        0,
        0.01,
        "18 cross-domain\nnearest-neighbor links",
        ha="center",
        va="center",
        fontsize=12.5,
        weight="bold",
        color="#333333",
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        zorder=6,
    )
    ax.set_title("Taxonomy breach chord map", fontsize=16, pad=18)
    ax.set_xlim(-2.05, 2.05)
    ax.set_ylim(-1.48, 1.48)
    save(fig, "fig_08u_domain_breach_chords_candidate")


def figure_08u_attractor_spider_map() -> None:
    nearest = pd.read_csv(ROOT / "outputs" / "10_global_map_candidates" / "field_nearest_morphological_neighbors.csv")
    fd = field_domains()
    incoming = nearest.groupby("nearest_field").size().sort_values(ascending=False)
    attractors = incoming[incoming >= 2].index.tolist()[:6]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.8))
    axes = axes.reshape(-1)
    for ax, attractor in zip(axes, attractors):
        inc = nearest[nearest["nearest_field"] == attractor].copy()
        attractor_domain = fd.loc[fd["field_display_name"] == attractor, "domain_display_name"].iloc[0]
        ax.set_aspect("equal")
        ax.axis("off")
        ax.scatter(0, 0, s=520, color=DOMAIN_COLORS[attractor_domain], edgecolor="white", linewidth=1.2, zorder=3)
        ax.text(0, -0.02, short_field(attractor), ha="center", va="center", color="white", fontsize=8.2, weight="bold", zorder=4)
        angles = np.linspace(0, 2 * pi, len(inc), endpoint=False) if len(inc) else []
        for angle, row in zip(angles, inc.itertuples(index=False)):
            source_domain = fd.loc[fd["field_display_name"] == row.field, "domain_display_name"].iloc[0]
            x, y = 0.88 * cos(angle), 0.88 * sin(angle)
            cross = source_domain != attractor_domain
            ax.add_patch(FancyArrowPatch((x, y), (0, 0), arrowstyle="-|>", mutation_scale=12, color="#222222" if cross else "#8a8a8a", linewidth=1.2, alpha=0.60, shrinkA=10, shrinkB=22))
            ax.scatter(x, y, s=190, color=DOMAIN_COLORS[source_domain], edgecolor="white", linewidth=0.8, zorder=3)
            ax.text(1.08 * x, 1.08 * y, short_field(row.field), ha="center", va="center", fontsize=7.2)
        ax.set_title(f"{len(inc)} incoming links", fontsize=9, color=DOMAIN_COLORS[attractor_domain])
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.25, 1.25)
    for ax in axes[len(attractors) :]:
        ax.axis("off")
    fig.suptitle("Morphological attractor spider maps", fontsize=16, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig_08u_attractor_spider_map_candidate")


def figure_09u_orbital_typology_flow() -> None:
    assignments = pd.read_csv(
        ROOT / "outputs" / "09_morphological_typologies" / "temporal_trajectory_clustering" / "trajectory_cluster_assignments.csv"
    )
    counts = assignments.groupby(["typology_id", "trajectory_cluster_id"]).size().reset_index(name="n")
    static_order = [tid for tid, _ in TYPOLOGY_ORDER]
    dynamic_order = [1, 2, 3, 4]
    static_angles = dict(zip(static_order, np.linspace(pi * 0.74, pi * 1.26, len(static_order))))
    dynamic_angles = dict(zip(dynamic_order, np.linspace(-pi * 0.25, pi * 0.25, len(dynamic_order))))
    r_static = 1.0
    r_dynamic = 1.0
    fig, ax = plt.subplots(figsize=(10.5, 8.8))
    ax.set_aspect("equal")
    ax.axis("off")
    for row in counts.itertuples(index=False):
        a0 = static_angles[row.typology_id]
        a1 = dynamic_angles[row.trajectory_cluster_id]
        p0 = (r_static * cos(a0), r_static * sin(a0))
        p1 = (r_dynamic * cos(a1), r_dynamic * sin(a1))
        rad = 0.10 + 0.03 * (row.trajectory_cluster_id - 2)
        ax.add_patch(
            FancyArrowPatch(
                p0,
                p1,
                arrowstyle="-",
                linewidth=0.8 + row.n * 0.16,
                color=TYPOLOGY_COLORS[row.typology_id],
                alpha=0.33,
                connectionstyle=f"arc3,rad={rad}",
                zorder=1,
            )
        )
    for sid in static_order:
        a = static_angles[sid]
        x, y = r_static * cos(a), r_static * sin(a)
        total = int(counts[counts["typology_id"] == sid]["n"].sum())
        ax.scatter(x, y, s=420 + total * 4, color=TYPOLOGY_COLORS[sid], edgecolor="white", linewidth=1.0, zorder=3)
        ax.text(x - 0.18, y, f"{sid}\n{TYPOLOGY_LABELS[sid]}\n{total}", ha="right", va="center", fontsize=7.6, color=TYPOLOGY_COLORS[sid], weight="bold")
    for did in dynamic_order:
        a = dynamic_angles[did]
        x, y = r_dynamic * cos(a), r_dynamic * sin(a)
        total = int(counts[counts["trajectory_cluster_id"] == did]["n"].sum())
        ax.scatter(x, y, s=420 + total * 4, color=DYNAMIC_COLORS[did], edgecolor="white", linewidth=1.0, zorder=3)
        ax.text(x + 0.18, y, f"D{did}\n{DYNAMIC_LABELS[did]}\n{total}", ha="left", va="center", fontsize=7.6, color=DYNAMIC_COLORS[did], weight="bold")
    ax.text(0, 0.06, "shape\nvs\nchange", ha="center", va="center", fontsize=15, weight="bold", color="#333333")
    ax.set_title("Orbital static-dynamic typology flow", fontsize=16, pad=18)
    ax.set_xlim(-1.72, 1.78)
    ax.set_ylim(-1.25, 1.25)
    save(fig, "fig_09u_orbital_typology_flow_candidate")


def figure_09u_split_typology_constellation() -> None:
    static = pd.read_csv(ROOT / "outputs" / "09_morphological_typologies" / "typology_pca_scores.csv")
    dynamic = pd.read_csv(
        ROOT / "outputs" / "09_morphological_typologies" / "temporal_trajectory_clustering" / "trajectory_cluster_assignments.csv"
    )[["subfield_id", "trajectory_cluster_id"]]
    frame = static.merge(dynamic, on="subfield_id", how="inner")
    fig, ax = plt.subplots(figsize=(10.4, 8.2))
    markers = {1: "o", 2: "s", 3: "^", 4: "D"}
    for tid, label in TYPOLOGY_ORDER:
        sub = frame[frame["typology_id"] == tid]
        for did in sorted(sub["trajectory_cluster_id"].unique()):
            ss = sub[sub["trajectory_cluster_id"] == did]
            ax.scatter(ss["pca_1"], ss["pca_2"], s=38, marker=markers[int(did)], color=TYPOLOGY_COLORS[tid], edgecolor="white", linewidth=0.45, alpha=0.72)
    prototypes = pd.read_csv(ROOT / "outputs" / "09_morphological_typologies" / "subfield_typology_assignments.csv")
    proto_ids = prototypes[prototypes["is_typology_prototype"] == True]["subfield_id"].astype(str)
    top = frame[frame["subfield_id"].astype(str).isin(proto_ids)]
    ax.scatter(top["pca_1"], top["pca_2"], s=150, facecolor="none", edgecolor="#111111", linewidth=1.2, zorder=4)
    for row in top.itertuples(index=False):
        label = str(row.subfield_display_name)
        if len(label) > 24:
            label = label[:22] + "..."
        ax.text(row.pca_1 + 0.05, row.pca_2 + 0.05, label, fontsize=7.0)
    typ_handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=TYPOLOGY_COLORS[tid], markeredgecolor="white", markersize=8, label=f"{tid} {label}") for tid, label in TYPOLOGY_ORDER]
    dyn_handles = [Line2D([0], [0], marker=markers[did], color="#444444", markerfacecolor="#cccccc", linestyle="none", markersize=7, label=f"D{did}") for did in markers]
    leg1 = ax.legend(handles=typ_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, fontsize=7.5, title="Static")
    ax.add_artist(leg1)
    ax.legend(handles=dyn_handles, loc="lower left", bbox_to_anchor=(1.01, 0.0), frameon=True, fontsize=8, title="Dynamic")
    ax.set_title("Split typology constellation", fontsize=16, pad=16)
    ax.set_xlabel("Static morphology PCA 1")
    ax.set_ylabel("Static morphology PCA 2")
    ax.grid(color="#eeeeee", linewidth=0.55)
    save(fig, "fig_09u_split_typology_constellation_candidate")


def build_gallery(stems: list[str]) -> None:
    ncols = 2
    nrows = int(np.ceil(len(stems) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.0, nrows * 5.2))
    axes = np.array(axes).reshape(-1)
    for ax, stem in zip(axes, stems):
        img = mpimg.imread(OUTPUT_DIR / f"{stem}.png")
        ax.imshow(img)
        ax.set_title(stem.replace("_candidate", "").replace("_", " "), fontsize=10)
        ax.axis("off")
    for ax in axes[len(stems) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "unconventional_candidate_gallery.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


def write_report() -> None:
    text = """# Unconventional Visualization Candidates

Generated from existing TFM outputs only. These are experimental sketches; no thesis file was modified.

## Candidates

1. `fig_06u_domain_morphology_crests_candidate`
   - Location: Chapter 6 after the domain heatmap.
   - Idea: domain-level morphology as signed crests across four metric families.
   - Why it is different: it behaves like a visual profile emblem, not a matrix.

2. `fig_06u_field_morphology_islands_candidate`
   - Location: Chapter 6 after the field heatmap or near the existing PCA profile map.
   - Idea: fields become islands in profile space, with density contours as terrain.
   - Why it is different: it reads as a morphology landscape.

3. `fig_06u_field_extreme_glyph_ring_candidate`
   - Location: Chapter 6 after the field heatmap.
   - Idea: fields are arranged in a ring by domain; each field carries a small four-family glyph.
   - Risk: visually distinctive but more complex to read.

4. `fig_07u_field_delta_rose_candidate`
   - Location: Chapter 7 after the field temporal delta heatmap.
   - Idea: field-level temporal change as a circular rose; bar length is change magnitude and color is change archetype.
   - Why it is different: it makes the heatmap feel like a result constellation.

5. `fig_07u_subfield_change_constellation_candidate`
   - Location: Chapter 7 in the dynamic-region section.
   - Idea: all subfields projected by temporal-change vectors; strongest movers highlighted.
   - Why it is different: it shows heterogeneity beneath fields.

6. `fig_07u_domain_change_braids_candidate`
   - Location: Chapter 7 after temporal-change results.
   - Idea: domains flow into temporal change archetypes.
   - Why it is different: it directly visualizes that domains do not share one temporal destiny.

7. `fig_08u_domain_breach_chords_candidate`
   - Location: Chapter 8 after the nearest-neighbor ring or domain-pair heatmap.
   - Idea: aggregate cross-domain nearest-neighbor breaches as directed chords.
   - Why it is different: it distills the taxonomy-breaking result to four domain nodes.

8. `fig_08u_attractor_spider_map_candidate`
   - Location: Chapter 8 nearest-neighbor section.
   - Idea: each attractor field becomes an ego-network spider map.
   - Risk: memorable, but probably supplementary rather than main.

9. `fig_09u_orbital_typology_flow_candidate`
   - Location: Chapter 9 after the dynamic heatmap.
   - Idea: circular/orbital version of the static-to-dynamic flow.
   - Why it is different: less conventional than the alluvial, more poster-like.

10. `fig_09u_split_typology_constellation_candidate`
   - Location: Chapter 9 after static/dynamic diagnostics.
   - Idea: static PCA space colored by static type and marked by dynamic type.
   - Why it is different: it shows disagreement inside the same coordinate system.

## Initial Read

Best candidates to consider seriously:

1. `fig_07u_domain_change_braids_candidate`
2. `fig_09u_orbital_typology_flow_candidate`
3. `fig_07u_field_delta_rose_candidate`
4. `fig_06u_field_morphology_islands_candidate`
5. `fig_08u_domain_breach_chords_candidate`

Most experimental but risky:

- `fig_06u_field_extreme_glyph_ring_candidate`
- `fig_08u_attractor_spider_map_candidate`
"""
    (OUTPUT_DIR / "unconventional_visual_candidates_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    setup()
    stems = [
        "fig_06u_domain_morphology_crests_candidate",
        "fig_06u_field_morphology_islands_candidate",
        "fig_06u_field_extreme_glyph_ring_candidate",
        "fig_07u_field_delta_rose_candidate",
        "fig_07u_subfield_change_constellation_candidate",
        "fig_07u_domain_change_braids_candidate",
        "fig_08u_domain_breach_chords_candidate",
        "fig_08u_attractor_spider_map_candidate",
        "fig_09u_orbital_typology_flow_candidate",
        "fig_09u_split_typology_constellation_candidate",
    ]
    figure_06u_domain_morphology_crests()
    figure_06u_field_morphology_islands()
    figure_06u_field_extreme_glyph_ring()
    figure_07u_field_delta_rose()
    figure_07u_subfield_change_constellation()
    figure_07u_domain_change_braids()
    figure_08u_domain_breach_chords()
    figure_08u_attractor_spider_map()
    figure_09u_orbital_typology_flow()
    figure_09u_split_typology_constellation()
    build_gallery(stems)
    write_report()


if __name__ == "__main__":
    main()
