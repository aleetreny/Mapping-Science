from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.morphological_typologies import DOMAIN_COLORS, TYPOLOGY_COLORS, TYPOLOGY_ORDER
from src.temporal_common import REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS, WINDOW_STRUCTURAL_METRICS


OUTPUT_DIR = ROOT / "outputs" / "11_heatmap_followup_candidates"

DOMAIN_ORDER = ["Life Sciences", "Social Sciences", "Physical Sciences", "Health Sciences"]
DYNAMIC_COLORS = {1: "#4c78a8", 2: "#c47a2c", 3: "#6b8e23", 4: "#8e5ea2"}
DYNAMIC_LABELS = {
    1: "D1 Compacting\nlocally uneven",
    2: "D2 Broadening\nhub-diffusing",
    3: "D3 Smoothing\nsparse-neighborhood",
    4: "D4 Dimensionalizing\nhub-concentrating",
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

METRIC_LABELS = {
    "embedding_distance_to_centroid_median": "Centroid median",
    "embedding_distance_to_centroid_iqr": "Centroid IQR",
    "embedding_distance_to_centroid_p90": "Centroid P90",
    "embedding_knn_median_distance": "kNN median",
    "embedding_knn_distance_cv": "kNN CV",
    "embedding_knn_indegree_gini": "Hub Gini",
    "embedding_pca_dim_80": "PCA D80",
    "embedding_pca_spectral_entropy": "PCA entropy",
    "embedding_centroid_drift_early_late": "Centroid drift",
    "embedding_radial_expansion_slope": "Radial slope",
    "embedding_recent_novelty_score": "Recent novelty",
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
STATIC_FAMILIES = {
    "Dispersion /\nsparsity": [
        "embedding_distance_to_centroid_median",
        "embedding_distance_to_centroid_p90",
        "embedding_knn_median_distance",
    ],
    "Local unevenness /\nhubness": [
        "embedding_distance_to_centroid_iqr",
        "embedding_knn_distance_cv",
        "embedding_knn_indegree_gini",
    ],
    "Spectral\nstructure": [
        "embedding_pca_dim_80",
        "embedding_pca_spectral_entropy",
    ],
    "Temporal\nmovement": [
        "embedding_centroid_drift_early_late",
        "embedding_radial_expansion_slope",
        "embedding_recent_novelty_score",
    ],
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


def legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=DOMAIN_COLORS[domain],
            markeredgecolor="white",
            markersize=8,
            label=domain,
        )
        for domain in DOMAIN_ORDER
    ]


def label_points(ax: plt.Axes, frame: pd.DataFrame, n: int, *, x: str, y: str, label: str) -> None:
    ranked = frame.assign(_radius=np.hypot(frame[x], frame[y])).nlargest(n, "_radius")
    for row in ranked.itertuples(index=False):
        xv = getattr(row, x)
        yv = getattr(row, y)
        text = short_field(getattr(row, label))
        dx = 0.045 if xv >= 0 else -0.045
        dy = 0.045 if yv >= 0 else -0.045
        ax.text(
            xv + dx,
            yv + dy,
            text,
            fontsize=7.6,
            ha="left" if dx > 0 else "right",
            va="bottom" if dy > 0 else "top",
            color="#222222",
        )


def build_static_compass_scores(level: str) -> pd.DataFrame:
    static = pd.read_csv(ROOT / "outputs" / "05_static_comparison" / "static_discipline_profiles.csv")
    frame = static[static["level"] == level].copy()
    scaled = zscore(frame, REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS)
    for metric in REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS:
        frame[f"z_{metric}"] = scaled[metric].to_numpy()
    frame["spread_sparse_score"] = scaled[SPREAD_METRICS].mean(axis=1)
    frame["local_concentration_score"] = scaled[CONCENTRATION_METRICS].mean(axis=1)
    return frame


def figure_06a_domain_morphology_compass() -> None:
    frame = build_static_compass_scores("domain")
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    ax.axhline(0, color="#d0d0d0", linewidth=0.9)
    ax.axvline(0, color="#d0d0d0", linewidth=0.9)

    for row in frame.itertuples(index=False):
        color = DOMAIN_COLORS[row.entity_display_name]
        ax.scatter(
            row.spread_sparse_score,
            row.local_concentration_score,
            s=190 + row.n_subfields * 4.3,
            color=color,
            edgecolor="white",
            linewidth=1.2,
            alpha=0.96,
            zorder=3,
        )
        ax.text(
            row.spread_sparse_score + 0.05,
            row.local_concentration_score + 0.05,
            row.entity_display_name,
            color=color,
            fontsize=10,
            weight="bold",
        )

    ax.text(-1.24, 1.05, "compact but\nuneven", color="#666666", fontsize=8.3)
    ax.text(0.72, 1.05, "broad and\nhub-prone", color="#666666", fontsize=8.3)
    ax.text(-1.24, -1.24, "compact and\nsmooth", color="#666666", fontsize=8.3)
    ax.text(0.74, -1.24, "broad but\nless centralized", color="#666666", fontsize=8.3)
    ax.set_title("Domain morphology compass", fontsize=15, pad=16)
    ax.set_xlabel("Global spread / sparse neighborhoods score")
    ax.set_ylabel("Local unevenness / hub concentration score")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.grid(color="#eeeeee", linewidth=0.6)
    save(fig, "fig_06a_domain_morphology_compass_candidate")


def figure_06b_field_morphology_compass() -> None:
    frame = build_static_compass_scores("field")
    field_domains = pd.read_csv(ROOT / "outputs" / "09_morphological_typologies" / "subfield_typology_assignments.csv")[
        ["field_display_name", "domain_display_name"]
    ].drop_duplicates()
    frame = frame.merge(field_domains, left_on="entity_display_name", right_on="field_display_name", how="left")

    fig, ax = plt.subplots(figsize=(10.6, 8.2))
    ax.axhline(0, color="#d3d3d3", linewidth=0.9)
    ax.axvline(0, color="#d3d3d3", linewidth=0.9)
    for domain in DOMAIN_ORDER:
        subset = frame[frame["domain_display_name"] == domain]
        ax.scatter(
            subset["spread_sparse_score"],
            subset["local_concentration_score"],
            s=85 + subset["n_subfields"] * 14,
            color=DOMAIN_COLORS[domain],
            edgecolor="white",
            linewidth=0.9,
            alpha=0.88,
            label=domain,
            zorder=3,
        )
    label_points(ax, frame, 12, x="spread_sparse_score", y="local_concentration_score", label="entity_display_name")
    ax.text(-1.58, 1.42, "locally uneven /\nhub concentrated", color="#666666", fontsize=8.2)
    ax.text(0.95, 1.42, "broad + uneven", color="#666666", fontsize=8.2)
    ax.text(-1.58, -1.45, "compact /\nregular", color="#666666", fontsize=8.2)
    ax.text(1.00, -1.45, "broad /\nlow hubness", color="#666666", fontsize=8.2)
    ax.set_title("Field morphology compass", fontsize=15.5, pad=16)
    ax.set_xlabel("Global spread / sparse neighborhoods score")
    ax.set_ylabel("Local unevenness / hub concentration score")
    ax.grid(color="#eeeeee", linewidth=0.55)
    ax.legend(handles=legend_handles(), loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, fontsize=8.5)
    save(fig, "fig_06b_field_morphology_compass_candidate")


def trajectory_scores(frame: pd.DataFrame, entity_col: str) -> pd.DataFrame:
    rows = []
    for (entity, label), group in frame.groupby([entity_col, "window_label"], sort=False):
        values = group.set_index("metric")["mean_standardized_value"]
        rows.append(
            {
                entity_col: entity,
                "window_label": label,
                "window_index": int(group["window_index"].iloc[0]),
                "spread_sparse_score": float(values.reindex(SPREAD_METRICS).mean()),
                "local_concentration_score": float(values.reindex(CONCENTRATION_METRICS).mean()),
                "n_subfields": int(group["n_subfields"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def figure_07a_domain_temporal_paths() -> None:
    raw = pd.read_parquet(ROOT / "data" / "processed" / "temporal" / "domain_window_embedding_metric_trajectories.parquet")
    scores = trajectory_scores(raw, "domain_display_name")
    fig, ax = plt.subplots(figsize=(9.5, 7.4))
    ax.axhline(0, color="#d3d3d3", linewidth=0.9)
    ax.axvline(0, color="#d3d3d3", linewidth=0.9)
    for domain in DOMAIN_ORDER:
        sub = scores[scores["domain_display_name"] == domain].sort_values("window_index")
        color = DOMAIN_COLORS[domain]
        ax.plot(
            sub["spread_sparse_score"],
            sub["local_concentration_score"],
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=5,
            zorder=3,
        )
        for idx in range(len(sub) - 1):
            a = sub.iloc[idx]
            b = sub.iloc[idx + 1]
            ax.add_patch(
                FancyArrowPatch(
                    (a.spread_sparse_score, a.local_concentration_score),
                    (b.spread_sparse_score, b.local_concentration_score),
                    arrowstyle="-|>",
                    mutation_scale=11,
                    color=color,
                    linewidth=1.2,
                    alpha=0.65,
                    shrinkA=4,
                    shrinkB=4,
                    zorder=2,
                )
            )
        first = sub.iloc[0]
        last = sub.iloc[-1]
        ax.text(first.spread_sparse_score, first.local_concentration_score - 0.08, "2000-04", color=color, fontsize=7.5, ha="center")
        ax.text(last.spread_sparse_score + 0.04, last.local_concentration_score + 0.04, f"{domain}\n2020-24", color=color, fontsize=9, weight="bold")
    ax.set_title("Domain temporal paths in morphology space", fontsize=15.5, pad=16)
    ax.set_xlabel("Global spread / sparse neighborhoods score")
    ax.set_ylabel("Local unevenness / hub concentration score")
    ax.grid(color="#eeeeee", linewidth=0.55)
    save(fig, "fig_07a_domain_temporal_paths_candidate")


def figure_07b_field_change_compass() -> None:
    raw = pd.read_parquet(ROOT / "data" / "processed" / "temporal" / "field_window_embedding_metric_trajectories.parquet")
    scores = trajectory_scores(raw, "field_display_name")
    wide = scores.pivot_table(
        index="field_display_name",
        columns="window_index",
        values=["spread_sparse_score", "local_concentration_score"],
        aggfunc="first",
    )
    deltas = pd.DataFrame(
        {
            "field_display_name": wide.index,
            "spread_delta": wide[("spread_sparse_score", 4)] - wide[("spread_sparse_score", 0)],
            "concentration_delta": wide[("local_concentration_score", 4)] - wide[("local_concentration_score", 0)],
        }
    ).reset_index(drop=True)
    field_domains = pd.read_csv(ROOT / "outputs" / "09_morphological_typologies" / "subfield_typology_assignments.csv")[
        ["field_display_name", "domain_display_name"]
    ].drop_duplicates()
    deltas = deltas.merge(field_domains, on="field_display_name", how="left")
    deltas["radius"] = np.hypot(deltas["spread_delta"], deltas["concentration_delta"])

    fig, ax = plt.subplots(figsize=(10.6, 8.2))
    ax.axhline(0, color="#d3d3d3", linewidth=0.9)
    ax.axvline(0, color="#d3d3d3", linewidth=0.9)
    for row in deltas.itertuples(index=False):
        color = DOMAIN_COLORS[row.domain_display_name]
        ax.add_patch(
            FancyArrowPatch(
                (0, 0),
                (row.spread_delta, row.concentration_delta),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.05,
                color=color,
                alpha=0.38,
                zorder=1,
            )
        )
        ax.scatter(row.spread_delta, row.concentration_delta, s=70, color=color, edgecolor="white", linewidth=0.7, zorder=3)
    for row in deltas.nlargest(12, "radius").itertuples(index=False):
        dx = 0.035 if row.spread_delta >= 0 else -0.035
        dy = 0.035 if row.concentration_delta >= 0 else -0.035
        ax.text(
            row.spread_delta + dx,
            row.concentration_delta + dy,
            short_field(row.field_display_name),
            fontsize=7.7,
            ha="left" if dx > 0 else "right",
            va="bottom" if dy > 0 else "top",
        )
    ax.text(-1.56, 1.17, "compacting but\nmore uneven", fontsize=8.3, color="#666666")
    ax.text(0.92, 1.17, "broadening and\nmore uneven", fontsize=8.3, color="#666666")
    ax.text(-1.56, -1.28, "compacting and\nsmoothing", fontsize=8.3, color="#666666")
    ax.text(0.92, -1.28, "broadening but\nsmoothing", fontsize=8.3, color="#666666")
    ax.set_title("Field temporal change compass", fontsize=15.5, pad=16)
    ax.set_xlabel("Final-minus-initial spread / sparsity score")
    ax.set_ylabel("Final-minus-initial local unevenness / hubness score")
    ax.grid(color="#eeeeee", linewidth=0.55)
    ax.legend(handles=legend_handles(), loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, fontsize=8.5)
    save(fig, "fig_07b_field_change_compass_candidate")


def figure_08a_similarity_bridge_load() -> None:
    nearest = pd.read_csv(ROOT / "outputs" / "10_global_map_candidates" / "field_nearest_morphological_neighbors.csv")
    field_domains = pd.read_csv(ROOT / "outputs" / "09_morphological_typologies" / "subfield_typology_assignments.csv")[
        ["field_display_name", "domain_display_name"]
    ].drop_duplicates()
    static = build_static_compass_scores("field")[["entity_display_name", "n_subfields"]]
    inbound = nearest["nearest_field"].value_counts().rename_axis("field").reset_index(name="incoming_links")
    frame = nearest.rename(columns={"field": "field_display_name", "distance": "nearest_distance"}).merge(
        inbound, left_on="field_display_name", right_on="field", how="left"
    )
    frame["incoming_links"] = frame["incoming_links"].fillna(0).astype(int)
    frame = frame.merge(field_domains, on="field_display_name", how="left")
    frame = frame.merge(static, left_on="field_display_name", right_on="entity_display_name", how="left")
    frame["isolate_score"] = frame["nearest_distance"] * (1 + 0.25 * (frame["incoming_links"] == 0))

    fig, ax = plt.subplots(figsize=(10.4, 7.6))
    for domain in DOMAIN_ORDER:
        subset = frame[frame["domain_display_name"] == domain]
        ax.scatter(
            subset["nearest_distance"],
            subset["incoming_links"],
            s=90 + subset["n_subfields"] * 13,
            color=DOMAIN_COLORS[domain],
            edgecolor="white",
            linewidth=0.9,
            alpha=0.88,
            label=domain,
            zorder=3,
        )
    to_label = pd.concat(
        [
            frame.nlargest(6, "incoming_links"),
            frame.nlargest(5, "nearest_distance"),
            frame[frame["field_display_name"].isin(["Computer Science", "Materials Science", "Neuroscience", "Medicine"])],
        ]
    ).drop_duplicates("field_display_name")
    for row in to_label.itertuples(index=False):
        ax.text(
            row.nearest_distance + 0.035,
            row.incoming_links + 0.05,
            short_field(row.field_display_name),
            fontsize=7.8,
            ha="left",
            va="bottom",
        )
    ax.set_title("Nearest-neighbor attractors and isolates", fontsize=15.5, pad=16)
    ax.set_xlabel("Distance to own nearest morphology neighbor")
    ax.set_ylabel("Incoming nearest-neighbor links from other fields")
    ax.grid(color="#eeeeee", linewidth=0.55)
    ax.set_ylim(-0.35, frame["incoming_links"].max() + 0.9)
    ax.legend(handles=legend_handles(), loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, fontsize=8.5)
    save(fig, "fig_08a_similarity_bridge_load_candidate")


def figure_09a_static_typology_signature_cards() -> None:
    profile = pd.read_csv(ROOT / "outputs" / "09_morphological_typologies" / "typology_profile_summary.csv")
    rows = []
    for _, row in profile.iterrows():
        item = {
            "typology_id": row["typology_id"],
            "typology_label": row["typology_label"],
            "n_subfields": int(row["n_subfields"]),
        }
        for family, metrics in STATIC_FAMILIES.items():
            item[family] = float(row[metrics].mean())
        rows.append(item)
    frame = pd.DataFrame(rows)

    fig, axes = plt.subplots(len(frame), 1, figsize=(10.8, 8.8), sharex=True)
    families = list(STATIC_FAMILIES)
    y = np.arange(len(families))
    for ax, (_, row) in zip(axes, frame.iterrows()):
        color = TYPOLOGY_COLORS[row["typology_id"]]
        vals = np.array([row[family] for family in families], dtype=float)
        ax.axvline(0, color="#d0d0d0", linewidth=0.9)
        ax.hlines(y, 0, vals, color=color, linewidth=3.0, alpha=0.78)
        ax.scatter(vals, y, s=85, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_yticks(y, families, fontsize=8.4)
        ax.invert_yaxis()
        ax.set_title(
            f"{row['typology_id']}: {row['typology_label']} (n={int(row['n_subfields'])})",
            loc="left",
            fontsize=10.2,
            color=color,
            pad=2,
        )
        ax.grid(axis="x", color="#eeeeee", linewidth=0.55)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[-1].set_xlabel("Mean standardized family score")
    fig.suptitle("Static typology signatures by metric family", fontsize=15.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, "fig_09a_static_typology_signature_cards_candidate")


def figure_09b_dynamic_typology_motion_compass() -> None:
    profile = pd.read_csv(
        ROOT
        / "outputs"
        / "09_morphological_typologies"
        / "temporal_trajectory_clustering"
        / "trajectory_cluster_profile_summary.csv"
    )
    rows = []
    for _, row in profile.iterrows():
        cid = int(row["trajectory_cluster_id"])
        spread = np.mean([row[f"slope_{metric}"] for metric in SPREAD_METRICS])
        concentration = np.mean([row[f"slope_{metric}"] for metric in CONCENTRATION_METRICS])
        rows.append(
            {
                "trajectory_cluster_id": cid,
                "n_subfields": int(row["n_subfields"]),
                "dominant_domain": row["dominant_domain"],
                "spread_slope": float(spread),
                "concentration_slope": float(concentration),
            }
        )
    frame = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    ax.axhline(0, color="#d3d3d3", linewidth=0.9)
    ax.axvline(0, color="#d3d3d3", linewidth=0.9)
    for row in frame.itertuples(index=False):
        color = DYNAMIC_COLORS[row.trajectory_cluster_id]
        ax.add_patch(
            FancyArrowPatch(
                (0, 0),
                (row.spread_slope, row.concentration_slope),
                arrowstyle="-|>",
                mutation_scale=18,
                linewidth=3.0,
                color=color,
                alpha=0.78,
                zorder=2,
            )
        )
        ax.scatter(row.spread_slope, row.concentration_slope, s=180 + row.n_subfields, color=color, edgecolor="white", linewidth=1.0, zorder=3)
        offsets = {
            1: (-0.06, 0.06, "right", "bottom"),
            2: (0.07, -0.06, "left", "top"),
            3: (0.06, -0.02, "left", "top"),
            4: (0.06, 0.05, "left", "bottom"),
        }
        dx, dy, ha, va = offsets[row.trajectory_cluster_id]
        ax.text(
            row.spread_slope + dx,
            row.concentration_slope + dy,
            f"{DYNAMIC_LABELS[row.trajectory_cluster_id]}\nn={row.n_subfields}",
            color=color,
            fontsize=9.0,
            weight="bold",
            ha=ha,
            va=va,
        )
    ax.text(0.02, 0.98, "compacting but\nmore uneven", transform=ax.transAxes, fontsize=8.4, color="#666666", va="top")
    ax.text(0.77, 0.98, "broadening and\nmore uneven", transform=ax.transAxes, fontsize=8.4, color="#666666", va="top")
    ax.text(0.02, 0.04, "compacting and\nsmoothing", transform=ax.transAxes, fontsize=8.4, color="#666666", va="bottom")
    ax.text(0.77, 0.04, "broadening but\nsmoothing", transform=ax.transAxes, fontsize=8.4, color="#666666", va="bottom")
    ax.set_title("Dynamic typology motion compass", fontsize=15.5, pad=16)
    ax.set_xlabel("Mean slope of spread / sparsity metrics")
    ax.set_ylabel("Mean slope of unevenness / hubness metrics")
    ax.set_xlim(-0.58, 0.74)
    ax.set_ylim(-0.82, 0.72)
    ax.grid(color="#eeeeee", linewidth=0.55)
    save(fig, "fig_09b_dynamic_typology_motion_compass_candidate")


def draw_ribbon(
    ax: plt.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width: float,
    color: str,
    alpha: float,
) -> None:
    ctrl = 0.38
    top0 = y0 + width / 2
    bot0 = y0 - width / 2
    top1 = y1 + width / 2
    bot1 = y1 - width / 2
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


def stacked_positions(counts: pd.DataFrame, order: list, total: float, gap: float) -> tuple[dict, dict]:
    heights = {key: counts.loc[key].sum() / total for key in order}
    scale = (1 - gap * (len(order) - 1)) / sum(heights.values())
    heights = {key: heights[key] * scale for key in order}
    y_top = 1.0
    ranges = {}
    centers = {}
    for key in order:
        h = heights[key]
        ranges[key] = (y_top - h, y_top)
        centers[key] = y_top - h / 2
        y_top -= h + gap
    return ranges, centers


def figure_09c_static_dynamic_alluvial() -> None:
    assignments = pd.read_csv(
        ROOT
        / "outputs"
        / "09_morphological_typologies"
        / "temporal_trajectory_clustering"
        / "trajectory_cluster_assignments.csv"
    )
    static_order = [tid for tid, _ in TYPOLOGY_ORDER]
    dynamic_order = [1, 2, 3, 4]
    counts = (
        assignments.groupby(["typology_id", "trajectory_cluster_id"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=static_order, columns=dynamic_order, fill_value=0)
    )
    total = float(counts.to_numpy().sum())

    left_ranges, _ = stacked_positions(counts, static_order, total, 0.035)
    right_counts = counts.T
    right_ranges, _ = stacked_positions(right_counts, dynamic_order, total, 0.045)

    left_offsets = {key: left_ranges[key][1] for key in static_order}
    right_offsets = {key: right_ranges[key][1] for key in dynamic_order}

    fig, ax = plt.subplots(figsize=(12.4, 7.8))
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.04, 1.12)
    ax.axis("off")

    min_width = 0.003
    for sid in static_order:
        for did in dynamic_order:
            n = int(counts.loc[sid, did])
            if n == 0:
                continue
            h = max(n / total * 0.86, min_width)
            y0 = left_offsets[sid] - h / 2
            y1 = right_offsets[did] - h / 2
            draw_ribbon(ax, 0.18, y0, 0.82, y1, h, TYPOLOGY_COLORS[sid], 0.46)
            left_offsets[sid] -= h
            right_offsets[did] -= h

    for sid in static_order:
        bottom, top = left_ranges[sid]
        color = TYPOLOGY_COLORS[sid]
        ax.add_patch(Rectangle((0.03, bottom), 0.12, top - bottom, facecolor=color, edgecolor="white", linewidth=1.0, alpha=0.95, zorder=3))
        label = dict(TYPOLOGY_ORDER)[sid]
        ax.text(0.01, (bottom + top) / 2, f"{sid}\n{label}\n{int(counts.loc[sid].sum())}", ha="right", va="center", fontsize=8.0, color=color, weight="bold")
    for did in dynamic_order:
        bottom, top = right_ranges[did]
        color = DYNAMIC_COLORS[did]
        ax.add_patch(Rectangle((0.85, bottom), 0.12, top - bottom, facecolor=color, edgecolor="white", linewidth=1.0, alpha=0.95, zorder=3))
        dynamic_label = DYNAMIC_LABELS[did].replace(f"D{did} ", "").replace(chr(10), " ")
        ax.text(0.99, (bottom + top) / 2, f"D{did}\n{dynamic_label}\n{int(counts[did].sum())}", ha="left", va="center", fontsize=8.0, color=color, weight="bold")
    ax.text(0.50, 1.095, "Static-to-dynamic typology flow", ha="center", fontsize=15.5, weight="bold")
    ax.text(
        0.50,
        1.055,
        "Each ribbon is a group of subfields sharing one full-period shape type and one temporal trajectory type",
        ha="center",
        fontsize=9.0,
        color="#444444",
    )
    ax.text(0.09, 1.015, "Static morphology", ha="center", fontsize=11.5, weight="bold")
    ax.text(0.91, 1.015, "Dynamic trajectories", ha="center", fontsize=11.5, weight="bold")
    save(fig, "fig_09c_static_dynamic_alluvial_candidate")


def build_gallery(stems: list[str]) -> None:
    ncols = 2
    nrows = int(np.ceil(len(stems) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.0, nrows * 5.1))
    axes = np.array(axes).reshape(-1)
    for ax, stem in zip(axes, stems):
        img = mpimg.imread(OUTPUT_DIR / f"{stem}.png")
        ax.imshow(img)
        ax.set_title(stem.replace("_candidate", "").replace("_", " "), fontsize=10)
        ax.axis("off")
    for ax in axes[len(stems) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "candidate_gallery.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


def write_report() -> None:
    report = """# Heatmap Follow-up Visualization Candidates

Generated from existing TFM outputs only. These are exploratory sketches; no thesis chapter was modified.

## Candidate Locations

1. `fig_06a_domain_morphology_compass_candidate`
   - Thesis location: after Figure 6 domain-level morphological profiles.
   - Replaces no figure; complements the domain heatmap.
   - Purpose: converts the 11-metric domain heatmap into two readable axes: global spread/sparse neighborhoods and local unevenness/hub concentration.
   - Risk: only four points, so it is interpretive and should not be oversold.

2. `fig_06b_field_morphology_compass_candidate`
   - Thesis location: after Figure 6 field-level morphological profiles.
   - Purpose: highlights why field variation breaks domain-level simplicity.
   - Strongest use: this is the most direct post-heatmap map for Chapter 6.

3. `fig_07a_domain_temporal_paths_candidate`
   - Thesis location: after Figure 7 domain temporal trajectories.
   - Purpose: shows the temporal path of each domain through a compact morphology space.
   - Strongest use: reader sees direction of movement, not separate line panels.

4. `fig_07b_field_change_compass_candidate`
   - Thesis location: after Figure 7 field-level temporal change heatmap.
   - Purpose: turns final-minus-initial field deltas into a directional map: compacting/broadening vs smoothing/unevening.
   - Strongest use: makes the non-universal temporal story visually memorable.

5. `fig_08a_similarity_bridge_load_candidate`
   - Thesis location: optional, after the static distance heatmap or after the nearest-neighbor ring already inserted.
   - Purpose: separates attractor fields from isolates by incoming nearest-neighbor links and outgoing nearest-neighbor distance.
   - Risk: less visually distinctive than the ring, but analytically useful.

6. `fig_09a_static_typology_signature_cards_candidate`
   - Thesis location: after Figure 9 static typology profile heatmap.
   - Purpose: compresses the 11-metric typology heatmap into four profile families.
   - Strongest use: helps readers remember what T1-T5 mean without rereading the heatmap.

7. `fig_09b_dynamic_typology_motion_compass_candidate`
   - Thesis location: after Figure 9 dynamic temporal typology profile heatmap.
   - Purpose: gives each dynamic typology an arrow in change-space.
   - Strongest use: makes D1-D4 feel like motion types rather than rows of numbers.

8. `fig_09c_static_dynamic_alluvial_candidate`
   - Thesis location: after the dynamic typology heatmap or before the static/dynamic projection comparison.
   - Purpose: shows that static shape groups and dynamic trajectory groups are not the same partition.
   - Strongest use: probably the best Chapter 9 candidate because it visualizes the chapter's central contrast.

## My Initial Ranking

1. Chapter 9 alluvial (`fig_09c`) - strongest narrative payoff.
2. Chapter 7 field change compass (`fig_07b`) - most useful replacement for heatmap fatigue.
3. Chapter 6 field morphology compass (`fig_06b`) - cleanest static morphology map.
4. Chapter 9 dynamic motion compass (`fig_09b`) - good companion to the dynamic heatmap.
5. Chapter 7 domain temporal paths (`fig_07a`) - elegant but broader.
6. Chapter 6 domain compass (`fig_06a`) - useful, but only four points.
7. Chapter 8 bridge load (`fig_08a`) - useful if we want a small analytical sidebar, less necessary after the ring map.
8. Chapter 9 signature cards (`fig_09a`) - explanatory, less stunning.
"""
    (OUTPUT_DIR / "heatmap_followup_candidates_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    setup()
    stems = [
        "fig_06a_domain_morphology_compass_candidate",
        "fig_06b_field_morphology_compass_candidate",
        "fig_07a_domain_temporal_paths_candidate",
        "fig_07b_field_change_compass_candidate",
        "fig_08a_similarity_bridge_load_candidate",
        "fig_09a_static_typology_signature_cards_candidate",
        "fig_09b_dynamic_typology_motion_compass_candidate",
        "fig_09c_static_dynamic_alluvial_candidate",
    ]
    figure_06a_domain_morphology_compass()
    figure_06b_field_morphology_compass()
    figure_07a_domain_temporal_paths()
    figure_07b_field_change_compass()
    figure_08a_similarity_bridge_load()
    figure_09a_static_typology_signature_cards()
    figure_09b_dynamic_typology_motion_compass()
    figure_09c_static_dynamic_alluvial()
    build_gallery(stems)
    write_report()


if __name__ == "__main__":
    main()
