from __future__ import annotations

import sys
from math import isnan
from pathlib import Path
from textwrap import shorten

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from sklearn.metrics import adjusted_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.morphological_typologies import DOMAIN_COLORS, TYPOLOGY_COLORS, TYPOLOGY_ORDER
from src.temporal_common import REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS


OUTPUT_DIR = ROOT / "outputs" / "14_value_added_ch06_ch09_candidates"

DOMAIN_ORDER = ["Life Sciences", "Social Sciences", "Physical Sciences", "Health Sciences"]
TYPOLOGY_IDS = [typology_id for typology_id, _label in TYPOLOGY_ORDER]
TYPOLOGY_LABELS = dict(TYPOLOGY_ORDER)
METRICS = list(REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS)

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

DYNAMIC_ORDER = [1, 2, 3, 4]
DYNAMIC_COLORS = {1: "#4c78a8", 2: "#c47a2c", 3: "#6b8e23", 4: "#8e5ea2"}
DYNAMIC_LABELS = {
    1: "Compacting locally uneven",
    2: "Broadening hub-diffusing",
    3: "Smoothing sparse-neighborhood",
    4: "Dimensionalizing hub-concentrating",
}

BORDER_COLORS = {
    "Ambiguous": "#b5483a",
    "Near border": "#c47a2c",
    "Core": "#5f8f3a",
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
            "figure.dpi": 130,
            "savefig.dpi": 260,
            "xtick.color": "#222222",
            "ytick.color": "#222222",
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def assignments() -> pd.DataFrame:
    return pd.read_csv(ROOT / "outputs" / "09_morphological_typologies" / "subfield_typology_assignments.csv")


def pca_scores() -> pd.DataFrame:
    return pd.read_csv(ROOT / "outputs" / "09_morphological_typologies" / "typology_pca_scores.csv")


def trajectory_assignments() -> pd.DataFrame:
    return pd.read_csv(
        ROOT
        / "outputs"
        / "09_morphological_typologies"
        / "temporal_trajectory_clustering"
        / "trajectory_cluster_assignments.csv"
    )


def short_name(value: str, width: int = 38) -> str:
    return shorten(str(value), width=width, placeholder="...")


def eta_squared(frame: pd.DataFrame, group_col: str, metric: str) -> float:
    x = frame[metric].astype(float)
    total = float(((x - x.mean()) ** 2).sum())
    if total <= 0:
        return float("nan")
    grouped = frame.groupby(group_col)[metric]
    between = float(sum(len(values) * (values.mean() - x.mean()) ** 2 for _key, values in grouped))
    return between / total


def taxonomy_explanatory_power(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        rows.append(
            {
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "eta2_domain": eta_squared(frame, "domain_display_name", metric),
                "eta2_field": eta_squared(frame, "field_display_name", metric),
                "eta2_typology": eta_squared(frame, "typology_id", metric),
            }
        )
    out = pd.DataFrame(rows).sort_values("eta2_domain", ascending=True)
    out.to_csv(OUTPUT_DIR / "ch06_taxonomy_explanatory_power.csv", index=False)
    return out


def scaled_metric_columns() -> list[str]:
    return [f"scaled_{metric}" for metric in METRICS]


def distance_to_centroids(values: np.ndarray, centroids: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, vector in enumerate(values):
        row: dict[str, object] = {"row_index": idx}
        for centroid_id, centroid in centroids.iterrows():
            row[str(centroid_id)] = float(np.linalg.norm(vector - centroid.to_numpy(dtype=float)))
        rows.append(row)
    return pd.DataFrame(rows).set_index("row_index")


def domain_passport(frame: pd.DataFrame) -> pd.DataFrame:
    cols = scaled_metric_columns()
    centroids = frame.groupby("domain_display_name")[cols].mean().reindex(DOMAIN_ORDER)
    distances = distance_to_centroids(frame[cols].to_numpy(dtype=float), centroids)
    passport = frame[
        [
            "subfield_id",
            "subfield_display_name",
            "field_display_name",
            "domain_display_name",
            "typology_id",
        ]
    ].copy()
    closest = distances.idxmin(axis=1).to_numpy()
    passport["closest_morphology_domain"] = closest
    passport["own_domain_distance"] = [
        float(distances.iloc[i][domain]) for i, domain in enumerate(passport["domain_display_name"])
    ]
    nearest_other_domains: list[str] = []
    nearest_other_distances: list[float] = []
    for i, own_domain in enumerate(passport["domain_display_name"]):
        ordered = distances.iloc[i].drop(labels=[own_domain]).sort_values()
        nearest_other_domains.append(str(ordered.index[0]))
        nearest_other_distances.append(float(ordered.iloc[0]))
    passport["nearest_other_domain"] = nearest_other_domains
    passport["nearest_other_domain_distance"] = nearest_other_distances
    passport["domain_home_margin"] = passport["nearest_other_domain_distance"] - passport["own_domain_distance"]
    passport["cross_domain_passport"] = passport["closest_morphology_domain"] != passport["domain_display_name"]
    passport.to_csv(OUTPUT_DIR / "ch06_domain_passport_assignments.csv", index=False)

    counts = (
        passport.groupby(["domain_display_name", "closest_morphology_domain"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=DOMAIN_ORDER, columns=DOMAIN_ORDER, fill_value=0)
    )
    counts.to_csv(OUTPUT_DIR / "ch06_domain_passport_counts.csv")
    return passport


def typology_boundary_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cols = scaled_metric_columns()
    centroids = frame.groupby("typology_id")[cols].mean().reindex(TYPOLOGY_IDS)
    distances = distance_to_centroids(frame[cols].to_numpy(dtype=float), centroids)
    out = frame[
        [
            "subfield_id",
            "subfield_display_name",
            "field_display_name",
            "domain_display_name",
            "typology_id",
            "typology_label",
            "border_margin",
            "distance_to_typology_centroid",
        ]
    ].copy()
    nearest_rivals: list[str] = []
    nearest_rival_distances: list[float] = []
    for i, typology_id in enumerate(out["typology_id"]):
        ordered = distances.iloc[i].drop(labels=[typology_id]).sort_values()
        nearest_rivals.append(str(ordered.index[0]))
        nearest_rival_distances.append(float(ordered.iloc[0]))
    out["nearest_rival_typology"] = nearest_rivals
    out["nearest_rival_distance"] = nearest_rival_distances
    out["border_status"] = np.select(
        [
            out["border_margin"] < 0,
            out["border_margin"].between(0, 0.5, inclusive="left"),
            out["border_margin"] >= 0.5,
        ],
        ["Ambiguous", "Near border", "Core"],
        default="Core",
    )
    out.to_csv(OUTPUT_DIR / "ch09_typology_boundary_assignments.csv", index=False)
    summary = (
        out.groupby(["typology_id", "border_status"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=TYPOLOGY_IDS, columns=list(BORDER_COLORS), fill_value=0)
    )
    summary.to_csv(OUTPUT_DIR / "ch09_typology_border_summary.csv")
    flows = (
        out[out["border_status"] != "Core"]
        .groupby(["typology_id", "nearest_rival_typology"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=TYPOLOGY_IDS, columns=TYPOLOGY_IDS, fill_value=0)
    )
    flows.to_csv(OUTPUT_DIR / "ch09_typology_boundary_flows.csv")
    return out


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


def stacked_ranges(
    counts: pd.DataFrame,
    order: list,
    total: float,
    gap: float,
    *,
    axis: int,
) -> dict:
    if axis == 0:
        heights = {key: float(counts.loc[key].sum()) / total for key in order}
    else:
        heights = {key: float(counts[key].sum()) / total for key in order}
    nonzero = sum(value for value in heights.values() if value > 0)
    scale = (1.0 - gap * (len(order) - 1)) / nonzero
    y = 1.0
    ranges = {}
    for key in order:
        h = heights[key] * scale if heights[key] > 0 else 0.018
        ranges[key] = (y - h, y)
        y -= h + gap
    return ranges


def figure_06_taxonomy_explanatory_power(explained: pd.DataFrame) -> None:
    y = np.arange(len(explained))
    fig, ax = plt.subplots(figsize=(10.8, 6.8))
    ax.barh(y + 0.18, explained["eta2_domain"], height=0.32, color="#6b7280", label="OpenAlex domain")
    ax.barh(y - 0.18, explained["eta2_field"], height=0.32, color="#2563eb", alpha=0.76, label="OpenAlex field")
    for yi, row in zip(y, explained.itertuples(index=False)):
        if row.eta2_domain >= 0.15:
            ax.text(row.eta2_domain + 0.012, yi + 0.18, f"{row.eta2_domain:.2f}", va="center", fontsize=8.2)
        if row.eta2_field >= 0.45:
            ax.text(row.eta2_field + 0.012, yi - 0.18, f"{row.eta2_field:.2f}", va="center", fontsize=8.2)
    ax.set_yticks(y, explained["metric_label"])
    ax.set_xlim(0, 0.62)
    ax.set_xlabel(r"Explained variance, $\eta^2$")
    fig.suptitle("How much taxonomy explains morphology", fontsize=15.5, y=0.985)
    fig.text(
        0.5,
        0.925,
        "Domain labels explain at most 21% of any metric; field labels explain more, but still leave large internal variation.",
        ha="center",
        fontsize=9.4,
        color="#444444",
    )
    ax.grid(axis="x", color="#eeeeee", linewidth=0.6)
    ax.legend(loc="lower right", frameon=True, fontsize=8.7)
    fig.tight_layout(rect=[0, 0, 1, 0.885])
    save(fig, "fig_06x_taxonomy_explanatory_power_candidate")


def figure_06_domain_passport_flow(passport: pd.DataFrame) -> None:
    counts = (
        passport.groupby(["domain_display_name", "closest_morphology_domain"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=DOMAIN_ORDER, columns=DOMAIN_ORDER, fill_value=0)
    )
    total = float(counts.to_numpy().sum())
    cross = int(counts.to_numpy().sum() - np.trace(counts.to_numpy()))
    left_ranges = stacked_ranges(counts, DOMAIN_ORDER, total, 0.045, axis=0)
    right_ranges = stacked_ranges(counts, DOMAIN_ORDER, total, 0.045, axis=1)
    left_offsets = {key: left_ranges[key][1] for key in DOMAIN_ORDER}
    right_offsets = {key: right_ranges[key][1] for key in DOMAIN_ORDER}
    fig, ax = plt.subplots(figsize=(11.9, 7.9))
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, 1.12)
    ax.axis("off")
    for source in DOMAIN_ORDER:
        for target in DOMAIN_ORDER:
            n = int(counts.loc[source, target])
            if n == 0:
                continue
            h = max(n / total * 0.86, 0.005)
            y0 = left_offsets[source] - h / 2
            y1 = right_offsets[target] - h / 2
            alpha = 0.30 if source == target else 0.62
            draw_ribbon(ax, 0.19, y0, 0.81, y1, h, DOMAIN_COLORS[source], alpha)
            left_offsets[source] -= h
            right_offsets[target] -= h
    for domain in DOMAIN_ORDER:
        bottom, top = left_ranges[domain]
        color = DOMAIN_COLORS[domain]
        ax.add_patch(Rectangle((0.03, bottom), 0.12, top - bottom, facecolor=color, edgecolor="white", linewidth=1.0, zorder=3))
        ax.text(0.01, (bottom + top) / 2, f"{domain}\n{int(counts.loc[domain].sum())}", ha="right", va="center", fontsize=8.3, color=color, weight="bold")
    for domain in DOMAIN_ORDER:
        bottom, top = right_ranges[domain]
        color = DOMAIN_COLORS[domain]
        ax.add_patch(Rectangle((0.85, bottom), 0.12, top - bottom, facecolor=color, edgecolor="white", linewidth=1.0, zorder=3))
        ax.text(0.99, (bottom + top) / 2, f"{domain}\n{int(counts[domain].sum())}", ha="left", va="center", fontsize=8.3, color=color, weight="bold")
    ax.text(0.50, 1.095, "Morphological domain passport check", ha="center", fontsize=15.7, weight="bold")
    ax.text(
        0.50,
        1.055,
        f"Assigned OpenAlex domain -> closest domain centroid in eleven-metric morphology space; {cross}/241 subfields are closer to another domain.",
        ha="center",
        fontsize=9.0,
        color="#444444",
    )
    ax.text(0.05, -0.055, "Assigned domain", ha="center", fontsize=9.2, color="#444444")
    ax.text(0.91, -0.055, "Closest morphology-domain centroid", ha="center", fontsize=9.2, color="#444444")
    save(fig, "fig_06x_domain_passport_flow_candidate")


def figure_06_domain_passport_cases(passport: pd.DataFrame) -> None:
    cases = passport.loc[passport["cross_domain_passport"]].nsmallest(17, "domain_home_margin").copy()
    cases = cases.sort_values("domain_home_margin", ascending=True)
    y = np.arange(len(cases))
    colors = [DOMAIN_COLORS[d] for d in cases["domain_display_name"]]
    fig, ax = plt.subplots(figsize=(11.3, 7.7))
    ax.axvline(0, color="#222222", linewidth=0.85)
    ax.barh(y, cases["domain_home_margin"], color=colors, alpha=0.82, height=0.62)
    ax.set_yticks(y, [short_name(v, 42) for v in cases["subfield_display_name"]], fontsize=8.0)
    for yi, row in zip(y, cases.itertuples(index=False)):
        ax.text(
            -0.02,
            yi,
            f"{row.domain_display_name.split()[0]} -> {row.closest_morphology_domain.split()[0]}",
            ha="right",
            va="center",
            fontsize=7.5,
            color="#222222",
        )
        ax.text(row.domain_home_margin - 0.015, yi, f"{row.domain_home_margin:.2f}", ha="right", va="center", fontsize=7.4, color="white")
    ax.set_xlim(min(cases["domain_home_margin"]) - 0.15, 0.18)
    ax.set_xlabel("Home-domain margin (negative = closer to another domain centroid)")
    fig.suptitle("Strongest taxonomy-leakage cases", fontsize=15.4, y=0.985)
    fig.text(
        0.5,
        0.925,
        "These subfields are morphologically nearer to a different domain centroid than to their own assigned domain centroid.",
        ha="center",
        fontsize=9.3,
        color="#444444",
    )
    ax.grid(axis="x", color="#eeeeee", linewidth=0.55)
    ax.invert_yaxis()
    fig.tight_layout(rect=[0, 0, 1, 0.885])
    save(fig, "fig_06x_domain_passport_cases_candidate")


def figure_09_typology_borderland(boundaries: pd.DataFrame) -> None:
    summary = (
        boundaries.groupby(["typology_id", "border_status"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=TYPOLOGY_IDS, columns=list(BORDER_COLORS), fill_value=0)
    )
    totals = summary.sum(axis=1)
    y = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(11.5, 5.9))
    left = np.zeros(len(summary))
    for status, color in BORDER_COLORS.items():
        values = summary[status].to_numpy(dtype=float) / totals.to_numpy(dtype=float)
        bars = ax.barh(y, values, left=left, height=0.58, color=color, edgecolor="white", linewidth=0.7, label=status)
        for bar, share, count in zip(bars, values, summary[status].to_numpy(dtype=int)):
            if share >= 0.085:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{count}",
                    ha="center",
                    va="center",
                    fontsize=8.4,
                    color="white",
                    weight="bold",
                )
        left += values
    labels = [f"{tid} ({int(totals.loc[tid])}) {TYPOLOGY_LABELS[tid]}" for tid in TYPOLOGY_IDS]
    ax.set_yticks(y, labels, fontsize=8.3)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of subfields in typology")
    ambiguous = int(summary["Ambiguous"].sum())
    near = int(summary["Near border"].sum())
    core = int(summary["Core"].sum())
    fig.suptitle("Static typology borderlands", fontsize=15.4, y=0.985)
    fig.text(
        0.5,
        0.925,
        f"Centroid-margin diagnostic: {ambiguous} ambiguous, {near} near-border, {core} core subfields.",
        ha="center",
        fontsize=9.3,
        color="#444444",
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=True, fontsize=8.6)
    ax.grid(axis="x", color="#eeeeee", linewidth=0.55)
    fig.tight_layout(rect=[0, 0.08, 1, 0.885])
    save(fig, "fig_09x_typology_borderlands_candidate")


def figure_09_typology_boundary_flow(boundaries: pd.DataFrame) -> None:
    border = boundaries.loc[boundaries["border_status"] != "Core"].copy()
    counts = (
        border.groupby(["typology_id", "nearest_rival_typology"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=TYPOLOGY_IDS, columns=TYPOLOGY_IDS, fill_value=0)
    )
    total = float(counts.to_numpy().sum())
    left_ranges = stacked_ranges(counts, TYPOLOGY_IDS, total, 0.040, axis=0)
    right_ranges = stacked_ranges(counts, TYPOLOGY_IDS, total, 0.040, axis=1)
    left_offsets = {key: left_ranges[key][1] for key in TYPOLOGY_IDS}
    right_offsets = {key: right_ranges[key][1] for key in TYPOLOGY_IDS}
    fig, ax = plt.subplots(figsize=(12.2, 7.7))
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, 1.12)
    ax.axis("off")
    for source in TYPOLOGY_IDS:
        for target in TYPOLOGY_IDS:
            n = int(counts.loc[source, target])
            if n == 0:
                continue
            h = max(n / total * 0.86, 0.006)
            y0 = left_offsets[source] - h / 2
            y1 = right_offsets[target] - h / 2
            draw_ribbon(ax, 0.19, y0, 0.81, y1, h, TYPOLOGY_COLORS[source], 0.54)
            left_offsets[source] -= h
            right_offsets[target] -= h
    for tid in TYPOLOGY_IDS:
        bottom, top = left_ranges[tid]
        color = TYPOLOGY_COLORS[tid]
        n = int(counts.loc[tid].sum())
        ax.add_patch(Rectangle((0.03, bottom), 0.12, max(top - bottom, 0.018), facecolor=color, edgecolor="white", linewidth=1.0, zorder=3))
        ax.text(0.01, (bottom + top) / 2, f"{tid}\n{n}", ha="right", va="center", fontsize=8.4, color=color, weight="bold")
    for tid in TYPOLOGY_IDS:
        bottom, top = right_ranges[tid]
        color = TYPOLOGY_COLORS[tid]
        n = int(counts[tid].sum())
        alpha = 0.95 if n > 0 else 0.18
        ax.add_patch(Rectangle((0.85, bottom), 0.12, max(top - bottom, 0.018), facecolor=color, edgecolor="white", linewidth=1.0, alpha=alpha, zorder=3))
        ax.text(0.99, (bottom + top) / 2, f"{tid}\n{n}", ha="left", va="center", fontsize=8.4, color=color, weight="bold", alpha=1.0 if n > 0 else 0.4)
    ax.text(0.50, 1.095, "Where border cases would go next", ha="center", fontsize=15.5, weight="bold")
    ax.text(
        0.50,
        1.055,
        f"Only ambiguous and near-border subfields are shown (margin < 0.5; n={int(total)}); ribbons point to the nearest rival typology centroid.",
        ha="center",
        fontsize=9.0,
        color="#444444",
    )
    ax.text(0.05, -0.055, "Assigned static typology", ha="center", fontsize=9.2, color="#444444")
    ax.text(0.91, -0.055, "Nearest rival typology", ha="center", fontsize=9.2, color="#444444")
    save(fig, "fig_09x_typology_boundary_flow_candidate")


def figure_09_static_dynamic_transition_flow(static: pd.DataFrame, dynamic: pd.DataFrame) -> None:
    merged = static[["subfield_id", "typology_id"]].merge(
        dynamic[["subfield_id", "trajectory_cluster_id"]],
        on="subfield_id",
        how="inner",
    )
    merged["trajectory_cluster_id"] = merged["trajectory_cluster_id"].astype(int)
    counts = (
        merged.groupby(["typology_id", "trajectory_cluster_id"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=TYPOLOGY_IDS, columns=DYNAMIC_ORDER, fill_value=0)
    )
    counts.to_csv(OUTPUT_DIR / "ch09_static_dynamic_flow_counts.csv")
    ami = adjusted_mutual_info_score(merged["typology_id"], merged["trajectory_cluster_id"])
    total = float(counts.to_numpy().sum())
    left_ranges = stacked_ranges(counts, TYPOLOGY_IDS, total, 0.040, axis=0)
    right_ranges = stacked_ranges(counts, DYNAMIC_ORDER, total, 0.052, axis=1)
    left_offsets = {key: left_ranges[key][1] for key in TYPOLOGY_IDS}
    right_offsets = {key: right_ranges[key][1] for key in DYNAMIC_ORDER}
    fig, ax = plt.subplots(figsize=(12.4, 7.8))
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, 1.12)
    ax.axis("off")
    for tid in TYPOLOGY_IDS:
        for did in DYNAMIC_ORDER:
            n = int(counts.loc[tid, did])
            if n == 0:
                continue
            h = max(n / total * 0.86, 0.005)
            y0 = left_offsets[tid] - h / 2
            y1 = right_offsets[did] - h / 2
            draw_ribbon(ax, 0.19, y0, 0.81, y1, h, TYPOLOGY_COLORS[tid], 0.49)
            left_offsets[tid] -= h
            right_offsets[did] -= h
    for tid in TYPOLOGY_IDS:
        bottom, top = left_ranges[tid]
        color = TYPOLOGY_COLORS[tid]
        ax.add_patch(Rectangle((0.03, bottom), 0.12, top - bottom, facecolor=color, edgecolor="white", linewidth=1.0, zorder=3))
        ax.text(0.01, (bottom + top) / 2, f"{tid}\n{int(counts.loc[tid].sum())}", ha="right", va="center", fontsize=8.5, color=color, weight="bold")
    for did in DYNAMIC_ORDER:
        bottom, top = right_ranges[did]
        color = DYNAMIC_COLORS[did]
        ax.add_patch(Rectangle((0.85, bottom), 0.12, top - bottom, facecolor=color, edgecolor="white", linewidth=1.0, zorder=3))
        ax.text(
            0.99,
            (bottom + top) / 2,
            f"D{did}\n{int(counts[did].sum())}",
            ha="left",
            va="center",
            fontsize=8.5,
            color=color,
            weight="bold",
        )
    ax.text(0.50, 1.095, "Static shape does not lock temporal motion", ha="center", fontsize=15.5, weight="bold")
    ax.text(
        0.50,
        1.055,
        f"Static morphology typology -> dynamic trajectory typology for {int(total)} complete trajectories; AMI={ami:.3f}.",
        ha="center",
        fontsize=9.0,
        color="#444444",
    )
    ax.text(0.05, -0.055, "Static full-period profile", ha="center", fontsize=9.2, color="#444444")
    ax.text(0.91, -0.055, "Temporal movement profile", ha="center", fontsize=9.2, color="#444444")
    save(fig, "fig_09x_static_dynamic_transition_flow_candidate")


def figure_09_typology_confidence_constellation(pca: pd.DataFrame, boundaries: pd.DataFrame) -> None:
    frame = pca.merge(boundaries[["subfield_id", "border_status", "nearest_rival_typology"]], on="subfield_id", how="left")
    x_min, x_max = -3.35, 5.25
    y_min, y_max = -3.65, 3.25
    fig, ax = plt.subplots(figsize=(10.9, 8.1))
    status_order = ["Core", "Near border", "Ambiguous"]
    sizes = {"Core": 30, "Near border": 56, "Ambiguous": 92}
    alphas = {"Core": 0.48, "Near border": 0.78, "Ambiguous": 0.96}
    edges = {"Core": "none", "Near border": "#333333", "Ambiguous": "#111111"}
    for tid in TYPOLOGY_IDS:
        for status in status_order:
            sub = frame[(frame["typology_id"] == tid) & (frame["border_status"] == status)]
            if sub.empty:
                continue
            ax.scatter(
                np.clip(sub["pca_1"], x_min, x_max),
                np.clip(sub["pca_2"], y_min, y_max),
                s=sizes[status],
                color=TYPOLOGY_COLORS[tid],
                alpha=alphas[status],
                edgecolor=edges[status],
                linewidth=0.45 if status != "Core" else 0.0,
                zorder=3 if status == "Ambiguous" else 2,
            )
    centroids = frame.groupby("typology_id")[["pca_1", "pca_2"]].mean()
    for tid, row in centroids.iterrows():
        ax.scatter(row["pca_1"], row["pca_2"], marker="*", s=250, color=TYPOLOGY_COLORS[tid], edgecolor="white", linewidth=0.9, zorder=5)
        ax.text(row["pca_1"] + 0.07, row["pca_2"] + 0.07, tid, fontsize=9.5, weight="bold", color=TYPOLOGY_COLORS[tid])
    labels = frame.nsmallest(12, "border_margin")
    for row in labels.itertuples(index=False):
        x = float(np.clip(row.pca_1, x_min, x_max))
        y = float(np.clip(row.pca_2, y_min, y_max))
        label = short_name(row.subfield_display_name, 29)
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(7, 6),
            textcoords="offset points",
            fontsize=7.0,
            color="#222222",
            arrowprops={"arrowstyle": "-", "color": "#777777", "linewidth": 0.55},
        )
    if (frame["pca_1"] > x_max).any():
        ax.text(x_max - 0.04, y_min + 0.18, "right-edge clipped outlier(s)", ha="right", fontsize=7.2, color="#555555")
    ax.axhline(0, color="#dddddd", linewidth=0.75)
    ax.axvline(0, color="#dddddd", linewidth=0.75)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Static morphology PCA 1")
    ax.set_ylabel("Static morphology PCA 2")
    fig.suptitle("Typology confidence constellation", fontsize=15.5, y=0.985)
    fig.text(
        0.5,
        0.925,
        "Point size and outline encode centroid-margin confidence; labels mark the most ambiguous cases.",
        ha="center",
        fontsize=9.2,
        color="#444444",
    )
    typ_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=TYPOLOGY_COLORS[tid], markeredgecolor="white", markersize=8, label=f"{tid}")
        for tid in TYPOLOGY_IDS
    ]
    status_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#bfbfbf", markeredgecolor=edges[status] if edges[status] != "none" else "#bfbfbf", markersize=np.sqrt(sizes[status]), label=status)
        for status in status_order
    ]
    leg1 = ax.legend(handles=typ_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, fontsize=8.0, title="Static")
    ax.add_artist(leg1)
    ax.legend(handles=status_handles, loc="lower left", bbox_to_anchor=(1.01, 0.0), frameon=True, fontsize=8.0, title="Margin")
    ax.grid(color="#eeeeee", linewidth=0.5)
    fig.tight_layout(rect=[0, 0, 1, 0.885])
    save(fig, "fig_09x_typology_confidence_constellation_candidate")


def build_gallery(stems: list[str]) -> None:
    ncols = 2
    nrows = int(np.ceil(len(stems) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.0, nrows * 5.1))
    axes = np.array(axes).reshape(-1)
    for ax, stem in zip(axes, stems):
        img = mpimg.imread(OUTPUT_DIR / f"{stem}.png")
        ax.imshow(img)
        ax.set_title(stem.replace("_candidate", "").replace("_", " "), fontsize=9.6)
        ax.axis("off")
    for ax in axes[len(stems) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "value_added_candidates_gallery.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


def write_report(explained: pd.DataFrame, passport: pd.DataFrame, boundaries: pd.DataFrame) -> None:
    counts = (
        passport.groupby(["domain_display_name", "closest_morphology_domain"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=DOMAIN_ORDER, columns=DOMAIN_ORDER, fill_value=0)
    )
    cross = int(counts.to_numpy().sum() - np.trace(counts.to_numpy()))
    border_summary = (
        boundaries.groupby(["typology_id", "border_status"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=TYPOLOGY_IDS, columns=list(BORDER_COLORS), fill_value=0)
    )
    top_domain_metric = explained.sort_values("eta2_domain", ascending=False).iloc[0]
    top_field_metric = explained.sort_values("eta2_field", ascending=False).iloc[0]
    text = f"""# Value-added candidates for Chapters 6 and 9

Generated from existing TFM outputs only. No thesis `.tex` file was modified.

## Chapter 6 candidates

1. `fig_06x_domain_passport_flow_candidate`
   - Best insertion point: after `fig_06_field_metric_profile_heatmap`, before the paragraph beginning "The most visible contrast is local structure."
   - Argument added: OpenAlex domains are useful descriptors but not morphological containers.
   - Main result: {cross}/241 subfields are closest to a different domain centroid in eleven-metric morphology space.
   - My preference: strongest Chapter 6 addition. It is visual, surprising, and supports the chapter's central claim.

2. `fig_06x_taxonomy_explanatory_power_candidate`
   - Best insertion point: after `fig_06_subfield_metric_distributions_by_domain` or before the final "What this result means" paragraph.
   - Argument added: a quantitative ceiling on taxonomy as an explanation of morphology.
   - Main result: the strongest domain-level eta-squared is {top_domain_metric.metric_label} ({top_domain_metric.eta2_domain:.3f}); the strongest field-level eta-squared is {top_field_metric.metric_label} ({top_field_metric.eta2_field:.3f}).
   - My preference: very robust, but less spectacular than the passport flow.

3. `fig_06x_domain_passport_cases_candidate`
   - Best insertion point: immediately after the passport flow if we want named examples.
   - Argument added: not all taxonomy leakage is abstract; the strongest cases are interpretable subfields.
   - My preference: good as a supporting figure, maybe appendix if Chapter 6 becomes too heavy.

## Chapter 9 candidates

1. `fig_09x_static_dynamic_transition_flow_candidate`
   - Best insertion point: after `fig_09_dynamic_typology_profile_heatmap`.
   - Argument added: static morphology does not lock temporal behavior; shape and movement are different analytical dimensions.
   - My preference: strongest Chapter 9 addition because it creates a bridge between the two heatmaps.

2. `fig_09x_typology_borderlands_candidate`
   - Best insertion point: after `fig_09_typology_profile_heatmap`.
   - Argument added: the static clusters are reference profiles with measurable cores and borderlands.
   - Main result: {int(border_summary['Ambiguous'].sum())} ambiguous, {int(border_summary['Near border'].sum())} near-border, {int(border_summary['Core'].sum())} core subfields.
   - My preference: very publishable diagnostic; it makes the low silhouette concrete.

3. `fig_09x_typology_boundary_flow_candidate`
   - Best insertion point: after the borderlands figure if we want a more visual version of the same diagnostic.
   - Argument added: when static cases are uncertain, they do not all drift to one rival; uncertainty has structure.
   - My preference: visually strong, but use only if we do not overload Chapter 9.

4. `fig_09x_typology_confidence_constellation_candidate`
   - Best insertion point: after the static typology heatmap or in Appendix C.
   - Argument added: shows where the centroid-margin uncertainty lives in profile space.
   - My preference: attractive, but more exploratory; I would keep it as backup unless you love it.

## Short recommendation

For the main thesis, I would add:

- Chapter 6: `fig_06x_domain_passport_flow_candidate`
- Chapter 9: `fig_09x_static_dynamic_transition_flow_candidate`
- Optional second Chapter 9 figure: `fig_09x_typology_borderlands_candidate`

These add new evidence instead of just redrawing heatmap values.
"""
    (OUTPUT_DIR / "value_added_candidates_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    setup()
    static = assignments()
    explained = taxonomy_explanatory_power(static)
    passport = domain_passport(static)
    boundaries = typology_boundary_frame(static)
    dynamic = trajectory_assignments()
    pca = pca_scores()

    stems = [
        "fig_06x_taxonomy_explanatory_power_candidate",
        "fig_06x_domain_passport_flow_candidate",
        "fig_06x_domain_passport_cases_candidate",
        "fig_09x_typology_borderlands_candidate",
        "fig_09x_typology_boundary_flow_candidate",
        "fig_09x_static_dynamic_transition_flow_candidate",
        "fig_09x_typology_confidence_constellation_candidate",
    ]

    figure_06_taxonomy_explanatory_power(explained)
    figure_06_domain_passport_flow(passport)
    figure_06_domain_passport_cases(passport)
    figure_09_typology_borderland(boundaries)
    figure_09_typology_boundary_flow(boundaries)
    figure_09_static_dynamic_transition_flow(static, dynamic)
    figure_09_typology_confidence_constellation(pca, boundaries)
    build_gallery(stems)
    write_report(explained, passport, boundaries)


if __name__ == "__main__":
    main()
