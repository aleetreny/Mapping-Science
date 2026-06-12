from __future__ import annotations

import json
import sys
from pathlib import Path
from textwrap import shorten

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from sklearn.metrics import adjusted_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.morphological_typologies import DOMAIN_COLORS, TYPOLOGY_COLORS, TYPOLOGY_ORDER


OUTPUT_DIR = ROOT / "outputs" / "15_glanceable_visual_candidates"

DOMAIN_ORDER = ["Life Sciences", "Social Sciences", "Physical Sciences", "Health Sciences"]
TYPOLOGY_IDS = [typology_id for typology_id, _label in TYPOLOGY_ORDER]
TYPOLOGY_LABELS = dict(TYPOLOGY_ORDER)
DYNAMIC_ORDER = [1, 2, 3, 4]
DYNAMIC_COLORS = {1: "#4c78a8", 2: "#c47a2c", 3: "#6b8e23", 4: "#8e5ea2"}
DYNAMIC_LABELS = {
    1: "Compacting locally uneven",
    2: "Broadening hub-diffusing",
    3: "Smoothing sparse-neighborhood",
    4: "Dimensionalizing hub-concentrating",
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
TEMPORAL_NOVELTY_METRICS = [
    "embedding_centroid_drift_early_late",
    "embedding_radial_expansion_slope",
    "embedding_recent_novelty_score",
]

SHORT_FIELD = {
    "Artificial Intelligence": "Artificial Intelligence",
    "Applied Microbiology and Biotechnology": "Applied Microbiology",
    "Human Factors and Ergonomics": "Human Factors",
    "Research and Theory": "Research & Theory",
    "Electrical and Electronic Engineering": "Electrical Engineering",
    "Computer Vision and Pattern Recognition": "Computer Vision",
    "General Materials Science": "General Materials",
    "Molecular Biology": "Molecular Biology",
    "Biomedical Engineering": "Biomedical Eng.",
    "General Energy": "General Energy",
    "Medical Laboratory Technology": "Med. Lab Technology",
    "Philosophy": "Philosophy",
    "Molecular Medicine": "Molecular Medicine",
    "Pharmaceutical Science": "Pharmaceutical Sci.",
    "General Dentistry": "General Dentistry",
    "Computational Mathematics": "Computational Math.",
}


def setup() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.dpi": 130,
            "savefig.dpi": 260,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "xtick.color": "#222222",
            "ytick.color": "#222222",
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def short(value: str, width: int = 28) -> str:
    if value in SHORT_FIELD:
        return SHORT_FIELD[value]
    return shorten(str(value), width=width, placeholder="...")


def load_static() -> pd.DataFrame:
    frame = pd.read_csv(ROOT / "outputs" / "09_morphological_typologies" / "subfield_typology_assignments.csv")
    frame["semantic_breadth"] = frame[[f"scaled_{metric}" for metric in SPREAD_METRICS]].mean(axis=1)
    frame["local_concentration"] = frame[[f"scaled_{metric}" for metric in CONCENTRATION_METRICS]].mean(axis=1)
    frame["temporal_novelty"] = frame[[f"scaled_{metric}" for metric in TEMPORAL_NOVELTY_METRICS]].mean(axis=1)
    frame["compass_extremity"] = np.hypot(frame["semantic_breadth"], frame["local_concentration"])
    frame["quadrant"] = np.select(
        [
            (frame["semantic_breadth"] >= 0) & (frame["local_concentration"] >= 0),
            (frame["semantic_breadth"] >= 0) & (frame["local_concentration"] < 0),
            (frame["semantic_breadth"] < 0) & (frame["local_concentration"] >= 0),
            (frame["semantic_breadth"] < 0) & (frame["local_concentration"] < 0),
        ],
        ["Broad + concentrated", "Broad + smooth", "Compact + concentrated", "Compact + smooth"],
        default="",
    )
    frame.to_csv(OUTPUT_DIR / "ch06_compass_scores.csv", index=False)
    return frame


def load_dynamic() -> pd.DataFrame:
    return pd.read_csv(
        ROOT
        / "outputs"
        / "09_morphological_typologies"
        / "temporal_trajectory_clustering"
        / "trajectory_cluster_assignments.csv"
    )


def load_dynamic_summary() -> pd.DataFrame:
    return pd.read_csv(
        ROOT
        / "outputs"
        / "09_morphological_typologies"
        / "temporal_trajectory_clustering"
        / "trajectory_cluster_profile_summary.csv"
    )


def domain_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=DOMAIN_COLORS[d], markeredgecolor="white", markersize=8, label=d)
        for d in DOMAIN_ORDER
    ]


def figure_06_morphology_compass(frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 8.5))
    xlim = (-1.65, 1.20)
    ylim = (-1.55, 2.55)
    ax.add_patch(Rectangle((0, 0), xlim[1], ylim[1], facecolor="#f3e5d4", edgecolor="none", alpha=0.82, zorder=0))
    ax.add_patch(Rectangle((0, ylim[0]), xlim[1], -ylim[0], facecolor="#e2eef7", edgecolor="none", alpha=0.82, zorder=0))
    ax.add_patch(Rectangle((xlim[0], 0), -xlim[0], ylim[1], facecolor="#eee3f2", edgecolor="none", alpha=0.78, zorder=0))
    ax.add_patch(Rectangle((xlim[0], ylim[0]), -xlim[0], -ylim[0], facecolor="#e6f0dc", edgecolor="none", alpha=0.78, zorder=0))
    for domain in DOMAIN_ORDER:
        sub = frame[frame["domain_display_name"] == domain]
        novelty = np.clip(sub["temporal_novelty"], -0.3, 2.2)
        sizes = 38 + 34 * np.maximum(novelty, 0)
        ax.scatter(
            sub["semantic_breadth"],
            sub["local_concentration"],
            s=sizes,
            color=DOMAIN_COLORS[domain],
            edgecolor="white",
            linewidth=0.65,
            alpha=0.82,
            label=domain,
            zorder=3,
        )
    label_names = [
        "Applied Microbiology and Biotechnology",
        "Artificial Intelligence",
        "Human Factors and Ergonomics",
        "Philosophy",
        "Research and Theory",
        "Biomedical Engineering",
        "Molecular Medicine",
        "Electrical and Electronic Engineering",
        "General Energy",
        "Computer Vision and Pattern Recognition",
    ]
    offsets = {
        "Applied Microbiology and Biotechnology": (8, 7),
        "Artificial Intelligence": (9, -14),
        "Human Factors and Ergonomics": (8, 9),
        "Philosophy": (8, 7),
        "Research and Theory": (8, -12),
        "Biomedical Engineering": (-78, 9),
        "Molecular Medicine": (8, 7),
        "Electrical and Electronic Engineering": (8, -14),
        "General Energy": (8, 7),
        "Computer Vision and Pattern Recognition": (8, 7),
    }
    for row in frame[frame["subfield_display_name"].isin(label_names)].itertuples(index=False):
        ax.annotate(
            short(row.subfield_display_name),
            xy=(row.semantic_breadth, row.local_concentration),
            xytext=offsets.get(row.subfield_display_name, (7, 6)),
            textcoords="offset points",
            fontsize=7.7,
            color="#202020",
            arrowprops={"arrowstyle": "-", "color": "#555555", "linewidth": 0.55},
            zorder=5,
        )
    ax.axhline(0, color="#222222", linewidth=1.0)
    ax.axvline(0, color="#222222", linewidth=1.0)
    ax.text(0.62, 2.25, "Broad +\nconcentrated", ha="center", va="center", fontsize=15, weight="bold", color="#8a4f22")
    ax.text(0.62, -1.25, "Broad +\nsmooth", ha="center", va="center", fontsize=15, weight="bold", color="#2f5f8f")
    ax.text(-1.08, 2.25, "Compact +\nconcentrated", ha="center", va="center", fontsize=15, weight="bold", color="#6d4a80")
    ax.text(-1.08, -1.25, "Compact +\nsmooth", ha="center", va="center", fontsize=15, weight="bold", color="#4f722e")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Semantic breadth  <- compact      broad ->", fontsize=10.5)
    ax.set_ylabel("Local concentration  <- smooth      hub-dominated ->", fontsize=10.5)
    ax.set_title("A morphology compass for subfields", fontsize=17, weight="bold", pad=18)
    ax.text(
        0.5,
        1.02,
        "Two readable axes turn the eleven static metrics into four structural zones.",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        color="#444444",
    )
    ax.grid(color="white", linewidth=0.7, alpha=0.7)
    ax.legend(handles=domain_handles(), loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig_06y_morphology_compass_candidate")


def quadrant_examples(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        quadrant: (
            frame[frame["quadrant"] == quadrant]
            .sort_values("compass_extremity", ascending=False)
            .head(5)
            .copy()
        )
        for quadrant in ["Broad + concentrated", "Broad + smooth", "Compact + concentrated", "Compact + smooth"]
    }


def figure_06_quadrant_cards(frame: pd.DataFrame) -> None:
    examples = quadrant_examples(frame)
    counts = frame["quadrant"].value_counts()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.3))
    axes = axes.reshape(-1)
    card_defs = [
        ("Broad + concentrated", "#f3e5d4", "#8a4f22", "wide territory, strong hubs"),
        ("Broad + smooth", "#e2eef7", "#2f5f8f", "wide territory, weak hubs"),
        ("Compact + concentrated", "#eee3f2", "#6d4a80", "tight territory, strong hubs"),
        ("Compact + smooth", "#e6f0dc", "#4f722e", "tight territory, weak hubs"),
    ]
    for ax, (quadrant, fill, accent, subtitle) in zip(axes, card_defs):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(
            FancyBboxPatch(
                (0.02, 0.02),
                0.96,
                0.96,
                boxstyle="round,pad=0.018,rounding_size=0.03",
                facecolor=fill,
                edgecolor=accent,
                linewidth=1.4,
                alpha=0.96,
            )
        )
        ax.text(0.06, 0.91, quadrant, fontsize=17, weight="bold", color=accent, va="top")
        ax.text(0.06, 0.825, subtitle, fontsize=11.0, color="#333333", va="top")
        ax.text(0.91, 0.89, f"n={int(counts[quadrant])}", fontsize=15, weight="bold", color=accent, ha="right", va="top")
        rows = examples[quadrant].reset_index(drop=True)
        for i, row in rows.iterrows():
            y = 0.70 - i * 0.125
            color = DOMAIN_COLORS[row["domain_display_name"]]
            ax.add_patch(Rectangle((0.06, y - 0.031), 0.028, 0.062, facecolor=color, edgecolor="none"))
            ax.text(0.105, y + 0.017, short(row["subfield_display_name"], 40), fontsize=10.8, color="#111111", va="center", weight="bold")
            ax.text(0.105, y - 0.033, row["domain_display_name"], fontsize=9.0, color=color, va="center")
    fig.suptitle("The four easiest-to-read static morphology zones", fontsize=17, weight="bold", y=0.99)
    fig.text(
        0.5,
        0.94,
        "Each card shows the most extreme named subfields inside one simple structural quadrant.",
        ha="center",
        fontsize=10.8,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    save(fig, "fig_06y_quadrant_extreme_cards_candidate")


def figure_06_rebel_spotlight(frame: pd.DataFrame) -> None:
    spotlight = pd.DataFrame(
        [
            {
                "title": "AI is broad,\nbut not hubbed",
                "name": "Artificial Intelligence",
                "score": "high breadth\nlow hubness",
                "x": 0,
                "color": "#2f5f8f",
            },
            {
                "title": "Applied Microbiology\nis compact and hubbed",
                "name": "Applied Microbiology and Biotechnology",
                "score": "low breadth\nhigh hubness",
                "x": 1,
                "color": "#6d4a80",
            },
            {
                "title": "Philosophy is broad\nand hubbed",
                "name": "Philosophy",
                "score": "high breadth\nhigh hubness",
                "x": 2,
                "color": "#8a4f22",
            },
            {
                "title": "Human Factors is\nthe novelty outlier",
                "name": "Human Factors and Ergonomics",
                "score": "largest temporal\nnovelty score",
                "x": 3,
                "color": "#b5483a",
            },
        ]
    )
    data = frame.set_index("subfield_display_name")
    fig, ax = plt.subplots(figsize=(13.5, 6.7))
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for item in spotlight.itertuples(index=False):
        row = data.loc[item.name]
        ax.add_patch(
            FancyBboxPatch(
                (item.x - 0.43, 0.08),
                0.86,
                0.78,
                boxstyle="round,pad=0.02,rounding_size=0.035",
                facecolor="#f7f7f4",
                edgecolor=item.color,
                linewidth=1.5,
            )
        )
        ax.text(item.x, 0.79, item.title, ha="center", va="top", fontsize=10.2, weight="bold", color=item.color)
        ax.text(item.x, 0.61, short(item.name, 29), ha="center", va="center", fontsize=12.5, weight="bold", color="#111111")
        ax.text(item.x, 0.48, row["domain_display_name"], ha="center", va="center", fontsize=8.5, color=DOMAIN_COLORS[row["domain_display_name"]])
        metrics = [
            ("Breadth", row["semantic_breadth"]),
            ("Hubness", row["local_concentration"]),
            ("Novelty", row["temporal_novelty"]),
        ]
        for j, (label, value) in enumerate(metrics):
            y = 0.34 - j * 0.085
            ax.text(item.x - 0.31, y, label, ha="left", va="center", fontsize=8.2, color="#444444")
            ax.add_patch(Rectangle((item.x - 0.04, y - 0.018), 0.25, 0.036, facecolor="#e5e5e5", edgecolor="none"))
            width = np.clip(abs(value) / 2.5, 0.03, 0.25)
            color = item.color if value >= 0 else "#777777"
            if value >= 0:
                ax.add_patch(Rectangle((item.x - 0.04, y - 0.018), width, 0.036, facecolor=color, edgecolor="none", alpha=0.86))
            else:
                ax.add_patch(Rectangle((item.x - 0.04 + 0.25 - width, y - 0.018), width, 0.036, facecolor=color, edgecolor="none", alpha=0.86))
            ax.text(item.x + 0.25, y, f"{value:+.1f}", ha="right", va="center", fontsize=8.0, color="#333333")
        ax.text(item.x, 0.105, item.score, ha="center", va="bottom", fontsize=9.0, color="#333333")
    fig.suptitle("Four subfields that break the simple domain story", fontsize=17, weight="bold", y=0.98)
    fig.text(0.5, 0.90, "These are not just outliers; each one breaks a different intuitive expectation.", ha="center", fontsize=10, color="#444444")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    save(fig, "fig_06y_rebel_spotlight_candidate")


def diagnostic_values(static: pd.DataFrame, dynamic: pd.DataFrame) -> dict[str, float]:
    static_summary = json.loads((ROOT / "outputs" / "09_morphological_typologies" / "summary.json").read_text(encoding="utf-8"))
    dynamic_summary = json.loads(
        (
            ROOT
            / "outputs"
            / "09_morphological_typologies"
            / "temporal_trajectory_clustering"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )
    merged = static[["subfield_id", "typology_id", "domain_display_name"]].merge(
        dynamic[["subfield_id", "trajectory_cluster_id"]],
        on="subfield_id",
        how="inner",
    )
    return {
        "static_silhouette": float(static_summary["selected_solution"]["silhouette"]),
        "dynamic_silhouette": float(dynamic_summary["selected_solution"]["silhouette"]),
        "static_ari": float(static_summary["stability_mean_subsample_ari"]),
        "dynamic_ari": float(dynamic_summary["stability_mean_ari"]),
        "static_domain_ami": float(adjusted_mutual_info_score(static["typology_id"], static["domain_display_name"])),
        "dynamic_domain_ami": float(dynamic_summary["selected_solution"]["ami_vs_domain"]),
        "dynamic_static_ami": float(adjusted_mutual_info_score(merged["trajectory_cluster_id"], merged["typology_id"])),
    }


def figure_09_verdict_scoreboard(static: pd.DataFrame, dynamic: pd.DataFrame) -> None:
    values = diagnostic_values(static, dynamic)
    metrics = [
        ("Separation", "silhouette", values["static_silhouette"], values["dynamic_silhouette"], "3.0x clearer"),
        ("Stability", "subsample ARI", values["static_ari"], values["dynamic_ari"], "2.0x steadier"),
        ("Domain overlap", "AMI vs OpenAlex", values["static_domain_ami"], values["dynamic_domain_ami"], "both near zero"),
    ]
    fig, ax = plt.subplots(figsize=(12.4, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.94, "Temporal motion is the cleaner signal", ha="center", fontsize=18, weight="bold")
    ax.text(
        0.5,
        0.885,
        "Dynamic trajectories separate and reproduce better than static shape, while both stay weakly tied to OpenAlex domains.",
        ha="center",
        fontsize=10.2,
        color="#444444",
    )
    ax.add_patch(
        FancyBboxPatch((0.18, 0.22), 0.30, 0.58, boxstyle="round,pad=0.018,rounding_size=0.025", facecolor="#f0f0ee", edgecolor="#d4d4d0")
    )
    ax.add_patch(
        FancyBboxPatch((0.52, 0.22), 0.30, 0.58, boxstyle="round,pad=0.018,rounding_size=0.025", facecolor="#eef5ef", edgecolor="#b8d5bf")
    )
    ax.text(0.33, 0.755, "Static shape", ha="center", fontsize=15, weight="bold", color="#555555")
    ax.text(0.67, 0.755, "Temporal motion", ha="center", fontsize=15, weight="bold", color="#2f7f5f")
    y_positions = [0.62, 0.47, 0.32]
    for (name, unit, static_value, dynamic_value, note), y in zip(metrics, y_positions):
        ax.text(0.08, y, name, ha="left", va="center", fontsize=12.0, weight="bold", color="#222222")
        ax.text(0.08, y - 0.043, unit, ha="left", va="center", fontsize=8.5, color="#666666")
        ax.text(0.33, y + 0.025, f"{static_value:.3f}", ha="center", va="center", fontsize=18, weight="bold", color="#555555")
        ax.text(0.67, y + 0.025, f"{dynamic_value:.3f}", ha="center", va="center", fontsize=18, weight="bold", color="#2f7f5f")
        scale_max = 0.80 if name != "Domain overlap" else 0.25
        ax.add_patch(Rectangle((0.235, y - 0.045), 0.19, 0.035, facecolor="#dddddd", edgecolor="none"))
        ax.add_patch(Rectangle((0.575, y - 0.045), 0.19, 0.035, facecolor="#d9e8dc", edgecolor="none"))
        ax.add_patch(Rectangle((0.235, y - 0.045), 0.19 * min(static_value / scale_max, 1), 0.035, facecolor="#8a8a8a", edgecolor="none"))
        ax.add_patch(Rectangle((0.575, y - 0.045), 0.19 * min(dynamic_value / scale_max, 1), 0.035, facecolor="#2f7f5f", edgecolor="none"))
        ax.text(0.90, y, note, ha="right", va="center", fontsize=10.0, color="#555555", weight="bold" if name != "Domain overlap" else "normal")
    ax.text(
        0.50,
        0.12,
        f"Headline: silhouette is {values['dynamic_silhouette'] / values['static_silhouette']:.1f}x higher and stability is {values['dynamic_ari'] / values['static_ari']:.1f}x higher for temporal trajectories.",
        ha="center",
        fontsize=12.5,
        weight="bold",
        color="#111111",
    )
    ax.text(
        0.50,
        0.06,
        f"Static-dynamic AMI is only {values['dynamic_static_ami']:.3f}: temporal movement adds a genuinely different view.",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )
    save(fig, "fig_09y_temporal_signal_scoreboard_candidate")


def figure_09_static_fate_cards(static: pd.DataFrame, dynamic: pd.DataFrame) -> None:
    merged = static[["subfield_id", "typology_id"]].merge(
        dynamic[["subfield_id", "trajectory_cluster_id"]],
        on="subfield_id",
        how="inner",
    )
    counts = (
        merged.groupby(["typology_id", "trajectory_cluster_id"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=TYPOLOGY_IDS, columns=DYNAMIC_ORDER, fill_value=0)
    )
    counts.to_csv(OUTPUT_DIR / "ch09_static_dynamic_fate_counts.csv")
    fig, axes = plt.subplots(5, 1, figsize=(12.2, 8.6), sharex=True)
    for ax, tid in zip(axes, TYPOLOGY_IDS):
        total = int(counts.loc[tid].sum())
        shares = counts.loc[tid] / total
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(Rectangle((0.00, 0.18), 1.0, 0.64, facecolor="#f7f7f4", edgecolor="#e0e0dd", linewidth=0.8))
        ax.text(0.02, 0.62, f"{tid}", ha="left", va="center", fontsize=15, weight="bold", color=TYPOLOGY_COLORS[tid])
        ax.text(0.08, 0.62, TYPOLOGY_LABELS[tid], ha="left", va="center", fontsize=10.4, color="#222222", weight="bold")
        ax.text(0.08, 0.38, f"{total} subfields", ha="left", va="center", fontsize=8.7, color="#555555")
        bar_left = 0.35
        bar_width = 0.46
        x = bar_left
        for did in DYNAMIC_ORDER:
            width = bar_width * float(shares.loc[did])
            ax.add_patch(Rectangle((x, 0.37), width, 0.25, facecolor=DYNAMIC_COLORS[did], edgecolor="white", linewidth=0.7))
            if width > 0.055:
                ax.text(x + width / 2, 0.495, f"D{did}", ha="center", va="center", fontsize=8.5, color="white", weight="bold")
            x += width
        top_did = int(shares.idxmax())
        top_share = float(shares.loc[top_did])
        split_note = "split" if top_share < 0.50 else "main fate"
        ax.text(0.84, 0.56, f"{split_note}: D{top_did}", ha="left", va="center", fontsize=9.5, weight="bold", color=DYNAMIC_COLORS[top_did])
        ax.text(0.84, 0.40, f"{top_share:.0%}", ha="left", va="center", fontsize=15, weight="bold", color=DYNAMIC_COLORS[top_did])
    handles = [
        Line2D([0], [0], color=DYNAMIC_COLORS[did], linewidth=8, label=f"D{did} {DYNAMIC_LABELS[did]}")
        for did in DYNAMIC_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.005), ncol=2, frameon=True, fontsize=8.3)
    fig.suptitle("Every static type splits into several temporal fates", fontsize=17, weight="bold", y=0.985)
    fig.text(0.5, 0.94, "A readable replacement for the static-to-dynamic alluvial: each row is one static typology.", ha="center", fontsize=10, color="#444444")
    fig.tight_layout(rect=[0, 0.055, 1, 0.91])
    save(fig, "fig_09y_static_fate_cards_candidate")


def arrow_label(value: float) -> str:
    if value >= 0.20:
        return "up"
    if value <= -0.20:
        return "down"
    return "flat"


def figure_09_dynamic_movement_cards(summary: pd.DataFrame) -> None:
    metrics = [
        ("Territory", "slope_embedding_distance_to_centroid_median"),
        ("Neighborhood", "slope_embedding_knn_median_distance"),
        ("Hubness", "slope_embedding_knn_indegree_gini"),
        ("Dimensionality", "slope_embedding_pca_dim_80"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2))
    axes = axes.reshape(-1)
    for ax, row in zip(axes, summary.sort_values("trajectory_cluster_id").itertuples(index=False)):
        did = int(row.trajectory_cluster_id)
        color = DYNAMIC_COLORS[did]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(
            FancyBboxPatch(
                (0.02, 0.04),
                0.96,
                0.90,
                boxstyle="round,pad=0.018,rounding_size=0.035",
                facecolor="#f7f7f4",
                edgecolor=color,
                linewidth=1.5,
            )
        )
        ax.text(0.07, 0.86, f"D{did}", fontsize=18, weight="bold", color=color, va="center")
        ax.text(0.19, 0.86, DYNAMIC_LABELS[did], fontsize=12.4, weight="bold", color="#222222", va="center")
        ax.text(0.07, 0.755, f"{int(row.n_subfields)} subfields | dominant domain: {row.dominant_domain} ({row.dominant_domain_share:.0%})", fontsize=9.5, color="#555555")
        for i, (label, col) in enumerate(metrics):
            value = float(getattr(row, col))
            status = arrow_label(value)
            y = 0.61 - i * 0.13
            ax.text(0.08, y, label, ha="left", va="center", fontsize=10.8, color="#333333", weight="bold")
            ax.add_patch(Rectangle((0.37, y - 0.025), 0.26, 0.05, facecolor="#e5e5e5", edgecolor="none"))
            if value >= 0:
                ax.add_patch(Rectangle((0.50, y - 0.025), min(abs(value) / 1.2, 0.13), 0.05, facecolor=color, edgecolor="none", alpha=0.85))
            else:
                ax.add_patch(Rectangle((0.50 - min(abs(value) / 1.2, 0.13), y - 0.025), min(abs(value) / 1.2, 0.13), 0.05, facecolor=color, edgecolor="none", alpha=0.85))
            symbol = {"up": "UP", "down": "DOWN", "flat": "FLAT"}[status]
            ax.text(0.70, y, symbol, ha="left", va="center", fontsize=10.4, color=color, weight="bold")
            ax.text(0.91, y, f"{value:+.2f}", ha="right", va="center", fontsize=9.5, color="#444444")
    fig.suptitle("The four temporal movements in plain language", fontsize=17, weight="bold", y=0.99)
    fig.text(0.5, 0.94, "Each card compresses the dynamic heatmap into four interpretable movement directions.", ha="center", fontsize=10.8, color="#444444")
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    save(fig, "fig_09y_dynamic_movement_cards_candidate")


def build_gallery(stems: list[str]) -> None:
    ncols = 2
    nrows = int(np.ceil(len(stems) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.0, nrows * 5.2))
    axes = np.array(axes).reshape(-1)
    for ax, stem in zip(axes, stems):
        img = mpimg.imread(OUTPUT_DIR / f"{stem}.png")
        ax.imshow(img)
        ax.set_title(stem.replace("_candidate", "").replace("_", " "), fontsize=9.6)
        ax.axis("off")
    for ax in axes[len(stems) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "glanceable_candidates_gallery.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


def write_report(static: pd.DataFrame, dynamic: pd.DataFrame) -> None:
    values = diagnostic_values(static, dynamic)
    quadrant_counts = static["quadrant"].value_counts()
    lines = [
        "# Glanceable Visual Candidates",
        "",
        "Generated from existing TFM outputs only. No thesis file was modified.",
        "",
        "## Chapter 6",
        "",
        "1. `fig_06y_morphology_compass_candidate`",
        "   - Best insertion point: after the field-level heatmap.",
        "   - One-glance message: subfields occupy four intuitive morphology zones: broad/smooth, broad/concentrated, compact/smooth, compact/concentrated.",
        "   - Result counts: "
        + "; ".join(f"{label}={int(quadrant_counts[label])}" for label in quadrant_counts.index)
        + ".",
        "   - My read: strongest Chapter 6 replacement. It is immediately understandable and still grounded in the eleven metrics.",
        "",
        "2. `fig_06y_quadrant_extreme_cards_candidate`",
        "   - Best insertion point: immediately after the compass, or instead of the compass if we want zero scatterplot complexity.",
        "   - One-glance message: each structural zone has named, interpretable extreme cases.",
        "   - My read: very clear; maybe more magazine-like than thesis-like, but useful if the reader needs names fast.",
        "",
        "3. `fig_06y_rebel_spotlight_candidate`",
        "   - Best insertion point: near the outlier ranking.",
        "   - One-glance message: four subfields break four different intuitive expectations.",
        "   - My read: visually direct, but I would use it only if you want a punchier narrative moment.",
        "",
        "## Chapter 9",
        "",
        "1. `fig_09y_temporal_signal_scoreboard_candidate`",
        "   - Best insertion point: after the diagnostics figure or as the chapter's interpretive bridge before the static heatmap.",
        f"   - One-glance message: temporal motion has {values['dynamic_silhouette'] / values['static_silhouette']:.1f}x higher silhouette and {values['dynamic_ari'] / values['static_ari']:.1f}x higher stability than static shape.",
        "   - My read: strongest Chapter 9 addition. It gives the reader the punchline before the heatmaps.",
        "",
        "2. `fig_09y_static_fate_cards_candidate`",
        "   - Best insertion point: after the dynamic heatmap.",
        "   - One-glance message: every static type splits into several temporal fates.",
        "   - My read: much easier than the alluvial version; good if we want to explain AMI=0.048 visually.",
        "",
        "3. `fig_09y_dynamic_movement_cards_candidate`",
        "   - Best insertion point: immediately after the dynamic heatmap.",
        "   - One-glance message: the four dynamic clusters are four movement verbs, not four opaque heatmap rows.",
        "   - My read: clearest for nontechnical readers, but more explanatory than revelatory.",
        "",
        "## Recommendation",
        "",
        "- Section 6: try `fig_06y_morphology_compass_candidate` first.",
        "- Section 9: try `fig_09y_temporal_signal_scoreboard_candidate` first.",
        "- If you want a second figure in section 9, use `fig_09y_static_fate_cards_candidate`.",
    ]
    (OUTPUT_DIR / "glanceable_candidates_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup()
    static = load_static()
    dynamic = load_dynamic()
    dynamic_summary = load_dynamic_summary()
    stems = [
        "fig_06y_morphology_compass_candidate",
        "fig_06y_quadrant_extreme_cards_candidate",
        "fig_06y_rebel_spotlight_candidate",
        "fig_09y_temporal_signal_scoreboard_candidate",
        "fig_09y_static_fate_cards_candidate",
        "fig_09y_dynamic_movement_cards_candidate",
    ]
    figure_06_morphology_compass(static)
    figure_06_quadrant_cards(static)
    figure_06_rebel_spotlight(static)
    figure_09_verdict_scoreboard(static, dynamic)
    figure_09_static_fate_cards(static, dynamic)
    figure_09_dynamic_movement_cards(dynamic_summary)
    build_gallery(stems)
    write_report(static, dynamic)


if __name__ == "__main__":
    main()
