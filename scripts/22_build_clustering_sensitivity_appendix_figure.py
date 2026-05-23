from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.temporal_common import display_path, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact appendix figure for temporal and hybrid clustering sensitivity."
    )
    parser.add_argument(
        "--temporal-assignments-path",
        default=(
            "outputs/09_morphological_typologies/temporal_trajectory_clustering/"
            "trajectory_cluster_assignments.csv"
        ),
    )
    parser.add_argument(
        "--hybrid-composition-path",
        default=(
            "outputs/09_morphological_typologies/hybrid_centroid_morphology_clustering/"
            "hybrid_cluster_domain_composition.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/09_morphological_typologies/clustering_sensitivity_appendix",
    )
    parser.add_argument("--figure-dir", default="memory/figures")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    temporal_path = resolve_path(args.temporal_assignments_path, root=ROOT)
    hybrid_path = resolve_path(args.hybrid_composition_path, root=ROOT)
    output_dir = resolve_path(args.output_dir, root=ROOT)
    figure_dir = resolve_path(args.figure_dir, root=ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = "fig_c_temporal_hybrid_clustering_sensitivity"
    outputs = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf", figure_dir / f"{stem}.png"]
    if any(path.exists() for path in outputs) and not args.overwrite:
        existing = ", ".join(display_path(path, root=ROOT) for path in outputs if path.exists())
        raise FileExistsError(f"Refusing to overwrite existing figure without --overwrite: {existing}")

    temporal = pd.read_csv(temporal_path)
    composition = pd.read_csv(hybrid_path)
    required_temporal = {
        "trajectory_cluster_id",
        "trajectory_pca_1",
        "trajectory_pca_2",
        "trajectory_pca_1_explained_variance",
        "trajectory_pca_2_explained_variance",
    }
    required_composition = {
        "hybrid_cluster_id",
        "domain_display_name",
        "share_within_cluster",
    }
    missing_temporal = sorted(required_temporal - set(temporal.columns))
    missing_composition = sorted(required_composition - set(composition.columns))
    if missing_temporal or missing_composition:
        raise ValueError(
            f"Missing temporal columns={missing_temporal}; missing hybrid columns={missing_composition}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), constrained_layout=True)
    cmap = plt.get_cmap("tab10")

    ax = axes[0]
    for idx, cluster_id in enumerate(sorted(temporal["trajectory_cluster_id"].unique())):
        subset = temporal.loc[temporal["trajectory_cluster_id"] == cluster_id]
        ax.scatter(
            subset["trajectory_pca_1"],
            subset["trajectory_pca_2"],
            s=33,
            color=cmap(idx % 10),
            alpha=0.84,
            edgecolor="white",
            linewidth=0.35,
            label=f"Dyn{int(cluster_id)} ({len(subset)})",
        )
    pca1 = float(temporal["trajectory_pca_1_explained_variance"].iloc[0]) * 100
    pca2 = float(temporal["trajectory_pca_2_explained_variance"].iloc[0]) * 100
    ax.set_title("A. Temporal trajectory clusters", loc="left", fontsize=12)
    ax.set_xlabel(f"Trajectory PCA 1 ({pca1:.1f}%)")
    ax.set_ylabel(f"Trajectory PCA 2 ({pca2:.1f}%)")
    ax.grid(True, alpha=0.18)
    ax.legend(fontsize=7.5, frameon=False)

    ax = axes[1]
    pivot = composition.pivot_table(
        index="hybrid_cluster_id",
        columns="domain_display_name",
        values="share_within_cluster",
        fill_value=0.0,
    )
    domain_order = [
        domain for domain in ["Life Sciences", "Social Sciences", "Physical Sciences", "Health Sciences"]
        if domain in pivot.columns
    ]
    pivot = pivot.reindex(columns=domain_order)
    left = np.zeros(len(pivot))
    colors = {
        "Life Sciences": "#2f7f5f",
        "Social Sciences": "#8b6bb5",
        "Physical Sciences": "#ba6a36",
        "Health Sciences": "#4c78a8",
    }
    y = np.arange(len(pivot))
    for domain in pivot.columns:
        values = pivot[domain].to_numpy(dtype=float)
        ax.barh(
            y,
            values,
            left=left,
            color=colors.get(domain, "#999999"),
            edgecolor="white",
            linewidth=0.5,
            label=domain,
        )
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels([f"H{int(value)}" for value in pivot.index])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share within hybrid cluster")
    ax.set_title("B. Hybrid clusters are domain-heavy", loc="left", fontsize=12)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.grid(True, axis="x", alpha=0.15)

    fig.savefig(output_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {display_path(figure_dir / f'{stem}.png', root=ROOT)}")


if __name__ == "__main__":
    main()
