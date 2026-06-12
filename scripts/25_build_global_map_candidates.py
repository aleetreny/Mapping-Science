from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "10_global_map_candidates"
FIGURE_DIR = ROOT / "memory" / "figures"

DOMAIN_ORDER = [
    "Life Sciences",
    "Social Sciences",
    "Physical Sciences",
    "Health Sciences",
]

DOMAIN_COLORS = {
    "Life Sciences": "#2f7f5f",
    "Social Sciences": "#8b6bb5",
    "Physical Sciences": "#ba6a36",
    "Health Sciences": "#4c78a8",
}

STATIC_COLORS = {
    "T1": "#3776ab",
    "T2": "#5f8f3a",
    "T3": "#8e5ea2",
    "T4": "#c47a2c",
    "T5": "#b5483a",
}

DYNAMIC_COLORS = {
    "1": "#4c78a8",
    "2": "#c47a2c",
    "3": "#6b8e23",
    "4": "#8e5ea2",
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
    "embedding_distance_to_centroid_median": "Centroid distance",
    "embedding_distance_to_centroid_iqr": "Radial unevenness",
    "embedding_distance_to_centroid_p90": "Outer radius",
    "embedding_knn_median_distance": "Local sparsity",
    "embedding_knn_distance_cv": "Density unevenness",
    "embedding_knn_indegree_gini": "Hub concentration",
    "embedding_pca_dim_80": "Spectral dimensionality",
    "embedding_pca_spectral_entropy": "Spectral entropy",
}


def _setup() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
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
            "savefig.dpi": 300,
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    for folder in [OUTPUT_DIR, FIGURE_DIR]:
        fig.savefig(folder / f"{stem}.png", bbox_inches="tight")
        fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _pcoa(distance: pd.DataFrame) -> tuple[pd.DataFrame, tuple[float, float]]:
    labels = list(distance.index)
    d = distance.to_numpy(dtype=float)
    n = d.shape[0]
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ (d ** 2) @ j
    eigvals, eigvecs = np.linalg.eigh(b)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    positive = eigvals > 1e-12
    total = eigvals[positive].sum()
    coords = eigvecs[:, :2] * np.sqrt(np.maximum(eigvals[:2], 0))
    shares = (
        float(eigvals[0] / total) if total > 0 else 0.0,
        float(eigvals[1] / total) if total > 0 else 0.0,
    )
    return pd.DataFrame(coords, index=labels, columns=["x", "y"]), shares


def _field_domain_table() -> pd.DataFrame:
    assignments = pd.read_csv(ROOT / "outputs" / "09_morphological_typologies" / "subfield_typology_assignments.csv")
    fields = (
        assignments.groupby(["field_id", "field_display_name", "domain_display_name"], as_index=False)
        .agg(n_subfields=("subfield_id", "nunique"))
        .sort_values("field_display_name")
    )
    fields["field_id"] = fields["field_id"].astype(str)
    return fields


def _field_distance_matrix() -> tuple[pd.DataFrame, pd.DataFrame]:
    distances = pd.read_parquet(ROOT / "data" / "processed" / "temporal" / "morphological_pair_distances_static.parquet")
    distances = distances[
        (distances["metric_set"] == "static_full_reduced")
        & (distances["level"] == "field")
        & (distances["distance_metric"] == "euclidean")
    ].copy()
    fields = _field_domain_table()
    field_names = sorted(fields["field_display_name"].tolist())
    matrix = pd.DataFrame(0.0, index=field_names, columns=field_names)
    for row in distances.itertuples(index=False):
        a = str(row.entity_display_name_a)
        b = str(row.entity_display_name_b)
        value = float(row.morphological_distance)
        if a in matrix.index and b in matrix.columns:
            matrix.loc[a, b] = value
            matrix.loc[b, a] = value
    meta = fields.set_index("field_display_name").loc[field_names].reset_index()
    return matrix, meta


def build_field_neighbor_map() -> dict[str, float]:
    matrix, meta = _field_distance_matrix()
    coords, shares = _pcoa(matrix)
    meta = meta.join(coords, on="field_display_name")
    domain_by_field = dict(zip(meta["field_display_name"], meta["domain_display_name"]))

    nearest_rows = []
    for field in matrix.index:
        distances = matrix.loc[field].copy()
        distances.loc[field] = np.inf
        nearest = str(distances.idxmin())
        nearest_rows.append(
            {
                "field": field,
                "nearest_field": nearest,
                "distance": float(distances.loc[nearest]),
                "cross_domain": domain_by_field[field] != domain_by_field[nearest],
            }
        )
    nearest = pd.DataFrame(nearest_rows)
    nearest.to_csv(OUTPUT_DIR / "field_nearest_morphological_neighbors.csv", index=False)

    fig, ax = plt.subplots(figsize=(11.5, 8.2))
    ax.grid(color="#e5e5e5", linewidth=0.55, alpha=0.85)

    drawn_pairs: set[tuple[str, str]] = set()
    for row in nearest.itertuples(index=False):
        pair = tuple(sorted([row.field, row.nearest_field]))
        if pair in drawn_pairs:
            continue
        drawn_pairs.add(pair)
        a = coords.loc[row.field]
        b = coords.loc[row.nearest_field]
        cross = domain_by_field[row.field] != domain_by_field[row.nearest_field]
        ax.plot(
            [a.x, b.x],
            [a.y, b.y],
            color="#222222" if cross else "#b7b7b7",
            linewidth=1.35 if cross else 0.85,
            alpha=0.74 if cross else 0.48,
            zorder=1,
        )

    for domain in DOMAIN_ORDER:
        subset = meta[meta["domain_display_name"] == domain]
        ax.scatter(
            subset["x"],
            subset["y"],
            s=80 + subset["n_subfields"] * 16,
            color=DOMAIN_COLORS[domain],
            edgecolor="white",
            linewidth=1.0,
            alpha=0.94,
            label=domain,
            zorder=3,
        )

    label_offsets = {
        "Computer Science": (0.05, -0.11),
        "Pharmacology, Toxicology and Pharmaceutics": (0.05, 0.06),
        "Dentistry": (0.05, 0.04),
        "Energy": (0.05, 0.08),
        "Immunology and Microbiology": (-0.48, 0.03),
        "Medicine": (0.05, -0.08),
        "Chemistry": (0.05, 0.05),
        "Materials Science": (0.05, -0.08),
        "Physics and Astronomy": (0.05, -0.06),
        "Neuroscience": (0.05, 0.05),
        "Psychology": (0.05, -0.10),
        "Decision Sciences": (0.05, 0.14),
        "Social Sciences": (-0.42, -0.02),
        "Materials Science": (-0.62, -0.08),
        "Environmental Science": (0.05, 0.07),
    }
    for row in meta.itertuples(index=False):
        dx, dy = label_offsets.get(row.field_display_name, (0.035, 0.035))
        ax.text(
            row.x + dx,
            row.y + dy,
            FIELD_LABELS.get(row.field_display_name, row.field_display_name),
            fontsize=8.2,
            color="#222222",
            zorder=4,
        )

    cross_count = int(nearest["cross_domain"].sum())
    ax.set_xlabel(f"Morphology PCoA 1 ({shares[0] * 100:.1f}% of positive-axis inertia)")
    ax.set_ylabel(f"Morphology PCoA 2 ({shares[1] * 100:.1f}% of positive-axis inertia)")
    ax.set_title("Field morphology neighbor map", fontsize=16, pad=14)
    legend_handles = [
        *[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=DOMAIN_COLORS[d], markeredgecolor="white", markersize=9, label=d)
            for d in DOMAIN_ORDER
        ],
        Line2D([0], [0], color="#222222", linewidth=1.6, label="Cross-domain nearest-neighbor link"),
        Line2D([0], [0], color="#b7b7b7", linewidth=1.1, alpha=0.8, label="Within-domain nearest-neighbor link"),
    ]
    ax.legend(
        handles=legend_handles,
        title=f"{cross_count}/{len(nearest)} fields link outside their domain",
        loc="upper right",
        frameon=True,
        framealpha=0.92,
        fontsize=9,
        title_fontsize=9.5,
    )
    _save(fig, "fig_08_field_morphology_neighbor_map_candidate")
    return {"cross_domain_nearest_neighbors": cross_count, "n_fields": len(nearest)}


def build_field_trajectory_map() -> dict[str, float]:
    frame = pd.read_parquet(ROOT / "data" / "processed" / "temporal" / "field_window_embedding_metric_trajectories.parquet")
    metrics = list(METRIC_LABELS)
    frame = frame[frame["metric"].isin(metrics)].copy()
    pivot = frame.pivot_table(
        index=["field_id", "field_display_name", "window_label", "window_index", "window_mid_year"],
        columns="metric",
        values="mean_standardized_value",
        aggfunc="mean",
    ).reset_index()
    pivot = pivot.dropna(subset=metrics)
    features = pivot[metrics].to_numpy(dtype=float)
    features = (features - features.mean(axis=0)) / features.std(axis=0, ddof=0)
    pca = PCA(n_components=2, random_state=0)
    xy = pca.fit_transform(features)
    pivot["x"] = xy[:, 0]
    pivot["y"] = xy[:, 1]

    meta = _field_domain_table()[["field_id", "field_display_name", "domain_display_name", "n_subfields"]].copy()
    meta["field_id"] = meta["field_id"].astype(str)
    pivot["field_id"] = pivot["field_id"].astype(str)
    pivot = pivot.merge(meta, on=["field_id", "field_display_name"], how="left")

    first_last = []
    for field, group in pivot.groupby("field_display_name"):
        ordered = group.sort_values("window_index")
        first = ordered.iloc[0]
        last = ordered.iloc[-1]
        path = ordered[["x", "y"]].to_numpy()
        segment_lengths = np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1))
        first_last.append(
            {
                "field_display_name": field,
                "domain_display_name": first.domain_display_name,
                "direct_distance": float(np.linalg.norm(path[-1] - path[0])),
                "path_length": float(segment_lengths.sum()),
                "start_x": float(first.x),
                "start_y": float(first.y),
                "end_x": float(last.x),
                "end_y": float(last.y),
            }
        )
    movement = pd.DataFrame(first_last).sort_values("path_length", ascending=False)
    movement.to_csv(OUTPUT_DIR / "field_morphology_trajectory_distances.csv", index=False)
    top_labels = set(movement.head(10)["field_display_name"])

    fig, ax = plt.subplots(figsize=(11.5, 8.2))
    ax.grid(color="#e5e5e5", linewidth=0.55, alpha=0.85)

    for field, group in pivot.groupby("field_display_name", sort=False):
        group = group.sort_values("window_index")
        domain = str(group["domain_display_name"].iloc[0])
        color = DOMAIN_COLORS.get(domain, "#999999")
        ax.plot(group["x"], group["y"], color=color, linewidth=1.2, alpha=0.45, zorder=1)
        ax.scatter(group["x"], group["y"], color=color, s=18, alpha=0.55, linewidth=0, zorder=2)
        start = group.iloc[0]
        end = group.iloc[-1]
        ax.scatter([start.x], [start.y], s=64, facecolor="white", edgecolor=color, linewidth=1.5, zorder=4)
        ax.scatter([end.x], [end.y], s=70, marker="s", facecolor=color, edgecolor="white", linewidth=0.8, zorder=4)
        ax.annotate(
            "",
            xy=(end.x, end.y),
            xytext=(start.x, start.y),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.0, "alpha": 0.32},
            zorder=1,
        )
        if field in top_labels:
            ax.text(
                end.x + 0.05,
                end.y + 0.05,
                FIELD_LABELS.get(field, field),
                fontsize=8.4,
                color="#222222",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
                zorder=5,
            )

    ax.axhline(0, color="#bbbbbb", linewidth=0.7, alpha=0.75)
    ax.axvline(0, color="#bbbbbb", linewidth=0.7, alpha=0.75)
    ax.set_xlabel(f"Morphology trajectory PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"Morphology trajectory PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    ax.set_title("Field morphology trajectories, 2000-2004 to 2020-2024", fontsize=16, pad=14)
    ax.text(
        0.012,
        0.985,
        "Open circles mark 2000-2004; filled squares mark 2020-2024. Coordinates use eight windowed structural metrics.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.7,
        color="#222222",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.94},
    )
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=DOMAIN_COLORS[d], markeredgecolor="white", markersize=9, label=d)
        for d in DOMAIN_ORDER
    ]
    legend_handles.extend(
        [
            Line2D([0], [0], marker="o", color="#555555", markerfacecolor="white", markersize=8, linestyle="None", label="2000-2004"),
            Line2D([0], [0], marker="s", color="#555555", markerfacecolor="#555555", markersize=8, linestyle="None", label="2020-2024"),
        ]
    )
    ax.legend(handles=legend_handles, loc="lower left", frameon=True, framealpha=0.92, fontsize=9)
    _save(fig, "fig_07_field_morphology_trajectory_map_candidate")
    return {
        "pc1_variance": float(pca.explained_variance_ratio_[0]),
        "pc2_variance": float(pca.explained_variance_ratio_[1]),
        "top_path_field": str(movement.iloc[0]["field_display_name"]),
    }


def build_static_dynamic_reader_map() -> dict[str, float]:
    static = pd.read_csv(ROOT / "outputs" / "09_morphological_typologies" / "projection_sensitivity" / "static_projection_coordinates.csv")
    dynamic = pd.read_csv(ROOT / "outputs" / "09_morphological_typologies" / "projection_sensitivity" / "dynamic_projection_coordinates.csv")
    static = static[(static["projection_method"] == "PCA") & (static["metric"] == "linear")].copy()
    dynamic = dynamic[(dynamic["projection_method"] == "PCoA") & (dynamic["metric"] == "correlation")].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6))
    for ax in axes:
        ax.grid(color="#e5e5e5", linewidth=0.5, alpha=0.85)
        ax.axhline(0, color="#cccccc", linewidth=0.6)
        ax.axvline(0, color="#cccccc", linewidth=0.6)

    for typology_id, group in static.groupby("typology_id"):
        color = STATIC_COLORS.get(str(typology_id), "#999999")
        axes[0].scatter(group["x"], group["y"], s=26, color=color, alpha=0.78, edgecolor="white", linewidth=0.35, label=str(typology_id))
    axes[0].set_title("A. Static morphology: weakly separated profiles", loc="left", fontsize=13.5)
    axes[0].set_xlabel(str(static["axis_1_label"].iloc[0]))
    axes[0].set_ylabel(str(static["axis_2_label"].iloc[0]))
    axes[0].text(
        0.03,
        0.97,
        "selected k = 5; silhouette = 0.133",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.94},
    )

    for cluster_id, group in dynamic.groupby("typology_id"):
        color = DYNAMIC_COLORS.get(str(cluster_id), "#999999")
        axes[1].scatter(group["x"], group["y"], s=28, color=color, alpha=0.82, edgecolor="white", linewidth=0.35, label=f"D{cluster_id}")
    axes[1].set_title("B. Dynamic morphology: clearer trajectory regimes", loc="left", fontsize=13.5)
    axes[1].set_xlabel(str(dynamic["axis_1_label"].iloc[0]))
    axes[1].set_ylabel(str(dynamic["axis_2_label"].iloc[0]))
    axes[1].text(
        0.03,
        0.97,
        "selected k = 4; silhouette = 0.396; mean subsample ARI = 0.725",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.94},
    )

    static_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=STATIC_COLORS[k], markeredgecolor="white", markersize=7, label=k)
        for k in ["T1", "T2", "T3", "T4", "T5"]
    ]
    dynamic_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=DYNAMIC_COLORS[k], markeredgecolor="white", markersize=7, label=f"D{k}")
        for k in ["1", "2", "3", "4"]
    ]
    axes[0].legend(handles=static_handles, title="Static typology", loc="lower left", frameon=True, fontsize=8.5, title_fontsize=9)
    axes[1].legend(handles=dynamic_handles, title="Dynamic typology", loc="lower left", frameon=True, fontsize=8.5, title_fontsize=9)

    fig.suptitle("Static shape is continuous; temporal morphology separates more clearly", fontsize=16, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_09_static_dynamic_reader_map_candidate")
    return {"static_points": len(static), "dynamic_points": len(dynamic)}


def write_report(stats: dict[str, dict[str, float]]) -> None:
    lines = [
        "# Global Map Candidates",
        "",
        "Generated from existing TFM outputs. These figures are candidates only; thesis chapter files were not modified.",
        "",
        "## Candidate Figures",
        "",
        "- `memory/figures/fig_08_field_morphology_neighbor_map_candidate.png`: field-level morphology neighbor map based on full-period Euclidean profile distances.",
        "- `memory/figures/fig_07_field_morphology_trajectory_map_candidate.png`: field-level temporal morphology trajectory map using eight windowed structural metrics.",
        "- `memory/figures/fig_09_static_dynamic_reader_map_candidate.png`: reader-facing static-vs-dynamic separation map for Chapter 9.",
        "",
        "## Suggested Use",
        "",
        "1. Use the neighbor map in Chapter 8 before or instead of the field distance heatmap. It makes cross-domain nearest-neighbor results visible immediately.",
        "2. Use the trajectory map in Chapter 7 after the domain trajectories. It gives temporal morphology a global map instead of only heatmaps and rankings.",
        "3. Use the reader map in Chapter 9 instead of the six-panel projection comparison if the main text needs a cleaner argument-first figure. Keep the six-panel version in the appendix.",
        "",
        "## Computed Checks",
        "",
        f"- Cross-domain nearest-neighbor fields: {int(stats['neighbor']['cross_domain_nearest_neighbors'])} of {int(stats['neighbor']['n_fields'])}.",
        f"- Field trajectory PCA variance: PC1 {stats['trajectory']['pc1_variance'] * 100:.1f}%, PC2 {stats['trajectory']['pc2_variance'] * 100:.1f}%.",
        f"- Largest field trajectory path in candidate map: {stats['trajectory']['top_path_field']}.",
    ]
    (OUTPUT_DIR / "global_map_candidates_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    _setup()
    stats = {
        "neighbor": build_field_neighbor_map(),
        "trajectory": build_field_trajectory_map(),
        "reader": build_static_dynamic_reader_map(),
    }
    write_report(stats)


if __name__ == "__main__":
    main()
