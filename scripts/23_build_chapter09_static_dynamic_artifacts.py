from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.morphological_typologies import DOMAIN_COLORS, TYPOLOGY_COLORS, TYPOLOGY_ORDER
from src.temporal_common import REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS, display_path, resolve_path


DYNAMIC_LABELS: dict[int, str] = {
    1: "Compacting locally uneven trajectories",
    2: "Broadening hub-diffusing trajectories",
    3: "Smoothing sparse-neighborhood trajectories",
    4: "Dimensionalizing hub-concentrating trajectories",
}

DYNAMIC_COLORS: dict[int, str] = {
    1: "#4c78a8",
    2: "#c47a2c",
    3: "#6b8e23",
    4: "#8e5ea2",
}

SLOPE_METRICS: list[str] = [
    "embedding_distance_to_centroid_median",
    "embedding_distance_to_centroid_iqr",
    "embedding_distance_to_centroid_p90",
    "embedding_knn_median_distance",
    "embedding_knn_distance_cv",
    "embedding_knn_indegree_gini",
    "embedding_pca_dim_80",
    "embedding_pca_spectral_entropy",
]

SLOPE_LABELS: dict[str, str] = {
    "embedding_distance_to_centroid_median": "Centroid\nmedian",
    "embedding_distance_to_centroid_iqr": "Centroid\nIQR",
    "embedding_distance_to_centroid_p90": "Centroid\nP90",
    "embedding_knn_median_distance": "kNN\nmedian",
    "embedding_knn_distance_cv": "kNN\nCV",
    "embedding_knn_indegree_gini": "Hub\nGini",
    "embedding_pca_dim_80": "PCA\nD80",
    "embedding_pca_spectral_entropy": "PCA\nentropy",
}

DOMAIN_ORDER: list[str] = [
    "Life Sciences",
    "Social Sciences",
    "Physical Sciences",
    "Health Sciences",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Chapter 9 static-vs-dynamic clustering figures and table."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/09_morphological_typologies/projection_sensitivity",
    )
    parser.add_argument("--figure-dir", default="memory/figures")
    parser.add_argument("--table-dir", default="memory/tables")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_csv(relative_path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative_path)


def _save_figure(fig: plt.Figure, stem: str, output_dir: Path, figure_dir: Path) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(figure_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _cluster_sizes(size_string: str) -> list[int]:
    return [int(float(part)) for part in str(size_string).split(";") if str(part).strip()]


def _mean_ari(path: str) -> float:
    frame = _read_csv(path)
    return float(frame["adjusted_rand_index"].mean())


def build_diagnostics(output_dir: Path, figure_dir: Path) -> None:
    static = _read_csv("outputs/09_morphological_typologies/cluster_solution_selected.csv").iloc[0]
    temporal = _read_csv(
        "outputs/09_morphological_typologies/temporal_trajectory_clustering/"
        "trajectory_cluster_solution_selected.csv"
    ).iloc[0]
    centroid = _read_csv(
        "outputs/09_morphological_typologies/centroid_embedding_clustering/"
        "centroid_cluster_solution_selected.csv"
    ).iloc[0]
    hybrid = _read_csv(
        "outputs/09_morphological_typologies/hybrid_centroid_morphology_clustering/"
        "hybrid_cluster_solution_selected.csv"
    ).iloc[0]

    labels = ["Static\nmorphology", "Temporal\nslopes", "Centroid\nlocation", "Hybrid"]
    silhouettes = [
        float(static["silhouette"]),
        float(temporal["silhouette"]),
        float(centroid["silhouette"]),
        float(hybrid["silhouette"]),
    ]
    stability_labels = ["Static\nmorphology", "Temporal\nslopes", "Hybrid"]
    stabilities = [
        _mean_ari("outputs/09_morphological_typologies/typology_bootstrap_stability.csv"),
        _mean_ari(
            "outputs/09_morphological_typologies/temporal_trajectory_clustering/"
            "trajectory_cluster_stability_summary.csv"
        ),
        _mean_ari(
            "outputs/09_morphological_typologies/hybrid_centroid_morphology_clustering/"
            "hybrid_cluster_stability_summary.csv"
        ),
    ]
    size_rows = [
        ("Static", _cluster_sizes(static["cluster_sizes_sorted"]), "#7a7a7a"),
        ("Temporal", _cluster_sizes(temporal["cluster_sizes_sorted"]), "#4c78a8"),
        ("Centroid", _cluster_sizes(centroid["cluster_sizes_sorted"]), "#ba6a36"),
        ("Hybrid", _cluster_sizes(hybrid["cluster_sizes_sorted"]), "#6b8e23"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), constrained_layout=True)

    ax = axes[0]
    colors = ["#7a7a7a", "#4c78a8", "#ba6a36", "#6b8e23"]
    ax.bar(np.arange(len(labels)), silhouettes, color=colors, width=0.68)
    for idx, value in enumerate(silhouettes):
        ax.text(idx, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 0.46)
    ax.set_ylabel("Silhouette")
    ax.set_title("A. Separation", loc="left", fontsize=12)
    ax.grid(True, axis="y", alpha=0.18)

    ax = axes[1]
    ax.bar(np.arange(len(stability_labels)), stabilities, color=["#7a7a7a", "#4c78a8", "#6b8e23"])
    for idx, value in enumerate(stabilities):
        ax.text(idx, value + 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(np.arange(len(stability_labels)))
    ax.set_xticklabels(stability_labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean subsample ARI")
    ax.set_title("B. Stability", loc="left", fontsize=12)
    ax.grid(True, axis="y", alpha=0.18)
    ax.text(
        0.03,
        0.96,
        "centroid stability\nnot bootstrapped",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color="#555555",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
    )

    ax = axes[2]
    for y_pos, (row_label, sizes, color) in enumerate(size_rows):
        left = 0
        total = sum(sizes)
        for size in sizes:
            alpha = 0.45 + 0.45 * (size / max(sizes))
            ax.barh(y_pos, size, left=left, color=color, alpha=alpha, edgecolor="white", linewidth=0.6)
            if size / total > 0.09:
                ax.text(left + size / 2, y_pos, str(size), ha="center", va="center", fontsize=8)
            left += size
    ax.set_yticks(np.arange(len(size_rows)))
    ax.set_yticklabels([row[0] for row in size_rows], fontsize=9)
    ax.set_xlabel("Subfields")
    ax.set_title("C. Cluster-size balance", loc="left", fontsize=12)
    ax.grid(True, axis="x", alpha=0.16)

    _save_figure(fig, "fig_09_static_dynamic_diagnostics", output_dir, figure_dir)


def build_dynamic_heatmap(output_dir: Path, figure_dir: Path) -> None:
    profile = _read_csv(
        "outputs/09_morphological_typologies/temporal_trajectory_clustering/"
        "trajectory_cluster_profile_summary.csv"
    )
    profile = profile.sort_values("trajectory_cluster_id", kind="mergesort")
    value_columns = [f"slope_{metric}" for metric in SLOPE_METRICS]
    values = profile[value_columns].to_numpy(dtype=float)
    row_labels = [
        f"D{int(row.trajectory_cluster_id)} {DYNAMIC_LABELS[int(row.trajectory_cluster_id)]}\n(n={int(row.n_subfields)})"
        for row in profile.itertuples()
    ]

    fig, ax = plt.subplots(figsize=(11.3, 4.6), constrained_layout=True)
    limit = max(1.2, float(np.nanmax(np.abs(values))))
    image = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            color = "white" if abs(value) > 0.72 * limit else "#222222"
            ax.text(col_idx, row_idx, f"{value:+.2f}", ha="center", va="center", fontsize=8, color=color)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8.8)
    ax.set_xticks(np.arange(len(value_columns)))
    ax.set_xticklabels([SLOPE_LABELS[metric] for metric in SLOPE_METRICS], fontsize=8.8)
    ax.set_title("Temporal trajectory slope profiles", fontsize=13)
    ax.tick_params(axis="both", length=0)
    for boundary in [2.5, 5.5]:
        ax.axvline(boundary, color="#2d2d2d", linewidth=0.8)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.84, pad=0.015)
    colorbar.set_label("Mean standardized slope", fontsize=9)
    _save_figure(fig, "fig_09_dynamic_typology_profile_heatmap", output_dir, figure_dir)


def build_profile_maps(output_dir: Path, figure_dir: Path) -> None:
    static = _read_csv("outputs/09_morphological_typologies/typology_pca_scores.csv")
    temporal = _read_csv(
        "outputs/09_morphological_typologies/temporal_trajectory_clustering/"
        "trajectory_cluster_assignments.csv"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.1), constrained_layout=True)

    ax = axes[0]
    for typology_id, label in TYPOLOGY_ORDER:
        subset = static.loc[static["typology_id"] == typology_id]
        ax.scatter(
            subset["pca_1"],
            subset["pca_2"],
            s=28,
            color=TYPOLOGY_COLORS[typology_id],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.86,
            label=f"{typology_id} ({len(subset)})",
        )
    pca1 = float(static["pca_1_explained_variance"].iloc[0]) * 100
    pca2 = float(static["pca_2_explained_variance"].iloc[0]) * 100
    ax.set_title("A. Static morphology typology", loc="left", fontsize=12)
    ax.set_xlabel(f"Profile PCA 1 ({pca1:.1f}%)")
    ax.set_ylabel(f"Profile PCA 2 ({pca2:.1f}%)")
    ax.grid(True, alpha=0.16)
    ax.legend(fontsize=7.8, frameon=False, loc="best")

    ax = axes[1]
    for cluster_id in sorted(temporal["trajectory_cluster_id"].unique()):
        subset = temporal.loc[temporal["trajectory_cluster_id"] == cluster_id]
        ax.scatter(
            subset["trajectory_pca_1"],
            subset["trajectory_pca_2"],
            s=28,
            color=DYNAMIC_COLORS[int(cluster_id)],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.86,
            label=f"D{int(cluster_id)} ({len(subset)})",
        )
    pca1 = float(temporal["trajectory_pca_1_explained_variance"].iloc[0]) * 100
    pca2 = float(temporal["trajectory_pca_2_explained_variance"].iloc[0]) * 100
    ax.set_title("B. Temporal trajectory typology", loc="left", fontsize=12)
    ax.set_xlabel(f"Trajectory PCA 1 ({pca1:.1f}%)")
    ax.set_ylabel(f"Trajectory PCA 2 ({pca2:.1f}%)")
    ax.grid(True, alpha=0.16)
    ax.legend(fontsize=7.8, frameon=False, loc="best")

    _save_figure(fig, "fig_09_static_dynamic_profile_maps", output_dir, figure_dir)


def _load_static_projection_space() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    core = _read_csv("outputs/04_reduced_metric_core/reduced_interpretable_core_metrics.csv")
    assignments = _read_csv("outputs/09_morphological_typologies/subfield_typology_assignments.csv")
    core["subfield_id"] = core["subfield_id"].astype(str)
    assignments["subfield_id"] = assignments["subfield_id"].astype(str)
    if "metric_status" in core.columns:
        core = core.loc[core["metric_status"].isin({"completed", "completed_with_warnings"})].copy()
    columns = list(REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS)
    metadata = (
        assignments[
            [
                "subfield_id",
                "subfield_display_name",
                "field_display_name",
                "domain_display_name",
                "typology_id",
                "typology_label",
            ]
        ]
        .merge(core[["subfield_id", *columns]], on="subfield_id", how="left")
        .sort_values("subfield_id", kind="mergesort")
        .reset_index(drop=True)
    )
    matrix = RobustScaler().fit_transform(metadata[columns].to_numpy(dtype=float))
    labels = metadata["typology_id"].to_numpy()
    return metadata, matrix, labels


def _load_dynamic_projection_space() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    features = _read_csv(
        "outputs/09_morphological_typologies/temporal_trajectory_clustering/"
        "trajectory_feature_matrix.csv"
    )
    assignments = _read_csv(
        "outputs/09_morphological_typologies/temporal_trajectory_clustering/"
        "trajectory_cluster_assignments.csv"
    )
    features["subfield_id"] = features["subfield_id"].astype(str)
    assignments["subfield_id"] = assignments["subfield_id"].astype(str)
    columns = [f"linear_slope_{idx:02d}" for idx in range(8)]
    metadata = (
        assignments[
            [
                "subfield_id",
                "subfield_display_name",
                "field_display_name",
                "domain_display_name",
                "trajectory_cluster_id",
            ]
        ]
        .merge(features[["subfield_id", *columns]], on="subfield_id", how="left")
        .sort_values("subfield_id", kind="mergesort")
        .reset_index(drop=True)
    )
    metadata["trajectory_cluster_label"] = metadata["trajectory_cluster_id"].map(
        lambda value: f"D{int(value)} {DYNAMIC_LABELS[int(value)]}"
    )
    matrix = metadata[columns].to_numpy(dtype=float)
    labels = metadata["trajectory_cluster_id"].to_numpy(dtype=int)
    return metadata, matrix, labels


def _pca_projection(matrix: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    model = PCA(n_components=2, random_state=42)
    coordinates = model.fit_transform(matrix)
    variance = tuple(float(value) for value in model.explained_variance_ratio_)
    return coordinates, variance


def _pcoa_projection(matrix: np.ndarray, *, metric: str) -> tuple[np.ndarray, tuple[float, float]]:
    distances = squareform(pdist(matrix, metric=metric))
    n_obs = distances.shape[0]
    centering = np.eye(n_obs) - np.ones((n_obs, n_obs)) / n_obs
    gram = -0.5 * centering @ (distances**2) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive = np.clip(eigenvalues, a_min=0.0, a_max=None)
    coordinates = eigenvectors[:, :2] * np.sqrt(positive[:2])
    denominator = float(positive.sum())
    variance = tuple(float(value / denominator) if denominator > 0 else 0.0 for value in positive[:2])
    return coordinates, variance


def _umap_grid(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    space: str,
    metric: str,
) -> tuple[dict[tuple[int, float], np.ndarray], pd.DataFrame]:
    import umap

    embeddings: dict[tuple[int, float], np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for n_neighbors in [10, 15, 30]:
        for min_dist in [0.05, 0.15, 0.30]:
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                random_state=42,
            )
            embedding = reducer.fit_transform(matrix)
            embeddings[(n_neighbors, min_dist)] = embedding
            trust = trustworthiness(matrix, embedding, n_neighbors=10, metric=metric)
            projection_silhouette = silhouette_score(embedding, labels, metric="euclidean")
            rows.append(
                {
                    "space": space,
                    "metric": metric,
                    "n_neighbors": n_neighbors,
                    "min_dist": min_dist,
                    "trustworthiness_k10": float(trust),
                    "projection_silhouette": float(projection_silhouette),
                    "selected": n_neighbors == 15 and np.isclose(min_dist, 0.15),
                    "selection_rule": "central conservative grid setting",
                }
            )
    return embeddings, pd.DataFrame(rows)


def _projection_coordinate_frame(
    metadata: pd.DataFrame,
    projection_name: str,
    coordinates: np.ndarray,
    *,
    space: str,
    label_column: str,
    display_column: str,
    axis_1_label: str,
    axis_2_label: str,
    metric: str,
    n_neighbors: int | None = None,
    min_dist: float | None = None,
) -> pd.DataFrame:
    frame = metadata[["subfield_id", "subfield_display_name", "field_display_name", "domain_display_name"]].copy()
    frame["space"] = space
    frame["typology_id"] = metadata[label_column].to_numpy()
    frame["typology_label"] = metadata[display_column].to_numpy()
    frame["projection_method"] = projection_name
    frame["x"] = coordinates[:, 0]
    frame["y"] = coordinates[:, 1]
    frame["axis_1_label"] = axis_1_label
    frame["axis_2_label"] = axis_2_label
    frame["metric"] = metric
    frame["n_neighbors"] = n_neighbors
    frame["min_dist"] = min_dist
    return frame


def _scatter_projection(
    ax: plt.Axes,
    coordinates: np.ndarray,
    labels: np.ndarray,
    *,
    ordered_labels: list,
    color_map: dict,
    title: str,
    xlabel: str,
    ylabel: str,
    legend: bool,
) -> None:
    for label in ordered_labels:
        mask = labels == label
        label_text = str(label) if isinstance(label, str) else f"D{int(label)}"
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=18,
            color=color_map[label],
            edgecolor="white",
            linewidth=0.28,
            alpha=0.86,
            label=f"{label_text} ({int(mask.sum())})",
        )
    ax.set_title(title, loc="left", fontsize=11.5)
    ax.set_xlabel(xlabel, fontsize=9.2)
    ax.set_ylabel(ylabel, fontsize=9.2)
    ax.tick_params(labelsize=8.2)
    ax.grid(True, alpha=0.14)
    if legend:
        ax.legend(fontsize=7.0, frameon=False, loc="best", handletextpad=0.2)


def build_projection_comparison(output_dir: Path, figure_dir: Path) -> None:
    static_meta, static_matrix, static_labels = _load_static_projection_space()
    dynamic_meta, dynamic_matrix, dynamic_labels = _load_dynamic_projection_space()

    static_pca, static_pca_var = _pca_projection(static_matrix)
    dynamic_pca, dynamic_pca_var = _pca_projection(dynamic_matrix)
    static_pcoa, static_pcoa_var = _pcoa_projection(static_matrix, metric="euclidean")
    dynamic_pcoa, dynamic_pcoa_var = _pcoa_projection(dynamic_matrix, metric="correlation")
    static_umaps, static_umap_rows = _umap_grid(static_matrix, static_labels, space="static", metric="euclidean")
    dynamic_umaps, dynamic_umap_rows = _umap_grid(dynamic_matrix, dynamic_labels, space="dynamic", metric="correlation")
    selected_umap_key = (15, 0.15)
    static_umap = static_umaps[selected_umap_key]
    dynamic_umap = dynamic_umaps[selected_umap_key]

    static_rows = [
        _projection_coordinate_frame(
            static_meta,
            "PCA",
            static_pca,
            space="static",
            label_column="typology_id",
            display_column="typology_label",
            axis_1_label=f"PCA 1 ({static_pca_var[0] * 100:.1f}%)",
            axis_2_label=f"PCA 2 ({static_pca_var[1] * 100:.1f}%)",
            metric="linear",
        ),
        _projection_coordinate_frame(
            static_meta,
            "PCoA",
            static_pcoa,
            space="static",
            label_column="typology_id",
            display_column="typology_label",
            axis_1_label=f"PCoA 1 ({static_pcoa_var[0] * 100:.1f}%)",
            axis_2_label=f"PCoA 2 ({static_pcoa_var[1] * 100:.1f}%)",
            metric="euclidean",
        ),
        _projection_coordinate_frame(
            static_meta,
            "UMAP",
            static_umap,
            space="static",
            label_column="typology_id",
            display_column="typology_label",
            axis_1_label="UMAP 1",
            axis_2_label="UMAP 2",
            metric="euclidean",
            n_neighbors=15,
            min_dist=0.15,
        ),
    ]
    dynamic_rows = [
        _projection_coordinate_frame(
            dynamic_meta,
            "PCA",
            dynamic_pca,
            space="dynamic",
            label_column="trajectory_cluster_id",
            display_column="trajectory_cluster_label",
            axis_1_label=f"PCA 1 ({dynamic_pca_var[0] * 100:.1f}%)",
            axis_2_label=f"PCA 2 ({dynamic_pca_var[1] * 100:.1f}%)",
            metric="linear",
        ),
        _projection_coordinate_frame(
            dynamic_meta,
            "PCoA",
            dynamic_pcoa,
            space="dynamic",
            label_column="trajectory_cluster_id",
            display_column="trajectory_cluster_label",
            axis_1_label=f"PCoA 1 ({dynamic_pcoa_var[0] * 100:.1f}%)",
            axis_2_label=f"PCoA 2 ({dynamic_pcoa_var[1] * 100:.1f}%)",
            metric="correlation",
        ),
        _projection_coordinate_frame(
            dynamic_meta,
            "UMAP",
            dynamic_umap,
            space="dynamic",
            label_column="trajectory_cluster_id",
            display_column="trajectory_cluster_label",
            axis_1_label="UMAP 1",
            axis_2_label="UMAP 2",
            metric="correlation",
            n_neighbors=15,
            min_dist=0.15,
        ),
    ]

    static_coordinates = pd.concat(static_rows, ignore_index=True)
    dynamic_coordinates = pd.concat(dynamic_rows, ignore_index=True)
    umap_selection = pd.concat([static_umap_rows, dynamic_umap_rows], ignore_index=True)
    static_coordinates.to_csv(output_dir / "static_projection_coordinates.csv", index=False)
    dynamic_coordinates.to_csv(output_dir / "dynamic_projection_coordinates.csv", index=False)
    umap_selection.to_csv(output_dir / "umap_parameter_selection.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.0), constrained_layout=True)
    static_order = [typology_id for typology_id, _ in TYPOLOGY_ORDER]
    dynamic_order = sorted(DYNAMIC_LABELS)
    panels = [
        (
            axes[0, 0],
            static_pca,
            static_labels,
            static_order,
            TYPOLOGY_COLORS,
            "A. Static PCA",
            f"PCA 1 ({static_pca_var[0] * 100:.1f}%)",
            f"PCA 2 ({static_pca_var[1] * 100:.1f}%)",
            False,
        ),
        (
            axes[0, 1],
            static_pcoa,
            static_labels,
            static_order,
            TYPOLOGY_COLORS,
            "B. Static PCoA",
            f"PCoA 1 ({static_pcoa_var[0] * 100:.1f}%)",
            f"PCoA 2 ({static_pcoa_var[1] * 100:.1f}%)",
            False,
        ),
        (
            axes[0, 2],
            static_umap,
            static_labels,
            static_order,
            TYPOLOGY_COLORS,
            "C. Static UMAP",
            "UMAP 1",
            "UMAP 2",
            True,
        ),
        (
            axes[1, 0],
            dynamic_pca,
            dynamic_labels,
            dynamic_order,
            DYNAMIC_COLORS,
            "D. Dynamic PCA",
            f"PCA 1 ({dynamic_pca_var[0] * 100:.1f}%)",
            f"PCA 2 ({dynamic_pca_var[1] * 100:.1f}%)",
            False,
        ),
        (
            axes[1, 1],
            dynamic_pcoa,
            dynamic_labels,
            dynamic_order,
            DYNAMIC_COLORS,
            "E. Dynamic PCoA",
            f"PCoA 1 ({dynamic_pcoa_var[0] * 100:.1f}%)",
            f"PCoA 2 ({dynamic_pcoa_var[1] * 100:.1f}%)",
            False,
        ),
        (
            axes[1, 2],
            dynamic_umap,
            dynamic_labels,
            dynamic_order,
            DYNAMIC_COLORS,
            "F. Dynamic UMAP",
            "UMAP 1",
            "UMAP 2",
            True,
        ),
    ]
    for panel in panels:
        _scatter_projection(
            panel[0],
            panel[1],
            panel[2],
            ordered_labels=panel[3],
            color_map=panel[4],
            title=panel[5],
            xlabel=panel[6],
            ylabel=panel[7],
            legend=panel[8],
        )
    _save_figure(fig, "fig_09_static_dynamic_projection_comparison", output_dir, figure_dir)

    static_umap_selected = static_umap_rows.loc[static_umap_rows["selected"]].iloc[0]
    dynamic_umap_selected = dynamic_umap_rows.loc[dynamic_umap_rows["selected"]].iloc[0]
    lines = [
        "# Chapter 9 Projection Sensitivity",
        "",
        "Projection coordinates are visualization diagnostics only. Clustering assignments are unchanged.",
        "",
        "## Inputs",
        "",
        "- Static morphology: robust-scaled eleven-metric full-period profiles; labels T1--T5.",
        "- Dynamic morphology: standardized linear slopes of eight windowed structural metrics; labels D1--D4.",
        "",
        "## Projection Methods",
        "",
        f"- Static PCA variance: PC1 {static_pca_var[0] * 100:.1f}%, PC2 {static_pca_var[1] * 100:.1f}%.",
        f"- Dynamic PCA variance: PC1 {dynamic_pca_var[0] * 100:.1f}%, PC2 {dynamic_pca_var[1] * 100:.1f}%.",
        f"- Static PCoA uses Euclidean distance; first two positive-axis shares {static_pcoa_var[0] * 100:.1f}% and {static_pcoa_var[1] * 100:.1f}%.",
        f"- Dynamic PCoA uses correlation distance; first two positive-axis shares {dynamic_pcoa_var[0] * 100:.1f}% and {dynamic_pcoa_var[1] * 100:.1f}%.",
        "- UMAP grid tested n_neighbors in {10, 15, 30} and min_dist in {0.05, 0.15, 0.30}.",
        "- Selected UMAP parameters for both spaces: n_neighbors=15, min_dist=0.15, a central conservative setting rather than the maximum-separation plot.",
        "",
        "## Selected UMAP Diagnostics",
        "",
        f"- Static selected UMAP trustworthiness@10: {float(static_umap_selected['trustworthiness_k10']):.3f}; projection silhouette: {float(static_umap_selected['projection_silhouette']):.3f}.",
        f"- Dynamic selected UMAP trustworthiness@10: {float(dynamic_umap_selected['trustworthiness_k10']):.3f}; projection silhouette: {float(dynamic_umap_selected['projection_silhouette']):.3f}.",
        "",
        "UMAP is not used as quantitative clustering evidence and may exaggerate local separation.",
        "",
    ]
    (output_dir / "projection_sensitivity_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _composition_pivot(frame: pd.DataFrame, group_col: str, label_map: dict[str, str]) -> pd.DataFrame:
    pivot = frame.pivot_table(
        index=group_col,
        columns="domain_display_name",
        values="share" if "share" in frame.columns else "share_within_cluster",
        fill_value=0.0,
    )
    pivot = pivot.reindex(columns=[domain for domain in DOMAIN_ORDER if domain in pivot.columns])
    pivot.index = [label_map[str(index_value)] for index_value in pivot.index]
    return pivot


def build_domain_composition(output_dir: Path, figure_dir: Path) -> None:
    static_comp = _read_csv("outputs/09_morphological_typologies/typology_domain_composition.csv")
    dynamic_comp = _read_csv(
        "outputs/09_morphological_typologies/temporal_trajectory_clustering/"
        "trajectory_cluster_domain_composition.csv"
    )

    static_labels = {
        typology_id: f"{typology_id}\n({int(static_comp.loc[static_comp['typology_id'] == typology_id, 'typology_total'].iloc[0])})"
        for typology_id, _ in TYPOLOGY_ORDER
    }
    dynamic_labels = {}
    for cluster_id in sorted(dynamic_comp["trajectory_cluster_id"].unique()):
        total = int(dynamic_comp.loc[dynamic_comp["trajectory_cluster_id"] == cluster_id, "n_subfields"].sum())
        dynamic_labels[str(cluster_id)] = f"D{int(cluster_id)}\n({total})"

    static_pivot = _composition_pivot(static_comp, "typology_id", static_labels)
    dynamic_pivot = _composition_pivot(dynamic_comp, "trajectory_cluster_id", dynamic_labels)

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 6.8), sharex=True)
    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.08, right=0.96, hspace=0.28)
    for ax, pivot, title in [
        (axes[0], static_pivot, "A. Static morphology typologies"),
        (axes[1], dynamic_pivot, "B. Temporal trajectory typologies"),
    ]:
        left = np.zeros(len(pivot))
        x = np.arange(len(pivot))
        for domain in pivot.columns:
            values = pivot[domain].to_numpy(dtype=float)
            ax.bar(
                x,
                values,
                bottom=left,
                color=DOMAIN_COLORS.get(domain, "#999999"),
                edgecolor="white",
                linewidth=0.55,
                label=domain,
            )
            left += values
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Share")
        ax.set_title(title, loc="left", fontsize=12)
        ax.grid(True, axis="y", alpha=0.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=8.5,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
    )
    axes[1].set_xlabel("Typology (subfield count)")
    _save_figure(fig, "fig_c_static_dynamic_domain_composition", output_dir, figure_dir)


def _shorten(value: str, limit: int = 30) -> str:
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    text = str(value)
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _representatives() -> dict[int, str]:
    features = _read_csv(
        "outputs/09_morphological_typologies/temporal_trajectory_clustering/"
        "trajectory_feature_matrix.csv"
    )
    assignments = _read_csv(
        "outputs/09_morphological_typologies/temporal_trajectory_clustering/"
        "trajectory_cluster_assignments.csv"
    )
    columns = [f"linear_slope_{idx:02d}" for idx in range(8)]
    merged = assignments[["subfield_id", "trajectory_cluster_id"]].merge(
        features[["subfield_id", "subfield_display_name", *columns]],
        on="subfield_id",
        how="left",
    )
    representatives: dict[int, str] = {}
    used: set[str] = set()
    for cluster_id in sorted(merged["trajectory_cluster_id"].unique()):
        subset = merged.loc[merged["trajectory_cluster_id"] == cluster_id].copy()
        centroid = subset[columns].mean(axis=0).to_numpy(dtype=float)
        distances = np.linalg.norm(subset[columns].to_numpy(dtype=float) - centroid, axis=1)
        subset["distance_to_dynamic_centroid"] = distances
        chosen: list[str] = []
        for name in subset.sort_values("distance_to_dynamic_centroid")["subfield_display_name"]:
            name = str(name)
            if name not in used:
                chosen.append(_shorten(name, 32))
                used.add(name)
            if len(chosen) == 3:
                break
        representatives[int(cluster_id)] = "; ".join(chosen)
    return representatives


def _domain_mix() -> dict[int, str]:
    composition = _read_csv(
        "outputs/09_morphological_typologies/temporal_trajectory_clustering/"
        "trajectory_cluster_domain_composition.csv"
    )
    mixes: dict[int, str] = {}
    for cluster_id, subset in composition.groupby("trajectory_cluster_id", sort=True):
        top = subset.sort_values("n_subfields", ascending=False).head(2)
        mixes[int(cluster_id)] = "; ".join(
            f"{row.domain_display_name} {int(row.n_subfields)}" for row in top.itertuples()
        )
    return mixes


def build_dynamic_table(table_dir: Path) -> None:
    profile = _read_csv(
        "outputs/09_morphological_typologies/temporal_trajectory_clustering/"
        "trajectory_cluster_profile_summary.csv"
    ).sort_values("trajectory_cluster_id", kind="mergesort")
    representatives = _representatives()
    domain_mix = _domain_mix()
    patterns = {
        1: "centroid and kNN distances fall; kNN CV and hubness rise",
        2: "centroid, P90, and kNN distance rise; hubness and spectral complexity fall",
        3: "kNN distance rises; kNN CV and hubness fall; spectral spread rises",
        4: "PCA D80 and hubness rise; outer spread and kNN CV fall",
    }

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption[Dynamic temporal typology summary]{Selected temporal-trajectory morphological typology}",
        r"\label{tab:dynamic_typology_summary}",
        r"\scriptsize",
        r"\begingroup",
        r"\setlength{\tabcolsep}{1.2pt}",
        (
            r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.23\textwidth} "
            r">{\raggedleft\arraybackslash}p{0.05\textwidth} "
            r">{\raggedright\arraybackslash}p{0.31\textwidth} "
            r">{\raggedright\arraybackslash}p{0.25\textwidth} "
            r">{\raggedright\arraybackslash}p{0.12\textwidth}@{}}"
        ),
        r"\hline",
        (
            r"\textbf{Dynamic typology} & \textbf{$n$} & "
            r"\textbf{Dominant slope pattern} & \textbf{Representative subfields} & "
            r"\textbf{Largest domains} \\"
        ),
        r"\hline",
    ]
    for row in profile.itertuples():
        cluster_id = int(row.trajectory_cluster_id)
        label = f"D{cluster_id} {DYNAMIC_LABELS[cluster_id]}"
        fields = [
            _latex_escape(label),
            str(int(row.n_subfields)),
            _latex_escape(patterns[cluster_id]),
            _latex_escape(representatives[cluster_id]),
            _latex_escape(domain_mix[cluster_id]),
        ]
        lines.append(" & ".join(fields) + r" \\")
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\endgroup",
            r"\par\vspace{0.15cm}",
            (
                r"\footnotesize\textit{Note.} Dynamic typologies are obtained from "
                r"average-link clustering with correlation distance on standardized "
                r"linear slopes of the eight windowed structural metrics. Representative "
                r"subfields are closest to each dynamic-cluster centroid in slope space."
            ),
            r"\end{table}",
            "",
        ]
    )
    (table_dir / "tab_09_dynamic_typology_summary.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir, root=ROOT)
    figure_dir = resolve_path(args.figure_dir, root=ROOT)
    table_dir = resolve_path(args.table_dir, root=ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    stems = [
        "fig_09_static_dynamic_diagnostics",
        "fig_09_dynamic_typology_profile_heatmap",
        "fig_09_static_dynamic_projection_comparison",
        "fig_c_static_dynamic_domain_composition",
    ]
    candidate_outputs = [
        path
        for stem in stems
        for path in [
            output_dir / f"{stem}.png",
            output_dir / f"{stem}.pdf",
            figure_dir / f"{stem}.png",
            figure_dir / f"{stem}.pdf",
        ]
    ]
    candidate_outputs.extend(
        [
            table_dir / "tab_09_dynamic_typology_summary.tex",
            output_dir / "static_projection_coordinates.csv",
            output_dir / "dynamic_projection_coordinates.csv",
            output_dir / "umap_parameter_selection.csv",
            output_dir / "projection_sensitivity_summary.md",
        ]
    )
    existing = [path for path in candidate_outputs if path.exists()]
    if existing and not args.overwrite:
        rendered = ", ".join(display_path(path, root=ROOT) for path in existing[:5])
        raise FileExistsError(f"Refusing to overwrite outputs without --overwrite: {rendered}")

    build_diagnostics(output_dir, figure_dir)
    build_dynamic_heatmap(output_dir, figure_dir)
    build_projection_comparison(output_dir, figure_dir)
    build_domain_composition(output_dir, figure_dir)
    build_dynamic_table(table_dir)
    print(f"Wrote Chapter 9 artifacts to {display_path(output_dir, root=ROOT)}")


if __name__ == "__main__":
    main()
