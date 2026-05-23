from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

from src.clustering_sensitivity_common import (
    bootstrap_subsample_stability,
    canonicalize_cluster_labels,
    dataframe_to_markdown,
    error_row,
    evaluate_labels,
    labels_hierarchical_features,
    labels_hierarchical_precomputed,
    labels_kmeans,
    labels_spectral_features,
    labels_spectral_precomputed,
    normalize_distance_matrix,
    save_figure,
    standardize_matrix,
    write_json,
)
from src.morphological_typologies import REDUCED_METRICS
from src.temporal_common import display_path, ensure_outputs_do_not_exist, utc_now_iso


HYBRID_OUTPUTS: dict[str, str] = {
    "model_comparison": "hybrid_cluster_model_comparison.csv",
    "selected_solution": "hybrid_cluster_solution_selected.csv",
    "assignments": "hybrid_cluster_assignments.csv",
    "profile_summary": "hybrid_cluster_profile_summary.csv",
    "domain_composition": "hybrid_cluster_domain_composition.csv",
    "stability_summary": "hybrid_cluster_stability_summary.csv",
    "summary_md": "hybrid_clustering_summary.md",
    "summary_json": "summary.json",
}

HYBRID_FIGURE_STEMS = [
    "fig_c_hybrid_quality_by_alpha",
    "fig_c_hybrid_embedding_vs_morphology_map",
    "fig_c_hybrid_domain_composition",
]


def output_paths(output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    return {key: output_dir / filename for key, filename in HYBRID_OUTPUTS.items()}


def figure_paths(output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    paths: dict[str, Path] = {}
    for stem in HYBRID_FIGURE_STEMS:
        paths[f"{stem}_png"] = output_dir / f"{stem}.png"
        paths[f"{stem}_pdf"] = output_dir / f"{stem}.pdf"
    return paths


def existing_outputs(output_dir: str | Path) -> list[Path]:
    paths = list(output_paths(output_dir).values())
    paths.extend(figure_paths(output_dir).values())
    return [path for path in paths if path.exists()]


def ensure_hybrid_outputs(
    output_dir: str | Path,
    *,
    overwrite: bool,
    root: Path,
) -> None:
    ensure_outputs_do_not_exist(
        existing_outputs(output_dir),
        overwrite=overwrite,
        root=root,
        description="hybrid centroid morphology clustering outputs",
    )


def load_hybrid_inputs(
    *,
    typology_assignments_path: Path,
    centroid_embeddings_path: Path,
    centroid_assignments_path: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    assignments = pd.read_csv(typology_assignments_path)
    assignments["subfield_id"] = assignments["subfield_id"].astype(str)
    scaled_columns = [f"scaled_{metric}" for metric in REDUCED_METRICS]
    required = {
        "subfield_id",
        "subfield_display_name",
        "field_display_name",
        "domain_display_name",
        "typology_id",
        "typology_label",
        *scaled_columns,
    }
    missing = sorted(required - set(assignments.columns))
    if missing:
        raise ValueError(f"Typology assignments are missing columns: {missing}")

    centroid_assignments = pd.read_csv(centroid_assignments_path)
    centroid_assignments["subfield_id"] = centroid_assignments["subfield_id"].astype(str)
    if "centroid_cluster_id" not in centroid_assignments.columns:
        raise ValueError("Centroid assignments are missing centroid_cluster_id.")

    centroid_matrix = np.load(centroid_embeddings_path)
    if centroid_matrix.ndim != 2:
        raise ValueError(f"Expected 2D centroid matrix, got shape {centroid_matrix.shape}")
    if len(centroid_assignments) != centroid_matrix.shape[0]:
        raise ValueError("Centroid assignments and centroid embedding matrix row counts differ.")

    centroid_columns = pd.DataFrame(
        {
            "subfield_id": centroid_assignments["subfield_id"],
            "centroid_cluster_id": centroid_assignments["centroid_cluster_id"].astype(int),
            "centroid_row": np.arange(len(centroid_assignments)),
        }
    )
    metadata = assignments.merge(centroid_columns, on="subfield_id", how="inner")
    if len(metadata) != len(assignments):
        raise ValueError("Not all morphology subfields have centroid clustering rows.")
    metadata = metadata.sort_values("subfield_id").reset_index(drop=True)
    centroid_matrix = centroid_matrix[metadata["centroid_row"].to_numpy(dtype=int)]
    morphology_matrix = metadata[scaled_columns].to_numpy(dtype=float)
    if not np.isfinite(morphology_matrix).all() or not np.isfinite(centroid_matrix).all():
        raise ValueError("Hybrid inputs contain non-finite values.")
    return metadata, centroid_matrix.astype(float), morphology_matrix


def balanced_concat(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_scaled = standardize_matrix(first) / np.sqrt(first.shape[1])
    second_scaled = standardize_matrix(second) / np.sqrt(second.shape[1])
    return np.column_stack([first_scaled, second_scaled])


def build_hybrid_representations(
    centroid_matrix: np.ndarray,
    morphology_matrix: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    feature_representations: dict[str, np.ndarray] = {}
    distance_representations: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, Any]] = {}

    max_components = min(centroid_matrix.shape[0] - 1, centroid_matrix.shape[1])
    for requested_components in [10, 25]:
        n_components = min(requested_components, max_components)
        pca = PCA(n_components=n_components, random_state=0)
        centroid_scores = pca.fit_transform(centroid_matrix)
        name = f"concat_centroid_pca{requested_components}_morphology"
        feature_representations[name] = balanced_concat(centroid_scores, morphology_matrix)
        metadata[name] = {
            "kind": "feature_concat",
            "centroid_pca_components": n_components,
            "centroid_pca_variance": float(pca.explained_variance_ratio_.sum()),
        }

    pca_full = PCA(random_state=0).fit(centroid_matrix)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    n_80 = int(np.searchsorted(cumulative, 0.80) + 1)
    n_80 = min(max(n_80, 2), centroid_matrix.shape[1], centroid_matrix.shape[0] - 1)
    pca = PCA(n_components=n_80, random_state=0)
    centroid_scores = pca.fit_transform(centroid_matrix)
    name = "concat_centroid_pca80_morphology"
    feature_representations[name] = balanced_concat(centroid_scores, morphology_matrix)
    metadata[name] = {
        "kind": "feature_concat",
        "centroid_pca_components": n_80,
        "centroid_pca_variance": float(pca.explained_variance_ratio_.sum()),
    }

    centroid_distance = normalize_distance_matrix(cosine_distances(centroid_matrix))
    morphology_distance = normalize_distance_matrix(euclidean_distances(morphology_matrix))
    centroid_pc25 = PCA(n_components=min(25, max_components), random_state=0).fit_transform(centroid_matrix)
    centroid_pc25_scaled = standardize_matrix(centroid_pc25) / np.sqrt(centroid_pc25.shape[1])
    morphology_scaled = standardize_matrix(morphology_matrix) / np.sqrt(morphology_matrix.shape[1])
    for alpha in [0.25, 0.50, 0.75]:
        name = f"distance_alpha{alpha:.2f}".replace(".", "p")
        distance_representations[name] = alpha * centroid_distance + (1.0 - alpha) * morphology_distance
        feature_representations[f"{name}_approx_features"] = np.column_stack(
            [np.sqrt(alpha) * centroid_pc25_scaled, np.sqrt(1.0 - alpha) * morphology_scaled]
        )
        metadata[name] = {
            "kind": "weighted_distance",
            "alpha_centroid": alpha,
            "centroid_distance": "cosine",
            "morphology_distance": "euclidean",
        }
    return feature_representations, distance_representations, metadata


def explore_hybrid_models(
    feature_representations: dict[str, np.ndarray],
    distance_representations: dict[str, np.ndarray],
    representation_metadata: dict[str, dict[str, Any]],
    *,
    morph_labels: np.ndarray,
    domain_labels: np.ndarray,
    centroid_labels: np.ndarray,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    labels_by_solution: dict[str, np.ndarray] = {}
    features_by_solution: dict[str, np.ndarray] = {}
    representation_by_solution: dict[str, str] = {}

    def add_feature_candidate(representation: str, algorithm: str, metric: str, k: int, labels: np.ndarray) -> None:
        solution_id = f"hybrid_{representation}_{algorithm}_{metric}_k{k}"
        labels = canonicalize_cluster_labels(labels)
        labels_by_solution[solution_id] = labels
        features_by_solution[solution_id] = feature_representations[representation]
        representation_by_solution[solution_id] = representation
        rows.append(
            evaluate_labels(
                solution_id=solution_id,
                representation=representation,
                algorithm=algorithm,
                distance_metric=metric,
                k=k,
                labels=labels,
                feature_matrix=feature_representations[representation],
                morph_labels=morph_labels,
                domain_labels=domain_labels,
                centroid_labels=centroid_labels,
                silhouette_metric="correlation" if metric == "correlation" else "euclidean",
                extra=representation_metadata.get(representation, {}),
            )
        )

    for representation, matrix in feature_representations.items():
        if representation.endswith("_approx_features"):
            continue
        for k in range(3, 11):
            for method, metric in [
                ("ward", "euclidean"),
                ("average", "euclidean"),
                ("average", "correlation"),
            ]:
                try:
                    add_feature_candidate(
                        representation,
                        f"hierarchical_{method}",
                        metric,
                        k,
                        labels_hierarchical_features(matrix, k=k, method=method, metric=metric),
                    )
                except Exception as exc:
                    rows.append(
                        error_row(
                            solution_id=f"hybrid_{representation}_hierarchical_{method}_{metric}_k{k}",
                            representation=representation,
                            algorithm=f"hierarchical_{method}",
                            distance_metric=metric,
                            k=k,
                            exc=exc,
                            extra=representation_metadata.get(representation, {}),
                        )
                    )
            try:
                add_feature_candidate(
                    representation,
                    "kmeans",
                    "euclidean",
                    k,
                    labels_kmeans(matrix, k=k, random_seed=random_seed),
                )
            except Exception as exc:
                rows.append(
                    error_row(
                        solution_id=f"hybrid_{representation}_kmeans_euclidean_k{k}",
                        representation=representation,
                        algorithm="kmeans",
                        distance_metric="euclidean",
                        k=k,
                        exc=exc,
                        extra=representation_metadata.get(representation, {}),
                    )
                )
            try:
                add_feature_candidate(
                    representation,
                    "spectral",
                    "nearest_neighbors",
                    k,
                    labels_spectral_features(matrix, k=k, random_seed=random_seed),
                )
            except Exception as exc:
                rows.append(
                    error_row(
                        solution_id=f"hybrid_{representation}_spectral_nearest_neighbors_k{k}",
                        representation=representation,
                        algorithm="spectral",
                        distance_metric="nearest_neighbors",
                        k=k,
                        exc=exc,
                        extra=representation_metadata.get(representation, {}),
                    )
                )

    for representation, distance_matrix in distance_representations.items():
        approx_features = feature_representations[f"{representation}_approx_features"]
        for k in range(3, 11):
            for method in ["average", "complete"]:
                try:
                    labels = labels_hierarchical_precomputed(distance_matrix, k=k, method=method)
                    solution_id = f"hybrid_{representation}_hierarchical_{method}_precomputed_k{k}"
                    labels_by_solution[solution_id] = labels
                    features_by_solution[solution_id] = approx_features
                    representation_by_solution[solution_id] = representation
                    rows.append(
                        evaluate_labels(
                            solution_id=solution_id,
                            representation=representation,
                            algorithm=f"hierarchical_{method}",
                            distance_metric="hybrid_precomputed",
                            k=k,
                            labels=labels,
                            feature_matrix=approx_features,
                            morph_labels=morph_labels,
                            domain_labels=domain_labels,
                            centroid_labels=centroid_labels,
                            silhouette_matrix=distance_matrix,
                            precomputed_silhouette=True,
                            extra=representation_metadata.get(representation, {}),
                        )
                    )
                except Exception as exc:
                    rows.append(
                        error_row(
                            solution_id=f"hybrid_{representation}_hierarchical_{method}_precomputed_k{k}",
                            representation=representation,
                            algorithm=f"hierarchical_{method}",
                            distance_metric="hybrid_precomputed",
                            k=k,
                            exc=exc,
                            extra=representation_metadata.get(representation, {}),
                        )
                    )
            try:
                labels = labels_spectral_precomputed(distance_matrix, k=k, random_seed=random_seed)
                solution_id = f"hybrid_{representation}_spectral_precomputed_k{k}"
                labels_by_solution[solution_id] = labels
                features_by_solution[solution_id] = approx_features
                representation_by_solution[solution_id] = representation
                rows.append(
                    evaluate_labels(
                        solution_id=solution_id,
                        representation=representation,
                        algorithm="spectral",
                        distance_metric="hybrid_precomputed",
                        k=k,
                        labels=labels,
                        feature_matrix=approx_features,
                        morph_labels=morph_labels,
                        domain_labels=domain_labels,
                        centroid_labels=centroid_labels,
                        silhouette_matrix=distance_matrix,
                        precomputed_silhouette=True,
                        extra=representation_metadata.get(representation, {}),
                    )
                )
            except Exception as exc:
                rows.append(
                    error_row(
                        solution_id=f"hybrid_{representation}_spectral_precomputed_k{k}",
                        representation=representation,
                        algorithm="spectral",
                        distance_metric="hybrid_precomputed",
                        k=k,
                        exc=exc,
                        extra=representation_metadata.get(representation, {}),
                    )
                )
    return pd.DataFrame(rows), labels_by_solution, features_by_solution, representation_by_solution


def select_hybrid_solution(model_comparison: pd.DataFrame, *, n_subfields: int) -> pd.Series:
    ok = model_comparison.loc[model_comparison["status"].eq("ok")].copy()
    ok["silhouette"] = pd.to_numeric(ok["silhouette"], errors="coerce")
    balanced = ok.loc[
        (ok["silhouette"].notna())
        & (ok["min_cluster_size"] >= 8)
        & (ok["max_cluster_size"] <= max(30, int(round(0.72 * n_subfields))))
    ].copy()
    pool = balanced if not balanced.empty else ok.loc[ok["silhouette"].notna()].copy()
    pool["k_distance_from_5"] = (pd.to_numeric(pool["k"], errors="coerce") - 5).abs()
    return pool.sort_values(
        ["silhouette", "ami_vs_domain", "ami_vs_morphology", "k_distance_from_5"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).iloc[0]


def profile_summary(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    morphology_matrix: np.ndarray,
) -> pd.DataFrame:
    frame = metadata[["domain_display_name", "typology_id", "centroid_cluster_id"]].copy()
    frame["hybrid_cluster_id"] = labels
    for idx, metric in enumerate(REDUCED_METRICS):
        frame[f"mean_{metric}"] = morphology_matrix[:, idx]
    rows: list[dict[str, Any]] = []
    for cluster_id, group in frame.groupby("hybrid_cluster_id", sort=True):
        domain_counts = group["domain_display_name"].value_counts()
        typology_counts = group["typology_id"].value_counts()
        centroid_counts = group["centroid_cluster_id"].value_counts()
        row: dict[str, Any] = {
            "hybrid_cluster_id": int(cluster_id),
            "n_subfields": int(len(group)),
            "dominant_domain": str(domain_counts.index[0]),
            "dominant_domain_share": float(domain_counts.iloc[0] / len(group)),
            "dominant_morphology_typology": str(typology_counts.index[0]),
            "dominant_morphology_typology_share": float(typology_counts.iloc[0] / len(group)),
            "dominant_centroid_cluster": int(centroid_counts.index[0]),
            "dominant_centroid_cluster_share": float(centroid_counts.iloc[0] / len(group)),
        }
        metric_means = group[[f"mean_{metric}" for metric in REDUCED_METRICS]].mean()
        metric_means.index = REDUCED_METRICS
        row["largest_morphology_features"] = "; ".join(
            f"{idx}={value:+.2f}" for idx, value in metric_means.sort_values(key=lambda v: v.abs(), ascending=False).head(4).items()
        )
        for metric, value in metric_means.items():
            row[metric] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def domain_composition(metadata: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    frame = metadata.copy()
    frame["hybrid_cluster_id"] = labels
    counts = (
        frame.groupby(["hybrid_cluster_id", "domain_display_name"], as_index=False)
        .size()
        .rename(columns={"size": "n_subfields"})
    )
    totals = counts.groupby("hybrid_cluster_id")["n_subfields"].transform("sum")
    counts["share_within_cluster"] = counts["n_subfields"] / totals
    return counts


def plot_hybrid_quality(model_comparison: pd.DataFrame, *, output_dir: Path) -> None:
    ok = model_comparison.loc[model_comparison["status"].eq("ok")].copy()
    ok["k"] = pd.to_numeric(ok["k"], errors="coerce")
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    selected_reps = [
        "distance_alpha0p25",
        "distance_alpha0p50",
        "distance_alpha0p75",
        "concat_centroid_pca10_morphology",
        "concat_centroid_pca25_morphology",
    ]
    for representation in selected_reps:
        group = ok.loc[ok["representation"].eq(representation)]
        if group.empty:
            continue
        best = group.sort_values("silhouette", ascending=False).groupby("k", as_index=False).head(1)
        ax.plot(best["k"], best["silhouette"], marker="o", linewidth=1.6, label=representation.replace("_", " "))
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Best silhouette")
    ax.set_title("Hybrid clustering is strongest when centroid information dominates", loc="left", fontsize=12)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=7.4, frameon=False)
    fig.tight_layout()
    save_figure(fig, "fig_c_hybrid_quality_by_alpha", output_dir=output_dir)


def plot_hybrid_map(
    metadata: pd.DataFrame,
    selected_features: np.ndarray,
    labels: np.ndarray,
    selected_solution: pd.Series,
    *,
    output_dir: Path,
) -> pd.DataFrame:
    pca = PCA(n_components=2, random_state=0)
    scores = pca.fit_transform(selected_features)
    pca_frame = metadata.copy()
    pca_frame["hybrid_cluster_id"] = labels
    pca_frame["hybrid_pca_1"] = scores[:, 0]
    pca_frame["hybrid_pca_2"] = scores[:, 1]
    pca_frame["hybrid_pca_1_explained_variance"] = float(pca.explained_variance_ratio_[0])
    pca_frame["hybrid_pca_2_explained_variance"] = float(pca.explained_variance_ratio_[1])

    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    cmap = plt.get_cmap("tab10")
    for idx, cluster_id in enumerate(sorted(np.unique(labels))):
        subset = pca_frame.loc[pca_frame["hybrid_cluster_id"] == cluster_id]
        ax.scatter(
            subset["hybrid_pca_1"],
            subset["hybrid_pca_2"],
            s=34,
            color=cmap(idx % 10),
            alpha=0.82,
            edgecolor="white",
            linewidth=0.35,
            label=f"H{int(cluster_id)} ({len(subset)})",
        )
    ax.set_title(
        f"Hybrid clusters ({selected_solution['representation']}, k={int(selected_solution['k'])})",
        loc="left",
        fontsize=12,
    )
    ax.set_xlabel(f"Hybrid PCA 1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"Hybrid PCA 2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.grid(True, alpha=0.18)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save_figure(fig, "fig_c_hybrid_embedding_vs_morphology_map", output_dir=output_dir)
    return pca_frame


def plot_hybrid_domain_composition(composition: pd.DataFrame, *, output_dir: Path) -> None:
    pivot = composition.pivot_table(
        index="hybrid_cluster_id",
        columns="domain_display_name",
        values="share_within_cluster",
        fill_value=0.0,
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    left = np.zeros(len(pivot))
    colors = {
        "Life Sciences": "#2f7f5f",
        "Social Sciences": "#8b6bb5",
        "Physical Sciences": "#ba6a36",
        "Health Sciences": "#4c78a8",
    }
    for domain in pivot.columns:
        values = pivot[domain].to_numpy(dtype=float)
        ax.barh(
            np.arange(len(pivot)),
            values,
            left=left,
            color=colors.get(domain, "#999999"),
            label=domain,
            edgecolor="white",
            linewidth=0.5,
        )
        left += values
    ax.set_yticks(np.arange(len(pivot)))
    ax.set_yticklabels([f"H{int(value)}" for value in pivot.index])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share within hybrid cluster")
    ax.set_title("Hybrid cluster domain composition", loc="left", fontsize=12)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    save_figure(fig, "fig_c_hybrid_domain_composition", output_dir=output_dir)


def stability_for_selected(
    selected_solution: pd.Series,
    selected_labels: np.ndarray,
    selected_features: np.ndarray,
    representation_by_solution: dict[str, str],
    distance_representations: dict[str, np.ndarray],
    feature_representations: dict[str, np.ndarray],
    *,
    random_seed: int,
) -> pd.DataFrame:
    solution_id = str(selected_solution["solution_id"])
    representation = representation_by_solution[solution_id]
    algorithm = str(selected_solution["algorithm"])
    metric = str(selected_solution["distance_metric"])
    k = int(selected_solution["k"])

    def label_function(indices: np.ndarray) -> np.ndarray:
        if representation in distance_representations and metric == "hybrid_precomputed":
            subset_distance = distance_representations[representation][np.ix_(indices, indices)]
            if algorithm.startswith("hierarchical_"):
                method = algorithm.removeprefix("hierarchical_")
                return labels_hierarchical_precomputed(subset_distance, k=k, method=method)
            if algorithm == "spectral":
                return labels_spectral_precomputed(subset_distance, k=k, random_seed=random_seed)
        subset = selected_features[indices]
        if algorithm.startswith("hierarchical_"):
            method = algorithm.removeprefix("hierarchical_")
            return labels_hierarchical_features(subset, k=k, method=method, metric=metric)
        if algorithm == "kmeans":
            return labels_kmeans(subset, k=k, random_seed=random_seed)
        if algorithm == "spectral":
            return labels_spectral_features(subset, k=k, random_seed=random_seed)
        raise ValueError(f"Unsupported selected algorithm for stability: {algorithm}")

    return bootstrap_subsample_stability(
        selected_features,
        selected_labels,
        label_function=label_function,
        random_seed=random_seed,
        n_iterations=100,
    )


def render_summary(
    *,
    selected_solution: pd.Series,
    model_comparison: pd.DataFrame,
    stability: pd.DataFrame,
) -> str:
    top = (
        model_comparison.loc[model_comparison["status"].eq("ok")]
        .sort_values("silhouette", ascending=False)
        .head(10)
    )
    lines = [
        "# Hybrid Centroid-Morphology Clustering Sensitivity",
        "",
        f"Generated: {utc_now_iso()}",
        "",
        "## Selected Diagnostic Solution",
        "",
        f"- Solution: `{selected_solution['solution_id']}`",
        f"- Representation: {selected_solution['representation']}",
        f"- Silhouette: {float(selected_solution['silhouette']):.3f}",
        f"- Cluster sizes: {selected_solution['cluster_sizes_sorted']}",
        f"- AMI vs morphology typology: {float(selected_solution['ami_vs_morphology']):.3f}",
        f"- AMI vs centroid clusters: {float(selected_solution['ami_vs_centroid']):.3f}",
        f"- AMI vs OpenAlex domains: {float(selected_solution['ami_vs_domain']):.3f}",
        f"- Domain purity: {float(selected_solution['domain_purity']):.3f}",
        f"- Mean subsample ARI: {float(stability['adjusted_rand_index'].mean()):.3f}",
        "",
        "## Interpretation",
        "",
        "Hybrid clustering improves separation relative to morphology-only profiles when semantic-location information is included. The strongest solutions align more with centroid/domain structure than with morphology, so this is not a replacement morphology typology.",
        "",
        "## Top Candidates",
        "",
        dataframe_to_markdown(
            top[
                [
                    "solution_id",
                    "silhouette",
                    "min_cluster_size",
                    "max_cluster_size",
                    "ami_vs_morphology",
                    "ami_vs_centroid",
                    "ami_vs_domain",
                    "domain_purity",
                ]
            ]
        ),
    ]
    return "\n".join(lines) + "\n"


def build_hybrid_clustering_outputs(
    *,
    typology_assignments_path: Path,
    centroid_embeddings_path: Path,
    centroid_assignments_path: Path,
    output_dir: Path,
    root: Path,
    random_seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata, centroid_matrix, morphology_matrix = load_hybrid_inputs(
        typology_assignments_path=typology_assignments_path,
        centroid_embeddings_path=centroid_embeddings_path,
        centroid_assignments_path=centroid_assignments_path,
    )
    feature_representations, distance_representations, representation_metadata = build_hybrid_representations(
        centroid_matrix,
        morphology_matrix,
    )
    morph_labels = metadata["typology_id"].to_numpy()
    domain_labels = metadata["domain_display_name"].to_numpy()
    centroid_labels = metadata["centroid_cluster_id"].to_numpy()
    model_comparison, labels_by_solution, features_by_solution, representation_by_solution = explore_hybrid_models(
        feature_representations,
        distance_representations,
        representation_metadata,
        morph_labels=morph_labels,
        domain_labels=domain_labels,
        centroid_labels=centroid_labels,
        random_seed=random_seed,
    )
    selected_solution = select_hybrid_solution(model_comparison, n_subfields=len(metadata))
    selected_solution_id = str(selected_solution["solution_id"])
    selected_labels = labels_by_solution[selected_solution_id]
    selected_features = features_by_solution[selected_solution_id]

    pca_frame = plot_hybrid_map(
        metadata,
        selected_features,
        selected_labels,
        selected_solution,
        output_dir=output_dir,
    )
    profile = profile_summary(metadata, selected_labels, morphology_matrix)
    composition = domain_composition(metadata, selected_labels)
    stability = stability_for_selected(
        selected_solution,
        selected_labels,
        selected_features,
        representation_by_solution,
        distance_representations,
        feature_representations,
        random_seed=random_seed,
    )
    plot_hybrid_quality(model_comparison, output_dir=output_dir)
    plot_hybrid_domain_composition(composition, output_dir=output_dir)

    assignments = metadata[
        [
            "subfield_id",
            "subfield_display_name",
            "field_display_name",
            "domain_display_name",
            "typology_id",
            "typology_label",
            "centroid_cluster_id",
        ]
    ].copy()
    assignments["hybrid_cluster_id"] = selected_labels
    assignments["hybrid_cluster_label"] = [f"H{int(value)}" for value in selected_labels]
    assignments = assignments.merge(
        pca_frame[
            [
                "subfield_id",
                "hybrid_pca_1",
                "hybrid_pca_2",
                "hybrid_pca_1_explained_variance",
                "hybrid_pca_2_explained_variance",
            ]
        ],
        on="subfield_id",
        how="left",
    )

    paths = output_paths(output_dir)
    model_comparison.to_csv(paths["model_comparison"], index=False)
    pd.DataFrame([selected_solution]).to_csv(paths["selected_solution"], index=False)
    assignments.to_csv(paths["assignments"], index=False)
    profile.to_csv(paths["profile_summary"], index=False)
    composition.to_csv(paths["domain_composition"], index=False)
    stability.to_csv(paths["stability_summary"], index=False)
    paths["summary_md"].write_text(
        render_summary(
            selected_solution=selected_solution,
            model_comparison=model_comparison,
            stability=stability,
        ),
        encoding="utf-8",
    )
    summary = {
        "generated_at": utc_now_iso(),
        "n_subfields": int(len(metadata)),
        "selected_solution": selected_solution.to_dict(),
        "stability_mean_ari": float(stability["adjusted_rand_index"].mean()),
        "stability_mean_ami": float(stability["adjusted_mutual_information"].mean()),
        "input_paths": {
            "typology_assignments": display_path(typology_assignments_path, root=root),
            "centroid_embeddings": display_path(centroid_embeddings_path, root=root),
            "centroid_assignments": display_path(centroid_assignments_path, root=root),
        },
        "representations": representation_metadata,
        "output_paths": {key: display_path(path, root=root) for key, path in paths.items()},
        "figure_paths": {key: display_path(path, root=root) for key, path in figure_paths(output_dir).items()},
    }
    write_json(paths["summary_json"], summary)
    return summary
