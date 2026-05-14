from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, Isomap, trustworthiness
from sklearn.metrics import silhouette_score

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


METHOD_ORDER = ["pca", "mds", "tsne", "umap", "isomap", "phate"]
METHOD_DISPLAY_NAMES = {
    "pca": "PCA",
    "mds": "MDS",
    "tsne": "t-SNE",
    "umap": "UMAP",
    "isomap": "Isomap",
    "phate": "PHATE",
}
SPACE_DISPLAY_NAMES = {
    "umap_only": "UMAP-only metrics",
    "embedding_only": "Embedding-only metrics",
    "combined": "Combined metrics",
}


@dataclass(frozen=True)
class DimensionalityReductionConfig:
    random_state: int = 42
    n_neighbors: int = 15
    tsne_perplexity: float = 30.0
    mds_n_init: int = 4
    mds_max_iter: int = 1000
    phate_decay: int = 40


@dataclass(frozen=True)
class UMAPTuningConfig:
    random_state: int = 42
    n_neighbors_values: tuple[int, ...] = (5, 10, 15, 30, 50)
    min_dist_values: tuple[float, ...] = (0.0, 0.05, 0.1, 0.3, 0.6)


def effective_neighbors(n_rows: int, requested: int) -> int:
    if n_rows < 3:
        raise ValueError("dimensionality reduction comparison requires at least three rows")
    return min(max(2, requested), n_rows - 1)


def effective_tsne_perplexity(n_rows: int, requested: float) -> float:
    upper = max(2.0, min(float(requested), (n_rows - 1) / 3))
    return min(upper, n_rows - 1.0)


def fit_projection(
    features: pd.DataFrame,
    *,
    method: str,
    config: DimensionalityReductionConfig,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    if method not in METHOD_ORDER:
        raise ValueError(f"unknown dimensionality-reduction method: {method}")
    n_rows = len(features)
    n_neighbors = effective_neighbors(n_rows, config.n_neighbors)
    values = features.to_numpy(dtype=float)
    params: dict[str, Any] = {
        "n_neighbors": n_neighbors,
        "random_state": config.random_state,
    }
    warning = ""

    try:
        if method == "pca":
            model = PCA(n_components=2, random_state=config.random_state)
            coords = model.fit_transform(values)
            params["explained_variance_ratio_1"] = float(model.explained_variance_ratio_[0])
            params["explained_variance_ratio_2"] = float(model.explained_variance_ratio_[1])
        elif method == "mds":
            model = MDS(
                n_components=2,
                metric_mds=True,
                metric="euclidean",
                n_init=config.mds_n_init,
                max_iter=config.mds_max_iter,
                init="random",
                eps=1e-6,
                random_state=config.random_state,
                n_jobs=1,
                normalized_stress="auto",
            )
            coords = model.fit_transform(values)
            params["n_init"] = config.mds_n_init
            params["max_iter"] = config.mds_max_iter
        elif method == "tsne":
            perplexity = effective_tsne_perplexity(n_rows, config.tsne_perplexity)
            model = TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                random_state=config.random_state,
                max_iter=1000,
            )
            coords = model.fit_transform(values)
            params["perplexity"] = perplexity
            params["init"] = "pca"
            params["learning_rate"] = "auto"
        elif method == "umap":
            import umap

            model = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=0.1,
                metric="euclidean",
                random_state=config.random_state,
            )
            coords = model.fit_transform(values)
            params["min_dist"] = 0.1
            params["metric"] = "euclidean"
        elif method == "isomap":
            model = Isomap(n_components=2, n_neighbors=n_neighbors)
            coords = model.fit_transform(values)
        else:
            import phate

            model = phate.PHATE(
                n_components=2,
                knn=n_neighbors,
                decay=config.phate_decay,
                random_state=config.random_state,
                n_jobs=1,
                verbose=False,
            )
            coords = model.fit_transform(values)
            params["knn"] = n_neighbors
            params["decay"] = config.phate_decay
    except Exception as exc:
        coords = np.full((n_rows, 2), np.nan, dtype=float)
        warning = f"{METHOD_DISPLAY_NAMES[method]} projection failed: {exc}"
    return np.asarray(coords, dtype=float), warning, params


def fit_umap_tuned_projection(
    features: pd.DataFrame,
    *,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    effective = effective_neighbors(len(features), n_neighbors)
    params = {
        "n_neighbors": effective,
        "requested_n_neighbors": int(n_neighbors),
        "min_dist": float(min_dist),
        "metric": "euclidean",
        "random_state": int(random_state),
    }
    try:
        import umap

        model = umap.UMAP(
            n_components=2,
            n_neighbors=effective,
            min_dist=float(min_dist),
            metric="euclidean",
            random_state=random_state,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="n_jobs value 1 overridden.*",
                category=UserWarning,
            )
            coords = model.fit_transform(features.to_numpy(dtype=float))
        warning = ""
    except Exception as exc:
        coords = np.full((len(features), 2), np.nan, dtype=float)
        warning = f"UMAP projection failed for n_neighbors={n_neighbors}, min_dist={min_dist}: {exc}"
    return np.asarray(coords, dtype=float), warning, params


def projection_coordinate_frame(
    *,
    space: str,
    method: str,
    metadata: pd.DataFrame,
    coordinates: np.ndarray,
    cluster_column: str,
    warning: str = "",
) -> pd.DataFrame:
    required = {
        "subfield_id",
        "subfield_display_name",
        "field_display_name",
        "domain_display_name",
        cluster_column,
    }
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(
            "metadata for projection output is missing columns: "
            f"{', '.join(sorted(missing))}"
        )
    coords = np.asarray(coordinates, dtype=float)
    if coords.shape != (len(metadata), 2):
        raise ValueError(
            f"coordinates must have shape ({len(metadata)}, 2), got {coords.shape}"
        )
    frame = metadata[
        [
            "subfield_id",
            "subfield_display_name",
            "field_display_name",
            "domain_display_name",
            cluster_column,
        ]
    ].copy()
    frame["feature_set"] = space
    frame["method"] = method
    frame["method_display_name"] = METHOD_DISPLAY_NAMES[method]
    frame["x"] = coords[:, 0]
    frame["y"] = coords[:, 1]
    frame["cluster_label"] = frame[cluster_column]
    frame["projection_warning"] = warning
    ordered = [
        "feature_set",
        "method",
        "method_display_name",
        "subfield_id",
        "subfield_display_name",
        "field_display_name",
        "domain_display_name",
        "cluster_label",
        cluster_column,
        "x",
        "y",
        "projection_warning",
    ]
    return frame[ordered]


def umap_tuning_coordinate_frame(
    *,
    space: str,
    metadata: pd.DataFrame,
    coordinates: np.ndarray,
    cluster_column: str,
    n_neighbors: int,
    min_dist: float,
    warning: str = "",
) -> pd.DataFrame:
    base = projection_coordinate_frame(
        space=space,
        method="umap",
        metadata=metadata,
        coordinates=coordinates,
        cluster_column=cluster_column,
        warning=warning,
    )
    base["n_neighbors"] = int(n_neighbors)
    base["min_dist"] = float(min_dist)
    ordered = [
        "feature_set",
        "n_neighbors",
        "min_dist",
        "subfield_id",
        "subfield_display_name",
        "field_display_name",
        "domain_display_name",
        "cluster_label",
        cluster_column,
        "x",
        "y",
        "projection_warning",
    ]
    return base[ordered]


def projection_quality_rows(
    *,
    space: str,
    method: str,
    features: pd.DataFrame,
    coordinates: np.ndarray,
    labels: pd.Series,
    warning: str,
    params: dict[str, Any],
    config: DimensionalityReductionConfig,
) -> dict[str, Any]:
    coords = np.asarray(coordinates, dtype=float)
    finite = np.isfinite(coords).all(axis=1)
    n_neighbors = effective_neighbors(len(features), config.n_neighbors)
    trust = np.nan
    silhouette = np.nan
    if finite.all() and len(np.unique(labels)) > 1:
        try:
            trust = float(
                trustworthiness(
                    features.to_numpy(dtype=float),
                    coords,
                    n_neighbors=n_neighbors,
                )
            )
        except Exception:
            trust = np.nan
        try:
            silhouette = float(silhouette_score(coords, labels))
        except Exception:
            silhouette = np.nan
    return {
        "feature_set": space,
        "method": method,
        "method_display_name": METHOD_DISPLAY_NAMES[method],
        "n_subfields": int(len(features)),
        "n_features": int(features.shape[1]),
        "trustworthiness": trust,
        "cluster_silhouette_2d": silhouette,
        "warning": warning,
        "params_json": params,
    }


def umap_tuning_quality_row(
    *,
    space: str,
    features: pd.DataFrame,
    coordinates: np.ndarray,
    labels: pd.Series,
    n_neighbors: int,
    min_dist: float,
    warning: str,
) -> dict[str, Any]:
    coords = np.asarray(coordinates, dtype=float)
    finite = np.isfinite(coords).all(axis=1)
    effective = effective_neighbors(len(features), n_neighbors)
    trust = np.nan
    silhouette = np.nan
    if finite.all() and len(np.unique(labels)) > 1:
        try:
            trust = float(
                trustworthiness(
                    features.to_numpy(dtype=float),
                    coords,
                    n_neighbors=effective,
                )
            )
        except Exception:
            trust = np.nan
        try:
            silhouette = float(silhouette_score(coords, labels))
        except Exception:
            silhouette = np.nan
    return {
        "feature_set": space,
        "n_neighbors": int(n_neighbors),
        "effective_n_neighbors": int(effective),
        "min_dist": float(min_dist),
        "n_subfields": int(len(features)),
        "n_features": int(features.shape[1]),
        "trustworthiness": trust,
        "cluster_silhouette_2d": silhouette,
        "warning": warning,
    }


def _cluster_sort_key(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _short_label(value: Any, *, max_chars: int = 28) -> str:
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def representative_ids(representatives: pd.DataFrame, *, labels_per_cluster: int = 1) -> set[str]:
    if representatives.empty or "subfield_id" not in representatives.columns:
        return set()
    reps = representatives.copy()
    if "representative_rank" in reps.columns:
        rank_column = "representative_rank"
    elif "rank" in reps.columns:
        rank_column = "rank"
    else:
        rank_column = ""
    if rank_column:
        reps[rank_column] = pd.to_numeric(reps[rank_column], errors="coerce")
        reps = reps.dropna(subset=[rank_column]).sort_values(["cluster_id", rank_column])
        reps = reps.groupby("cluster_id", as_index=False).head(labels_per_cluster)
    else:
        reps = reps.groupby("cluster_id", as_index=False).head(labels_per_cluster)
    return set(reps["subfield_id"].astype(str))


def _label_rows(frame: pd.DataFrame, label_ids: set[str]) -> pd.DataFrame:
    if not label_ids:
        return frame.iloc[0:0].copy()
    labels = frame.loc[frame["subfield_id"].astype(str).isin(label_ids)].copy()
    labels = labels.sort_values(["cluster_label", "subfield_display_name"])
    return labels


def plot_projection_grid(
    *,
    space: str,
    coordinates: pd.DataFrame,
    representatives: pd.DataFrame,
    output_path: str | Path,
    labels_per_cluster: int = 1,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2), dpi=170)
    axes_flat = axes.ravel()
    clusters = sorted(coordinates["cluster_label"].dropna().unique(), key=_cluster_sort_key)
    domains = sorted(coordinates["domain_display_name"].dropna().unique())
    colors = plt.get_cmap("tab10").colors
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    color_by_cluster = {
        cluster_id: colors[idx % len(colors)]
        for idx, cluster_id in enumerate(clusters)
    }
    marker_by_domain = {
        domain: markers[idx % len(markers)]
        for idx, domain in enumerate(domains)
    }
    label_ids = representative_ids(representatives, labels_per_cluster=labels_per_cluster)

    for ax, method in zip(axes_flat, METHOD_ORDER):
        method_frame = coordinates.loc[coordinates["method"] == method].copy()
        warning_values = method_frame["projection_warning"].dropna().unique()
        warning = str(warning_values[0]) if len(warning_values) else ""
        if warning or method_frame[["x", "y"]].isna().any().any():
            ax.text(
                0.5,
                0.5,
                warning or "Projection unavailable",
                ha="center",
                va="center",
                wrap=True,
                fontsize=8,
            )
            ax.set_axis_off()
            ax.set_title(METHOD_DISPLAY_NAMES[method])
            continue
        for cluster_id in clusters:
            cluster_group = method_frame.loc[method_frame["cluster_label"] == cluster_id]
            for domain in domains:
                group = cluster_group.loc[cluster_group["domain_display_name"] == domain]
                if group.empty:
                    continue
                ax.scatter(
                    group["x"],
                    group["y"],
                    s=25,
                    alpha=0.82,
                    color=color_by_cluster[cluster_id],
                    marker=marker_by_domain[domain],
                    edgecolors="white",
                    linewidths=0.35,
                )
        labels = _label_rows(method_frame, label_ids)
        for idx, row in enumerate(labels.itertuples(index=False)):
            offset = (5, 4) if idx % 2 == 0 else (5, -8)
            ax.annotate(
                _short_label(row.subfield_display_name),
                (row.x, row.y),
                xytext=offset,
                textcoords="offset points",
                fontsize=6.2,
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "white",
                    "edgecolor": "#dddddd",
                    "alpha": 0.78,
                    "linewidth": 0.3,
                },
            )
        ax.set_title(METHOD_DISPLAY_NAMES[method], fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(color="#eeeeee", linewidth=0.5)

    cluster_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color_by_cluster[cluster_id],
            markeredgecolor="white",
            markersize=7,
            label=f"C{cluster_id}",
        )
        for cluster_id in clusters
    ]
    domain_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_by_domain[domain],
            color="#555555",
            linestyle="none",
            markerfacecolor="none",
            markersize=7,
            label=domain,
        )
        for domain in domains
    ]
    fig.legend(
        handles=cluster_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=max(1, len(cluster_handles)),
        frameon=False,
        title="Ward clusters for this feature set",
    )
    fig.legend(
        handles=domain_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=max(1, min(4, len(domain_handles))),
        frameon=False,
        title="Domain marker",
    )
    title = SPACE_DISPLAY_NAMES.get(space, space)
    fig.suptitle(f"Dimensionality-Reduction Comparison: {title}", y=0.995, fontsize=14)
    fig.tight_layout(rect=(0.02, 0.07, 0.98, 0.93))
    fig.savefig(output_path)
    plt.close(fig)


def plot_umap_tuning_grid(
    *,
    space: str,
    coordinates: pd.DataFrame,
    quality: pd.DataFrame,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    neighbor_values = sorted(coordinates["n_neighbors"].dropna().unique())
    min_dist_values = sorted(coordinates["min_dist"].dropna().unique())
    if not neighbor_values or not min_dist_values:
        fig, ax = plt.subplots(figsize=(6, 3.5), dpi=160)
        ax.text(0.5, 0.5, "No UMAP tuning coordinates available", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        return

    fig_width = max(13.5, 3.15 * len(min_dist_values))
    fig_height = max(11.5, 2.75 * len(neighbor_values))
    fig, axes = plt.subplots(
        len(neighbor_values),
        len(min_dist_values),
        figsize=(fig_width, fig_height),
        dpi=160,
        squeeze=False,
    )
    clusters = sorted(coordinates["cluster_label"].dropna().unique(), key=_cluster_sort_key)
    colors = plt.get_cmap("tab10").colors
    color_by_cluster = {
        cluster_id: colors[idx % len(colors)]
        for idx, cluster_id in enumerate(clusters)
    }
    quality_lookup = {
        (row.n_neighbors, row.min_dist): row
        for row in quality.itertuples(index=False)
    }

    for row_idx, n_neighbors in enumerate(neighbor_values):
        for col_idx, min_dist in enumerate(min_dist_values):
            ax = axes[row_idx, col_idx]
            panel = coordinates.loc[
                (coordinates["n_neighbors"] == n_neighbors)
                & (np.isclose(coordinates["min_dist"], min_dist))
            ]
            warning_values = panel["projection_warning"].dropna().unique()
            warning = str(warning_values[0]) if len(warning_values) else ""
            if warning or panel[["x", "y"]].isna().any().any():
                ax.text(
                    0.5,
                    0.5,
                    warning or "Projection unavailable",
                    ha="center",
                    va="center",
                    wrap=True,
                    fontsize=7,
                )
                ax.set_axis_off()
            else:
                for cluster_id in clusters:
                    group = panel.loc[panel["cluster_label"] == cluster_id]
                    if group.empty:
                        continue
                    ax.scatter(
                        group["x"],
                        group["y"],
                        s=18,
                        alpha=0.82,
                        color=color_by_cluster[cluster_id],
                        edgecolors="white",
                        linewidths=0.25,
                    )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(color="#eeeeee", linewidth=0.45)
            quality_row = quality_lookup.get((n_neighbors, min_dist))
            if quality_row is not None:
                trust = getattr(quality_row, "trustworthiness")
                sil = getattr(quality_row, "cluster_silhouette_2d")
                subtitle = (
                    f"T={trust:.3f}, S={sil:.3f}"
                    if pd.notna(trust) and pd.notna(sil)
                    else "T/S unavailable"
                )
            else:
                subtitle = ""
            ax.set_title(
                f"k={int(n_neighbors)}, min_dist={min_dist:g}\n{subtitle}",
                fontsize=8.2,
            )

    cluster_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color_by_cluster[cluster_id],
            markeredgecolor="white",
            markersize=7,
            label=f"C{cluster_id}",
        )
        for cluster_id in clusters
    ]
    fig.legend(
        handles=cluster_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=max(1, len(cluster_handles)),
        frameon=False,
        title="Ward clusters for this feature set",
    )
    title = SPACE_DISPLAY_NAMES.get(space, space)
    fig.suptitle(f"UMAP Hyperparameter Tuning: {title}", y=0.995, fontsize=14)
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.97))
    fig.savefig(output_path)
    plt.close(fig)


def umap_tuning_summary_markdown(
    *,
    quality: pd.DataFrame,
    warnings: list[str],
    config: UMAPTuningConfig,
) -> str:
    lines = [
        "# UMAP Hyperparameter Tuning",
        "",
        "This stage sweeps UMAP `n_neighbors` and `min_dist` on the same "
        "preprocessed feature matrices used in the dimensionality-reduction "
        "comparison. The goal is sensitivity analysis, not aesthetic overfitting: "
        "settings are compared using trustworthiness and 2D cluster silhouette, "
        "then inspected visually in the grid figures.",
        "",
        "## Grid",
        "",
        f"- Random state: {config.random_state}",
        f"- n_neighbors values: {', '.join(str(value) for value in config.n_neighbors_values)}",
        f"- min_dist values: {', '.join(str(value) for value in config.min_dist_values)}",
        "",
        "## Best Settings By Diagnostic",
        "",
    ]
    if quality.empty:
        lines.append("No UMAP tuning diagnostics were available.")
    else:
        for space in ["umap_only", "embedding_only", "combined"]:
            group = quality.loc[quality["feature_set"] == space].copy()
            if group.empty:
                continue
            best_trust = group.sort_values("trustworthiness", ascending=False).iloc[0]
            best_silhouette = group.sort_values("cluster_silhouette_2d", ascending=False).iloc[0]
            lines.append(
                f"- {SPACE_DISPLAY_NAMES.get(space, space)}: best trustworthiness at "
                f"k={int(best_trust.n_neighbors)}, min_dist={best_trust.min_dist:g} "
                f"({best_trust.trustworthiness:.3f}); best cluster silhouette at "
                f"k={int(best_silhouette.n_neighbors)}, min_dist={best_silhouette.min_dist:g} "
                f"({best_silhouette.cluster_silhouette_2d:.3f})."
            )
    lines.extend(
        [
            "",
            "## Reading Notes",
            "",
            "- Lower `n_neighbors` emphasizes very local neighborhoods and can create "
            "more apparent islands.",
            "- Higher `n_neighbors` emphasizes broader global organization and can "
            "merge local islands into smoother gradients.",
            "- Lower `min_dist` allows tighter clusters; higher values spread points "
            "more evenly.",
            "- Prefer settings that show stable structure across nearby parameter "
            "values rather than a single visually dramatic panel.",
            "",
        ]
    )
    if warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")
    return "\n".join(lines)


def comparison_summary_markdown(
    *,
    quality: pd.DataFrame,
    warnings: list[str],
    config: DimensionalityReductionConfig,
) -> str:
    lines = [
        "# Dimensionality-Reduction Comparison",
        "",
        "This stage compares PCA, MDS, t-SNE, UMAP, Isomap, and PHATE on the same "
        "preprocessed feature matrices used by the metric-clustering pipeline. "
        "For each metric family, retained metrics are median-imputed, scaled, and "
        "reduced by PCA to the configured variance threshold before the comparison "
        "methods are applied. The combined space uses the existing block-PCA "
        "representation that balances projected morphology and embedding-space "
        "metrics before visualization.",
        "",
        "Cluster colors refer to the Ward `k=5` solution for the same feature set. "
        "Marker shape indicates OpenAlex domain. Projection figures are visual "
        "diagnostics, not evidence of definitive disciplinary classes.",
        "",
        "## Parameters",
        "",
        f"- Random state: {config.random_state}",
        f"- Neighbor count for UMAP, Isomap, PHATE, and trustworthiness: {config.n_neighbors}",
        f"- t-SNE perplexity target: {config.tsne_perplexity}",
        f"- MDS initializations: {config.mds_n_init}",
        f"- PHATE decay: {config.phate_decay}",
        "",
        "## Quantitative Diagnostics",
        "",
    ]
    if quality.empty:
        lines.append("No quality diagnostics were available.")
    else:
        for space in ["umap_only", "embedding_only", "combined"]:
            group = quality.loc[quality["feature_set"] == space].copy()
            if group.empty:
                continue
            best_trust = group.sort_values("trustworthiness", ascending=False).head(1)
            best_silhouette = group.sort_values("cluster_silhouette_2d", ascending=False).head(1)
            trust_text = "unavailable"
            silhouette_text = "unavailable"
            if not best_trust.empty and pd.notna(best_trust.iloc[0]["trustworthiness"]):
                trust_text = (
                    f"{best_trust.iloc[0]['method_display_name']} "
                    f"({best_trust.iloc[0]['trustworthiness']:.3f})"
                )
            if not best_silhouette.empty and pd.notna(best_silhouette.iloc[0]["cluster_silhouette_2d"]):
                silhouette_text = (
                    f"{best_silhouette.iloc[0]['method_display_name']} "
                    f"({best_silhouette.iloc[0]['cluster_silhouette_2d']:.3f})"
                )
            lines.append(
                f"- {SPACE_DISPLAY_NAMES.get(space, space)}: highest trustworthiness = "
                f"{trust_text}; strongest 2D cluster separation = {silhouette_text}."
            )
    lines.extend(
        [
            "",
            "## Visual Reading Guide",
            "",
            "- PCA and MDS are conservative baselines: useful when broad linear or "
            "distance-preserving structure is enough, but often less separated.",
            "- t-SNE, UMAP, and PHATE emphasize local neighborhoods more strongly; "
            "clearer islands in these panels should be treated as exploratory.",
            "- Isomap is useful for checking whether a continuous manifold explains "
            "the metric space better than sharply separated groups.",
            "- Compare methods by looking for recurring neighborhoods across panels, "
            "not by picking the plot that looks most separated.",
            "",
        ]
    )
    if warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")
    return "\n".join(lines)
