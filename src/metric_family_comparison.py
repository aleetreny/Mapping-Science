from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from src.embedding_space_metrics import CORE_EMBEDDING_METRIC_COLUMNS
from src.morphology_metrics import CORE_METRIC_COLUMNS_V2

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


VALID_STATUSES = {"completed", "completed_with_warnings"}

ANALOGUE_METRIC_PAIRS = {
    "knn_median_distance": "embedding_knn_median_distance",
    "knn_distance_cv": "embedding_knn_distance_cv",
    "radial_tail_index": "embedding_tail_index_p90_median",
    "centroid_drift_early_late": "embedding_centroid_drift_early_late",
    "annual_centroid_path_length": "embedding_annual_centroid_path_length",
    "directionality_ratio": "embedding_directionality_ratio",
    "radial_expansion_slope": "embedding_radial_expansion_slope",
    "mst_gap_index": "embedding_graph_edge_distance_p90",
    "anisotropy_ratio": "embedding_pca_first_component_share",
}


def filter_valid_metric_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "metric_status" not in frame.columns:
        return frame.copy()
    return frame.loc[frame["metric_status"].isin(VALID_STATUSES)].copy()


def available_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def join_metric_families(
    umap_metrics: pd.DataFrame,
    embedding_metrics: pd.DataFrame,
) -> pd.DataFrame:
    if "subfield_id" not in umap_metrics.columns or "subfield_id" not in embedding_metrics.columns:
        raise ValueError("both metric tables must contain subfield_id")
    left = filter_valid_metric_rows(umap_metrics)
    right = filter_valid_metric_rows(embedding_metrics)
    joined = left.merge(
        right,
        on="subfield_id",
        how="inner",
        suffixes=("_umap", "_embedding"),
    )
    return joined


def pairwise_correlations(
    joined: pd.DataFrame,
    umap_columns: list[str] | None = None,
    embedding_columns: list[str] | None = None,
    *,
    method: str,
) -> pd.DataFrame:
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be pearson or spearman")
    umap_columns = available_columns(joined, umap_columns or CORE_METRIC_COLUMNS_V2)
    embedding_columns = available_columns(
        joined,
        embedding_columns or CORE_EMBEDDING_METRIC_COLUMNS,
    )
    rows: list[dict[str, Any]] = []
    for umap_metric in umap_columns:
        left = pd.to_numeric(joined[umap_metric], errors="coerce")
        for embedding_metric in embedding_columns:
            right = pd.to_numeric(joined[embedding_metric], errors="coerce")
            valid = left.notna() & right.notna()
            n_valid = int(valid.sum())
            if n_valid < 2:
                correlation = np.nan
            else:
                correlation = left.loc[valid].corr(right.loc[valid], method=method)
            rows.append(
                {
                    "umap_metric": umap_metric,
                    "embedding_metric": embedding_metric,
                    "method": method,
                    "correlation": float(correlation)
                    if pd.notna(correlation)
                    else np.nan,
                    "abs_correlation": abs(float(correlation))
                    if pd.notna(correlation)
                    else np.nan,
                    "n_valid_pairwise": n_valid,
                }
            )
    return pd.DataFrame(rows)


def analogue_pair_correlations(
    joined: pd.DataFrame,
    *,
    method: str = "spearman",
    analogue_pairs: dict[str, str] | None = None,
) -> pd.DataFrame:
    pairs = analogue_pairs or ANALOGUE_METRIC_PAIRS
    rows: list[dict[str, Any]] = []
    for umap_metric, embedding_metric in pairs.items():
        if umap_metric not in joined.columns or embedding_metric not in joined.columns:
            rows.append(
                {
                    "umap_metric": umap_metric,
                    "embedding_metric": embedding_metric,
                    "method": method,
                    "correlation": np.nan,
                    "abs_correlation": np.nan,
                    "n_valid_pairwise": 0,
                    "available": False,
                }
            )
            continue
        left = pd.to_numeric(joined[umap_metric], errors="coerce")
        right = pd.to_numeric(joined[embedding_metric], errors="coerce")
        valid = left.notna() & right.notna()
        n_valid = int(valid.sum())
        correlation = (
            left.loc[valid].corr(right.loc[valid], method=method)
            if n_valid >= 2
            else np.nan
        )
        rows.append(
            {
                "umap_metric": umap_metric,
                "embedding_metric": embedding_metric,
                "method": method,
                "correlation": float(correlation)
                if pd.notna(correlation)
                else np.nan,
                "abs_correlation": abs(float(correlation))
                if pd.notna(correlation)
                else np.nan,
                "n_valid_pairwise": n_valid,
                "available": True,
            }
        )
    return pd.DataFrame(rows)


def top_absolute_correlations(correlations: pd.DataFrame, *, n: int = 25) -> pd.DataFrame:
    if correlations.empty or "abs_correlation" not in correlations.columns:
        return correlations.head(0).copy()
    return (
        correlations.dropna(subset=["abs_correlation"])
        .sort_values("abs_correlation", ascending=False, kind="mergesort")
        .head(n)
        .reset_index(drop=True)
    )


def weak_umap_relationships(correlations: pd.DataFrame) -> pd.DataFrame:
    if correlations.empty:
        return pd.DataFrame(columns=["umap_metric", "max_abs_correlation"])
    grouped = (
        correlations.groupby("umap_metric", dropna=False)["abs_correlation"]
        .max()
        .reset_index(name="max_abs_correlation")
        .sort_values("max_abs_correlation", kind="mergesort")
        .reset_index(drop=True)
    )
    return grouped


def correlation_matrix(
    correlations: pd.DataFrame,
    *,
    value_column: str = "correlation",
) -> pd.DataFrame:
    if correlations.empty:
        return pd.DataFrame()
    return correlations.pivot(
        index="umap_metric",
        columns="embedding_metric",
        values=value_column,
    )


def plot_correlation_heatmap(correlations: pd.DataFrame, output_path: str | Path) -> None:
    matrix = correlation_matrix(correlations)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if matrix.empty:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
        ax.text(0.5, 0.5, "No correlations available", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        return

    fig_width = max(10, len(matrix.columns) * 0.42)
    fig_height = max(8, len(matrix.index) * 0.34)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=160)
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=6)
    ax.set_xlabel("Embedding-space metrics")
    ax.set_ylabel("Projected UMAP metrics")
    ax.set_title("Cross-Family Spearman Correlations")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def comparison_summary_payload(
    *,
    umap_rows: int,
    embedding_rows: int,
    joined: pd.DataFrame,
    spearman: pd.DataFrame,
    pearson: pd.DataFrame,
    analogue: pd.DataFrame,
) -> dict[str, Any]:
    top = top_absolute_correlations(spearman, n=10)
    weak = weak_umap_relationships(spearman).head(10)
    return {
        "n_umap_rows": int(umap_rows),
        "n_embedding_rows": int(embedding_rows),
        "n_joined_subfields": int(len(joined)),
        "n_spearman_pairs": int(len(spearman)),
        "n_pearson_pairs": int(len(pearson)),
        "n_analogue_pairs": int(len(analogue)),
        "top_absolute_spearman": top.to_dict("records"),
        "weakest_umap_max_relationships": weak.to_dict("records"),
    }


def markdown_summary(
    *,
    payload: dict[str, Any],
    top_spearman: pd.DataFrame,
    analogue: pd.DataFrame,
    weak_umap: pd.DataFrame,
) -> str:
    lines = [
        "# Metric Family Comparison",
        "",
        "This is a diagnostic comparison between projected UMAP morphology metrics "
        "and original embedding-space metrics. It is not a validation proof or a "
        "scientific conclusion.",
        "",
        f"- UMAP metric rows read: {payload['n_umap_rows']}",
        f"- Embedding metric rows read: {payload['n_embedding_rows']}",
        f"- Joined subfields compared: {payload['n_joined_subfields']}",
        "",
        "## Strongest Absolute Spearman Relationships",
        "",
    ]
    if top_spearman.empty:
        lines.append("No pairwise Spearman correlations were available.")
    else:
        for row in top_spearman.head(10).itertuples(index=False):
            lines.append(
                f"- `{row.umap_metric}` vs `{row.embedding_metric}`: "
                f"{row.correlation:.3f} (n={row.n_valid_pairwise})"
            )
    lines.extend(["", "## Analogue Metric Pairs", ""])
    if analogue.empty:
        lines.append("No analogue metric pairs were available.")
    else:
        lines.append(
            "These pairs are conceptual analogues, not exact equivalents."
        )
        for row in analogue.itertuples(index=False):
            if pd.isna(row.correlation):
                value = "NaN"
            else:
                value = f"{row.correlation:.3f}"
            lines.append(
                f"- `{row.umap_metric}` vs `{row.embedding_metric}`: "
                f"{value} (n={row.n_valid_pairwise})"
            )
    lines.extend(["", "## Weak UMAP Relationships", ""])
    if weak_umap.empty:
        lines.append("No weak-relationship diagnostic was available.")
    else:
        lines.append(
            "These projected metrics had the lowest maximum absolute Spearman "
            "relationship with any embedding-space core metric."
        )
        for row in weak_umap.head(10).itertuples(index=False):
            value = "NaN" if pd.isna(row.max_abs_correlation) else f"{row.max_abs_correlation:.3f}"
            lines.append(f"- `{row.umap_metric}`: max |rho| = {value}")
    lines.extend(
        [
            "",
            "Weak correlation is not automatically a problem. The two families "
            "summarize related but different views of subfield morphology.",
            "",
        ]
    )
    return "\n".join(lines)
