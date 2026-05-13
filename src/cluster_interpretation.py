from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from src.embedding_space_metrics import (
    CORE_EMBEDDING_METRIC_COLUMNS,
    metric_dictionary_frame as embedding_metric_dictionary_frame,
)
from src.morphology_metrics import (
    CORE_METRIC_COLUMNS_V2,
    metric_dictionary_frame as morphology_metric_dictionary_frame,
)

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


SUPPORTED_SPACES = ("embedding_only", "umap_only", "combined")

METADATA_COLUMNS = [
    "subfield_id",
    "subfield_display_name",
    "field_id",
    "field_display_name",
    "domain_id",
    "domain_display_name",
]

DEFAULT_MASTER_METRICS = [
    "radial_tail_index",
    "knn_median_distance",
    "density_entropy",
    "density_peak_count",
    "effective_area_90",
    "anisotropy_ratio",
    "centroid_drift_early_late",
    "embedding_centroid_norm",
    "embedding_distance_to_centroid_mean",
    "embedding_knn_median_distance",
    "embedding_pca_first_component_share",
    "embedding_pca_participation_ratio",
    "embedding_centroid_drift_early_late",
]

METRIC_HINTS = {
    "anisotropy_ratio": "elongated or directional map shape",
    "boundary_complexity": "irregular projected boundary",
    "centroid_drift_early_late": "projected centroid movement from early to late period",
    "component_separation_index": "separation between dense projected components",
    "core_periphery_ratio": "relative size of dense core versus broader footprint",
    "dense_component_count": "number of dense projected islands",
    "density_entropy": "evenness of projected density across the map",
    "density_entropy_slope_by_year": "change in projected density evenness over time",
    "density_peak_count": "number of dense local peaks in the projected map",
    "directionality_ratio": "directionality of projected temporal movement",
    "effective_area_90": "spatial area occupied by the densest 90 percent of mass",
    "knn_distance_cv": "unevenness of local projected density",
    "knn_median_distance": "typical local sparsity in the projected map",
    "largest_component_mass_share": "dominance of one dense projected component",
    "max_normalized_radius": "most extreme projected outlier distance",
    "mst_gap_index": "large gaps in the projected point cloud",
    "peak_dominance": "dominance of the strongest projected density peak",
    "peak_mass_entropy": "balance of mass across projected density peaks",
    "radial_expansion_slope": "projected radial expansion or contraction over time",
    "radial_iqr_index": "middle-range projected radial spread",
    "radial_tail_index": "strength of projected long tails",
    "support_solidity": "filled versus concave projected support",
    "embedding_centroid_drift_early_late": "change in semantic centroid between early and late periods",
    "embedding_centroid_norm": "strength of a common semantic direction",
    "embedding_distance_to_centroid_iqr": "middle-range semantic dispersion",
    "embedding_distance_to_centroid_mean": "semantic dispersion around the subfield centroid",
    "embedding_distance_to_centroid_median": "typical semantic dispersion around the centroid",
    "embedding_distance_to_centroid_p90": "semantic distance of the outer periphery",
    "embedding_distance_to_centroid_std": "heterogeneity in semantic dispersion",
    "embedding_graph_connected_component_count": "fragmentation of the semantic neighbor graph",
    "embedding_graph_edge_distance_median": "typical semantic graph edge length",
    "embedding_graph_edge_distance_p90": "long sparse-bridge semantic graph edges",
    "embedding_graph_largest_component_share": "global connectedness of the semantic neighbor graph",
    "embedding_knn_distance_cv": "unevenness of local semantic density",
    "embedding_knn_mean_distance": "average local semantic sparsity",
    "embedding_knn_median_distance": "typical local semantic sparsity",
    "embedding_knn_p90_distance": "sparse local semantic neighborhoods",
    "embedding_pca_dim_50": "components needed for half of embedding variance",
    "embedding_pca_dim_80": "components needed for most embedding variance",
    "embedding_pca_first_component_share": "concentration of embedding variance in one dominant axis",
    "embedding_pca_participation_ratio": "effective dimensionality of the embedding cloud",
    "embedding_pca_top3_variance_share": "variance concentration in the top three embedding axes",
    "embedding_pca_top5_variance_share": "variance concentration in the top five embedding axes",
    "embedding_radial_expansion_slope": "semantic expansion away from the period centroid",
    "embedding_tail_index_p90_median": "strength of the semantic distance tail",
}

EXPLORATORY_WORDING = (
    "These clusters are exploratory morphology profiles, not definitive "
    "disciplinary classes. They summarize similarities between subfields "
    "according to the selected metric space."
)


def require_existing_paths(paths: dict[str, str | Path]) -> None:
    missing = [
        f"{label}: {Path(path)}"
        for label, path in paths.items()
        if not Path(path).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required input file(s): " + "; ".join(missing)
        )


def normalize_subfield_id(frame: pd.DataFrame) -> pd.DataFrame:
    if "subfield_id" not in frame.columns:
        raise ValueError("frame must contain subfield_id")
    result = frame.copy()
    result["subfield_id"] = result["subfield_id"].astype(str)
    return result


def read_summary_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _cluster_k(column: str) -> int | None:
    prefix = "cluster_ward_k"
    if not column.startswith(prefix):
        return None
    try:
        return int(column.removeprefix(prefix))
    except ValueError:
        return None


def main_cluster_column(
    assignments: pd.DataFrame,
    *,
    summary: dict[str, Any] | None = None,
    preferred_k: int = 5,
) -> str:
    if summary:
        candidate = summary.get("main_cluster_column")
        if isinstance(candidate, str) and candidate in assignments.columns:
            return candidate
    ward_columns = [column for column in assignments.columns if _cluster_k(column) is not None]
    if not ward_columns:
        raise ValueError("cluster assignment table has no cluster_ward_k* columns")
    return sorted(
        ward_columns,
        key=lambda column: (abs((_cluster_k(column) or preferred_k) - preferred_k), _cluster_k(column) or 0),
    )[0]


def _metadata_frame(assignments: pd.DataFrame) -> pd.DataFrame:
    frame = normalize_subfield_id(assignments)
    for column in METADATA_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[METADATA_COLUMNS].drop_duplicates("subfield_id").reset_index(drop=True)


def _assignment_cluster_frame(
    assignments: pd.DataFrame,
    *,
    cluster_column: str,
    output_column: str,
) -> pd.DataFrame:
    frame = normalize_subfield_id(assignments)
    if cluster_column not in frame.columns:
        raise ValueError(f"cluster assignment table is missing {cluster_column}")
    return frame[["subfield_id", cluster_column]].rename(columns={cluster_column: output_column})


def _score_columns(pca_scores: pd.DataFrame) -> list[str]:
    preferred = [
        column
        for column in pca_scores.columns
        if "_PC" in column or column.startswith("PC")
    ]
    if preferred:
        return preferred
    numeric_columns = []
    for column in pca_scores.columns:
        if column in METADATA_COLUMNS:
            continue
        values = pd.to_numeric(pca_scores[column], errors="coerce")
        if values.notna().any():
            numeric_columns.append(column)
    return numeric_columns


def pca_coordinate_frame(pca_scores: pd.DataFrame, *, space: str) -> pd.DataFrame:
    frame = normalize_subfield_id(pca_scores)
    score_columns = _score_columns(frame)
    if not score_columns:
        raise ValueError(f"{space} pca_scores table has no numeric score columns")
    result = frame[["subfield_id"]].copy()
    result[f"{space}_pca1"] = pd.to_numeric(frame[score_columns[0]], errors="coerce")
    result[f"{space}_pca2"] = (
        pd.to_numeric(frame[score_columns[1]], errors="coerce")
        if len(score_columns) > 1
        else np.nan
    )
    return result


def _selected_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def build_master_table(
    *,
    assignments_by_space: dict[str, pd.DataFrame],
    pca_scores_by_space: dict[str, pd.DataFrame],
    cluster_columns_by_space: dict[str, str],
    umap_metrics: pd.DataFrame,
    embedding_metrics: pd.DataFrame,
    main_space: str = "combined",
    selected_metric_columns: list[str] | None = None,
    combined_umap_coordinates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if main_space not in assignments_by_space:
        raise ValueError(f"missing assignments for main space: {main_space}")
    selected_metric_columns = selected_metric_columns or DEFAULT_MASTER_METRICS

    master = _metadata_frame(assignments_by_space[main_space])
    for space in SUPPORTED_SPACES:
        if space not in assignments_by_space:
            continue
        cluster_column = cluster_columns_by_space[space]
        master = master.merge(
            _assignment_cluster_frame(
                assignments_by_space[space],
                cluster_column=cluster_column,
                output_column=f"cluster_{space}",
            ),
            on="subfield_id",
            how="left",
        )

    for space in SUPPORTED_SPACES:
        if space not in pca_scores_by_space:
            continue
        master = master.merge(
            pca_coordinate_frame(pca_scores_by_space[space], space=space),
            on="subfield_id",
            how="left",
        )

    if combined_umap_coordinates is not None:
        coords = normalize_subfield_id(combined_umap_coordinates)
        x_candidates = ["combined_umap_x", "umap_x", "x", "UMAP1", "umap_1"]
        y_candidates = ["combined_umap_y", "umap_y", "y", "UMAP2", "umap_2"]
        x_column = next((column for column in x_candidates if column in coords.columns), None)
        y_column = next((column for column in y_candidates if column in coords.columns), None)
        if x_column and y_column:
            coords = coords[["subfield_id", x_column, y_column]].rename(
                columns={x_column: "combined_umap_x", y_column: "combined_umap_y"}
            )
            master = master.merge(coords, on="subfield_id", how="left")

    if "combined_umap_x" not in master.columns:
        master["combined_umap_x"] = np.nan
    if "combined_umap_y" not in master.columns:
        master["combined_umap_y"] = np.nan

    embedding = normalize_subfield_id(embedding_metrics)
    embedding_join_columns = _selected_columns(
        embedding,
        ["subfield_id", "n_available", "n_used"] + selected_metric_columns,
    )
    if "subfield_id" in embedding_join_columns:
        master = master.merge(
            embedding[embedding_join_columns].drop_duplicates("subfield_id"),
            on="subfield_id",
            how="left",
        )

    umap = normalize_subfield_id(umap_metrics)
    umap_join_columns = _selected_columns(umap, ["subfield_id", "n_points"] + selected_metric_columns)
    if "subfield_id" in umap_join_columns:
        duplicate_metric_columns = [
            column
            for column in umap_join_columns
            if column != "subfield_id" and column in master.columns
        ]
        safe_umap = umap[umap_join_columns].drop(columns=duplicate_metric_columns)
        master = master.merge(
            safe_umap.drop_duplicates("subfield_id"),
            on="subfield_id",
            how="left",
        )

    ordered = [
        "subfield_id",
        "subfield_display_name",
        "field_id",
        "field_display_name",
        "domain_id",
        "domain_display_name",
        "cluster_embedding_only",
        "cluster_umap_only",
        "cluster_combined",
        "combined_pca1",
        "combined_pca2",
        "combined_umap_x",
        "combined_umap_y",
        "embedding_only_pca1",
        "embedding_only_pca2",
        "umap_only_pca1",
        "umap_only_pca2",
        "n_available",
        "n_used",
        "n_points",
    ]
    ordered.extend(column for column in selected_metric_columns if column in master.columns)
    present_ordered = [column for column in ordered if column in master.columns]
    remainder = [column for column in master.columns if column not in present_ordered]
    return master[present_ordered + remainder].sort_values("subfield_id").reset_index(drop=True)


def cluster_composition(master: pd.DataFrame, *, level: str) -> pd.DataFrame:
    if level not in {"domain", "field"}:
        raise ValueError("level must be domain or field")
    cluster_column = "cluster_combined"
    if cluster_column not in master.columns:
        raise ValueError("master table must contain cluster_combined")
    id_column = f"{level}_id"
    name_column = f"{level}_display_name"
    required = {cluster_column, id_column, name_column}
    missing = required - set(master.columns)
    if missing:
        raise ValueError(f"master table missing columns: {', '.join(sorted(missing))}")

    base = master[[cluster_column, id_column, name_column, "subfield_id"]].copy()
    base[cluster_column] = pd.to_numeric(base[cluster_column], errors="coerce")
    base = base.dropna(subset=[cluster_column])
    base[cluster_column] = base[cluster_column].astype(int)

    cluster_sizes = base.groupby(cluster_column)["subfield_id"].nunique().rename("n_subfields")
    level_sizes = base.groupby([id_column, name_column])["subfield_id"].nunique().rename("level_total")
    counts = (
        base.groupby([cluster_column, id_column, name_column])["subfield_id"]
        .nunique()
        .rename("count")
        .reset_index()
    )
    counts = counts.merge(cluster_sizes.reset_index(), on=cluster_column, how="left")
    counts = counts.merge(level_sizes.reset_index(), on=[id_column, name_column], how="left")
    counts["share_within_cluster"] = counts["count"] / counts["n_subfields"]
    counts["share_within_domain_or_field"] = counts["count"] / counts["level_total"]
    counts = counts.drop(columns=["level_total"]).rename(columns={cluster_column: "cluster_id"})
    return counts.sort_values(["cluster_id", "count", name_column], ascending=[True, False, True]).reset_index(drop=True)


def combined_cluster_size_summary(master: pd.DataFrame) -> pd.DataFrame:
    domain_comp = cluster_composition(master, level="domain")
    field_comp = cluster_composition(master, level="field")
    total = int(master["subfield_id"].nunique())
    rows = []
    for cluster_id, group in master.groupby("cluster_combined", dropna=True):
        n_subfields = int(group["subfield_id"].nunique())
        domains = domain_comp.loc[domain_comp["cluster_id"] == int(cluster_id)]
        fields = field_comp.loc[field_comp["cluster_id"] == int(cluster_id)]
        top_domain = domains.iloc[0] if not domains.empty else None
        top_field = fields.iloc[0] if not fields.empty else None
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "n_subfields": n_subfields,
                "share_of_subfields": n_subfields / total if total else np.nan,
                "n_domains": int(group["domain_id"].nunique()),
                "n_fields": int(group["field_id"].nunique()),
                "dominant_domain": top_domain["domain_display_name"] if top_domain is not None else "",
                "dominant_domain_share": float(top_domain["share_within_cluster"]) if top_domain is not None else np.nan,
                "dominant_field": top_field["field_display_name"] if top_field is not None else "",
                "dominant_field_share": float(top_field["share_within_cluster"]) if top_field is not None else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)


def _first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((column for column in candidates if column in frame.columns), None)


def readable_representatives(representatives: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    output_columns = [
        "cluster_id",
        "representative_rank",
        "subfield_id",
        "subfield_display_name",
        "field_display_name",
        "domain_display_name",
        "distance_to_cluster_center",
    ]
    if representatives.empty:
        return pd.DataFrame(columns=output_columns)

    reps = normalize_subfield_id(representatives)
    cluster_column = _first_existing_column(
        reps,
        ["cluster_id", "cluster", "cluster_label", "cluster_combined"],
    )
    if cluster_column is None:
        ward_columns = [column for column in reps.columns if _cluster_k(column) is not None]
        cluster_column = ward_columns[0] if ward_columns else None
    if cluster_column is None:
        raise ValueError("representative table has no recognizable cluster column")
    rank_column = _first_existing_column(reps, ["representative_rank", "rank", "rep_rank"])
    distance_column = _first_existing_column(
        reps,
        ["distance_to_cluster_center", "distance_to_centroid", "centroid_distance", "distance"],
    )

    result = pd.DataFrame(
        {
            "cluster_id": pd.to_numeric(reps[cluster_column], errors="coerce"),
            "subfield_id": reps["subfield_id"],
        }
    )
    if rank_column:
        result["representative_rank"] = pd.to_numeric(reps[rank_column], errors="coerce")
    else:
        result["representative_rank"] = (
            result.groupby("cluster_id").cumcount().astype(int) + 1
        )
    result["distance_to_cluster_center"] = (
        pd.to_numeric(reps[distance_column], errors="coerce") if distance_column else np.nan
    )

    metadata = normalize_subfield_id(master)[
        [
            "subfield_id",
            "subfield_display_name",
            "field_display_name",
            "domain_display_name",
        ]
    ].drop_duplicates("subfield_id")
    result = result.merge(metadata, on="subfield_id", how="left")

    for column in ["subfield_display_name", "field_display_name", "domain_display_name"]:
        if column in reps.columns:
            result[column] = result[column].combine_first(reps[column])

    result["cluster_id"] = result["cluster_id"].astype("Int64")
    result["representative_rank"] = result["representative_rank"].astype("Int64")
    return result[output_columns].sort_values(["cluster_id", "representative_rank"]).reset_index(drop=True)


def representative_fallback_from_master(master: pd.DataFrame, *, per_cluster: int = 5) -> pd.DataFrame:
    rows = []
    for cluster_id, group in master.sort_values("subfield_display_name").groupby("cluster_combined"):
        for rank, row in enumerate(group.head(per_cluster).itertuples(index=False), start=1):
            rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "representative_rank": rank,
                    "subfield_id": row.subfield_id,
                    "subfield_display_name": row.subfield_display_name,
                    "field_display_name": row.field_display_name,
                    "domain_display_name": row.domain_display_name,
                    "distance_to_cluster_center": np.nan,
                }
            )
    return pd.DataFrame(rows)


def metric_hint_lookup() -> dict[str, str]:
    hints = dict(METRIC_HINTS)
    for dictionary in [morphology_metric_dictionary_frame(), embedding_metric_dictionary_frame()]:
        for row in dictionary.itertuples(index=False):
            hints.setdefault(str(row.metric_name), str(row.higher_means))
    return hints


def build_metric_profile_long(
    *,
    master: pd.DataFrame,
    umap_metrics: pd.DataFrame,
    embedding_metrics: pd.DataFrame,
    cluster_column: str = "cluster_combined",
) -> pd.DataFrame:
    if cluster_column not in master.columns:
        raise ValueError(f"master table must contain {cluster_column}")
    hints = metric_hint_lookup()
    cluster_frame = normalize_subfield_id(master)[["subfield_id", cluster_column]].dropna()
    cluster_frame[cluster_column] = pd.to_numeric(cluster_frame[cluster_column], errors="coerce")
    cluster_frame = cluster_frame.dropna(subset=[cluster_column])
    cluster_frame[cluster_column] = cluster_frame[cluster_column].astype(int)

    metric_blocks = [
        (
            "umap_projected",
            normalize_subfield_id(umap_metrics),
            _selected_columns(umap_metrics, CORE_METRIC_COLUMNS_V2),
        ),
        (
            "embedding_space",
            normalize_subfield_id(embedding_metrics),
            _selected_columns(embedding_metrics, CORE_EMBEDDING_METRIC_COLUMNS),
        ),
    ]
    rows: list[dict[str, Any]] = []
    for family, metrics, metric_columns in metric_blocks:
        if not metric_columns:
            continue
        joined = cluster_frame.merge(
            metrics[["subfield_id"] + metric_columns].drop_duplicates("subfield_id"),
            on="subfield_id",
            how="inner",
        )
        for metric_name in metric_columns:
            values = pd.to_numeric(joined[metric_name], errors="coerce")
            global_mean = values.mean()
            global_std = values.std(ddof=0)
            for cluster_id, group in joined.groupby(cluster_column):
                cluster_values = pd.to_numeric(group[metric_name], errors="coerce")
                cluster_mean = cluster_values.mean()
                cluster_mean_z = (
                    (cluster_mean - global_mean) / global_std
                    if pd.notna(cluster_mean)
                    and pd.notna(global_mean)
                    and pd.notna(global_std)
                    and global_std > 0
                    else np.nan
                )
                rows.append(
                    {
                        "cluster_id": int(cluster_id),
                        "metric_name": metric_name,
                        "metric_family": family,
                        "n_subfields": int(group["subfield_id"].nunique()),
                        "n_non_missing": int(cluster_values.notna().sum()),
                        "cluster_mean": float(cluster_mean) if pd.notna(cluster_mean) else np.nan,
                        "global_mean": float(global_mean) if pd.notna(global_mean) else np.nan,
                        "global_std": float(global_std) if pd.notna(global_std) else np.nan,
                        "cluster_mean_z": float(cluster_mean_z) if pd.notna(cluster_mean_z) else np.nan,
                        "plain_language_hint": hints.get(metric_name, ""),
                    }
                )
    return pd.DataFrame(rows).sort_values(["cluster_id", "metric_family", "metric_name"]).reset_index(drop=True)


def top_metric_differences(profile_long: pd.DataFrame, *, top_n: int = 8) -> pd.DataFrame:
    rows = []
    profile = profile_long.dropna(subset=["cluster_mean_z"]).copy()
    for cluster_id, group in profile.groupby("cluster_id"):
        high = group.sort_values("cluster_mean_z", ascending=False).head(top_n).copy()
        high["direction"] = "high"
        low = group.sort_values("cluster_mean_z", ascending=True).head(top_n).copy()
        low["direction"] = "low"
        rows.append(pd.concat([high, low], ignore_index=True))
    if not rows:
        return pd.DataFrame(columns=list(profile_long.columns) + ["direction"])
    columns = [
        "cluster_id",
        "metric_name",
        "metric_family",
        "cluster_mean_z",
        "direction",
        "plain_language_hint",
        "n_subfields",
        "n_non_missing",
        "cluster_mean",
        "global_mean",
        "global_std",
    ]
    result = pd.concat(rows, ignore_index=True)
    return result[[column for column in columns if column in result.columns]]


def _format_share(name: Any, share: Any) -> str:
    if pd.isna(share):
        return str(name)
    return f"{name} ({float(share):.0%})"


def _short_metric_name(metric_name: str) -> str:
    return metric_name.replace("embedding_", "emb_").replace("_", " ")


def suggested_cluster_label(top_differences: pd.DataFrame) -> str:
    high = top_differences.loc[top_differences["direction"] == "high"].head(1)
    low = top_differences.loc[top_differences["direction"] == "low"].head(1)
    parts = []
    if not high.empty:
        parts.append(f"high {_short_metric_name(str(high.iloc[0]['metric_name']))}")
    if not low.empty:
        parts.append(f"low {_short_metric_name(str(low.iloc[0]['metric_name']))}")
    if not parts:
        return "Exploratory morphology profile"
    return "Exploratory profile with " + " and ".join(parts)


def cluster_interpretation_markdown(
    *,
    master: pd.DataFrame,
    size_summary: pd.DataFrame,
    domain_composition: pd.DataFrame,
    field_composition: pd.DataFrame,
    representatives: pd.DataFrame,
    top_differences: pd.DataFrame,
) -> str:
    lines = [
        "# Combined Cluster Interpretation Summary",
        "",
        EXPLORATORY_WORDING,
        "",
    ]
    for row in size_summary.itertuples(index=False):
        cluster_id = int(row.cluster_id)
        domains = domain_composition.loc[domain_composition["cluster_id"] == cluster_id].head(3)
        fields = field_composition.loc[field_composition["cluster_id"] == cluster_id].head(5)
        reps = representatives.loc[representatives["cluster_id"] == cluster_id].head(5)
        diffs = top_differences.loc[top_differences["cluster_id"] == cluster_id]
        highs = diffs.loc[diffs["direction"] == "high"].head(5)
        lows = diffs.loc[diffs["direction"] == "low"].head(5)
        lines.extend(
            [
                f"## Cluster {cluster_id}: {suggested_cluster_label(diffs)}",
                "",
                f"- Size: {int(row.n_subfields)} subfields ({float(row.share_of_subfields):.1%} of the master table).",
                "- Dominant domains: "
                + "; ".join(
                    _format_share(item.domain_display_name, item.share_within_cluster)
                    for item in domains.itertuples(index=False)
                ),
                "- Dominant fields: "
                + "; ".join(
                    _format_share(item.field_display_name, item.share_within_cluster)
                    for item in fields.itertuples(index=False)
                ),
                "- Representative subfields: "
                + "; ".join(
                    str(item.subfield_display_name) for item in reps.itertuples(index=False)
                ),
                "- Highest standardized metrics: "
                + "; ".join(
                    f"{item.metric_name} ({item.cluster_mean_z:.2f})"
                    for item in highs.itertuples(index=False)
                ),
                "- Lowest standardized metrics: "
                + "; ".join(
                    f"{item.metric_name} ({item.cluster_mean_z:.2f})"
                    for item in lows.itertuples(index=False)
                ),
                "",
            ]
        )
    return "\n".join(lines)


def agreement_summary_table(agreement: pd.DataFrame) -> pd.DataFrame:
    result = agreement.copy()
    if result.empty:
        return result

    def interpretation(ari: float) -> str:
        if pd.isna(ari):
            return "agreement unavailable"
        if ari < 0.20:
            return "low agreement; the metric families capture non-equivalent morphology views"
        if ari < 0.50:
            return "partial agreement; the spaces share structure but are not interchangeable"
        return "high agreement; the spaces produce similar partitions"

    result["interpretation"] = [
        interpretation(float(value)) for value in result["adjusted_rand_index"]
    ]
    return result


def contingency_top_cells(contingency: pd.DataFrame, *, top_n: int = 5) -> list[str]:
    if contingency.empty:
        return []
    rows = []
    for row_label in contingency.index:
        for column_label in contingency.columns:
            count = int(contingency.loc[row_label, column_label])
            if count > 0:
                rows.append((count, row_label, column_label))
    rows.sort(reverse=True)
    return [
        f"{row_label} x {column_label}: {count}"
        for count, row_label, column_label in rows[:top_n]
    ]


def agreement_summary_markdown(
    *,
    agreement: pd.DataFrame,
    contingencies: dict[str, pd.DataFrame],
) -> str:
    lines = [
        "# Cluster Family Agreement Summary",
        "",
        EXPLORATORY_WORDING,
        "",
        "Low ARI/NMI is not a failure by itself. It indicates that embedding-space "
        "metrics and projected UMAP metrics capture related but non-equivalent "
        "aspects of morphology, which supports reporting both families rather "
        "than collapsing everything into one view.",
        "",
    ]
    if agreement.empty:
        lines.append("No agreement comparisons were available.")
        return "\n".join(lines)

    for row in agreement.itertuples(index=False):
        lines.append(
            f"- {row.left_space} vs {row.right_space}: "
            f"ARI={row.adjusted_rand_index:.3f}, "
            f"NMI={row.normalized_mutual_info:.3f}, "
            f"n={int(row.n_common_subfields)}."
        )

    lines.extend(["", "## Largest Contingency Overlaps", ""])
    for name, contingency in contingencies.items():
        cells = contingency_top_cells(contingency)
        if not cells:
            continue
        lines.append(f"### {name}")
        lines.extend(f"- {cell}" for cell in cells)
        lines.append("")
    return "\n".join(lines)


def _cluster_sort_key(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def plot_domain_stacked_bar(domain_composition: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if domain_composition.empty:
        _plot_empty(output_path, "No domain composition available")
        return
    matrix = domain_composition.pivot_table(
        index="cluster_id",
        columns="domain_display_name",
        values="share_within_cluster",
        fill_value=0.0,
    ).sort_index(key=lambda index: [_cluster_sort_key(value) for value in index])
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
    bottom = np.zeros(len(matrix), dtype=float)
    colors = plt.get_cmap("tab20").colors
    x = np.arange(len(matrix))
    for idx, column in enumerate(matrix.columns):
        values = matrix[column].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, label=column, color=colors[idx % len(colors)], width=0.72)
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{cluster_id}" for cluster_id in matrix.index])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share within cluster")
    ax.set_xlabel("Combined cluster")
    ax.set_title("Domain Composition Of Combined Clusters")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_profile_heatmap(
    profile_long: pd.DataFrame,
    top_differences: pd.DataFrame,
    output_path: str | Path,
    *,
    max_metrics: int = 24,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if profile_long.empty:
        _plot_empty(output_path, "No metric profile available")
        return
    metric_order = (
        profile_long.assign(abs_z=profile_long["cluster_mean_z"].abs())
        .groupby("metric_name")["abs_z"]
        .max()
        .sort_values(ascending=False)
        .head(max_metrics)
        .index.tolist()
    )
    if not top_differences.empty:
        highlighted = (
            top_differences.assign(abs_z=top_differences["cluster_mean_z"].abs())
            .groupby("metric_name")["abs_z"]
            .max()
            .sort_values(ascending=False)
            .head(max_metrics)
            .index.tolist()
        )
        metric_order = highlighted or metric_order
    matrix = (
        profile_long.loc[profile_long["metric_name"].isin(metric_order)]
        .pivot_table(
            index="cluster_id",
            columns="metric_name",
            values="cluster_mean_z",
            fill_value=0.0,
        )
        .reindex(columns=metric_order)
        .sort_index()
    )
    fig_width = max(10, min(16, 0.55 * len(matrix.columns)))
    fig_height = max(4.5, 0.65 * len(matrix.index) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=160)
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap="coolwarm", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels([_short_metric_name(column) for column in matrix.columns], rotation=55, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels([f"C{cluster_id}" for cluster_id in matrix.index])
    ax.set_xlabel("Metric")
    ax.set_ylabel("Combined cluster")
    ax.set_title("Top Metric Differences By Combined Cluster")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="cluster mean z")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _short_label(value: Any, *, max_chars: int = 30) -> str:
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def plot_combined_pca_scatter(
    master: pd.DataFrame,
    representatives: pd.DataFrame,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    required = {"combined_pca1", "combined_pca2", "cluster_combined"}
    if not required.issubset(master.columns):
        _plot_empty(output_path, "Combined PCA coordinates unavailable")
        return
    plot_frame = master.dropna(subset=["combined_pca1", "cluster_combined"]).copy()
    if plot_frame.empty:
        _plot_empty(output_path, "Combined PCA coordinates unavailable")
        return
    y = pd.to_numeric(plot_frame["combined_pca2"], errors="coerce").fillna(0.0)
    clusters = sorted(plot_frame["cluster_combined"].dropna().unique(), key=_cluster_sort_key)
    colors = plt.get_cmap("tab10").colors
    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=160)
    for idx, cluster_id in enumerate(clusters):
        group = plot_frame.loc[plot_frame["cluster_combined"] == cluster_id]
        ax.scatter(
            group["combined_pca1"],
            y.loc[group.index],
            s=34,
            alpha=0.82,
            color=colors[idx % len(colors)],
            label=f"C{cluster_id}",
            linewidths=0,
        )
    top_reps = representatives.loc[representatives["representative_rank"] == 1]
    label_frame = plot_frame.merge(
        top_reps[["subfield_id", "representative_rank"]],
        on="subfield_id",
        how="inner",
    )
    for row in label_frame.itertuples(index=False):
        ax.annotate(
            _short_label(row.subfield_display_name),
            (row.combined_pca1, row.combined_pca2 if pd.notna(row.combined_pca2) else 0.0),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    ax.axhline(0, color="#d0d0d0", linewidth=0.7, zorder=0)
    ax.axvline(0, color="#d0d0d0", linewidth=0.7, zorder=0)
    ax.set_xlabel("Combined PCA 1")
    ax.set_ylabel("Combined PCA 2")
    ax.set_title("Combined Metric-Space PCA With Representative Labels")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="Cluster")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_contingency_heatmap(
    contingency: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if contingency.empty:
        _plot_empty(output_path, "No contingency table available")
        return
    matrix = contingency.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=160)
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels([str(column) for column in matrix.columns], fontsize=8)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels([str(index) for index in matrix.index], fontsize=8)
    ax.set_xlabel(str(contingency.columns.name or "right cluster"))
    ax.set_ylabel(str(contingency.index.name or "left cluster"))
    ax.set_title(title)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = int(matrix.iloc[row_idx, col_idx])
            if value:
                ax.text(col_idx, row_idx, str(value), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03, label="Subfields")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_empty(output_path: Path, message: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=160)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def top_domains_by_cluster(domain_composition: pd.DataFrame, *, top_n: int = 3) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for cluster_id, group in domain_composition.groupby("cluster_id"):
        result[str(cluster_id)] = [
            {
                "domain": str(row.domain_display_name),
                "count": int(row.count),
                "share_within_cluster": float(row.share_within_cluster),
            }
            for row in group.head(top_n).itertuples(index=False)
        ]
    return result


def top_profile_metrics_by_cluster(top_differences: pd.DataFrame, *, top_n: int = 4) -> dict[str, dict[str, list[dict[str, Any]]]]:
    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for cluster_id, group in top_differences.groupby("cluster_id"):
        result[str(cluster_id)] = {}
        for direction in ["high", "low"]:
            direction_group = group.loc[group["direction"] == direction].head(top_n)
            result[str(cluster_id)][direction] = [
                {
                    "metric_name": str(row.metric_name),
                    "metric_family": str(row.metric_family),
                    "cluster_mean_z": float(row.cluster_mean_z),
                    "plain_language_hint": str(row.plain_language_hint),
                }
                for row in direction_group.itertuples(index=False)
            ]
    return result
