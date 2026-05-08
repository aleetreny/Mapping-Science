from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.morphology_metrics import CORE_METRIC_COLUMNS_V2, DIAGNOSTIC_METRIC_COLUMNS
from src.storage import load_parquet, save_parquet
from src.subfield_labels import add_subfield_label_columns, duplicate_subfield_names_report


METADATA_COLUMNS = [
    "subfield_id",
    "subfield_label_unique",
    "subfield_display_name",
    "field_id",
    "field_display_name",
    "domain_id",
    "domain_display_name",
    "n_points",
]

FAMILY_SCORE_DEFINITIONS = {
    "diffuseness_score": [
        ("density_entropy", 1.0),
        ("effective_area_90", 1.0),
        ("radial_tail_index", 1.0),
    ],
    "concentration_score": [
        ("peak_dominance", 1.0),
        ("core_periphery_ratio", -1.0),
        ("density_entropy", -1.0),
        ("effective_area_90", -1.0),
    ],
    "fragmentation_score": [
        ("dense_component_count", 1.0),
        ("density_peak_count", 1.0),
        ("component_separation_index", 1.0),
        ("mst_gap_index", 1.0),
        ("largest_component_mass_share", -1.0),
    ],
    "elongation_score": [
        ("anisotropy_ratio", 1.0),
        ("boundary_complexity", 1.0),
        ("support_solidity", -1.0),
    ],
    "temporal_dynamism_score": [
        ("centroid_drift_early_late", 1.0),
        ("annual_centroid_path_length", 1.0),
        ("annual_centroid_step_cv", 1.0),
    ],
    "directional_change_score": [
        ("directionality_ratio", 1.0),
        ("centroid_drift_early_late", 1.0),
        ("radial_expansion_r2", 1.0),
    ],
    "expansion_score": [
        ("radial_expansion_slope", 1.0),
    ],
    "diversification_score": [
        ("density_entropy_slope_by_year", 1.0),
    ],
}


def parse_args() -> argparse.Namespace:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Analyze subfield morphology metrics before prediction work."
    )
    parser.add_argument(
        "--input-path",
        default="data/processed/subfield_morphology_metrics.parquet",
    )
    parser.add_argument(
        "--dictionary-path",
        default="outputs/metrics/subfield_morphology_metrics_dictionary.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/metrics/morphology_analysis",
    )
    parser.add_argument("--limit-subfields", type=int, default=None)
    parser.add_argument("--subfield-id", default=None)
    parser.add_argument("--high-correlation-threshold", type=float, default=0.85)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def display_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def output_paths(output_dir: Path) -> dict[str, Path]:
    figures = output_dir / "figures"
    return {
        "metric_descriptive_stats": output_dir / "metric_descriptive_stats.csv",
        "metric_missingness": output_dir / "metric_missingness.csv",
        "metric_correlation_matrix": output_dir / "metric_correlation_matrix.csv",
        "metric_spearman_correlation_matrix": output_dir
        / "metric_spearman_correlation_matrix.csv",
        "high_correlation_pairs": output_dir / "high_correlation_pairs.csv",
        "low_variance_metrics": output_dir / "low_variance_metrics.csv",
        "top_bottom_subfields_by_metric": output_dir
        / "top_bottom_subfields_by_metric.csv",
        "domain_metric_summary": output_dir / "domain_metric_summary.csv",
        "field_metric_summary": output_dir / "field_metric_summary.csv",
        "duplicate_subfield_names_report": output_dir
        / "duplicate_subfield_names_report.csv",
        "curated_model_features_parquet": output_dir / "curated_model_features.parquet",
        "curated_model_features_csv": output_dir / "curated_model_features.csv",
        "family_scores": output_dir / "family_scores.csv",
        "metric_pca_scores": output_dir / "metric_pca_scores.csv",
        "metric_pca_loadings": output_dir / "metric_pca_loadings.csv",
        "morphology_analysis_summary": output_dir / "morphology_analysis_summary.json",
        "core_metric_histograms": figures / "core_metric_histograms.png",
        "core_metric_correlation_heatmap": figures
        / "core_metric_correlation_heatmap.png",
        "domain_family_score_boxplots": figures
        / "domain_family_score_boxplots.png",
        "metric_pca_scatter": figures / "metric_pca_scatter.png",
        "metric_pca_loadings_figure": figures / "metric_pca_loadings.png",
    }


def ensure_outputs(paths: dict[str, Path], overwrite: bool) -> None:
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        formatted = ", ".join(display_path(path) for path in existing[:8])
        raise FileExistsError(
            "Refusing to overwrite morphology analysis outputs without --overwrite: "
            f"{formatted}"
        )
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)


def select_rows(
    metrics: pd.DataFrame,
    *,
    subfield_id: str | None,
    limit_subfields: int | None,
) -> pd.DataFrame:
    selected = metrics.copy()
    if "metric_status" in selected.columns:
        selected = selected[selected["metric_status"] != "failed"].copy()
    selected["_subfield_sort_key"] = selected["subfield_id"].astype(str)
    selected = selected.sort_values("_subfield_sort_key", kind="mergesort").drop(
        columns="_subfield_sort_key"
    )
    if subfield_id is not None:
        selected = selected[selected["subfield_id"].astype(str) == str(subfield_id)]
        if selected.empty:
            raise ValueError(f"No morphology row matched --subfield-id {subfield_id}")
    if limit_subfields is not None:
        if limit_subfields <= 0:
            raise ValueError("limit_subfields must be positive when provided")
        selected = selected.head(limit_subfields)
    if selected.empty:
        raise ValueError("No morphology rows are available to analyze")
    return add_subfield_label_columns(selected.reset_index(drop=True))


def validate_core_columns(metrics: pd.DataFrame) -> None:
    missing = [column for column in CORE_METRIC_COLUMNS_V2 if column not in metrics.columns]
    if missing:
        raise ValueError(f"morphology table missing core metrics: {', '.join(missing)}")


def metric_descriptive_stats(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n_rows = len(metrics)
    for column in CORE_METRIC_COLUMNS_V2:
        values = pd.to_numeric(metrics[column], errors="coerce")
        q = values.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
        mean = float(values.mean())
        std = float(values.std())
        iqr = float(q.loc[0.75] - q.loc[0.25])
        rows.append(
            {
                "metric_name": column,
                "count": int(values.count()),
                "missing": int(values.isna().sum()),
                "missing_share": float(values.isna().mean()) if n_rows else np.nan,
                "mean": mean,
                "std": std,
                "min": float(values.min()),
                "p05": float(q.loc[0.05]),
                "p25": float(q.loc[0.25]),
                "median": float(q.loc[0.50]),
                "p75": float(q.loc[0.75]),
                "p95": float(q.loc[0.95]),
                "max": float(values.max()),
                "iqr": iqr,
                "coefficient_of_variation": float(std / abs(mean))
                if np.isfinite(mean) and abs(mean) > 1e-12
                else np.nan,
                "n_unique": int(values.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def metric_missingness(stats: pd.DataFrame) -> pd.DataFrame:
    return stats[["metric_name", "count", "missing", "missing_share", "n_unique"]].copy()


def high_correlation_pairs(
    pearson: pd.DataFrame,
    spearman: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    rows = []
    columns = list(pearson.columns)
    for i, metric_a in enumerate(columns):
        for metric_b in columns[i + 1 :]:
            pearson_value = float(pearson.loc[metric_a, metric_b])
            spearman_value = float(spearman.loc[metric_a, metric_b])
            if not np.isfinite(pearson_value) or abs(pearson_value) < threshold:
                continue
            if abs(spearman_value) >= threshold:
                action = "keep_one_for_modeling"
            else:
                action = "keep_both_for_interpretation_only"
            rows.append(
                {
                    "metric_a": metric_a,
                    "metric_b": metric_b,
                    "pearson": pearson_value,
                    "spearman": spearman_value,
                    "abs_pearson": abs(pearson_value),
                    "suggested_action": action,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "metric_a",
                "metric_b",
                "pearson",
                "spearman",
                "abs_pearson",
                "suggested_action",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        "abs_pearson",
        ascending=False,
        kind="mergesort",
    )


def low_variance_metrics(stats: pd.DataFrame) -> pd.DataFrame:
    flagged = stats[
        (stats["n_unique"] <= 3)
        | (stats["std"].abs() <= 1e-12)
        | (stats["iqr"].abs() <= 1e-12)
    ].copy()
    if flagged.empty:
        return pd.DataFrame(
            columns=list(stats.columns) + ["low_variance_reason"]
        )
    reasons = []
    for row in flagged.itertuples(index=False):
        row_reasons = []
        if row.n_unique <= 3:
            row_reasons.append("n_unique <= 3")
        if abs(row.std) <= 1e-12:
            row_reasons.append("std close to 0")
        if abs(row.iqr) <= 1e-12:
            row_reasons.append("iqr close to 0")
        reasons.append("; ".join(row_reasons))
    flagged["low_variance_reason"] = reasons
    return flagged


def top_bottom_subfields(metrics: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    rows = []
    for metric_name in CORE_METRIC_COLUMNS_V2:
        ranked = metrics.dropna(subset=[metric_name]).copy()
        for direction, ascending in [("bottom", True), ("top", False)]:
            subset = ranked.sort_values(metric_name, ascending=ascending).head(n)
            for rank, row in enumerate(subset.itertuples(index=False), start=1):
                rows.append(
                    {
                        "metric_name": metric_name,
                        "rank_direction": direction,
                        "rank": rank,
                        "subfield_id": row.subfield_id,
                        "subfield_label_unique": row.subfield_label_unique,
                        "subfield_display_name": row.subfield_display_name,
                        "field_display_name": row.field_display_name,
                        "domain_display_name": row.domain_display_name,
                        "metric_value": float(getattr(row, metric_name)),
                    }
                )
    return pd.DataFrame(rows)


def group_metric_summary(metrics: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for group_value, group in metrics.groupby(group_column, dropna=False):
        for metric_name in CORE_METRIC_COLUMNS_V2:
            values = pd.to_numeric(group[metric_name], errors="coerce")
            rows.append(
                {
                    group_column: group_value,
                    "metric_name": metric_name,
                    "n_subfields": int(len(group)),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def robust_z_scores(metrics: pd.DataFrame) -> pd.DataFrame:
    values = metrics[CORE_METRIC_COLUMNS_V2].apply(pd.to_numeric, errors="coerce")
    medians = values.median()
    iqrs = values.quantile(0.75) - values.quantile(0.25)
    scaled = (values - medians) / iqrs.replace(0, np.nan)
    return scaled.clip(-5, 5)


def compute_family_scores(metrics: pd.DataFrame) -> pd.DataFrame:
    z = robust_z_scores(metrics)
    output = metrics[METADATA_COLUMNS].copy()
    for score_name, components in FAMILY_SCORE_DEFINITIONS.items():
        component_values = []
        used_names = []
        for metric_name, sign in components:
            if metric_name not in z.columns:
                continue
            component_values.append(z[metric_name] * sign)
            used_names.append(metric_name)
        if component_values:
            component_frame = pd.concat(component_values, axis=1)
            output[score_name] = component_frame.mean(axis=1, skipna=True)
            output[f"{score_name}_n_components"] = component_frame.notna().sum(axis=1)
            output[f"{score_name}_components"] = ", ".join(used_names)
        else:
            output[score_name] = np.nan
            output[f"{score_name}_n_components"] = 0
            output[f"{score_name}_components"] = ""
    return output


def curated_model_features(metrics: pd.DataFrame, family_scores: pd.DataFrame) -> pd.DataFrame:
    score_columns = list(FAMILY_SCORE_DEFINITIONS)
    return metrics[METADATA_COLUMNS + CORE_METRIC_COLUMNS_V2].merge(
        family_scores[["subfield_id"] + score_columns],
        on="subfield_id",
        how="left",
        validate="one_to_one",
    )


def standardized_metric_matrix(metrics: pd.DataFrame) -> pd.DataFrame:
    values = metrics[CORE_METRIC_COLUMNS_V2].apply(pd.to_numeric, errors="coerce")
    values = values.fillna(values.median())
    std = values.std(ddof=0).replace(0, np.nan)
    return ((values - values.mean()) / std).fillna(0.0)


def run_pca(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = standardized_metric_matrix(metrics)
    n_components = min(5, len(x), len(CORE_METRIC_COLUMNS_V2))
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(x)
    score_columns = [f"PC{i + 1}" for i in range(n_components)]
    score_frame = metrics[METADATA_COLUMNS].copy()
    for idx, column in enumerate(score_columns):
        score_frame[column] = scores[:, idx]
    score_frame["explained_variance_ratio_pc1"] = float(pca.explained_variance_ratio_[0])
    if n_components >= 2:
        score_frame["explained_variance_ratio_pc2"] = float(
            pca.explained_variance_ratio_[1]
        )

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=score_columns,
        index=CORE_METRIC_COLUMNS_V2,
    ).reset_index(names="metric_name")
    for idx, ratio in enumerate(pca.explained_variance_ratio_, start=1):
        loadings[f"explained_variance_ratio_PC{idx}"] = float(ratio)
    return score_frame, loadings


def plot_histograms(metrics: pd.DataFrame, path: Path) -> None:
    n_cols = 5
    n_rows = int(np.ceil(len(CORE_METRIC_COLUMNS_V2) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(17, 18), dpi=150)
    for ax, metric_name in zip(axes.ravel(), CORE_METRIC_COLUMNS_V2):
        ax.hist(metrics[metric_name].dropna(), bins=20, color="#4c78a8", alpha=0.85)
        ax.set_title(metric_name, fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(CORE_METRIC_COLUMNS_V2) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_heatmap(correlation: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    image = ax.imshow(correlation.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlation.columns)))
    ax.set_yticks(range(len(correlation.index)))
    ax.set_xticklabels(correlation.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(correlation.index, fontsize=6)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_domain_family_scores(family_scores: pd.DataFrame, path: Path) -> None:
    score_columns = list(FAMILY_SCORE_DEFINITIONS)
    domains = list(family_scores["domain_display_name"].dropna().unique())
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), dpi=150)
    for ax, score_name in zip(axes.ravel(), score_columns):
        data = [
            family_scores.loc[
                family_scores["domain_display_name"] == domain,
                score_name,
            ].dropna()
            for domain in domains
        ]
        ax.boxplot(data, tick_labels=domains, showfliers=False)
        ax.set_title(score_name, fontsize=9)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
    for ax in axes.ravel()[len(score_columns) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_pca_scatter(scores: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    domains = pd.Categorical(scores["domain_display_name"])
    scatter = ax.scatter(scores["PC1"], scores["PC2"], c=domains.codes, cmap="tab10", s=28)
    ax.set_xlabel("Metric PCA PC1")
    ax.set_ylabel("Metric PCA PC2")
    ax.set_title("Exploratory PCA of Core Morphology Metrics")
    handles = []
    for code, label in enumerate(domains.categories):
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=scatter.cmap(scatter.norm(code)),
                markersize=5,
                label=str(label),
            )
        )
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_pca_loadings(loadings: pd.DataFrame, path: Path) -> None:
    top = loadings.assign(abs_pc1=loadings["PC1"].abs()).sort_values(
        "abs_pc1",
        ascending=False,
    ).head(15)
    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    ax.barh(top["metric_name"], top["PC1"], color="#5f9ea0")
    ax.invert_yaxis()
    ax.set_xlabel("PC1 loading")
    ax.set_title("Top Metric PCA PC1 Loadings")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input_path)
    dictionary_path = resolve_path(args.dictionary_path)
    output_dir = resolve_path(args.output_dir)
    paths = output_paths(output_dir)
    ensure_outputs(paths, args.overwrite)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing morphology metrics table: {display_path(input_path)}. "
            "Run scripts/11_compute_subfield_morphology_metrics.py first."
        )

    metrics = load_parquet(input_path)
    validate_core_columns(metrics)
    selected = select_rows(
        metrics,
        subfield_id=args.subfield_id,
        limit_subfields=args.limit_subfields,
    )

    stats = metric_descriptive_stats(selected)
    missingness = metric_missingness(stats)
    pearson = selected[CORE_METRIC_COLUMNS_V2].corr(method="pearson")
    spearman = selected[CORE_METRIC_COLUMNS_V2].corr(method="spearman")
    high_pairs = high_correlation_pairs(
        pearson,
        spearman,
        threshold=args.high_correlation_threshold,
    )
    low_variance = low_variance_metrics(stats)
    top_bottom = top_bottom_subfields(selected)
    domain_summary = group_metric_summary(selected, "domain_display_name")
    field_summary = group_metric_summary(selected, "field_display_name")
    duplicate_report = duplicate_subfield_names_report(selected)
    family_scores = compute_family_scores(selected)
    curated = curated_model_features(selected, family_scores)
    pca_scores, pca_loadings = run_pca(selected)

    stats.to_csv(paths["metric_descriptive_stats"], index=False)
    missingness.to_csv(paths["metric_missingness"], index=False)
    pearson.to_csv(paths["metric_correlation_matrix"])
    spearman.to_csv(paths["metric_spearman_correlation_matrix"])
    high_pairs.to_csv(paths["high_correlation_pairs"], index=False)
    low_variance.to_csv(paths["low_variance_metrics"], index=False)
    top_bottom.to_csv(paths["top_bottom_subfields_by_metric"], index=False)
    domain_summary.to_csv(paths["domain_metric_summary"], index=False)
    field_summary.to_csv(paths["field_metric_summary"], index=False)
    duplicate_report.to_csv(paths["duplicate_subfield_names_report"], index=False)
    family_scores.to_csv(paths["family_scores"], index=False)
    save_parquet(curated, paths["curated_model_features_parquet"])
    curated.to_csv(paths["curated_model_features_csv"], index=False)
    pca_scores.to_csv(paths["metric_pca_scores"], index=False)
    pca_loadings.to_csv(paths["metric_pca_loadings"], index=False)

    plot_histograms(selected, paths["core_metric_histograms"])
    plot_heatmap(pearson, paths["core_metric_correlation_heatmap"])
    plot_domain_family_scores(family_scores, paths["domain_family_score_boxplots"])
    plot_pca_scatter(pca_scores, paths["metric_pca_scatter"])
    plot_pca_loadings(pca_loadings, paths["metric_pca_loadings_figure"])

    dictionary_exists = dictionary_path.exists()
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_path": display_path(input_path),
        "dictionary_path": display_path(dictionary_path) if dictionary_exists else "",
        "output_dir": display_path(output_dir),
        "n_input_rows": int(len(metrics)),
        "n_rows_analyzed": int(len(selected)),
        "core_metric_columns_v2": CORE_METRIC_COLUMNS_V2,
        "diagnostic_metric_columns": DIAGNOSTIC_METRIC_COLUMNS,
        "family_score_definitions": {
            score: [{"metric": metric, "sign": sign} for metric, sign in components]
            for score, components in FAMILY_SCORE_DEFINITIONS.items()
        },
        "n_high_correlation_pairs": int(len(high_pairs)),
        "n_low_variance_metrics": int(len(low_variance)),
        "n_duplicate_display_name_rows": int(len(duplicate_report)),
        "n_duplicated_display_names": int(
            duplicate_report["subfield_display_name"].nunique()
        )
        if len(duplicate_report)
        else 0,
        "pca_explained_variance": {
            column: float(pca_scores[f"explained_variance_ratio_{column.lower()}"].iloc[0])
            for column in ["PC1", "PC2"]
            if f"explained_variance_ratio_{column.lower()}" in pca_scores.columns
        },
        "outputs": {key: display_path(path) for key, path in paths.items()},
        "notes": [
            "PCA is exploratory and is applied only to the final morphology metric table.",
            "No growth targets or prediction models are joined or estimated here.",
        ],
    }
    write_json(paths["morphology_analysis_summary"], summary)

    print(f"Analyzed {len(selected)} subfields")
    print(f"Wrote outputs under {display_path(output_dir)}")
    print(
        f"High-correlation pairs: {len(high_pairs)}; "
        f"low-variance metrics: {len(low_variance)}; "
        f"duplicate-name report rows: {len(duplicate_report)}"
    )


if __name__ == "__main__":
    main()
