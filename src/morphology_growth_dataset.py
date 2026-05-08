from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.growth_targets import display_path, read_table, write_table
from src.morphology_metrics import CORE_METRIC_COLUMNS_V2, DIAGNOSTIC_METRIC_COLUMNS
from src.openalex import short_openalex_id


METADATA_COLUMNS = [
    "subfield_id",
    "subfield_display_name",
    "subfield_label_unique",
    "subfield_label_short",
    "subfield_display_name_is_duplicated",
    "field_id",
    "field_display_name",
    "domain_id",
    "domain_display_name",
    "n_points",
]

FAMILY_SCORE_COLUMNS = [
    "diffuseness_score",
    "concentration_score",
    "fragmentation_score",
    "elongation_score",
    "temporal_dynamism_score",
    "directional_change_score",
    "expansion_score",
    "diversification_score",
]

GROWTH_TARGET_COLUMNS = [
    "annualized_log_growth",
    "growth_above_median",
    "domain_adjusted_annualized_log_growth",
    "growth_above_domain_median",
    "growth_label_global_int",
    "growth_label_domain_int",
]

GROWTH_AUDIT_COLUMNS = [
    "papers_2010_2019",
    "papers_2020_2025",
    "annual_rate_2010_2019",
    "annual_rate_2020_2025",
    "annualized_growth_abs",
    "annualized_growth_ratio",
    "annualized_growth_pct",
    "global_growth_median",
    "growth_rank",
    "growth_percentile",
    "domain_growth_median",
    "domain_growth_rank",
    "domain_growth_percentile",
    "input_year_min",
    "input_year_max",
    "target_year_min",
    "target_year_max",
    "input_years",
    "target_years",
    "n_years_with_counts_2010_2019",
    "n_years_with_counts_2020_2025",
    "has_zero_input_count",
    "has_zero_target_count",
    "count_source",
]

DERIVED_CONTROL_COLUMNS = [
    "log_papers_2010_2019",
    "log_papers_2020_2025",
    "log_annual_rate_2010_2019",
    "log_annual_rate_2020_2025",
]

SAFE_BASELINE_CONTROL_COLUMNS = [
    "log_papers_2010_2019",
    "log_annual_rate_2010_2019",
]

RAW_MORPHOLOGY_CONTROL_COLUMNS = [
    "year_min",
    "year_max",
    "metric_status",
    "metric_error_message",
    "metric_warning_message",
    "n_years_available",
    "n_early_points",
    "n_late_points",
    "n_density_entropy_years",
]

TARGET_LIKE_PATTERNS = [
    "growth",
    "papers_2020",
    "annual_rate_2020",
    "log_papers_2020",
    "log_annual_rate_2020",
    "target_year",
    "has_zero_target",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def normalize_subfield_ids(frame: pd.DataFrame) -> pd.DataFrame:
    if "subfield_id" not in frame.columns:
        raise ValueError("Input table must contain subfield_id")
    output = frame.copy()
    output["subfield_id"] = output["subfield_id"].map(short_openalex_id)
    output = output[output["subfield_id"].notna()].copy()
    output["subfield_id"] = output["subfield_id"].astype(str)
    return output


def validate_unique_subfield_ids(frame: pd.DataFrame, name: str) -> None:
    if "subfield_id" not in frame.columns:
        raise ValueError(f"{name} table must contain subfield_id")
    missing = int(frame["subfield_id"].isna().sum())
    if missing:
        raise ValueError(f"{name} table has {missing} missing subfield_id values")
    duplicate_ids = frame.loc[frame["subfield_id"].duplicated(), "subfield_id"].tolist()
    if duplicate_ids:
        examples = ", ".join(str(value) for value in duplicate_ids[:10])
        raise ValueError(f"{name} table has duplicate subfield_id values: {examples}")


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} table is missing columns: {', '.join(missing)}")


def coerce_bool_label(series: pd.Series, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    valid = {"true", "false", "1", "0", "yes", "no", "y", "n"}
    invalid = sorted(set(normalized.dropna()) - valid)
    if invalid:
        raise ValueError(f"{column} contains non-boolean values: {invalid[:10]}")
    return normalized.isin({"true", "1", "yes", "y"})


def prefixed_pca_scores(pca_scores: pd.DataFrame | None) -> tuple[pd.DataFrame | None, list[str]]:
    if pca_scores is None or pca_scores.empty:
        return None, []
    pca = normalize_subfield_ids(pca_scores)
    validate_unique_subfield_ids(pca, "PCA scores")
    pc_columns = [
        column
        for column in pca.columns
        if column.upper().startswith("PC") and pd.api.types.is_numeric_dtype(pca[column])
    ]
    rename_map = {column: f"metric_pca_{column.upper()}" for column in pc_columns}
    pca = pca[["subfield_id"] + pc_columns].rename(columns=rename_map)
    return pca, list(rename_map.values())


def add_missing_columns_from_source(
    dataset: pd.DataFrame,
    source: pd.DataFrame | None,
    *,
    columns: list[str],
    source_name: str,
) -> pd.DataFrame:
    if source is None or source.empty:
        return dataset
    src = normalize_subfield_ids(source)
    validate_unique_subfield_ids(src, source_name)
    selected = ["subfield_id"] + [
        column for column in columns if column in src.columns and column not in dataset.columns
    ]
    if len(selected) == 1:
        return dataset
    return dataset.merge(src[selected], on="subfield_id", how="left", validate="one_to_one")


def add_derived_controls(dataset: pd.DataFrame) -> pd.DataFrame:
    output = dataset.copy()
    for source, target in [
        ("papers_2010_2019", "log_papers_2010_2019"),
        ("papers_2020_2025", "log_papers_2020_2025"),
        ("annual_rate_2010_2019", "log_annual_rate_2010_2019"),
        ("annual_rate_2020_2025", "log_annual_rate_2020_2025"),
    ]:
        if source not in output.columns:
            raise ValueError(f"Cannot derive {target}; missing {source}")
        values = pd.to_numeric(output[source], errors="coerce")
        if values.isna().any():
            raise ValueError(f"Cannot derive {target}; {source} contains missing values")
        output[target] = np.log1p(values.astype(float))

    output["growth_above_median"] = coerce_bool_label(
        output["growth_above_median"], "growth_above_median"
    )
    output["growth_above_domain_median"] = coerce_bool_label(
        output["growth_above_domain_median"], "growth_above_domain_median"
    )
    output["growth_label_global_int"] = output["growth_above_median"].astype(int)
    output["growth_label_domain_int"] = output["growth_above_domain_median"].astype(int)
    return output


def is_target_like(column: str) -> bool:
    lowered = column.lower()
    return any(pattern in lowered for pattern in TARGET_LIKE_PATTERNS)


def predictor_group_names(feature_groups: dict[str, Any]) -> list[str]:
    names = [
        "recommended_primary_feature_columns",
        "recommended_family_feature_columns",
        "recommended_core_metric_feature_columns",
    ]
    modelling = feature_groups.get("modeling_feature_sets", {})
    names.extend(f"modeling_feature_sets.{name}" for name in modelling)
    return names


def columns_for_group(feature_groups: dict[str, Any], group_name: str) -> list[str]:
    if group_name.startswith("modeling_feature_sets."):
        set_name = group_name.split(".", 1)[1]
        return list(feature_groups.get("modeling_feature_sets", {}).get(set_name, []))
    return list(feature_groups.get(group_name, []))


def validate_predictor_groups(feature_groups: dict[str, Any]) -> None:
    target_columns = set(feature_groups["growth_target_columns"])
    problems: list[str] = []
    for group_name in predictor_group_names(feature_groups):
        for column in columns_for_group(feature_groups, group_name):
            if column in target_columns or is_target_like(column):
                problems.append(f"{group_name}:{column}")
    if problems:
        raise ValueError(
            "Target-like columns appear in predictor feature groups: "
            + ", ".join(problems)
        )


def build_feature_groups(dataset: pd.DataFrame, pca_score_columns: list[str]) -> dict[str, Any]:
    metadata_columns = [column for column in METADATA_COLUMNS if column in dataset.columns]
    growth_target_columns = [
        column for column in GROWTH_TARGET_COLUMNS if column in dataset.columns
    ]
    growth_control_columns = [
        column
        for column in GROWTH_AUDIT_COLUMNS + DERIVED_CONTROL_COLUMNS
        if column in dataset.columns
    ]
    core_columns = [
        column for column in CORE_METRIC_COLUMNS_V2 if column in dataset.columns
    ]
    diagnostic_columns = [
        column
        for column in RAW_MORPHOLOGY_CONTROL_COLUMNS + DIAGNOSTIC_METRIC_COLUMNS
        if column in dataset.columns
    ]
    family_columns = [column for column in FAMILY_SCORE_COLUMNS if column in dataset.columns]
    pca_columns = [column for column in pca_score_columns if column in dataset.columns]

    groups: dict[str, Any] = {
        "metadata_columns": metadata_columns,
        "growth_target_columns": growth_target_columns,
        "growth_control_columns": growth_control_columns,
        "core_morphology_metric_columns": core_columns,
        "diagnostic_morphology_metric_columns": diagnostic_columns,
        "family_score_columns": family_columns,
        "pca_score_columns": pca_columns,
        "recommended_primary_feature_columns": [
            column
            for column in ["log_papers_2010_2019"] + family_columns + ["domain_display_name"]
            if column in dataset.columns
        ],
        "recommended_family_feature_columns": [
            column
            for column in SAFE_BASELINE_CONTROL_COLUMNS + family_columns
            if column in dataset.columns
        ],
        "recommended_core_metric_feature_columns": [
            column
            for column in ["log_papers_2010_2019"] + core_columns + ["domain_display_name"]
            if column in dataset.columns
        ],
        "modeling_feature_sets": {
            "baseline_size_only": [
                column for column in ["log_papers_2010_2019"] if column in dataset.columns
            ],
            "baseline_domain_only": [
                column for column in ["domain_display_name"] if column in dataset.columns
            ],
            "baseline_size_domain": [
                column
                for column in ["log_papers_2010_2019", "domain_display_name"]
                if column in dataset.columns
            ],
            "family_scores": family_columns,
            "core_morphology_metrics": core_columns,
            "combined_primary": [
                column
                for column in ["log_papers_2010_2019"] + family_columns + ["domain_display_name"]
                if column in dataset.columns
            ],
        },
        "notes": [
            "Categorical metadata such as domain_display_name should be encoded in Stage 15.",
            "PCA score columns are exploratory diagnostics, not primary morphology features.",
            "Growth target and future-count columns are excluded from predictor feature sets.",
        ],
    }
    validate_predictor_groups(groups)
    return groups


def build_join_audit(morphology: pd.DataFrame, growth: pd.DataFrame) -> pd.DataFrame:
    morph = morphology[["subfield_id"]].drop_duplicates().assign(in_morphology=True)
    target = growth[["subfield_id"]].drop_duplicates().assign(in_growth=True)
    audit = morph.merge(target, on="subfield_id", how="outer")
    audit["in_morphology"] = audit["in_morphology"].fillna(False).astype(bool)
    audit["in_growth"] = audit["in_growth"].fillna(False).astype(bool)
    audit["joined"] = audit["in_morphology"] & audit["in_growth"]
    audit["join_status"] = np.select(
        [
            audit["joined"],
            audit["in_morphology"] & ~audit["in_growth"],
            ~audit["in_morphology"] & audit["in_growth"],
        ],
        ["matched", "missing_growth", "missing_morphology"],
        default="unknown",
    )
    return audit.sort_values("subfield_id", kind="mergesort").reset_index(drop=True)


def validate_dataset(
    dataset: pd.DataFrame,
    *,
    feature_groups: dict[str, Any],
    expected_rows: int,
) -> dict[str, Any]:
    errors: list[str] = []
    if expected_rows > 0 and len(dataset) != expected_rows:
        errors.append(f"joined row count {len(dataset)} != expected {expected_rows}")
    if dataset["subfield_id"].duplicated().any():
        errors.append("joined dataset has duplicate subfield_id values")

    core_columns = feature_groups["core_morphology_metric_columns"]
    missing_core_values = int(dataset[core_columns].isna().sum().sum()) if core_columns else 0
    if missing_core_values:
        errors.append(f"core morphology metrics contain {missing_core_values} missing values")

    target_columns = feature_groups["growth_target_columns"]
    missing_target_values = (
        int(dataset[target_columns].isna().sum().sum()) if target_columns else 0
    )
    if missing_target_values:
        errors.append(f"growth target columns contain {missing_target_values} missing values")

    missing_family_scores = [
        column for column in FAMILY_SCORE_COLUMNS if column not in dataset.columns
    ]
    if missing_family_scores:
        errors.append("missing family scores: " + ", ".join(missing_family_scores))

    if "density_entropy_slope_by_year" not in dataset.columns:
        errors.append("density_entropy_slope_by_year is missing")
    if "outlier_share_r_gt_1_5" in feature_groups["core_morphology_metric_columns"]:
        errors.append("outlier_share_r_gt_1_5 must not be a core feature")
    if "outlier_share_r_gt_1_5" in dataset.columns and "outlier_share_r_gt_1_5" not in feature_groups[
        "diagnostic_morphology_metric_columns"
    ]:
        errors.append("outlier_share_r_gt_1_5 is present but not marked diagnostic")

    validate_predictor_groups(feature_groups)
    if errors:
        raise ValueError("; ".join(errors))

    return {
        "missing_core_metric_values": missing_core_values,
        "missing_growth_target_values": missing_target_values,
        "contains_all_family_scores": not missing_family_scores,
        "contains_density_entropy_slope_by_year": True,
        "outlier_share_r_gt_1_5_marked_diagnostic": (
            "outlier_share_r_gt_1_5"
            not in dataset.columns
            or "outlier_share_r_gt_1_5"
            in feature_groups["diagnostic_morphology_metric_columns"]
        ),
        "duplicated_display_name_rows": int(
            dataset.get("subfield_display_name_is_duplicated", pd.Series(dtype=bool))
            .fillna(False)
            .astype(bool)
            .sum()
        ),
    }


def build_morphology_growth_dataset(
    *,
    morphology: pd.DataFrame,
    growth: pd.DataFrame,
    raw_morphology: pd.DataFrame | None = None,
    family_scores: pd.DataFrame | None = None,
    pca_scores: pd.DataFrame | None = None,
    expected_rows: int = 240,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    morph = normalize_subfield_ids(morphology)
    growth_targets = normalize_subfield_ids(growth)
    validate_unique_subfield_ids(morph, "morphology")
    validate_unique_subfield_ids(growth_targets, "growth")
    if expected_rows > 0:
        if len(morph) != expected_rows:
            raise ValueError(f"morphology row count {len(morph)} != expected {expected_rows}")
        if len(growth_targets) != expected_rows:
            raise ValueError(f"growth row count {len(growth_targets)} != expected {expected_rows}")

    morph_ids = set(morph["subfield_id"])
    growth_ids = set(growth_targets["subfield_id"])
    missing_growth = sorted(morph_ids - growth_ids)
    missing_morphology = sorted(growth_ids - morph_ids)
    if missing_growth or missing_morphology:
        messages = []
        if missing_growth:
            messages.append(f"morphology rows missing growth targets: {missing_growth[:10]}")
        if missing_morphology:
            messages.append(f"growth rows missing morphology features: {missing_morphology[:10]}")
        raise ValueError("; ".join(messages))

    require_columns(morph, CORE_METRIC_COLUMNS_V2, "morphology")
    require_columns(growth_targets, GROWTH_TARGET_COLUMNS[:4], "growth")

    growth_columns = [
        "subfield_id"
    ] + [
        column
        for column in GROWTH_TARGET_COLUMNS[:4] + GROWTH_AUDIT_COLUMNS
        if column in growth_targets.columns
    ]
    dataset = morph.merge(
        growth_targets[growth_columns],
        on="subfield_id",
        how="inner",
        validate="one_to_one",
    )
    dataset = add_derived_controls(dataset)
    dataset = add_missing_columns_from_source(
        dataset,
        raw_morphology,
        columns=RAW_MORPHOLOGY_CONTROL_COLUMNS + DIAGNOSTIC_METRIC_COLUMNS,
        source_name="raw morphology",
    )
    dataset = add_missing_columns_from_source(
        dataset,
        family_scores,
        columns=FAMILY_SCORE_COLUMNS,
        source_name="family scores",
    )
    pca, pca_score_columns = prefixed_pca_scores(pca_scores)
    if pca is not None:
        dataset = dataset.merge(pca, on="subfield_id", how="left", validate="one_to_one")

    feature_groups = build_feature_groups(dataset, pca_score_columns)
    audit = build_join_audit(morph, growth_targets)
    validation = validate_dataset(
        dataset,
        feature_groups=feature_groups,
        expected_rows=expected_rows,
    )

    ordered_columns = []
    for group_name in [
        "metadata_columns",
        "growth_target_columns",
        "growth_control_columns",
        "core_morphology_metric_columns",
        "family_score_columns",
        "diagnostic_morphology_metric_columns",
        "pca_score_columns",
    ]:
        for column in feature_groups[group_name]:
            if column in dataset.columns and column not in ordered_columns:
                ordered_columns.append(column)
    ordered_columns.extend([column for column in dataset.columns if column not in ordered_columns])
    dataset = dataset[ordered_columns].sort_values("subfield_id", kind="mergesort").reset_index(
        drop=True
    )
    return dataset, feature_groups, audit, validation


def class_balance(dataset: pd.DataFrame, column: str) -> dict[str, int]:
    counts = dataset[column].value_counts(dropna=False).to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def build_domain_target_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    return (
        dataset.groupby(["domain_id", "domain_display_name"], dropna=False)
        .agg(
            n_subfields=("subfield_id", "nunique"),
            mean_annualized_log_growth=("annualized_log_growth", "mean"),
            median_annualized_log_growth=("annualized_log_growth", "median"),
            share_growth_above_median=("growth_above_median", "mean"),
            mean_domain_adjusted_annualized_log_growth=(
                "domain_adjusted_annualized_log_growth",
                "mean",
            ),
            share_growth_above_domain_median=("growth_above_domain_median", "mean"),
        )
        .reset_index()
        .sort_values("domain_id", kind="mergesort")
    )


def build_family_score_target_summary(dataset: pd.DataFrame, family_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    high = dataset["growth_above_median"].astype(bool)
    low = ~high
    for column in family_columns:
        values = pd.to_numeric(dataset[column], errors="coerce")
        rows.append(
            {
                "family_score": column,
                "mean_score_high_growth": float(values[high].mean()),
                "mean_score_low_growth": float(values[low].mean()),
                "high_minus_low_difference": float(values[high].mean() - values[low].mean()),
                "pearson_annualized_log_growth": float(values.corr(dataset["annualized_log_growth"], method="pearson")),
                "spearman_annualized_log_growth": float(values.corr(dataset["annualized_log_growth"], method="spearman")),
                "pearson_domain_adjusted_growth": float(values.corr(dataset["domain_adjusted_annualized_log_growth"], method="pearson")),
                "spearman_domain_adjusted_growth": float(values.corr(dataset["domain_adjusted_annualized_log_growth"], method="spearman")),
            }
        )
    return pd.DataFrame(rows)


def build_target_feature_correlations(
    dataset: pd.DataFrame,
    *,
    core_columns: list[str],
    family_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    targets = ["annualized_log_growth", "domain_adjusted_annualized_log_growth"]
    for group_name, columns in [
        ("core_morphology_metric", core_columns),
        ("family_score", family_columns),
    ]:
        for column in columns:
            values = pd.to_numeric(dataset[column], errors="coerce")
            for target in targets:
                rows.append(
                    {
                        "feature_group": group_name,
                        "feature": column,
                        "target": target,
                        "pearson": float(values.corr(dataset[target], method="pearson")),
                        "spearman": float(values.corr(dataset[target], method="spearman")),
                    }
                )
    return pd.DataFrame(rows)


def build_growth_rankings(dataset: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    sections = []
    for target in ["annualized_log_growth", "domain_adjusted_annualized_log_growth"]:
        for section, ascending in [("top", False), ("bottom", True)]:
            frame = dataset.sort_values(
                [target, "subfield_id"],
                ascending=[ascending, True],
                kind="mergesort",
            ).head(n)
            rows = frame[
                [
                    "subfield_id",
                    "subfield_label_short",
                    "domain_display_name",
                    "annualized_log_growth",
                    "domain_adjusted_annualized_log_growth",
                    "growth_above_median",
                    "growth_above_domain_median",
                ]
            ].copy()
            rows.insert(0, "ranking_section", f"{section}_{target}")
            rows.insert(1, "rank_within_section", range(1, len(rows) + 1))
            sections.append(rows)
    return pd.concat(sections, ignore_index=True)


def build_family_difference_rankings(
    family_summary: pd.DataFrame,
    n: int = 20,
) -> pd.DataFrame:
    if family_summary.empty:
        return family_summary
    return family_summary.reindex(
        family_summary["high_minus_low_difference"].abs().sort_values(ascending=False).index
    ).head(n).reset_index(drop=True)


def column_group_for(column: str, feature_groups: dict[str, Any]) -> str:
    for group_name in [
        "metadata_columns",
        "growth_target_columns",
        "growth_control_columns",
        "core_morphology_metric_columns",
        "family_score_columns",
        "diagnostic_morphology_metric_columns",
        "pca_score_columns",
    ]:
        if column in feature_groups[group_name]:
            return group_name
    return "other"


def description_for_column(column: str, group: str) -> str:
    descriptions = {
        "subfield_id": "OpenAlex subfield identifier used as the sole join key.",
        "subfield_label_unique": "Unique human-readable label including id, domain, field, and subfield name.",
        "annualized_log_growth": "Main continuous target: log1p target annual rate minus log1p input annual rate.",
        "growth_above_median": "Main global binary target using a strict global median threshold.",
        "domain_adjusted_annualized_log_growth": "Annualized log growth minus the within-domain median.",
        "growth_above_domain_median": "Domain-adjusted binary target using a strict within-domain median threshold.",
        "growth_label_global_int": "0/1 integer version of growth_above_median.",
        "growth_label_domain_int": "0/1 integer version of growth_above_domain_median.",
        "log_papers_2010_2019": "log1p of input-window publication count; safe size control.",
        "log_annual_rate_2010_2019": "log1p of input-window annual publication rate; safe size control.",
        "density_entropy_slope_by_year": "Core temporal morphology metric from 2010-2019 only.",
        "outlier_share_r_gt_1_5": "Diagnostic sparse outlier-share metric, not a core feature.",
    }
    if column in descriptions:
        return descriptions[column]
    if group == "core_morphology_metric_columns":
        return "Core 2010-2019 morphology metric recommended for modelling."
    if group == "family_score_columns":
        return "Interpretable robust-z morphology family score from Stage 12."
    if group == "growth_control_columns":
        return "Growth-window audit/control column; use only safe pre-2020 controls as predictors."
    if group == "diagnostic_morphology_metric_columns":
        return "Morphology diagnostic or quality-control column."
    if group == "pca_score_columns":
        return "Exploratory PCA score computed from the morphology metric table."
    if group == "metadata_columns":
        return "Identifier, label, or taxonomy metadata."
    return "Additional joined dataset column."


def build_data_dictionary(dataset: pd.DataFrame, feature_groups: dict[str, Any]) -> pd.DataFrame:
    predictor_candidates = set()
    for group_name in predictor_group_names(feature_groups):
        predictor_candidates.update(columns_for_group(feature_groups, group_name))
    target_columns = set(feature_groups["growth_target_columns"])
    control_columns = set(feature_groups["growth_control_columns"])
    diagnostic_columns = set(
        feature_groups["diagnostic_morphology_metric_columns"]
        + feature_groups["pca_score_columns"]
    )
    rows = []
    for column in dataset.columns:
        group = column_group_for(column, feature_groups)
        rows.append(
            {
                "column_name": column,
                "column_group": group,
                "description": description_for_column(column, group),
                "is_predictor_candidate": column in predictor_candidates,
                "is_target": column in target_columns,
                "is_control": column in control_columns,
                "is_diagnostic": column in diagnostic_columns,
            }
        )
    return pd.DataFrame(rows)


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def build_summary_payload(
    *,
    dataset: pd.DataFrame,
    morphology_rows: int,
    growth_rows: int,
    audit: pd.DataFrame,
    feature_groups: dict[str, Any],
    validation: dict[str, Any],
    input_paths: dict[str, str],
    output_paths: dict[str, str],
    warnings: list[str],
    assumptions: list[str],
) -> dict[str, Any]:
    return {
        "created_at": utc_now_iso(),
        "input_paths": input_paths,
        "output_paths": output_paths,
        "n_morphology_rows": int(morphology_rows),
        "n_growth_rows": int(growth_rows),
        "n_joined_rows": int(len(dataset)),
        "n_unmatched_morphology_rows": int((audit["join_status"] == "missing_growth").sum()),
        "n_unmatched_growth_rows": int((audit["join_status"] == "missing_morphology").sum()),
        "class_balance_growth_above_median": class_balance(dataset, "growth_above_median"),
        "class_balance_growth_above_domain_median": class_balance(
            dataset, "growth_above_domain_median"
        ),
        "n_duplicated_display_name_rows": validation["duplicated_display_name_rows"],
        "feature_group_sizes": {
            key: len(value)
            for key, value in feature_groups.items()
            if isinstance(value, list)
        },
        "modeling_feature_set_sizes": {
            key: len(value)
            for key, value in feature_groups.get("modeling_feature_sets", {}).items()
        },
        "validation": validation,
        "warnings": warnings,
        "assumptions": assumptions,
    }


def save_outputs(
    *,
    dataset: pd.DataFrame,
    feature_groups: dict[str, Any],
    audit: pd.DataFrame,
    dictionary: pd.DataFrame,
    domain_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
    correlations: pd.DataFrame,
    growth_rankings: pd.DataFrame,
    family_difference_rankings: pd.DataFrame,
    summary: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    write_table(dataset, paths["output_path"])
    write_table(dataset, paths["output_csv"])
    write_json(paths["feature_groups_path"], feature_groups)
    write_json(paths["summary_path"], summary)
    write_table(dictionary, paths["dictionary_path"])
    write_table(audit, paths["join_audit_path"])
    write_table(correlations, paths["target_feature_correlations_path"])
    write_table(family_summary, paths["family_score_target_summary_path"])
    write_table(domain_summary, paths["domain_target_summary_path"])
    write_table(growth_rankings, paths["growth_rankings_path"])
    write_table(family_difference_rankings, paths["family_difference_rankings_path"])


def short_text(value: object, max_len: int = 40) -> str:
    text = "" if value is None or pd.isna(value) else str(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_target_distribution(dataset: pd.DataFrame, path: Path, dpi: int = 150) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(dataset["annualized_log_growth"], bins=24, color="#4C78A8", edgecolor="white")
    axes[0].axvline(dataset["annualized_log_growth"].median(), color="#C44E52", linewidth=2)
    axes[0].set_title("Annualized Log Growth")
    axes[0].set_xlabel("annualized_log_growth")
    counts = dataset["growth_above_median"].value_counts().sort_index()
    axes[1].bar([str(index) for index in counts.index], counts.values, color="#59A14F")
    axes[1].set_title("Global Growth Label Balance")
    axes[1].set_xlabel("growth_above_median")
    axes[1].set_ylabel("Subfields")
    save_figure(fig, path, dpi)


def plot_domain_adjusted_target_distribution(
    dataset: pd.DataFrame, path: Path, dpi: int = 150
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(
        dataset["domain_adjusted_annualized_log_growth"],
        bins=24,
        color="#F28E2B",
        edgecolor="white",
    )
    axes[0].axvline(0, color="#C44E52", linewidth=2)
    axes[0].set_title("Domain-Adjusted Growth")
    axes[0].set_xlabel("domain_adjusted_annualized_log_growth")
    counts = dataset["growth_above_domain_median"].value_counts().sort_index()
    axes[1].bar([str(index) for index in counts.index], counts.values, color="#4C78A8")
    axes[1].set_title("Domain Growth Label Balance")
    axes[1].set_xlabel("growth_above_domain_median")
    axes[1].set_ylabel("Subfields")
    save_figure(fig, path, dpi)


def plot_growth_by_domain_boxplot(dataset: pd.DataFrame, path: Path, dpi: int = 150) -> None:
    plot_frame = dataset.copy()
    plot_frame["_domain_label"] = plot_frame["domain_display_name"].fillna("Unknown").astype(str)
    order = (
        plot_frame.groupby("_domain_label")["annualized_log_growth"]
        .median()
        .sort_values()
        .index
    )
    values = [
        plot_frame.loc[plot_frame["_domain_label"] == domain, "annualized_log_growth"].to_numpy()
        for domain in order
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.boxplot(values, labels=[short_text(domain, 26) for domain in order], patch_artist=True)
    ax.axhline(dataset["annualized_log_growth"].median(), color="#C44E52", linestyle="--")
    ax.set_title("Annualized Growth By Domain")
    ax.set_ylabel("annualized_log_growth")
    ax.tick_params(axis="x", rotation=30)
    save_figure(fig, path, dpi)


def plot_family_scores_vs_growth(
    dataset: pd.DataFrame,
    family_columns: list[str],
    path: Path,
    dpi: int = 150,
) -> None:
    n_cols = 4
    n_rows = math.ceil(len(family_columns) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.2 * n_rows), squeeze=False)
    for ax, column in zip(axes.ravel(), family_columns):
        ax.scatter(dataset[column], dataset["annualized_log_growth"], s=18, alpha=0.75, color="#4C78A8")
        ax.axhline(0, color="#333333", linestyle="--", linewidth=0.8)
        ax.set_title(short_text(column, 28))
        ax.set_xlabel(column)
        ax.set_ylabel("growth")
    for ax in axes.ravel()[len(family_columns):]:
        ax.axis("off")
    save_figure(fig, path, dpi)


def plot_family_scores_by_growth_label(
    dataset: pd.DataFrame,
    family_columns: list[str],
    path: Path,
    dpi: int = 150,
) -> None:
    n_cols = 4
    n_rows = math.ceil(len(family_columns) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.2 * n_rows), squeeze=False)
    low = dataset.loc[~dataset["growth_above_median"].astype(bool)]
    high = dataset.loc[dataset["growth_above_median"].astype(bool)]
    for ax, column in zip(axes.ravel(), family_columns):
        ax.boxplot([low[column].dropna(), high[column].dropna()], labels=["low", "high"])
        ax.set_title(short_text(column, 28))
        ax.set_ylabel("score")
    for ax in axes.ravel()[len(family_columns):]:
        ax.axis("off")
    save_figure(fig, path, dpi)


def plot_core_metric_target_correlation_barplot(
    correlations: pd.DataFrame,
    path: Path,
    dpi: int = 150,
) -> None:
    subset = correlations[
        (correlations["feature_group"] == "core_morphology_metric")
        & (correlations["target"] == "annualized_log_growth")
    ].copy()
    subset = subset.sort_values("pearson", kind="mergesort")
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = np.where(subset["pearson"] >= 0, "#4C78A8", "#C44E52")
    ax.barh(subset["feature"], subset["pearson"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title("Core Metric Correlation With Annualized Growth")
    ax.set_xlabel("Pearson correlation")
    ax.tick_params(axis="y", labelsize=8)
    save_figure(fig, path, dpi)


def plot_pca_or_family_scatter(
    dataset: pd.DataFrame,
    feature_groups: dict[str, Any],
    path: Path,
    dpi: int = 150,
) -> None:
    pca_columns = feature_groups["pca_score_columns"]
    if len(pca_columns) >= 2:
        x_col, y_col = pca_columns[:2]
        title = "Metric PCA Scores By Growth Label"
    else:
        family_columns = feature_groups["family_score_columns"]
        x_col, y_col = family_columns[:2]
        title = "Family Score Space By Growth Label"
    fig, ax = plt.subplots(figsize=(7, 5.5))
    labels = dataset["growth_above_median"].astype(bool)
    ax.scatter(dataset.loc[~labels, x_col], dataset.loc[~labels, y_col], label="low", alpha=0.75, s=28)
    ax.scatter(dataset.loc[labels, x_col], dataset.loc[labels, y_col], label="high", alpha=0.75, s=28)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend(frameon=False)
    save_figure(fig, path, dpi)


def write_stage14_figures(
    *,
    dataset: pd.DataFrame,
    feature_groups: dict[str, Any],
    correlations: pd.DataFrame,
    figures_dir: Path,
    dpi: int = 150,
) -> dict[str, Path]:
    paths = {
        "target_distribution": figures_dir / "target_distribution.png",
        "domain_adjusted_target_distribution": figures_dir
        / "domain_adjusted_target_distribution.png",
        "growth_by_domain_boxplot": figures_dir / "growth_by_domain_boxplot.png",
        "family_scores_vs_growth": figures_dir / "family_scores_vs_growth.png",
        "family_scores_by_growth_label": figures_dir / "family_scores_by_growth_label.png",
        "core_metric_target_correlation_barplot": figures_dir
        / "core_metric_target_correlation_barplot.png",
        "morphology_growth_pca_or_family_scatter": figures_dir
        / "morphology_growth_pca_or_family_scatter.png",
    }
    family_columns = feature_groups["family_score_columns"]
    plot_target_distribution(dataset, paths["target_distribution"], dpi=dpi)
    plot_domain_adjusted_target_distribution(
        dataset, paths["domain_adjusted_target_distribution"], dpi=dpi
    )
    plot_growth_by_domain_boxplot(dataset, paths["growth_by_domain_boxplot"], dpi=dpi)
    plot_family_scores_vs_growth(
        dataset, family_columns, paths["family_scores_vs_growth"], dpi=dpi
    )
    plot_family_scores_by_growth_label(
        dataset, family_columns, paths["family_scores_by_growth_label"], dpi=dpi
    )
    plot_core_metric_target_correlation_barplot(
        correlations, paths["core_metric_target_correlation_barplot"], dpi=dpi
    )
    plot_pca_or_family_scatter(
        dataset, feature_groups, paths["morphology_growth_pca_or_family_scatter"], dpi=dpi
    )
    return paths
