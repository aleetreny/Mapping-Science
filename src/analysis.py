from __future__ import annotations

import pandas as pd

from src.extraction_versions import (
    analysis_years,
    n_downloaded_works_column,
    n_valid_works_column,
    target_per_year_from_config,
    target_total_from_config,
)


ANALYSIS_SUBFIELDS_COLUMNS = [
    "subfield_id",
    "subfield_display_name",
    "field_id",
    "field_display_name",
    "domain_id",
    "domain_display_name",
    "n_valid_works",
    "planned_works",
    "shortfall",
    "main_analysis_eligible_2500",
    "robustness_eligible_500",
    "is_low_sample",
    "exclusion_reason",
]

VERSIONED_ANALYSIS_SUBFIELDS_COLUMNS = [
    "subfield_id",
    "subfield_display_name",
    "field_id",
    "field_display_name",
    "domain_id",
    "domain_display_name",
    "n_valid_works_period",
    "n_downloaded_works_period",
    "n_years_with_downloaded_works",
    "n_years_reaching_400",
    "n_years_below_400",
    "planned_works",
    "shortfall",
    "eligible_10000_full_period",
    "eligible_min_5000_full_period",
    "eligible_for_temporal_5year_exploration",
]


def eligibility_flags(n_valid_works: int) -> dict[str, object]:
    n_valid = int(n_valid_works)
    main_eligible = n_valid >= 2500
    robustness_eligible = n_valid >= 500
    if main_eligible:
        reason = "main_analysis_included"
    elif robustness_eligible:
        reason = "below_2500_valid_works"
    else:
        reason = "below_500_valid_works"

    return {
        "main_analysis_eligible_2500": main_eligible,
        "robustness_eligible_500": robustness_eligible,
        "is_low_sample": n_valid < 2500,
        "exclusion_reason": reason,
    }


def build_analysis_subfields(
    works_text: pd.DataFrame,
    plan: pd.DataFrame,
) -> pd.DataFrame:
    if plan.empty:
        return pd.DataFrame(columns=ANALYSIS_SUBFIELDS_COLUMNS)

    plan = plan.copy()
    works = works_text.copy()
    plan["subfield_id"] = plan["subfield_id"].astype(str)
    if not works.empty:
        works["subfield_id"] = works["subfield_id"].astype(str)

    metadata_cols = [
        "subfield_id",
        "subfield_display_name",
        "field_id",
        "field_display_name",
        "domain_id",
        "domain_display_name",
    ]
    metadata = plan[metadata_cols].drop_duplicates("subfield_id")

    if "planned_sample_size" in plan.columns:
        planned = (
            plan.groupby("subfield_id")["planned_sample_size"]
            .sum()
            .reset_index(name="planned_works")
        )
    else:
        planned = plan[["subfield_id", "planned_works"]].drop_duplicates("subfield_id")

    if works.empty:
        counts = pd.DataFrame({"subfield_id": metadata["subfield_id"], "n_valid_works": 0})
    else:
        counts = (
            works.groupby("subfield_id")
            .size()
            .reset_index(name="n_valid_works")
        )

    result = metadata.merge(planned, on="subfield_id", how="left").merge(
        counts, on="subfield_id", how="left"
    )
    result["planned_works"] = result["planned_works"].fillna(0).astype("int64")
    result["n_valid_works"] = result["n_valid_works"].fillna(0).astype("int64")
    result["shortfall"] = (
        result["planned_works"] - result["n_valid_works"]
    ).clip(lower=0).astype("int64")

    flags = result["n_valid_works"].apply(eligibility_flags).apply(pd.Series)
    result = pd.concat([result, flags], axis=1)

    return result[ANALYSIS_SUBFIELDS_COLUMNS].sort_values("subfield_id").reset_index(
        drop=True
    )


def build_versioned_analysis_subfields(
    works_text: pd.DataFrame,
    sample_plan: pd.DataFrame,
    corpus_plan: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    if sample_plan.empty and corpus_plan.empty:
        return pd.DataFrame(columns=VERSIONED_ANALYSIS_SUBFIELDS_COLUMNS)

    years = analysis_years(config)
    n_years = len(years)
    target_per_year = target_per_year_from_config(config)
    target_total = target_total_from_config(config)
    min_temporal_downloaded = max(5000, target_per_year * 10)
    valid_col = n_valid_works_column(config)

    plan_source = sample_plan.copy() if not sample_plan.empty else corpus_plan.copy()
    plan_source["subfield_id"] = plan_source["subfield_id"].astype(str)
    metadata_cols = [
        "subfield_id",
        "subfield_display_name",
        "field_id",
        "field_display_name",
        "domain_id",
        "domain_display_name",
    ]
    for column in metadata_cols:
        if column not in plan_source.columns:
            plan_source[column] = pd.NA
    metadata = plan_source[metadata_cols].drop_duplicates("subfield_id")

    if "planned_sample_size" in sample_plan.columns:
        planned = (
            sample_plan.assign(subfield_id=sample_plan["subfield_id"].astype(str))
            .groupby("subfield_id")["planned_sample_size"]
            .sum()
            .reset_index(name="planned_works")
        )
    elif "planned_sample_size" in corpus_plan.columns:
        planned = corpus_plan[["subfield_id", "planned_sample_size"]].rename(
            columns={"planned_sample_size": "planned_works"}
        )
        planned["subfield_id"] = planned["subfield_id"].astype(str)
    else:
        planned = pd.DataFrame({"subfield_id": metadata["subfield_id"], "planned_works": 0})

    if valid_col in corpus_plan.columns:
        valid = corpus_plan[["subfield_id", valid_col]].rename(
            columns={valid_col: "n_valid_works_period"}
        )
        valid["subfield_id"] = valid["subfield_id"].astype(str)
    else:
        valid = pd.DataFrame(
            {"subfield_id": metadata["subfield_id"], "n_valid_works_period": 0}
        )

    works = works_text.copy()
    if works.empty or "subfield_id" not in works.columns:
        downloaded = pd.DataFrame(
            {"subfield_id": metadata["subfield_id"], "n_downloaded_works_period": 0}
        )
        annual = pd.DataFrame(
            {
                "subfield_id": metadata["subfield_id"],
                "n_years_with_downloaded_works": 0,
                "n_years_reaching_400": 0,
            }
        )
    else:
        works["subfield_id"] = works["subfield_id"].astype(str)
        if "publication_year" in works.columns:
            years_numeric = pd.to_numeric(works["publication_year"], errors="coerce")
            works = works.loc[years_numeric.isin(years)].copy()
        downloaded = (
            works.groupby("subfield_id")
            .size()
            .reset_index(name="n_downloaded_works_period")
        )
        yearly_counts = (
            works.groupby(["subfield_id", "publication_year"])
            .size()
            .reset_index(name="downloaded_works")
        )
        annual = (
            yearly_counts.groupby("subfield_id")["downloaded_works"]
            .agg(
                n_years_with_downloaded_works=lambda values: int((values > 0).sum()),
                n_years_reaching_400=lambda values: int(
                    (values >= target_per_year).sum()
                ),
            )
            .reset_index()
        )

    result = (
        metadata.merge(valid, on="subfield_id", how="left")
        .merge(downloaded, on="subfield_id", how="left")
        .merge(annual, on="subfield_id", how="left")
        .merge(planned, on="subfield_id", how="left")
    )
    numeric_cols = [
        "n_valid_works_period",
        "n_downloaded_works_period",
        "n_years_with_downloaded_works",
        "n_years_reaching_400",
        "planned_works",
    ]
    for column in numeric_cols:
        result[column] = result[column].fillna(0).astype("int64")
    result["n_years_below_400"] = (n_years - result["n_years_reaching_400"]).clip(
        lower=0
    )
    result["shortfall"] = (
        result["planned_works"] - result["n_downloaded_works_period"]
    ).clip(lower=0)
    result["eligible_10000_full_period"] = (
        result["n_downloaded_works_period"] >= target_total
    )
    result["eligible_min_5000_full_period"] = (
        result["n_downloaded_works_period"] >= 5000
    )
    result["eligible_for_temporal_5year_exploration"] = (
        (result["n_downloaded_works_period"] >= min_temporal_downloaded)
        & (result["n_years_with_downloaded_works"] >= min(20, n_years))
    )

    valid_name = n_valid_works_column(config)
    downloaded_name = n_downloaded_works_column(config)
    result = result.rename(
        columns={
            "n_valid_works_period": valid_name,
            "n_downloaded_works_period": downloaded_name,
        }
    )
    columns = [
        valid_name if column == "n_valid_works_period" else column
        for column in VERSIONED_ANALYSIS_SUBFIELDS_COLUMNS
    ]
    columns = [
        downloaded_name if column == "n_downloaded_works_period" else column
        for column in columns
    ]
    return result[columns].sort_values("subfield_id").reset_index(drop=True)
