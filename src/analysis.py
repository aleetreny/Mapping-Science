from __future__ import annotations

import pandas as pd


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
