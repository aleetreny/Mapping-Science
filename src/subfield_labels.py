from __future__ import annotations

import pandas as pd


LABEL_COLUMNS = [
    "subfield_label_unique",
    "subfield_label_short",
    "subfield_display_name_is_duplicated",
]

DUPLICATE_REPORT_COLUMNS = [
    "subfield_display_name",
    "n_distinct_subfield_ids",
    "subfield_id",
    "domain_display_name",
    "field_display_name",
    "subfield_label_unique",
]


def label_part(value: object, default: str = "Unknown") -> str:
    if value is None or pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def build_subfield_label_unique(
    *,
    subfield_id: object,
    domain_display_name: object,
    field_display_name: object,
    subfield_display_name: object,
) -> str:
    return (
        f"{label_part(subfield_id)} | "
        f"{label_part(domain_display_name)} / "
        f"{label_part(field_display_name)} / "
        f"{label_part(subfield_display_name)}"
    )


def build_subfield_label_short(
    *,
    subfield_id: object,
    field_display_name: object,
    subfield_display_name: object,
) -> str:
    return (
        f"{label_part(subfield_id)} | "
        f"{label_part(field_display_name)} / "
        f"{label_part(subfield_display_name)}"
    )


def canonical_subfield_display_column(frame: pd.DataFrame) -> str:
    if "subfield_display_name" in frame.columns:
        return "subfield_display_name"
    if "subfield_name" in frame.columns:
        return "subfield_name"
    raise ValueError("frame must contain subfield_display_name or subfield_name")


def add_subfield_label_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if "subfield_id" not in frame.columns:
        raise ValueError("frame must contain subfield_id")

    output = frame.copy()
    display_column = canonical_subfield_display_column(output)
    for column in ["domain_display_name", "field_display_name"]:
        if column not in output.columns:
            output[column] = "Unknown"

    output["subfield_label_unique"] = [
        build_subfield_label_unique(
            subfield_id=row.subfield_id,
            domain_display_name=row.domain_display_name,
            field_display_name=row.field_display_name,
            subfield_display_name=getattr(row, display_column),
        )
        for row in output[
            ["subfield_id", "domain_display_name", "field_display_name", display_column]
        ].itertuples(index=False)
    ]
    output["subfield_label_short"] = [
        build_subfield_label_short(
            subfield_id=row.subfield_id,
            field_display_name=row.field_display_name,
            subfield_display_name=getattr(row, display_column),
        )
        for row in output[
            ["subfield_id", "field_display_name", display_column]
        ].itertuples(index=False)
    ]

    duplicate_counts = output.groupby(display_column, dropna=False)[
        "subfield_id"
    ].transform(lambda values: values.astype(str).nunique())
    output["subfield_display_name_is_duplicated"] = duplicate_counts > 1
    return output


def duplicate_subfield_names_report(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=DUPLICATE_REPORT_COLUMNS)

    labelled = add_subfield_label_columns(frame)
    display_column = canonical_subfield_display_column(labelled)
    distinct_counts = labelled.groupby(display_column, dropna=False)["subfield_id"].transform(
        lambda values: values.astype(str).nunique()
    )
    duplicates = labelled.loc[distinct_counts > 1].copy()
    if duplicates.empty:
        return pd.DataFrame(columns=DUPLICATE_REPORT_COLUMNS)

    duplicates["n_distinct_subfield_ids"] = distinct_counts.loc[duplicates.index].astype(int)
    if display_column != "subfield_display_name":
        duplicates["subfield_display_name"] = duplicates[display_column]

    report = duplicates[
        [
            "subfield_display_name",
            "n_distinct_subfield_ids",
            "subfield_id",
            "domain_display_name",
            "field_display_name",
            "subfield_label_unique",
        ]
    ].drop_duplicates()
    return report.sort_values(
        ["subfield_display_name", "subfield_id"],
        kind="mergesort",
    ).reset_index(drop=True)
