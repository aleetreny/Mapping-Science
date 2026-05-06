from __future__ import annotations

import numpy as np
import pandas as pd


UMAP_OUTPUT_COLUMNS = [
    "work_id",
    "analysis_row_id",
    "subfield_id",
    "subfield_display_name",
    "field_id",
    "field_display_name",
    "domain_id",
    "domain_display_name",
    "primary_topic_id",
    "primary_topic_display_name",
    "publication_year",
    "umap_x",
    "umap_y",
]


def balanced_sample_by_subfield(
    analysis_index: pd.DataFrame,
    sample_per_subfield: int,
    random_seed: int,
) -> pd.DataFrame:
    required = {"analysis_row_id", "subfield_id", "publication_year", "work_id"}
    missing = required - set(analysis_index.columns)
    if missing:
        raise ValueError(f"analysis index missing columns: {', '.join(sorted(missing))}")
    if sample_per_subfield <= 0:
        raise ValueError("sample_per_subfield must be positive")

    ordered = analysis_index.sort_values(
        ["subfield_id", "publication_year", "work_id"],
        kind="mergesort",
    )
    rng = np.random.default_rng(random_seed)
    sampled_groups: list[pd.DataFrame] = []
    for _, group in ordered.groupby("subfield_id", sort=True):
        if len(group) <= sample_per_subfield:
            sampled = group.copy()
        else:
            state = int(rng.integers(0, np.iinfo(np.int32).max))
            sampled = group.sample(n=sample_per_subfield, random_state=state)
        sampled_groups.append(sampled)

    if not sampled_groups:
        return ordered.head(0).copy()

    return (
        pd.concat(sampled_groups, ignore_index=True)
        .sort_values(["subfield_id", "publication_year", "work_id"], kind="mergesort")
        .reset_index(drop=True)
    )


def build_umap_output_frame(
    sampled_index: pd.DataFrame,
    coordinates: np.ndarray,
) -> pd.DataFrame:
    if coordinates.shape != (len(sampled_index), 2):
        raise ValueError(
            f"coordinates shape {coordinates.shape} does not match sampled rows "
            f"({len(sampled_index)}, 2)"
        )

    output = sampled_index.copy()
    output["umap_x"] = coordinates[:, 0]
    output["umap_y"] = coordinates[:, 1]
    for column in UMAP_OUTPUT_COLUMNS:
        if column not in output.columns:
            output[column] = pd.NA
    return output[UMAP_OUTPUT_COLUMNS]
