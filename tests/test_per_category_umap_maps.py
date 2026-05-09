from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.per_category_umap_maps import (
    MANIFEST_COLUMNS,
    category_manifest_frame,
    category_manifest_row,
    filter_category_input_window,
    list_groups,
    make_group_lookup,
    sample_group_rows,
    select_attempted_groups,
    validate_category_index_columns,
)


def synthetic_index() -> pd.DataFrame:
    records = []
    row_id = 0
    for domain_id, domain_name, field_id, field_name, subfield_id in [
        ("D1", "Domain 1", "F1", "Field 1", "S1"),
        ("D1", "Domain 1", "F1", "Field 1", "S2"),
        ("D1", "Domain 1", "F2", "Field 2", "S3"),
        ("D2", "Domain 2", "F3", "Field 3", "S4"),
    ]:
        for offset in range(4):
            records.append(
                {
                    "analysis_row_id": row_id,
                    "work_id": f"W{row_id:03d}",
                    "subfield_id": subfield_id,
                    "subfield_display_name": f"Subfield {subfield_id}",
                    "field_id": field_id,
                    "field_display_name": field_name,
                    "domain_id": domain_id,
                    "domain_display_name": domain_name,
                    "primary_topic_id": f"T{row_id}",
                    "primary_topic_display_name": f"Topic {row_id}",
                    "publication_year": 2010 + offset,
                    "main_analysis_eligible_2500": True,
                }
            )
            row_id += 1
    records.append(
        {
            "analysis_row_id": row_id,
            "work_id": "W-outside",
            "subfield_id": "S1",
            "subfield_display_name": "Subfield S1",
            "field_id": "F1",
            "field_display_name": "Field 1",
            "domain_id": "D1",
            "domain_display_name": "Domain 1",
            "primary_topic_id": f"T{row_id}",
            "primary_topic_display_name": f"Topic {row_id}",
            "publication_year": 2026,
            "main_analysis_eligible_2500": True,
        }
    )
    return pd.DataFrame(records)


def test_validate_category_index_columns_fails_clearly() -> None:
    incomplete = synthetic_index().drop(columns=["field_id"])

    with pytest.raises(ValueError, match="field_id"):
        validate_category_index_columns(incomplete)


def test_grouping_by_field_and_domain_works() -> None:
    index = filter_category_input_window(synthetic_index(), year_min=2010, year_max=2025)
    fields = list_groups(index, level="field")
    domains = list_groups(index, level="domain")

    assert fields["group_id"].tolist() == ["F1", "F2", "F3"]
    assert domains["group_id"].tolist() == ["D1", "D2"]
    assert make_group_lookup(index, level="field")["F1"]["subfield_id"].nunique() == 2


def test_select_attempted_groups_filters_and_limits() -> None:
    groups = list_groups(synthetic_index(), level="field")

    selected = select_attempted_groups(groups, group_id=None, limit_groups=2)
    one = select_attempted_groups(groups, group_id="F2", limit_groups=None)

    assert selected["group_id"].tolist() == ["F1", "F2"]
    assert one["group_id"].tolist() == ["F2"]


def test_sample_group_rows_is_deterministic() -> None:
    index = synthetic_index().sample(frac=1, random_state=10)

    first = sample_group_rows(
        index,
        max_papers=5,
        random_state=42,
        group_id="F1",
    )
    second = sample_group_rows(
        index.sample(frac=1, random_state=11),
        max_papers=5,
        random_state=42,
        group_id="F1",
    )

    assert first["analysis_row_id"].tolist() == second["analysis_row_id"].tolist()
    assert len(first) == 5


def test_manifest_has_expected_columns_and_skip_status() -> None:
    manifest = category_manifest_frame(
        [
            category_manifest_row(
                level="field",
                group_id="F1",
                group_name="Field 1",
                n_available=3,
                n_used=0,
                year_min=2010,
                year_max=2025,
                status="skipped",
                error_message="Only 3 papers available",
                umap_n_neighbors=30,
                umap_min_dist=0.05,
                umap_metric="cosine",
                random_state=42,
                max_papers_per_group=10000,
                sampling_applied=False,
                color_by="subfield_display_name",
            )
        ]
    )

    assert manifest.columns.tolist() == MANIFEST_COLUMNS
    assert manifest.loc[0, "status"] == "skipped"
    assert manifest.loc[0, "n_available"] == 3
