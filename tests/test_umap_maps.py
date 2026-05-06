import numpy as np
import pandas as pd

from src.umap_maps import (
    UMAP_OUTPUT_COLUMNS,
    balanced_sample_by_subfield,
    build_umap_output_frame,
)


def analysis_index_fixture() -> pd.DataFrame:
    records = []
    row_id = 0
    for subfield_id, n_rows in [("s1", 5), ("s2", 3), ("s3", 1)]:
        for offset in range(n_rows):
            records.append(
                {
                    "work_id": f"{subfield_id}-w{offset}",
                    "analysis_row_id": row_id,
                    "subfield_id": subfield_id,
                    "subfield_display_name": f"Subfield {subfield_id}",
                    "field_id": "f1",
                    "field_display_name": "Field 1",
                    "domain_id": "d1",
                    "domain_display_name": "Domain 1",
                    "primary_topic_id": f"topic-{row_id}",
                    "primary_topic_display_name": f"Topic {row_id}",
                    "publication_year": 2018 + (offset % 2),
                }
            )
            row_id += 1
    return pd.DataFrame(records)


def test_balanced_sampling_limits_rows_per_subfield() -> None:
    sampled = balanced_sample_by_subfield(
        analysis_index_fixture(),
        sample_per_subfield=2,
        random_seed=42,
    )

    counts = sampled.groupby("subfield_id").size().to_dict()
    assert counts == {"s1": 2, "s2": 2, "s3": 1}
    assert "analysis_row_id" in sampled.columns


def test_balanced_sampling_is_reproducible() -> None:
    first = balanced_sample_by_subfield(
        analysis_index_fixture(),
        sample_per_subfield=2,
        random_seed=7,
    )
    second = balanced_sample_by_subfield(
        analysis_index_fixture(),
        sample_per_subfield=2,
        random_seed=7,
    )

    assert first["analysis_row_id"].tolist() == second["analysis_row_id"].tolist()


def test_umap_output_frame_has_required_schema() -> None:
    sampled = balanced_sample_by_subfield(
        analysis_index_fixture(),
        sample_per_subfield=1,
        random_seed=42,
    )
    coordinates = np.column_stack(
        [
            np.arange(len(sampled), dtype=float),
            np.arange(len(sampled), dtype=float) + 10,
        ]
    )

    output = build_umap_output_frame(sampled, coordinates)

    assert output.columns.tolist() == UMAP_OUTPUT_COLUMNS
    assert {"umap_x", "umap_y"}.issubset(output.columns)
    assert output["analysis_row_id"].tolist() == sampled["analysis_row_id"].tolist()
