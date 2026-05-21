from __future__ import annotations

import pandas as pd

from src.static_discipline_profiles import (
    build_static_discipline_profile_outputs,
    metric_rankings,
    pairwise_profile_distances,
    profile_table,
)
from src.temporal_common import REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS


def tiny_reduced_core() -> pd.DataFrame:
    rows = []
    for idx, (subfield_id, field_id, domain_id, base) in enumerate(
        [
            ("S1", "F1", "D1", 0.1),
            ("S2", "F1", "D1", 0.2),
            ("S3", "F2", "D1", 0.8),
        ]
    ):
        row = {
            "subfield_id": subfield_id,
            "subfield_display_name": f"Subfield {idx}",
            "field_id": field_id,
            "field_display_name": f"Field {field_id}",
            "domain_id": domain_id,
            "domain_display_name": f"Domain {domain_id}",
            "metric_status": "completed",
        }
        for metric_idx, metric in enumerate(REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS):
            row[metric] = base + metric_idx * 0.01
        rows.append(row)
    return pd.DataFrame(rows)


def test_profile_table_aggregates_subfield_field_and_domain_levels() -> None:
    profiles = profile_table(tiny_reduced_core())

    assert set(profiles["level"]) == {"subfield", "field", "domain"}
    assert profiles.loc[profiles["level"] == "field", "n_subfields"].tolist() == [2, 1]


def test_pairwise_distances_and_rankings_are_written(tmp_path) -> None:
    profiles = profile_table(tiny_reduced_core())
    distances = pairwise_profile_distances(
        profiles,
        metrics=REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS,
    )
    rankings = metric_rankings(
        profiles,
        metrics=REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS,
        top_n=1,
    )
    summary = build_static_discipline_profile_outputs(
        tiny_reduced_core(),
        input_path="synthetic.parquet",
        output_dir=tmp_path,
        top_n=1,
    )

    assert {"euclidean", "correlation"} <= set(distances["distance_metric"])
    assert {"lowest", "highest"} <= set(rankings["direction"])
    assert (tmp_path / "static_discipline_profiles.parquet").exists()
    assert (tmp_path / "top_static_profile_pairs.csv").exists()
    assert summary["n_profile_rows"] == 6
