from __future__ import annotations

import numpy as np
import pandas as pd

from src.morphological_similarity_evolution import (
    TEMPORAL_METRIC_SET,
    add_top_pair_driver_diagnostics,
    entity_retention_diagnostics,
    pairwise_profile_distances,
    profile_frame_for_level,
    profile_distance,
    robust_scale_metrics,
    top_morphological_pairs,
    top_pair_entity_frequency,
)
from src.temporal_centroid_paths import compute_centroid_path_metrics
from src.temporal_common import WINDOW_STRUCTURAL_METRICS, TemporalWindow
from src.temporal_embedding_metric_evolution import (
    metric_long_frame,
    overall_change_ranking,
    temporal_change_frame,
    top_changes_by_metric,
    window_sample_size_diagnostics,
)
from src.temporal_semantic_distances import (
    add_driver_diagnostics,
    matrix_from_pairs,
    pairwise_semantic_distances,
    qa_checks_by_level,
    semantic_distance_changes,
    top_converging_diverging_pairs,
)


def five_windows() -> list[TemporalWindow]:
    return [
        TemporalWindow(2000, 2004, 0),
        TemporalWindow(2005, 2009, 1),
        TemporalWindow(2010, 2014, 2),
        TemporalWindow(2015, 2019, 3),
        TemporalWindow(2020, 2024, 4),
    ]


def synthetic_window_metrics() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subfield_id, base in [("S1", 0.1), ("S2", 0.5)]:
        for window in five_windows():
            row = {
                "subfield_id": subfield_id,
                "subfield_display_name": f"Subfield {subfield_id}",
                "subfield_label_unique": f"{subfield_id} | Synthetic",
                "subfield_label_short": subfield_id,
                "field_id": "F1",
                "field_display_name": "Field",
                "domain_id": "D1",
                "domain_display_name": "Domain",
                "window_start": window.window_start,
                "window_end": window.window_end,
                "window_index": window.window_index,
                "window_label": window.window_label,
                "window_mid_year": window.window_mid_year,
                "n_available": 100,
                "n_used": 100,
                "metric_status": "completed",
            }
            for metric_idx, metric in enumerate(WINDOW_STRUCTURAL_METRICS):
                row[metric] = base + window.window_index * 0.1 + metric_idx * 0.01
            rows.append(row)
    return pd.DataFrame(rows)


def test_temporal_change_frame_labels_increases_and_ranks_overall_change() -> None:
    long = metric_long_frame(synthetic_window_metrics())
    changes = temporal_change_frame(long, windows=five_windows())
    ranking = overall_change_ranking(changes)
    top_changes = top_changes_by_metric(changes, top_n=1)
    diagnostics = window_sample_size_diagnostics(synthetic_window_metrics())

    assert set(changes["direction_label"]) == {"increase"}
    assert changes["n_windows_available"].min() == 5
    assert ranking["overall_metric_change_rank"].tolist() == [1, 2]
    assert ranking["overall_metric_change_l2"].notna().all()
    assert set(top_changes["rank_type"]) == {
        "highest_standardized_delta",
        "lowest_standardized_delta",
        "largest_absolute_standardized_delta",
    }
    assert diagnostics["n_completed"].tolist() == [2, 2, 2, 2, 2]


def test_centroid_path_metrics_are_computed_in_original_vector_space() -> None:
    rows = []
    angles = np.deg2rad([0, 30, 60, 90, 120])
    vectors = [
        np.array([np.cos(angle), np.sin(angle)])
        for angle in angles
    ]
    for window, vector in zip(five_windows(), vectors, strict=True):
        rows.append(
            {
                "subfield_id": "S1",
                "subfield_display_name": "Subfield",
                "subfield_label_unique": "S1 | Synthetic",
                "subfield_label_short": "S1",
                "field_id": "F1",
                "field_display_name": "Field",
                "domain_id": "D1",
                "domain_display_name": "Domain",
                "window_start": window.window_start,
                "window_end": window.window_end,
                "window_index": window.window_index,
                "window_label": window.window_label,
                "window_mid_year": window.window_mid_year,
                "n_used": 10,
                "centroid_status": "completed",
                "centroid_dim_000": vector[0],
                "centroid_dim_001": vector[1],
            }
        )
    metrics = compute_centroid_path_metrics(pd.DataFrame(rows), windows=five_windows())

    assert metrics.loc[0, "n_windows_available"] == 5
    assert metrics.loc[0, "early_late_drift"] > metrics.loc[0, "total_path_length"]
    assert metrics.loc[0, "directionality_ratio_cosine_diagnostic"] > 1
    assert 0 < metrics.loc[0, "directionality_ratio_euclidean"] <= 1
    assert (
        metrics.loc[0, "early_late_drift_euclidean_normalized"]
        <= metrics.loc[0, "total_path_length_euclidean_normalized"]
    )


def test_semantic_distance_changes_detect_convergence() -> None:
    rows = []
    for window in [five_windows()[0], five_windows()[-1]]:
        rows.extend(
            [
                {
                    "level": "subfield",
                    "entity_id": "A",
                    "entity_display_name": "A",
                    "window_label": window.window_label,
                    "window_start": window.window_start,
                    "window_end": window.window_end,
                    "window_index": window.window_index,
                    "window_mid_year": window.window_mid_year,
                    "n_used": 10,
                    "centroid_dim_000": 1.0,
                    "centroid_dim_001": 0.0,
                },
                {
                    "level": "subfield",
                    "entity_id": "B",
                    "entity_display_name": "B",
                    "window_label": window.window_label,
                    "window_start": window.window_start,
                    "window_end": window.window_end,
                    "window_index": window.window_index,
                    "window_mid_year": window.window_mid_year,
                    "n_used": 10,
                    "centroid_dim_000": 0.0 if window.window_index == 0 else 0.9,
                    "centroid_dim_001": 1.0 if window.window_index == 0 else 0.1,
                },
            ]
        )
    pair_distances = pairwise_semantic_distances(pd.DataFrame(rows))
    changes = semantic_distance_changes(pair_distances, windows=five_windows())

    assert len(changes) == 1
    assert changes.loc[0, "delta_distance"] < 0


def test_semantic_top_tables_are_strictly_signed() -> None:
    changes = pd.DataFrame(
        {
            "level": ["domain", "domain", "domain"],
            "pair_id": ["A__B", "A__C", "B__C"],
            "entity_id_a": ["A", "A", "B"],
            "entity_display_name_a": ["A", "A", "B"],
            "entity_id_b": ["B", "C", "C"],
            "entity_display_name_b": ["B", "C", "C"],
            "delta_distance": [-0.2, 0.1, 0.0],
        }
    )

    converging = top_converging_diverging_pairs(changes, top_n=10, direction="converging")
    diverging = top_converging_diverging_pairs(changes, top_n=10, direction="diverging")

    assert (converging["delta_distance"] < 0).all()
    assert (diverging["delta_distance"] > 0).all()
    assert len(converging) == 1
    assert len(diverging) == 1


def test_semantic_delta_matrix_is_symmetric_with_zero_diagonal() -> None:
    changes = pd.DataFrame(
        {
            "entity_display_name_a": ["A", "A", "B"],
            "entity_display_name_b": ["B", "C", "C"],
            "delta_distance": [-0.2, 0.1, 0.0],
        }
    )

    matrix = matrix_from_pairs(changes, value_column="delta_distance")

    assert np.allclose(matrix.to_numpy(dtype=float), matrix.to_numpy(dtype=float).T)
    assert np.allclose(np.diag(matrix.to_numpy(dtype=float)), 0.0)


def test_semantic_pair_ids_are_deterministic() -> None:
    rows = []
    for name, vector in [
        ("B", [1.0, 0.0]),
        ("A", [0.0, 1.0]),
        ("C", [1.0, 1.0]),
    ]:
        rows.append(
            {
                "level": "field",
                "entity_id": name,
                "entity_display_name": name,
                "window_label": "2000-2004",
                "window_start": 2000,
                "window_end": 2004,
                "window_index": 0,
                "window_mid_year": 2002.0,
                "n_used": 10,
                "centroid_dim_000": vector[0],
                "centroid_dim_001": vector[1],
            }
        )

    pairs = pairwise_semantic_distances(pd.DataFrame(rows))

    assert pairs["pair_id"].tolist() == ["A__B", "A__C", "B__C"]


def test_driver_diagnostics_skip_when_path_metrics_missing() -> None:
    pairs = pd.DataFrame(
        {
            "level": ["subfield"],
            "pair_id": ["A__B"],
            "entity_id_a": ["A"],
            "entity_display_name_a": ["A"],
            "entity_id_b": ["B"],
            "entity_display_name_b": ["B"],
            "delta_distance": [-0.2],
        }
    )

    output, merged = add_driver_diagnostics(pairs, path_metrics=None)

    assert not merged
    assert output.equals(pairs)


def test_semantic_qa_checks_include_signed_counts_and_summary_metadata() -> None:
    changes = pd.DataFrame(
        {
            "level": ["field", "field", "field"],
            "delta_distance": [-0.2, 0.1, 0.0],
        }
    )

    checks = qa_checks_by_level(changes)
    summary = {"aggregation_mode_used": "weighted_subfield_centroids"}

    assert checks["field"]["n_converging_pairs"] == 1
    assert checks["field"]["n_diverging_pairs"] == 1
    assert checks["field"]["n_stable_zero_delta_pairs"] == 1
    assert summary["aggregation_mode_used"] == "weighted_subfield_centroids"


def test_morphological_profile_distance_uses_scaled_metric_vectors() -> None:
    frame = pd.DataFrame(
        {
            "subfield_id": ["A", "B", "C"],
            "subfield_display_name": ["A", "B", "C"],
            "m1": [1.0, 2.0, 10.0],
            "m2": [2.0, 3.0, 20.0],
        }
    )
    scaled, params = robust_scale_metrics(frame, metrics=["m1", "m2"], metric_set="test")
    profiles = scaled.assign(
        metric_set="test",
        level="subfield",
        entity_id=scaled["subfield_id"],
        entity_display_name=scaled["subfield_display_name"],
    )
    pairs = pairwise_profile_distances(
        profiles,
        metrics=["m1", "m2"],
        distance_metrics=["euclidean", "correlation"],
        temporal=False,
    )

    assert params["metric"].tolist() == ["m1", "m2"]
    assert profile_distance(np.array([0.0, 0.0]), np.array([3.0, 4.0]), distance_metric="euclidean") == 5.0
    assert set(pairs["distance_metric"]) == {"euclidean", "correlation"}
    assert pairs["morphological_distance"].notna().any()


def test_morphological_top_tables_are_strictly_signed() -> None:
    changes = pd.DataFrame(
        {
            "level": ["field", "field", "field"],
            "distance_metric": ["euclidean", "euclidean", "euclidean"],
            "pair_id": ["A__B", "A__C", "B__C"],
            "entity_id_a": ["A", "A", "B"],
            "entity_display_name_a": ["A", "A", "B"],
            "entity_id_b": ["B", "C", "C"],
            "entity_display_name_b": ["B", "C", "C"],
            "delta_morphological_distance": [-0.2, 0.1, 0.0],
        }
    )

    converging = top_morphological_pairs(changes, top_n=10, direction="converging")
    diverging = top_morphological_pairs(changes, top_n=10, direction="diverging")

    assert (converging["delta_morphological_distance"] < 0).all()
    assert (diverging["delta_morphological_distance"] > 0).all()
    assert len(converging) == 1
    assert len(diverging) == 1


def test_morphological_distance_types_are_both_produced() -> None:
    frame = pd.DataFrame(
        {
            "subfield_id": ["A", "B", "C"],
            "subfield_display_name": ["A", "B", "C"],
            "m1": [1.0, 2.0, 4.0],
            "m2": [1.0, 3.0, 2.0],
            "m3": [2.0, 5.0, 1.0],
        }
    )
    scaled, _params = robust_scale_metrics(frame, metrics=["m1", "m2", "m3"], metric_set="test")
    profiles = scaled.assign(
        metric_set="test",
        level="subfield",
        entity_id=scaled["subfield_id"],
        entity_display_name=scaled["subfield_display_name"],
    )

    pairs = pairwise_profile_distances(
        profiles,
        metrics=["m1", "m2", "m3"],
        distance_metrics=["euclidean", "correlation"],
        temporal=False,
    )

    assert set(pairs["distance_metric"]) == {"euclidean", "correlation"}


def test_entity_retention_diagnostics_are_produced() -> None:
    temporal = synthetic_window_metrics()
    scaled, _params = robust_scale_metrics(
        temporal,
        metrics=WINDOW_STRUCTURAL_METRICS,
        metric_set=TEMPORAL_METRIC_SET,
    )
    profiles = pd.concat(
        [
            profile_frame_for_level(
                scaled,
                metrics=WINDOW_STRUCTURAL_METRICS,
                level=level,
                metric_set=TEMPORAL_METRIC_SET,
                temporal=True,
            )
            for level in ["subfield", "field", "domain"]
        ],
        ignore_index=True,
    )

    diagnostics, dropped = entity_retention_diagnostics(
        temporal,
        profiles,
        levels=["subfield", "field", "domain"],
        windows=five_windows(),
        metrics=WINDOW_STRUCTURAL_METRICS,
    )

    assert diagnostics.set_index("level").loc["subfield", "n_retained_entities"] == 2
    assert diagnostics.set_index("level").loc["field", "n_retained_entities"] == 1
    assert diagnostics.set_index("level").loc["domain", "n_retained_entities"] == 1
    assert dropped.empty


def test_top_pair_entity_frequency_counts_are_correct() -> None:
    top_pairs = pd.DataFrame(
        {
            "level": ["field", "field"],
            "distance_metric": ["euclidean", "euclidean"],
            "direction": ["converging", "diverging"],
            "entity_id_a": ["A", "A"],
            "entity_display_name_a": ["A", "A"],
            "entity_field_a": ["A", "A"],
            "entity_domain_a": ["Domain", "Domain"],
            "entity_id_b": ["B", "C"],
            "entity_display_name_b": ["B", "C"],
            "entity_field_b": ["B", "C"],
            "entity_domain_b": ["Domain", "Domain"],
            "delta_morphological_distance": [-0.2, 0.3],
        }
    )

    frequency = top_pair_entity_frequency(top_pairs)
    entity_a = frequency.loc[frequency["entity_id"] == "A"].iloc[0]

    assert entity_a["n_top_converging_pairs"] == 1
    assert entity_a["n_top_diverging_pairs"] == 1
    assert entity_a["n_top_pairs_total"] == 2


def test_morphological_driver_diagnostics_skip_when_script19_outputs_missing(tmp_path) -> None:
    pairs = pd.DataFrame(
        {
            "level": ["subfield"],
            "distance_metric": ["euclidean"],
            "direction": ["converging"],
            "entity_id_a": ["A"],
            "entity_display_name_a": ["A"],
            "entity_id_b": ["B"],
            "entity_display_name_b": ["B"],
            "delta_morphological_distance": [-0.2],
        }
    )

    output, merged, warnings = add_top_pair_driver_diagnostics(pairs, root=tmp_path)

    assert not merged
    assert output.equals(pairs)
    assert warnings
