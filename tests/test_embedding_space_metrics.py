from __future__ import annotations

import numpy as np
import pandas as pd

from src.embedding_space_metrics import (
    CORE_EMBEDDING_METRIC_COLUMNS,
    DIAGNOSTIC_COLUMNS,
    EXCLUDED_FROM_CORE_COLUMNS,
    OUTPUT_COLUMNS,
    compute_embedding_metrics,
    compute_subfield_embedding_metric_row,
    embedding_metric_row_frame,
    gini_coefficient,
)
from src.embedding_metric_diagnostics import available_core_metric_columns
from src.per_subfield_umap_maps import sample_subfield_rows


def synthetic_index(
    n: int,
    *,
    subfield_id: str = "1100",
    subfield_name: str = "Synthetic Subfield",
    years: np.ndarray | None = None,
) -> pd.DataFrame:
    if years is None:
        years = np.resize(np.arange(2010, 2026), n)
    return pd.DataFrame(
        {
            "analysis_row_id": np.arange(n),
            "work_id": [f"W{subfield_id}-{row:03d}" for row in range(n)],
            "subfield_id": subfield_id,
            "subfield_display_name": subfield_name,
            "field_id": f"F{subfield_id}",
            "field_display_name": f"Field {subfield_id}",
            "domain_id": "D1",
            "domain_display_name": "Domain 1",
            "publication_year": years.astype(int),
            "main_analysis_eligible_2500": True,
        }
    )


def compact_embeddings(n: int = 96, dim: int = 12, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.zeros(dim, dtype=np.float32)
    base[0] = 1.0
    return (base + rng.normal(scale=0.01, size=(n, dim))).astype(np.float32)


def dispersed_embeddings(n: int = 96, dim: int = 12, seed: int = 43) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, dim)).astype(np.float32)


def metric_row(
    embeddings: np.ndarray,
    *,
    years: np.ndarray | None = None,
    subfield_id: str = "1100",
    subfield_name: str = "Synthetic Subfield",
) -> dict[str, object]:
    index = synthetic_index(
        len(embeddings),
        subfield_id=subfield_id,
        subfield_name=subfield_name,
        years=years,
    )
    return compute_subfield_embedding_metric_row(
        index,
        embeddings,
        n_available=len(index),
        year_min=2010,
        year_max=2025,
        analysis_index_path="index.parquet",
        embedding_matrix_path="matrix.npy",
        k_neighbors=5,
        random_state=42,
        sampling_applied=False,
        max_papers_per_subfield=None,
    )


def test_core_embedding_metric_columns_are_curated_26() -> None:
    assert len(CORE_EMBEDDING_METRIC_COLUMNS) == 26
    assert "embedding_knn_indegree_gini" in CORE_EMBEDDING_METRIC_COLUMNS
    assert "embedding_pca_spectral_entropy" in CORE_EMBEDDING_METRIC_COLUMNS
    assert "embedding_recent_novelty_score" in CORE_EMBEDDING_METRIC_COLUMNS
    assert "embedding_radial_expansion_r2" not in CORE_EMBEDDING_METRIC_COLUMNS
    assert "embedding_radial_expansion_r2" in DIAGNOSTIC_COLUMNS
    for column in EXCLUDED_FROM_CORE_COLUMNS:
        assert column not in CORE_EMBEDDING_METRIC_COLUMNS


def test_core_diagnostic_plot_column_selection_excludes_controls() -> None:
    frame = pd.DataFrame({column: [1.0, 2.0] for column in CORE_EMBEDDING_METRIC_COLUMNS})
    frame["embedding_graph_connected_component_count"] = [1, 1]
    frame["embedding_graph_largest_component_share"] = [1.0, 1.0]
    frame["n_valid_embedding_rows"] = [100, 120]

    selected = available_core_metric_columns(frame)

    assert selected == CORE_EMBEDDING_METRIC_COLUMNS
    assert "embedding_graph_connected_component_count" not in selected
    assert "embedding_graph_largest_component_share" not in selected
    assert "n_valid_embedding_rows" not in selected


def test_pca_spectral_entropy_is_finite_unit_interval_for_non_degenerate_matrix() -> None:
    metrics, _, warnings = compute_embedding_metrics(
        dispersed_embeddings(n=80, dim=16, seed=101),
        np.resize(np.arange(2000, 2025), 80),
        year_min=2000,
        year_max=2024,
        k_neighbors=5,
    )

    entropy = metrics["embedding_pca_spectral_entropy"]

    assert np.isfinite(entropy)
    assert 0.0 <= entropy <= 1.0
    assert not any("embedding_pca_spectral_entropy unavailable" in item for item in warnings)


def test_gini_coefficient_is_zero_for_uniform_and_higher_for_concentrated() -> None:
    uniform = gini_coefficient(np.array([3, 3, 3, 3]))
    concentrated = gini_coefficient(np.array([0, 0, 0, 12]))

    assert uniform == 0.0
    assert concentrated > uniform


def test_recent_novelty_score_positive_when_late_points_shift_from_early_core() -> None:
    years = np.array([2000, 2001, 2002, 2003, 2004] * 4 + [2020, 2021, 2022, 2023, 2024] * 4)
    early = np.tile(np.array([1.0, 0.02, 0.0, 0.0], dtype=np.float32), (20, 1))
    late = np.tile(np.array([0.0, 1.0, 0.02, 0.0], dtype=np.float32), (20, 1))
    embeddings = np.vstack([early, late])

    metrics, _, _ = compute_embedding_metrics(
        embeddings,
        years,
        year_min=2000,
        year_max=2024,
        k_neighbors=5,
    )

    assert metrics["embedding_recent_novelty_score"] > 0.5


def test_compact_subfield_has_lower_centroid_distance_than_dispersed_subfield() -> None:
    compact = metric_row(compact_embeddings())
    dispersed = metric_row(dispersed_embeddings())

    assert compact["metric_status"] == "completed"
    assert dispersed["metric_status"] == "completed"
    assert (
        compact["embedding_distance_to_centroid_mean"]
        < dispersed["embedding_distance_to_centroid_mean"]
    )
    assert compact["embedding_centroid_norm"] > dispersed["embedding_centroid_norm"]


def test_tiny_subfield_returns_warnings_and_nans() -> None:
    row = metric_row(compact_embeddings(n=1))

    assert row["metric_status"] == "completed_with_warnings"
    assert np.isnan(row["embedding_knn_mean_distance"])
    assert np.isnan(row["embedding_pca_first_component_share"])
    assert "fewer than two embeddings" in row["metric_warning_message"]


def test_missing_early_late_windows_warns_without_crashing() -> None:
    years = np.resize(np.arange(2015, 2019), 32)
    row = metric_row(compact_embeddings(n=32), years=years)

    assert row["metric_status"] == "completed_with_warnings"
    assert np.isnan(row["embedding_centroid_drift_early_late"])
    assert "early or late" in row["metric_warning_message"]


def test_duplicate_display_names_do_not_break_embedding_metric_rows() -> None:
    first = metric_row(
        compact_embeddings(n=48, seed=1),
        subfield_id="1303",
        subfield_name="Biochemistry",
    )
    second = metric_row(
        compact_embeddings(n=48, seed=2),
        subfield_id="2704",
        subfield_name="Biochemistry",
    )
    rows = embedding_metric_row_frame([first, second])

    assert rows["subfield_display_name_is_duplicated"].tolist() == [True, True]
    assert "1303 | Domain 1 / Field 1303 / Biochemistry" in set(
        rows["subfield_label_unique"]
    )
    assert "2704 | Domain 1 / Field 2704 / Biochemistry" in set(
        rows["subfield_label_unique"]
    )


def test_deterministic_sampling_gives_identical_embedding_metric_inputs() -> None:
    index = synthetic_index(80)
    shuffled = index.sample(frac=1, random_state=7)

    first = sample_subfield_rows(
        index,
        max_papers=20,
        random_state=42,
        subfield_id="1100",
    )
    second = sample_subfield_rows(
        shuffled,
        max_papers=20,
        random_state=42,
        subfield_id="1100",
    )

    assert first["analysis_row_id"].tolist() == second["analysis_row_id"].tolist()


def test_compute_embedding_metrics_output_columns_are_stable() -> None:
    metrics, controls, warnings = compute_embedding_metrics(
        compact_embeddings(n=64),
        np.resize(np.arange(2010, 2026), 64),
        year_min=2010,
        year_max=2025,
        k_neighbors=5,
    )
    row = embedding_metric_row_frame(
        [
            {
                "subfield_id": "1100",
                "subfield_display_name": "Synthetic Subfield",
                "field_id": "F1100",
                "field_display_name": "Field 1100",
                "domain_id": "D1",
                "domain_display_name": "Domain 1",
                "n_available": 64,
                "n_used": 64,
                "year_min": 2010,
                "year_max": 2025,
                "metric_status": "completed_with_warnings" if warnings else "completed",
                "metric_error_message": "",
                "metric_warning_message": "; ".join(warnings),
                "embedding_matrix_path": "matrix.npy",
                "analysis_index_path": "index.parquet",
                "k_neighbors": 5,
                "random_state": 42,
                "sampling_applied": False,
                "max_papers_per_subfield": None,
                **metrics,
                **controls,
            }
        ]
    )

    assert row.columns.tolist() == OUTPUT_COLUMNS
    assert set(CORE_EMBEDDING_METRIC_COLUMNS).issubset(row.columns)
    assert row.loc[0, "metric_status"] == "completed"
