from __future__ import annotations

import numpy as np
import pandas as pd

from src.metric_clustering import (
    ClusteringConfig,
    build_metric_space,
    cluster_representatives,
    compare_cluster_partitions,
    fit_pca_scores,
    metric_space_umap_output_frame,
    metric_block,
    preprocess_metric_block,
    run_metric_space_clustering,
    ward_labels,
)
from src.embedding_space_metrics import CORE_EMBEDDING_METRIC_COLUMNS
from src.morphology_metrics import CORE_METRIC_COLUMNS_V2


def synthetic_metric_tables(n: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    subfield_ids = [f"S{idx:02d}" for idx in range(n)]
    metadata = {
        "subfield_id": subfield_ids,
        "subfield_display_name": [f"Subfield {idx}" for idx in range(n)],
        "subfield_label_unique": [f"S{idx:02d} | Domain / Field / Subfield {idx}" for idx in range(n)],
        "subfield_label_short": [f"S{idx:02d} | Field / Subfield {idx}" for idx in range(n)],
        "field_id": [f"F{idx % 3}" for idx in range(n)],
        "field_display_name": [f"Field {idx % 3}" for idx in range(n)],
        "domain_id": [f"D{idx % 2}" for idx in range(n)],
        "domain_display_name": [f"Domain {idx % 2}" for idx in range(n)],
        "metric_status": ["completed"] * n,
    }
    base = np.linspace(-2.0, 2.0, n)
    umap = pd.DataFrame(metadata)
    umap["radial_tail_index"] = base + rng.normal(scale=0.05, size=n)
    umap["knn_median_distance"] = -base + rng.normal(scale=0.05, size=n)
    umap["density_entropy"] = np.sin(base) + rng.normal(scale=0.05, size=n)
    umap["mst_gap_index"] = rng.normal(size=n)
    umap["max_normalized_radius"] = 1.0
    umap["peak_dominance"] = [np.nan] * 8 + list(rng.normal(size=n - 8))

    embedding = pd.DataFrame(metadata)
    embedding["embedding_centroid_norm"] = base + rng.normal(scale=0.05, size=n)
    embedding["embedding_knn_median_distance"] = -base + rng.normal(scale=0.05, size=n)
    embedding["embedding_pca_first_component_share"] = rng.normal(size=n)
    embedding["embedding_graph_edge_distance_p90"] = np.cos(base) + rng.normal(scale=0.05, size=n)
    embedding["embedding_distance_to_centroid_mean"] = 0.5
    embedding["embedding_knn_distance_cv"] = [np.nan, np.nan] + list(rng.normal(size=n - 2))
    return umap, embedding


def config() -> ClusteringConfig:
    return ClusteringConfig(
        max_missing_share=0.30,
        min_unique_values=4,
        pca_variance_threshold=0.80,
        k_min=2,
        k_max=4,
        default_k=3,
        random_state=42,
    )


def test_core_metric_selection_uses_existing_constants() -> None:
    assert len(CORE_METRIC_COLUMNS_V2) == 25
    assert len(CORE_EMBEDDING_METRIC_COLUMNS) == 26
    assert "radial_tail_index" in CORE_METRIC_COLUMNS_V2
    assert "embedding_centroid_norm" in CORE_EMBEDDING_METRIC_COLUMNS

    frame = pd.DataFrame(
        {
            "subfield_id": ["S1", "S2"],
            "radial_tail_index": [1.0, 2.0],
            "knn_median_distance": [0.5, 0.6],
            "not_a_core_metric": [99.0, 100.0],
        }
    )

    selected = metric_block(frame, CORE_METRIC_COLUMNS_V2)

    assert selected.columns.tolist() == ["radial_tail_index", "knn_median_distance"]


def test_preprocessing_drops_high_missing_and_constant_metrics() -> None:
    block = pd.DataFrame(
        {
            "good": [1.0, 2.0, 3.0, 4.0, 5.0],
            "constant": [1.0, 1.0, 1.0, 1.0, 1.0],
            "high_missing": [1.0, np.nan, np.nan, np.nan, 5.0],
        }
    )

    scaled, report = preprocess_metric_block(
        block,
        metric_family="test",
        config=config(),
    )

    assert scaled.columns.tolist() == ["good"]
    reasons = report.set_index("metric_name")["drop_reason"].to_dict()
    assert reasons["constant"] == "constant"
    assert "high_missing" in reasons["high_missing"]


def test_preprocessing_median_imputation_removes_nans() -> None:
    block = pd.DataFrame(
        {
            "good": [1.0, 2.0, np.nan, 4.0, 5.0],
            "other": [5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )

    scaled, report = preprocess_metric_block(
        block,
        metric_family="test",
        config=config(),
    )

    assert not scaled.isna().any().any()
    assert report.set_index("metric_name").loc["good", "imputation_value"] == 3.0


def test_pca_dimension_selection_respects_variance_threshold() -> None:
    rng = np.random.default_rng(3)
    first = rng.normal(size=20)
    features = pd.DataFrame(
        {
            "a": first,
            "b": first + rng.normal(scale=0.01, size=20),
            "c": rng.normal(scale=0.05, size=20),
        }
    )

    scores, loadings, explained = fit_pca_scores(
        features,
        variance_threshold=0.80,
        block_name="test",
    )

    assert scores.shape[1] >= 1
    retained = explained.loc[explained["retained"]]
    assert retained["cumulative_explained_variance"].iloc[-1] >= 0.80
    assert set(loadings["component"]).issuperset(set(scores.columns))


def test_ward_clustering_returns_expected_number_of_clusters() -> None:
    features = pd.DataFrame(np.random.default_rng(4).normal(size=(12, 3)))

    labels = ward_labels(features, 3)

    assert len(labels) == 12
    assert len(set(labels)) == 3


def test_cluster_representatives_return_rows_per_cluster() -> None:
    assignments = pd.DataFrame(
        {
            "subfield_id": ["S1", "S2", "S3", "S4"],
            "subfield_display_name": ["A", "B", "C", "D"],
            "subfield_label_unique": ["A", "B", "C", "D"],
            "field_display_name": ["F", "F", "G", "G"],
            "domain_display_name": ["D", "D", "D", "D"],
            "cluster_ward_k2": [1, 1, 2, 2],
        }
    )
    features = pd.DataFrame({"PC1": [0.0, 0.1, 2.0, 2.1], "PC2": [0.0, 0.1, 0.0, 0.1]})

    reps = cluster_representatives(
        assignments,
        features,
        cluster_column="cluster_ward_k2",
        top_n=1,
    )

    assert reps["cluster_id"].tolist() == [1, 2]
    assert reps["rank"].tolist() == [1, 1]


def test_metric_space_umap_output_frame_includes_coordinates_and_cluster() -> None:
    assignments = pd.DataFrame(
        {
            "subfield_id": ["S1", "S2"],
            "subfield_display_name": ["A", "B"],
            "cluster_ward_k2": [1, 2],
        }
    )
    coords = np.array([[0.1, 0.2], [1.1, 1.2]])

    frame = metric_space_umap_output_frame(
        assignments,
        coords,
        cluster_column="cluster_ward_k2",
    )

    assert frame.columns.tolist() == [
        "subfield_id",
        "subfield_display_name",
        "metric_umap_x",
        "metric_umap_y",
        "cluster_ward_k2",
        "cluster_label",
    ]
    assert frame.loc[1, "metric_umap_y"] == 1.2
    assert frame.loc[0, "cluster_label"] == 1


def test_partition_comparison_computes_ari_nmi_and_contingencies() -> None:
    assignments = {
        "embedding_only": pd.DataFrame(
            {"subfield_id": ["S1", "S2", "S3"], "cluster_ward_k3": [1, 2, 3]}
        ),
        "umap_only": pd.DataFrame(
            {"subfield_id": ["S1", "S2", "S3"], "cluster_ward_k3": [1, 1, 2]}
        ),
        "combined": pd.DataFrame(
            {"subfield_id": ["S1", "S2", "S3"], "cluster_ward_k3": [1, 2, 3]}
        ),
    }

    agreement, contingencies = compare_cluster_partitions(
        assignments,
        cluster_column="cluster_ward_k3",
    )

    assert len(agreement) == 3
    assert {"adjusted_rand_index", "normalized_mutual_info"}.issubset(agreement.columns)
    assert ("embedding_only", "umap_only") in contingencies


def test_combined_block_pca_creates_valid_feature_matrix() -> None:
    umap, embedding = synthetic_metric_tables()
    space = build_metric_space(
        "combined",
        umap,
        embedding,
        config=config(),
    )

    assert len(space.metadata) == len(umap)
    assert any(column.startswith("umap_PC") for column in space.feature_matrix.columns)
    assert any(column.startswith("embedding_PC") for column in space.feature_matrix.columns)
    assert not space.feature_matrix.isna().any().any()


def test_clustering_helpers_do_not_crash_with_nans_and_small_samples() -> None:
    umap, embedding = synthetic_metric_tables(n=9)
    umap.loc[0, "radial_tail_index"] = np.nan
    embedding.loc[1, "embedding_centroid_norm"] = np.nan

    space = build_metric_space(
        "embedding_only",
        umap,
        embedding,
        config=config(),
    )
    result = run_metric_space_clustering(space, config=config())

    assert result.main_cluster_column == "cluster_ward_k3"
    assert result.assignments[result.main_cluster_column].nunique() == 3
    assert not result.cluster_profiles.empty
