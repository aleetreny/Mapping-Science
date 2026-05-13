from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cluster_interpretation import (
    build_master_table,
    build_metric_profile_long,
    cluster_composition,
    main_cluster_column,
    metric_umap_coordinate_frame,
    plot_combined_metric_space_umap,
    readable_representatives,
    require_existing_paths,
    top_metric_differences,
)


def synthetic_interpretation_inputs() -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, str],
    dict[str, pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame,
]:
    metadata = pd.DataFrame(
        {
            "subfield_id": ["S1", "S2", "S3", "S4"],
            "subfield_display_name": ["Alpha", "Beta", "Gamma", "Delta"],
            "field_id": ["F1", "F2", "F1", "F3"],
            "field_display_name": ["Field 1", "Field 2", "Field 1", "Field 3"],
            "domain_id": ["D1", "D1", "D1", "D2"],
            "domain_display_name": ["Domain 1", "Domain 1", "Domain 1", "Domain 2"],
        }
    )
    assignments_by_space = {
        "embedding_only": metadata.assign(cluster_ward_k2=[1, 1, 2, 2]),
        "umap_only": metadata.assign(cluster_ward_k2=[1, 2, 1, 2]),
        "combined": metadata.assign(cluster_ward_k2=[1, 1, 2, 2]),
    }
    pca_scores_by_space = {
        "embedding_only": metadata.assign(embedding_PC1=[0.0, 0.2, 1.0, 1.2], embedding_PC2=[1.0, 1.1, 0.0, 0.1]),
        "umap_only": metadata.assign(umap_PC1=[0.5, 0.6, -0.5, -0.6], umap_PC2=[0.0, 0.2, 1.0, 1.1]),
        "combined": metadata.assign(combined_PC1=[0.1, 0.2, 2.0, 2.1], combined_PC2=[1.0, 1.1, -1.0, -1.1]),
    }
    cluster_columns = {
        "embedding_only": "cluster_ward_k2",
        "umap_only": "cluster_ward_k2",
        "combined": "cluster_ward_k2",
    }
    metric_umap_coordinates = {
        "embedding_only": metadata.assign(
            metric_umap_x=[10.0, 11.0, 12.0, 13.0],
            metric_umap_y=[20.0, 21.0, 22.0, 23.0],
            cluster_ward_k2=[1, 1, 2, 2],
        ),
        "umap_only": metadata.assign(
            metric_umap_x=[-10.0, -11.0, -12.0, -13.0],
            metric_umap_y=[-20.0, -21.0, -22.0, -23.0],
            cluster_ward_k2=[1, 2, 1, 2],
        ),
        "combined": metadata.assign(
            metric_umap_x=[0.1, 0.2, 0.3, 0.4],
            metric_umap_y=[1.1, 1.2, 1.3, 1.4],
            cluster_ward_k2=[1, 1, 2, 2],
        ),
    }
    umap_metrics = metadata.assign(
        n_points=[400, 380, 390, 410],
        radial_tail_index=[10.0, 9.0, 1.0, 1.5],
        density_peak_count=[0.0, 1.0, 5.0, 6.0],
        anisotropy_ratio=[3.0, 2.8, 1.0, 1.1],
    )
    embedding_metrics = metadata.assign(
        n_available=[450, 430, 420, 440],
        n_used=[400, 400, 400, 400],
        embedding_centroid_norm=[0.9, 1.0, 0.2, 0.1],
        embedding_distance_to_centroid_mean=[0.1, 0.2, 0.9, 1.0],
        embedding_pca_first_component_share=[0.8, 0.7, 0.3, 0.2],
    )
    return (
        assignments_by_space,
        pca_scores_by_space,
        cluster_columns,
        metric_umap_coordinates,
        umap_metrics,
        embedding_metrics,
    )


def test_main_cluster_column_prefers_summary_column() -> None:
    assignments = pd.DataFrame(
        {
            "subfield_id": ["S1", "S2"],
            "cluster_ward_k3": [1, 2],
            "cluster_ward_k5": [2, 1],
        }
    )

    assert main_cluster_column(assignments, summary={"main_cluster_column": "cluster_ward_k3"}) == "cluster_ward_k3"


def test_master_table_joins_clusters_and_coordinates() -> None:
    (
        assignments,
        pca_scores,
        cluster_columns,
        metric_umap_coordinates,
        umap_metrics,
        embedding_metrics,
    ) = synthetic_interpretation_inputs()

    master = build_master_table(
        assignments_by_space=assignments,
        pca_scores_by_space=pca_scores,
        cluster_columns_by_space=cluster_columns,
        umap_metrics=umap_metrics,
        embedding_metrics=embedding_metrics,
        metric_umap_coordinates_by_space=metric_umap_coordinates,
    )

    assert len(master) == 4
    assert master.set_index("subfield_id").loc["S2", "cluster_combined"] == 1
    assert master.set_index("subfield_id").loc["S3", "cluster_umap_only"] == 1
    assert master.set_index("subfield_id").loc["S4", "embedding_only_pca2"] == 0.1
    assert master.set_index("subfield_id").loc["S4", "embedding_umap_y"] == 23.0
    assert master.set_index("subfield_id").loc["S3", "projected_umap_x"] == -12.0
    assert master.set_index("subfield_id").loc["S2", "combined_umap_y"] == 1.2
    assert master.set_index("subfield_id").loc["S1", "n_available"] == 450
    assert {
        "embedding_umap_x",
        "embedding_umap_y",
        "projected_umap_x",
        "projected_umap_y",
        "combined_umap_x",
        "combined_umap_y",
    }.issubset(master.columns)


def test_domain_and_field_composition_shares_sum_correctly() -> None:
    assignments, pca_scores, cluster_columns, _, umap_metrics, embedding_metrics = synthetic_interpretation_inputs()
    master = build_master_table(
        assignments_by_space=assignments,
        pca_scores_by_space=pca_scores,
        cluster_columns_by_space=cluster_columns,
        umap_metrics=umap_metrics,
        embedding_metrics=embedding_metrics,
    )

    domain_comp = cluster_composition(master, level="domain")
    cluster_share_sums = domain_comp.groupby("cluster_id")["share_within_cluster"].sum()
    domain1_cluster1 = domain_comp.loc[
        (domain_comp["cluster_id"] == 1)
        & (domain_comp["domain_display_name"] == "Domain 1")
    ].iloc[0]

    assert np.allclose(cluster_share_sums.to_numpy(), 1.0)
    assert domain1_cluster1["count"] == 2
    assert domain1_cluster1["share_within_domain_or_field"] == pytest.approx(2 / 3)


def test_top_high_low_metric_profiles_are_produced() -> None:
    assignments, pca_scores, cluster_columns, _, umap_metrics, embedding_metrics = synthetic_interpretation_inputs()
    master = build_master_table(
        assignments_by_space=assignments,
        pca_scores_by_space=pca_scores,
        cluster_columns_by_space=cluster_columns,
        umap_metrics=umap_metrics,
        embedding_metrics=embedding_metrics,
    )

    profile = build_metric_profile_long(
        master=master,
        umap_metrics=umap_metrics,
        embedding_metrics=embedding_metrics,
    )
    top = top_metric_differences(profile, top_n=2)

    assert {"high", "low"} == set(top["direction"])
    assert set(top["cluster_id"]) == {1, 2}
    assert "plain_language_hint" in top.columns
    assert top.groupby(["cluster_id", "direction"]).size().min() == 2


def test_missing_files_fail_with_clear_message(tmp_path) -> None:
    missing = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Missing required input file"):
        require_existing_paths({"test input": missing})


def test_representative_table_handles_expected_columns() -> None:
    assignments, pca_scores, cluster_columns, _, umap_metrics, embedding_metrics = synthetic_interpretation_inputs()
    master = build_master_table(
        assignments_by_space=assignments,
        pca_scores_by_space=pca_scores,
        cluster_columns_by_space=cluster_columns,
        umap_metrics=umap_metrics,
        embedding_metrics=embedding_metrics,
    )
    reps = pd.DataFrame(
        {
            "cluster": [1, 2],
            "rank": [1, 1],
            "subfield_id": ["S1", "S4"],
            "distance_to_centroid": [0.05, 0.07],
        }
    )

    readable = readable_representatives(reps, master)

    assert readable["cluster_id"].tolist() == [1, 2]
    assert readable["representative_rank"].tolist() == [1, 1]
    assert readable.loc[0, "subfield_display_name"] == "Alpha"
    assert readable.loc[1, "domain_display_name"] == "Domain 2"


def test_metric_umap_coordinate_frame_renames_by_space() -> None:
    coords = pd.DataFrame(
        {
            "subfield_id": ["S1"],
            "metric_umap_x": [1.5],
            "metric_umap_y": [-2.5],
        }
    )

    projected = metric_umap_coordinate_frame(coords, space="umap_only")

    assert projected.columns.tolist() == [
        "subfield_id",
        "projected_umap_x",
        "projected_umap_y",
    ]
    assert projected.loc[0, "projected_umap_y"] == -2.5


def test_combined_metric_space_umap_plot_writes_file(tmp_path) -> None:
    (
        assignments,
        pca_scores,
        cluster_columns,
        metric_umap_coordinates,
        umap_metrics,
        embedding_metrics,
    ) = synthetic_interpretation_inputs()
    master = build_master_table(
        assignments_by_space=assignments,
        pca_scores_by_space=pca_scores,
        cluster_columns_by_space=cluster_columns,
        umap_metrics=umap_metrics,
        embedding_metrics=embedding_metrics,
        metric_umap_coordinates_by_space=metric_umap_coordinates,
    )
    representatives = pd.DataFrame(
        {
            "cluster_id": [1, 1, 2, 2],
            "representative_rank": [1, 2, 1, 2],
            "subfield_id": ["S1", "S2", "S3", "S4"],
        }
    )
    output_path = tmp_path / "combined_metric_space_umap_clusters.png"

    plot_combined_metric_space_umap(master, representatives, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
