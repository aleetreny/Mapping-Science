from __future__ import annotations

import numpy as np
import pandas as pd

from src.dimensionality_reduction_comparison import (
    DimensionalityReductionConfig,
    effective_neighbors,
    effective_tsne_perplexity,
    fit_umap_tuned_projection,
    fit_projection,
    plot_projection_grid,
    plot_umap_tuning_grid,
    projection_coordinate_frame,
    umap_tuning_coordinate_frame,
    umap_tuning_quality_row,
)


def test_effective_neighbors_and_tsne_perplexity_are_bounded() -> None:
    assert effective_neighbors(10, 15) == 9
    assert effective_neighbors(10, 4) == 4
    assert effective_tsne_perplexity(12, 30.0) < 12


def test_pca_projection_returns_two_coordinates() -> None:
    rng = np.random.default_rng(7)
    features = pd.DataFrame(rng.normal(size=(12, 4)))

    coords, warning, params = fit_projection(
        features,
        method="pca",
        config=DimensionalityReductionConfig(random_state=7),
    )

    assert warning == ""
    assert coords.shape == (12, 2)
    assert "explained_variance_ratio_1" in params


def test_projection_coordinate_frame_includes_metadata_and_cluster() -> None:
    metadata = pd.DataFrame(
        {
            "subfield_id": ["S1", "S2"],
            "subfield_display_name": ["Alpha", "Beta"],
            "field_display_name": ["Field 1", "Field 2"],
            "domain_display_name": ["Domain 1", "Domain 2"],
            "cluster_ward_k2": [1, 2],
        }
    )
    coords = np.array([[0.1, 0.2], [1.1, 1.2]])

    frame = projection_coordinate_frame(
        space="combined",
        method="umap",
        metadata=metadata,
        coordinates=coords,
        cluster_column="cluster_ward_k2",
    )

    assert frame.loc[0, "feature_set"] == "combined"
    assert frame.loc[1, "method_display_name"] == "UMAP"
    assert frame.loc[1, "cluster_label"] == 2
    assert frame.loc[0, "x"] == 0.1


def test_projection_grid_writes_png(tmp_path) -> None:
    rows = []
    for method in ["pca", "mds", "tsne", "umap", "isomap", "phate"]:
        for idx in range(6):
            rows.append(
                {
                    "feature_set": "combined",
                    "method": method,
                    "method_display_name": method,
                    "subfield_id": f"S{idx}",
                    "subfield_display_name": f"Subfield {idx}",
                    "field_display_name": "Field",
                    "domain_display_name": "Domain 1" if idx < 3 else "Domain 2",
                    "cluster_label": 1 if idx < 3 else 2,
                    "x": float(idx),
                    "y": float(idx % 3),
                    "projection_warning": "",
                }
            )
    coordinates = pd.DataFrame(rows)
    representatives = pd.DataFrame(
        {
            "cluster_id": [1, 2],
            "rank": [1, 1],
            "subfield_id": ["S0", "S3"],
        }
    )
    output_path = tmp_path / "grid.png"

    plot_projection_grid(
        space="combined",
        coordinates=coordinates,
        representatives=representatives,
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_umap_tuning_coordinate_and_quality_rows() -> None:
    metadata = pd.DataFrame(
        {
            "subfield_id": ["S1", "S2", "S3"],
            "subfield_display_name": ["A", "B", "C"],
            "field_display_name": ["F", "F", "G"],
            "domain_display_name": ["D", "D", "E"],
            "cluster_ward_k2": [1, 1, 2],
        }
    )
    coords = np.array([[0.0, 0.0], [0.1, 0.1], [2.0, 2.0]])

    frame = umap_tuning_coordinate_frame(
        space="combined",
        metadata=metadata,
        coordinates=coords,
        cluster_column="cluster_ward_k2",
        n_neighbors=5,
        min_dist=0.1,
    )
    quality = umap_tuning_quality_row(
        space="combined",
        features=pd.DataFrame(np.random.default_rng(3).normal(size=(3, 2))),
        coordinates=coords,
        labels=metadata["cluster_ward_k2"],
        n_neighbors=5,
        min_dist=0.1,
        warning="",
    )

    assert frame["n_neighbors"].tolist() == [5, 5, 5]
    assert frame["min_dist"].tolist() == [0.1, 0.1, 0.1]
    assert quality["effective_n_neighbors"] == 2
    assert quality["feature_set"] == "combined"


def test_tuned_umap_projection_returns_two_coordinates() -> None:
    features = pd.DataFrame(np.random.default_rng(4).normal(size=(16, 4)))

    coords, warning, params = fit_umap_tuned_projection(
        features,
        n_neighbors=5,
        min_dist=0.2,
        random_state=4,
    )

    assert warning == ""
    assert coords.shape == (16, 2)
    assert params["min_dist"] == 0.2


def test_umap_tuning_grid_writes_png(tmp_path) -> None:
    rows = []
    for n_neighbors in [5, 10]:
        for min_dist in [0.0, 0.3]:
            for idx in range(6):
                rows.append(
                    {
                        "feature_set": "combined",
                        "n_neighbors": n_neighbors,
                        "min_dist": min_dist,
                        "subfield_id": f"S{idx}",
                        "subfield_display_name": f"Subfield {idx}",
                        "field_display_name": "Field",
                        "domain_display_name": "Domain",
                        "cluster_label": 1 if idx < 3 else 2,
                        "x": float(idx),
                        "y": float(idx % 3),
                        "projection_warning": "",
                    }
                )
    coordinates = pd.DataFrame(rows)
    quality = pd.DataFrame(
        [
            {
                "feature_set": "combined",
                "n_neighbors": n_neighbors,
                "min_dist": min_dist,
                "trustworthiness": 0.9,
                "cluster_silhouette_2d": 0.2,
            }
            for n_neighbors in [5, 10]
            for min_dist in [0.0, 0.3]
        ]
    )
    output_path = tmp_path / "umap_tuning_grid.png"

    plot_umap_tuning_grid(
        space="combined",
        coordinates=coordinates,
        quality=quality,
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
