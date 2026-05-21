# Clean Embedding Core Metrics

This step builds a reduced embedding-space metric set for the main analysis.
It does not replace or delete the broad table from script 12. The broad table
remains available for sensitivity analysis, diagnostics, and robustness checks.

The clean core exists to reduce redundancy, improve interpretability, avoid
near-constant/control variables, and keep conceptual coverage across semantic
coherence, dispersion, local density, hubness, intrinsic dimensionality, and
temporal semantic change.

## Command

```powershell
.\.venv\Scripts\python.exe scripts\16_build_clean_embedding_core_diagnostics.py --overwrite
```

Optional paths:

```powershell
.\.venv\Scripts\python.exe scripts\16_build_clean_embedding_core_diagnostics.py `
  --input-path data/processed/subfield_embedding_space_metrics.parquet `
  --output-dir outputs/analysis/clean_embedding_core_metrics `
  --min-non-missing-share 0.70 `
  --overwrite
```

## Inputs

```text
data/processed/subfield_embedding_space_metrics.parquet
```

## Outputs

```text
outputs/analysis/clean_embedding_core_metrics/clean_embedding_core_metrics.parquet
outputs/analysis/clean_embedding_core_metrics/clean_embedding_core_metrics.csv
outputs/analysis/clean_embedding_core_metrics/clean_core_spearman_correlation_matrix.csv
outputs/analysis/clean_embedding_core_metrics/clean_core_pearson_correlation_matrix.csv
outputs/analysis/clean_embedding_core_metrics/clean_core_spearman_correlation_heatmap.png
outputs/analysis/clean_embedding_core_metrics/clean_core_pearson_correlation_heatmap.png
outputs/analysis/clean_embedding_core_metrics/clean_core_top_abs_spearman_pairs.csv
outputs/analysis/clean_embedding_core_metrics/clean_core_top_abs_pearson_pairs.csv
outputs/analysis/clean_embedding_core_metrics/clean_core_distribution_diagnostics.csv
outputs/analysis/clean_embedding_core_metrics/clean_core_metric_groups.csv
outputs/analysis/clean_embedding_core_metrics/summary.md
outputs/analysis/clean_embedding_core_metrics/summary.json
```

## Final Clean Core

The final clean core contains 14 metrics:

```text
embedding_centroid_norm
embedding_distance_to_centroid_median
embedding_distance_to_centroid_iqr
embedding_distance_to_centroid_p90
embedding_knn_median_distance
embedding_knn_p90_distance
embedding_knn_distance_cv
embedding_knn_indegree_gini
embedding_pca_top5_variance_share
embedding_pca_dim_80
embedding_pca_spectral_entropy
embedding_centroid_drift_early_late
embedding_radial_expansion_slope
embedding_recent_novelty_score
```

Conceptual groups are written to `clean_core_metric_groups.csv`:

```text
semantic_coherence
semantic_dispersion
local_density_and_hubness
intrinsic_dimensionality
temporal_semantic_change
```

## Excluded From Final Core

These variables remain in the broad table but are not part of the clean core:

```text
embedding_distance_to_centroid_mean
embedding_distance_to_centroid_std
embedding_tail_index_p90_median
embedding_knn_mean_distance
embedding_pca_first_component_share
embedding_pca_top3_variance_share
embedding_pca_dim_50
embedding_pca_participation_ratio
embedding_graph_edge_distance_median
embedding_graph_edge_distance_p90
embedding_annual_centroid_path_length
embedding_directionality_ratio
```

They are excluded for conceptual compactness, reduced redundancy, and
robustness against outlier-driven or unstable summaries. They remain useful for
sensitivity checks.

## Diagnostic And Control Only

These should not be used as substantive morphology features:

```text
embedding_radial_expansion_r2
n_valid_embedding_rows
n_years_available
n_early_points
n_late_points
n_annual_centroids
n_pca_components_used
```

They remain useful for QA, filtering, and reproducibility.
