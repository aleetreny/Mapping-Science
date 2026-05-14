# Reduced Interpretable Embedding Core

This downstream step selects a compact embedding-space metric set from the full
script-12 metric table. It does not recompute embeddings, recompute
embedding-space metrics, or delete columns from the broad table.

The reduced core is intended to lower redundancy before PCA, clustering,
classification, disciplinary comparison, or other main downstream analysis. The
full embedding-space metric table remains available for sensitivity analysis
and robustness checks.

## Command

```powershell
.\.venv\Scripts\python.exe scripts\16b_build_reduced_interpretable_embedding_core.py --overwrite
```

Optional paths:

```powershell
.\.venv\Scripts\python.exe scripts\16b_build_reduced_interpretable_embedding_core.py `
  --input data/processed/subfield_embedding_space_metrics.parquet `
  --output-dir outputs/analysis/reduced_interpretable_embedding_core `
  --overwrite
```

## Input

```text
data/processed/subfield_embedding_space_metrics.parquet
```

Rows are filtered to metric statuses:

```text
completed
completed_with_warnings
```

## Reduced Core Metrics

The reduced core contains 11 metrics grouped into four interpretable blocks.

Global semantic dispersion:

```text
embedding_distance_to_centroid_median
embedding_distance_to_centroid_iqr
embedding_distance_to_centroid_p90
```

Local semantic density and hubness:

```text
embedding_knn_median_distance
embedding_knn_distance_cv
embedding_knn_indegree_gini
```

Intrinsic dimensionality:

```text
embedding_pca_dim_80
embedding_pca_spectral_entropy
```

Temporal semantic movement:

```text
embedding_centroid_drift_early_late
embedding_radial_expansion_slope
embedding_recent_novelty_score
```

The set is selected for interpretability, distributional stability, and reduced
redundancy. It should not be read as the objectively true or only possible
embedding metric set.

## Removed From The 14-Metric Clean Core

These metrics remain in the full table and in prior diagnostics, but are not in
the reduced core:

```text
embedding_centroid_norm
embedding_knn_p90_distance
embedding_pca_top5_variance_share
```

They are excluded because the clean-core correlation diagnostics showed high
redundancy with retained metrics: centroid norm is nearly inverse of
centroid-distance summaries; kNN P90 is highly redundant with kNN median; PCA
top-five variance share is highly redundant or anticorrelated with spectral
entropy and dim80.

## Outputs

```text
outputs/analysis/reduced_interpretable_embedding_core/reduced_interpretable_core_metrics.parquet
outputs/analysis/reduced_interpretable_embedding_core/reduced_interpretable_core_metrics.csv
outputs/analysis/reduced_interpretable_embedding_core/reduced_core_spearman_correlation_matrix.csv
outputs/analysis/reduced_interpretable_embedding_core/reduced_core_pearson_correlation_matrix.csv
outputs/analysis/reduced_interpretable_embedding_core/reduced_core_spearman_correlation_heatmap.png
outputs/analysis/reduced_interpretable_embedding_core/reduced_core_pearson_correlation_heatmap.png
outputs/analysis/reduced_interpretable_embedding_core/reduced_core_histograms.png
outputs/analysis/reduced_interpretable_embedding_core/reduced_core_top_absolute_spearman_pairs.csv
outputs/analysis/reduced_interpretable_embedding_core/reduced_core_top_absolute_pearson_pairs.csv
outputs/analysis/reduced_interpretable_embedding_core/reduced_core_distribution_diagnostics.csv
outputs/analysis/reduced_interpretable_embedding_core/summary.md
outputs/analysis/reduced_interpretable_embedding_core/summary.json
```
