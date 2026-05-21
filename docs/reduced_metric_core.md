# Reduced Metric Core

`scripts/12_build_reduced_metric_core.py` selects the final interpretable
embedding-space metric set used by the thesis.

Input:

```text
data/processed/subfield_embedding_space_metrics.parquet
```

Outputs:

```text
outputs/04_reduced_metric_core/reduced_interpretable_core_metrics.parquet
outputs/04_reduced_metric_core/reduced_interpretable_core_metrics.csv
outputs/04_reduced_metric_core/reduced_core_spearman_correlation_matrix.csv
outputs/04_reduced_metric_core/reduced_core_pearson_correlation_matrix.csv
outputs/04_reduced_metric_core/reduced_core_spearman_correlation_heatmap.png
outputs/04_reduced_metric_core/reduced_core_pearson_correlation_heatmap.png
outputs/04_reduced_metric_core/reduced_core_histograms.png
outputs/04_reduced_metric_core/summary.md
outputs/04_reduced_metric_core/summary.json
```

The 11 metrics are:

```text
embedding_distance_to_centroid_median
embedding_distance_to_centroid_iqr
embedding_distance_to_centroid_p90
embedding_knn_median_distance
embedding_knn_distance_cv
embedding_knn_indegree_gini
embedding_pca_dim_80
embedding_pca_spectral_entropy
embedding_centroid_drift_early_late
embedding_radial_expansion_slope
embedding_recent_novelty_score
```

Run:

```powershell
.\.venv\Scripts\python.exe scripts\12_build_reduced_metric_core.py --overwrite
```

This stage does not recompute embeddings and does not delete columns from the
full metric table.
