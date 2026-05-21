# Metric Family Comparison

This diagnostic compares the two active metric families:

```text
projected UMAP morphology metrics
original embedding-space metrics
```

It is not a proof that one representation is correct, and it does not make
scientific conclusions. It gives a first check of whether the projected 2D
landscape metrics and high-dimensional embedding metrics tell compatible or
distinct stories.

## Command

```bash
python scripts/13_compare_metric_families.py --overwrite
```

The script requires both metric tables to already exist:

```text
data/processed/subfield_morphology_metrics.parquet
data/processed/subfield_embedding_space_metrics.parquet
```

## Outputs

```text
outputs/analysis/metric_family_comparison/metric_family_comparison_summary.json
outputs/analysis/metric_family_comparison/metric_family_comparison_summary.md
outputs/analysis/metric_family_comparison/cross_family_spearman_correlations.csv
outputs/analysis/metric_family_comparison/cross_family_pearson_correlations.csv
outputs/analysis/metric_family_comparison/top_absolute_spearman_correlations.csv
outputs/analysis/metric_family_comparison/analogue_metric_pair_correlations.csv
outputs/analysis/metric_family_comparison/cross_family_spearman_heatmap.png
```

Rows are joined on `subfield_id`. Rows with `metric_status` equal to
`completed` or `completed_with_warnings` are eligible.

## Analogue Pairs

The analogue-pair table includes deliberately imperfect conceptual matches such
as:

```text
knn_median_distance <-> embedding_knn_median_distance
radial_tail_index <-> embedding_tail_index_p90_median
centroid_drift_early_late <-> embedding_centroid_drift_early_late
mst_gap_index <-> embedding_graph_edge_distance_p90
anisotropy_ratio <-> embedding_pca_first_component_share
```

Weak correlation is not automatically failure. Projection metrics and
embedding-space metrics summarize different views of the same corpus.
