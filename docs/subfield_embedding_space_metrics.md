# Subfield Embedding-Space Metrics

This stage computes one row of high-dimensional semantic structure metrics per
OpenAlex subfield directly from SPECTER2 embeddings. It complements the
projected UMAP morphology table rather than replacing it.

Conceptual split:

```text
projected UMAP landscape -> visible 2D morphology
original embedding space -> high-dimensional semantic structure
```

The table contains exactly 25 curated core embedding-space metrics plus
diagnostics and controls. It does not run clustering, PCA over the final metric
table, regression, classification, or visual interpretation.

## Inputs

```text
data/processed/analysis_embedding_index.parquet
embeddings/specter2_v1/analysis/main_embeddings.float16.npy
```

The matrix is loaded with:

```python
np.load(matrix_path, mmap_mode="r")
```

For each subfield, the script filters rows to the selected year window,
extracts `analysis_row_id`, reads only those rows from the memory-mapped matrix,
converts the subfield slice to `float32`, and L2-normalizes rows before
cosine-based metrics.

The default active window is:

```text
2010 <= publication_year <= 2025
```

Year 2026 is excluded because it is incomplete/current.

## Outputs

```text
data/processed/subfield_embedding_space_metrics.parquet
data/processed/subfield_embedding_space_metrics.csv
outputs/metrics/subfield_embedding_space_metrics_summary.json
outputs/metrics/subfield_embedding_space_metrics_dictionary.csv
```

## Control Columns

The output uses `subfield_id` as the primary key. Display names are included
for readability but are not unique.

Important controls:

```text
subfield_id
subfield_display_name
subfield_label_unique
subfield_label_short
subfield_display_name_is_duplicated
field_id
field_display_name
domain_id
domain_display_name
n_available
n_used
year_min
year_max
metric_status
metric_error_message
metric_warning_message
embedding_matrix_path
analysis_index_path
k_neighbors
effective_k_neighbors
random_state
sampling_applied
max_papers_per_subfield
```

Diagnostics include:

```text
embedding_radial_expansion_r2
n_valid_embedding_rows
n_years_available
n_early_points
n_late_points
n_annual_centroids
n_pca_components_used
```

## The 25 Core Metrics

Semantic concentration and dispersion:

```text
embedding_centroid_norm
embedding_distance_to_centroid_mean
embedding_distance_to_centroid_median
embedding_distance_to_centroid_std
embedding_distance_to_centroid_iqr
embedding_distance_to_centroid_p90
embedding_tail_index_p90_median
```

Local semantic density:

```text
embedding_knn_mean_distance
embedding_knn_median_distance
embedding_knn_p90_distance
embedding_knn_distance_cv
```

Intrinsic dimensionality:

```text
embedding_pca_first_component_share
embedding_pca_top3_variance_share
embedding_pca_top5_variance_share
embedding_pca_dim_50
embedding_pca_dim_80
embedding_pca_participation_ratio
```

kNN graph structure:

```text
embedding_graph_connected_component_count
embedding_graph_largest_component_share
embedding_graph_edge_distance_median
embedding_graph_edge_distance_p90
```

Temporal movement in embedding space:

```text
embedding_centroid_drift_early_late
embedding_annual_centroid_path_length
embedding_directionality_ratio
embedding_radial_expansion_slope
```

The fixed early window is 2010-2012 and the fixed late window is 2023-2025.
If a temporal metric cannot be computed because too few papers are available in
the needed windows, the metric is `NaN` and the row is marked
`completed_with_warnings`.

`embedding_radial_expansion_r2` is retained as a diagnostic fit-strength column
and does not count toward the 25 core metrics.

## Sampling

Conceptually, the embedding-space metrics use all eligible papers from
2010-2025. By default, `--max-papers-per-subfield` is unset, so no cap is
applied.

If a cap is provided for runtime reasons, sampling is deterministic for
`random_state` and `subfield_id`. The output records:

```text
n_available
n_used
sampling_applied
max_papers_per_subfield
```

## Commands

Smoke test:

```bash
python scripts/12_compute_subfield_embedding_space_metrics.py --limit-subfields 3 --year-min 2010 --year-max 2025 --overwrite
```

Single subfield:

```bash
python scripts/12_compute_subfield_embedding_space_metrics.py --subfield-id 1100 --year-min 2010 --year-max 2025 --overwrite
```

Full run:

```bash
python scripts/12_compute_subfield_embedding_space_metrics.py --year-min 2010 --year-max 2025 --overwrite
```

Optional deterministic cap:

```bash
python scripts/12_compute_subfield_embedding_space_metrics.py --year-min 2010 --year-max 2025 --max-papers-per-subfield 10000 --overwrite
```

## Interpretation Caveats

These metrics describe semantic structure in the original SPECTER2 embedding
space. They avoid the distortions of a 2D projection, but they still depend on
the embedding model and the title-plus-abstract text available in the sampled
corpus.

The projected and embedding-space families both contain 25 core metrics for a
balanced, interpretable design. This does not give them equal statistical
weight automatically. Later analysis should standardize features and may use
block weighting if needed.

After both projected and embedding-space tables exist, run:

```bash
python scripts/13_compare_metric_families.py --overwrite
python scripts/14_summarize_metric_distributions.py --overwrite
```

These scripts summarize cross-family correlations and simple metric
distribution diagnostics. They are not clustering, regression, or final
interpretive analysis.
