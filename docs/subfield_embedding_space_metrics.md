# Subfield Embedding-Space Metrics

This stage computes one row of high-dimensional semantic structure metrics per
OpenAlex subfield directly from SPECTER2 embeddings. It complements the
projected UMAP morphology table rather than replacing it.

Conceptual split:

```text
projected UMAP landscape -> visible 2D morphology
original embedding space -> high-dimensional semantic structure
```

The table contains 26 curated core embedding-space metrics plus diagnostics and
controls. The metric table stage itself does not run
clustering, PCA over the final metric table, regression, classification, or
visual interpretation.

## Inputs

```text
data/processed/analysis_embedding_index.parquet
<embedding-dir>/analysis/main_embeddings.float16.npy
```

The matrix is loaded with:

```python
np.load(matrix_path, mmap_mode="r")
```

For each subfield, the script filters rows to the selected year window,
extracts `analysis_row_id`, reads only those rows from the memory-mapped matrix,
converts the subfield slice to `float32`, and L2-normalizes rows before
cosine-based metrics.

`--embedding-dir` defaults to `LOCAL_EMBEDDINGS_DIR` from `.env` or the
environment. For `2000_2024_400py`, either set:

```powershell
$env:LOCAL_EMBEDDINGS_DIR = "embeddings/specter2_v1_2000_2024_400py"
```

or pass `--embedding-dir embeddings/specter2_v1_2000_2024_400py`.

The active `2000_2024_400py` metric window is:

```text
2000 <= publication_year <= 2024
```

The legacy `2010-2025` window is still supported for older artifacts; year
2026 remains excluded because it is incomplete/current.

## Outputs

```text
data/processed/subfield_embedding_space_metrics.parquet
data/processed/subfield_embedding_space_metrics.csv
outputs/metrics/subfield_embedding_space_metrics_summary.json
outputs/metrics/subfield_embedding_space_metrics_dictionary.csv
outputs/analysis/embedding_space_metric_diagnostics/
```

The diagnostic folder contains the core metric histogram grid, Spearman and
Pearson core-metric heatmaps, the corresponding correlation matrices, a ranked
file of high absolute Spearman pairs, and a distribution diagnostic table that
labels each candidate column as `core`, `diagnostic`, `control`, or
`excluded_from_core`.

```text
outputs/analysis/embedding_space_metric_diagnostics/embedding_metric_histograms_core.png
outputs/analysis/embedding_space_metric_diagnostics/embedding_metric_spearman_correlation_heatmap.png
outputs/analysis/embedding_space_metric_diagnostics/embedding_metric_pearson_correlation_heatmap.png
outputs/analysis/embedding_space_metric_diagnostics/embedding_metric_spearman_correlation_matrix.csv
outputs/analysis/embedding_space_metric_diagnostics/embedding_metric_pearson_correlation_matrix.csv
outputs/analysis/embedding_space_metric_diagnostics/embedding_metric_top_abs_spearman_pairs.csv
outputs/analysis/embedding_space_metric_diagnostics/embedding_metric_distribution_diagnostics.csv
outputs/analysis/embedding_space_metric_diagnostics/embedding_metric_diagnostics_summary.md
outputs/analysis/embedding_space_metric_diagnostics/embedding_metric_diagnostics_summary.json
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
early_year_min
early_year_max
late_year_min
late_year_max
metric_status
metric_error_message
metric_warning_message
embedding_matrix_path
analysis_index_path
k_neighbors
effective_k_neighbors
n_valid_embedding_rows
n_years_available
n_early_points
n_late_points
n_annual_centroids
n_pca_components_used
random_state
sampling_applied
max_papers_per_subfield
```

Diagnostics include:

```text
embedding_graph_connected_component_count
embedding_graph_largest_component_share
embedding_radial_expansion_r2
```

`embedding_graph_connected_component_count` and
`embedding_graph_largest_component_share` are retained as diagnostics because
they can help check kNN graph construction, but they are not core features. In
the current embedding-space runs they are close to constant and therefore add
little substantive discrimination between subfields.

## The 26 Core Metrics

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
embedding_knn_indegree_gini
```

Intrinsic dimensionality:

```text
embedding_pca_first_component_share
embedding_pca_top3_variance_share
embedding_pca_top5_variance_share
embedding_pca_dim_50
embedding_pca_dim_80
embedding_pca_participation_ratio
embedding_pca_spectral_entropy
```

kNN graph structure:

```text
embedding_graph_edge_distance_median
embedding_graph_edge_distance_p90
```

Temporal movement in embedding space:

```text
embedding_centroid_drift_early_late
embedding_annual_centroid_path_length
embedding_directionality_ratio
embedding_radial_expansion_slope
embedding_recent_novelty_score
```

The early window is the first five selected years and the late window is the
last five selected years. For `2000_2024_400py`, that means early 2000-2004 and
late 2020-2024. If a temporal metric cannot be computed because too few papers
are available in the needed windows, the metric is `NaN` and the row is marked
`completed_with_warnings`.

`embedding_radial_expansion_r2` is retained as a diagnostic fit-strength column
and does not count toward the 26 core metrics.

## New Metric Definitions

`embedding_pca_spectral_entropy` is the normalized Shannon entropy of the
positive PCA explained-variance spectrum. Low values mean variance is
concentrated in a few latent directions; high values mean variance is spread
more evenly across positive-variance directions.

`embedding_knn_indegree_gini` is the Gini coefficient of directed kNN in-degree
counts. Low values mean papers are selected as local semantic neighbors at more
similar rates; high values mean a smaller subset of papers acts as stronger
local semantic hubs.

`embedding_recent_novelty_score` compares late-window papers with the early
semantic core. It is the median cosine distance from late-window papers to the
early-window centroid minus the median cosine distance from early-window papers
to that same early centroid. Positive values indicate that recent papers sit
farther from the early core than early papers did.

## Sampling

Conceptually, the embedding-space metrics use all eligible papers in the
selected metric year window. For `2000_2024_400py`, that is 2000-2024. By
default, `--max-papers-per-subfield` is unset, so no cap is applied.

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

```powershell
.\.venv\Scripts\python.exe scripts\12_compute_subfield_embedding_space_metrics.py `
  --embedding-dir embeddings/specter2_v1_2000_2024_400py `
  --limit-subfields 3 `
  --year-min 2000 `
  --year-max 2024 `
  --overwrite
```

Single subfield:

```powershell
.\.venv\Scripts\python.exe scripts\12_compute_subfield_embedding_space_metrics.py `
  --embedding-dir embeddings/specter2_v1_2000_2024_400py `
  --subfield-id 1100 `
  --year-min 2000 `
  --year-max 2024 `
  --overwrite
```

Full run:

```powershell
.\.venv\Scripts\python.exe scripts\12_compute_subfield_embedding_space_metrics.py `
  --embedding-dir embeddings/specter2_v1_2000_2024_400py `
  --year-min 2000 `
  --year-max 2024 `
  --overwrite
```

Optional deterministic cap:

```powershell
.\.venv\Scripts\python.exe scripts\12_compute_subfield_embedding_space_metrics.py `
  --embedding-dir embeddings/specter2_v1_2000_2024_400py `
  --year-min 2000 `
  --year-max 2024 `
  --max-papers-per-subfield 10000 `
  --overwrite
```

## Interpretation Caveats

These metrics describe semantic structure in the original SPECTER2 embedding
space. They avoid the distortions of a 2D projection, but they still depend on
the embedding model and the title-plus-abstract text available in the sampled
corpus.

The projected and embedding-space families are related but no longer forced to
have the same metric count. This does not give either family equal statistical
weight automatically. Later analysis should standardize features and may use
block weighting if needed.

After both projected and embedding-space tables exist, run:

```bash
python scripts/13_compare_metric_families.py --overwrite
python scripts/14_summarize_metric_distributions.py --overwrite
python scripts/15_cluster_metric_spaces.py --default-k 5 --overwrite
```

These scripts summarize cross-family correlations, metric distributions, and
exploratory metric-space clusters. They are not regression or final
interpretive analysis.
