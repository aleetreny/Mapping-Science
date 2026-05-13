# Agent prompt: build cluster interpretation tables and figures for TFM

We have completed the metric pipeline for the new dataset version `2000_2024_400py`.

Current confirmed state:

- OpenAlex corpus: `2000–2024`
- Main-analysis subfields: `241`
- Embedding folder: `embeddings/specter2_v1_2000_2024_400py`
- UMAP morphology metrics: `241 completed, 0 failed`
- Embedding-space metrics: `241 completed, 0 failed`
- Metric-family comparison: completed
- Metric-distribution diagnostics: completed
- Metric-space clustering: completed for:
  - `embedding_only`
  - `umap_only`
  - `combined`

Current clustering outputs live in:

```text
outputs/analysis/metric_clustering/
```

Known files include:

```text
outputs/analysis/metric_clustering/embedding_only/cluster_assignments.csv
outputs/analysis/metric_clustering/embedding_only/cluster_profiles.csv
outputs/analysis/metric_clustering/embedding_only/cluster_representatives.csv
outputs/analysis/metric_clustering/embedding_only/pca_scores.csv
outputs/analysis/metric_clustering/embedding_only/pca_loadings.csv
outputs/analysis/metric_clustering/embedding_only/summary.md

outputs/analysis/metric_clustering/umap_only/cluster_assignments.csv
outputs/analysis/metric_clustering/umap_only/cluster_profiles.csv
outputs/analysis/metric_clustering/umap_only/cluster_representatives.csv
outputs/analysis/metric_clustering/umap_only/pca_scores.csv
outputs/analysis/metric_clustering/umap_only/pca_loadings.csv
outputs/analysis/metric_clustering/umap_only/summary.md

outputs/analysis/metric_clustering/combined/cluster_assignments.csv
outputs/analysis/metric_clustering/combined/cluster_profiles.csv
outputs/analysis/metric_clustering/combined/cluster_representatives.csv
outputs/analysis/metric_clustering/combined/pca_scores.csv
outputs/analysis/metric_clustering/combined/pca_loadings.csv
outputs/analysis/metric_clustering/combined/summary.md

outputs/analysis/metric_clustering/comparison/cluster_partition_agreement.csv
outputs/analysis/metric_clustering/comparison/cluster_contingency_embedding_vs_combined.csv
outputs/analysis/metric_clustering/comparison/cluster_contingency_embedding_vs_umap.csv
outputs/analysis/metric_clustering/comparison/cluster_contingency_umap_vs_combined.csv
outputs/analysis/metric_clustering/comparison/summary.md
```

## Goal

Create a clean interpretation layer for the thesis.

The goal is **not** to rerun embeddings, UMAPs, metrics, or clustering. The goal is to convert the clustering outputs into tables, summaries, and simple figures that are easy to interpret and include in the TFM.

Add a new script:

```text
scripts/16_build_cluster_interpretation_tables.py
```

and, if useful, a small module:

```text
src/cluster_interpretation.py
```

Output directory:

```text
outputs/analysis/cluster_interpretation/
```

## Conceptual framing

Use `combined` as the main exploratory typology, because it merges both sources of information:

- original high-dimensional embedding-space metrics;
- projected UMAP morphology metrics.

But also report how `combined` relates to:

- `embedding_only`
- `umap_only`

Be careful with language. Do **not** claim clusters are objectively true classes of science. Treat them as exploratory morphology profiles.

Recommended wording in summaries:

```text
These clusters are exploratory morphology profiles, not definitive disciplinary classes.
They summarize similarities between subfields according to the selected metric space.
```

## Required outputs

### 1. Master subfield table

Create:

```text
outputs/analysis/cluster_interpretation/subfield_cluster_master_table.csv
outputs/analysis/cluster_interpretation/subfield_cluster_master_table.parquet
```

One row per subfield.

Include at least:

```text
subfield_id
subfield_display_name
field_id
field_display_name
domain_id
domain_display_name

cluster_embedding_only
cluster_umap_only
cluster_combined

combined_pca1
combined_pca2
combined_umap_x
combined_umap_y

embedding_only_pca1
embedding_only_pca2
umap_only_pca1
umap_only_pca2

n_available / n_used if available
key selected metrics from both families if easy to join
```

Use existing `cluster_assignments`, `pca_scores`, and metric tables:

```text
data/processed/subfield_morphology_metrics.parquet
data/processed/subfield_embedding_space_metrics.parquet
outputs/analysis/metric_clustering/*/cluster_assignments.csv
outputs/analysis/metric_clustering/*/pca_scores.csv
```

If `metric_space_umap_clusters.png` exists but no UMAP coordinates table exists, do not reverse-engineer coordinates from the image. Just use PCA coordinates unless there is already a coordinate table.

### 2. Cluster composition tables

Create composition tables for the `combined` clusters:

```text
outputs/analysis/cluster_interpretation/combined_cluster_domain_composition.csv
outputs/analysis/cluster_interpretation/combined_cluster_field_composition.csv
outputs/analysis/cluster_interpretation/combined_cluster_size_summary.csv
```

For each combined cluster, include:

```text
cluster_id
n_subfields
domain / field
count
share_within_cluster
share_within_domain_or_field
```

Also create a compact human-readable markdown summary:

```text
outputs/analysis/cluster_interpretation/combined_cluster_interpretation_summary.md
```

The markdown should include, for each combined cluster:

- size;
- dominant domains;
- dominant fields;
- representative subfields;
- strongest high/low metrics;
- cautious suggested interpretation label.

### 3. Representative subfields

Create:

```text
outputs/analysis/cluster_interpretation/combined_cluster_representatives_readable.csv
```

Use existing `combined/cluster_representatives.csv` if available.

Make it more readable by including:

```text
cluster_id
rank / representative_rank
subfield_id
subfield_display_name
field_display_name
domain_display_name
distance_to_cluster_center or equivalent if available
```

Also include a fallback if the existing representative file has different column names.

### 4. Cluster metric profiles

Create:

```text
outputs/analysis/cluster_interpretation/combined_cluster_metric_profile_long.csv
outputs/analysis/cluster_interpretation/combined_cluster_metric_profile_top_differences.csv
```

Use the combined `cluster_profiles.csv` and/or recompute cluster-wise z-score means from the metric tables.

For each cluster, identify:

- top 8 metrics with highest standardized mean;
- top 8 metrics with lowest standardized mean.

Include:

```text
cluster_id
metric_name
metric_family: umap_projected / embedding_space
cluster_mean_z
direction: high / low
plain_language_hint
```

Add plain-language hints for the main metrics where possible. Keep them simple, for example:

```text
anisotropy_ratio: elongated or directional map shape
density_peak_count: number of dense local peaks in the projected map
effective_area_90: spatial area occupied by the densest 90% of mass
embedding_pca_first_component_share: concentration of embedding variance in one dominant axis
embedding_pca_participation_ratio: effective dimensionality of the embedding cloud
embedding_distance_to_centroid_mean: semantic dispersion around the subfield centroid
embedding_centroid_drift_early_late: change in semantic centroid between early and late periods
```

Do not overexplain every metric if too many; a practical mapping for the top-profile metrics is enough.

### 5. Cluster agreement summary

Create:

```text
outputs/analysis/cluster_interpretation/cluster_family_agreement_summary.csv
outputs/analysis/cluster_interpretation/cluster_family_agreement_summary.md
```

Use:

```text
outputs/analysis/metric_clustering/comparison/cluster_partition_agreement.csv
```

Also include contingency tables or readable summaries of:

```text
embedding_only vs combined
umap_only vs combined
embedding_only vs umap_only
```

Important interpretation:
- Low ARI/NMI is not a failure by itself.
- It means embedding-space and projected UMAP metrics capture related but non-equivalent aspects of morphology.
- This should support the decision to report both families rather than collapse everything into one.

### 6. Figures

Create simple, readable figures. Do not overcomplicate.

Required figures:

```text
outputs/analysis/cluster_interpretation/combined_cluster_domain_stacked_bar.png
outputs/analysis/cluster_interpretation/combined_cluster_profile_heatmap_top_metrics.png
outputs/analysis/cluster_interpretation/combined_pca_scatter_labeled.png
```

Optional, if easy:

```text
outputs/analysis/cluster_interpretation/cluster_agreement_contingency_heatmap_combined_vs_embedding.png
outputs/analysis/cluster_interpretation/cluster_agreement_contingency_heatmap_combined_vs_umap.png
```

Figure requirements:
- readable labels;
- no tiny unreadable legends;
- use sensible figure sizes;
- save as PNG;
- avoid overly dense plots;
- if labels overlap too much, label only representatives.

### 7. Final summary JSON

Create:

```text
outputs/analysis/cluster_interpretation/summary.json
```

Include:

```text
created_at
input_paths
output_paths
n_subfields
cluster_spaces_used
combined_cluster_sizes
dominant_domains_by_cluster
top_profile_metrics_by_cluster
warnings
```

## Robustness and simplicity

Important:
- Keep the implementation simple.
- Do not add a complex new clustering method.
- Do not rerun metrics or clustering.
- Do not mutate existing clustering outputs.
- Do not require manual interpretation hardcoded into the script beyond simple metric hints.

The script should fail clearly if required input files are missing.

It should also tolerate minor column-name differences in clustering outputs where reasonable.

## CLI

The script should be runnable with:

```powershell
.\.venv\Scripts\python.exe scripts\16_build_cluster_interpretation_tables.py --overwrite
```

Optionally support:

```powershell
--clustering-dir outputs/analysis/metric_clustering
--metrics-umap data/processed/subfield_morphology_metrics.parquet
--metrics-embedding data/processed/subfield_embedding_space_metrics.parquet
--output-dir outputs/analysis/cluster_interpretation
--main-space combined
```

## Tests

Add focused tests for the core functions, not huge integration tests.

Suggested tests:
- master table joins cluster assignments from three spaces correctly;
- domain/field composition shares sum correctly;
- top high/low metric profiles are produced;
- missing files fail with clear messages;
- representative table handles expected columns.

Run:

```powershell
.\.venv\Scripts\python.exe -m compileall scripts src
.\.venv\Scripts\python.exe -m pytest
```

## After implementation

Run the script against the real outputs:

```powershell
.\.venv\Scripts\python.exe scripts\16_build_cluster_interpretation_tables.py --overwrite
```

Then report:
- files written;
- number of subfields in master table;
- combined cluster sizes;
- where to find the readable markdown summary;
- tests passed.
