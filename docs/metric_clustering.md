# Metric-Space Clustering

This stage builds exploratory subfield typologies from the completed metric
tables. It clusters subfields in three metric spaces:

- `embedding_only`: the 25 core direct embedding-space metrics.
- `umap_only`: the 25 core projected UMAP morphology metrics.
- `combined`: both metric families, using block PCA by default.

The output is descriptive. Clusters are a compact way to inspect recurring
metric profiles, not evidence that subfields belong to true natural kinds.

## Command

```bash
python scripts/15_cluster_metric_spaces.py --default-k 5 --overwrite
```

Useful options:

```bash
python scripts/15_cluster_metric_spaces.py --spaces embedding_only umap_only combined --combined-strategy block_pca --scaler standard --pca-variance-threshold 0.90 --k-min 3 --k-max 8 --default-k 5 --overwrite
```

## Inputs

The script requires:

```text
data/processed/subfield_morphology_metrics.parquet
data/processed/subfield_embedding_space_metrics.parquet
```

If present, this diagnostic table is reported but not used as an automatic
drop list:

```text
outputs/analysis/metric_distributions/low_information_metrics.csv
```

Rows with `metric_status` equal to `completed` or
`completed_with_warnings` are eligible.

## Preprocessing

Each metric block is filtered before clustering:

- Drop metrics with missing share above `--max-missing-share` (default `0.30`).
- Drop constant metrics and metrics with fewer than `--min-unique-values`
  unique non-missing values (default `4`).
- Median-impute retained metrics.
- Scale retained metrics with `standard` or `robust` scaling.
- Reduce retained metrics with PCA until the requested cumulative variance
  threshold is reached.

For `combined --combined-strategy block_pca`, projected and embedding metrics
are preprocessed and reduced separately, then the retained PC scores are
concatenated. This prevents the combined space from being dominated simply by
one raw metric block having more variance.

## Clustering

The main assignment is Ward hierarchical clustering on the PCA scores. The
script evaluates all k values from `--k-min` through `--k-max`, bounded by the
number of available subfields, and writes `cluster_ward_k{k}` columns.

KMeans and GaussianMixture are included as robustness checks on the same PCA
score matrix. The default-k KMeans and GMM assignments are included in the
assignment table, while quality diagnostics are written for every evaluated k.

## Outputs

Per space:

```text
outputs/analysis/metric_clustering/embedding_only/
outputs/analysis/metric_clustering/umap_only/
outputs/analysis/metric_clustering/combined/
```

Each space contains:

```text
cluster_assignments.csv
cluster_assignments.parquet
preprocessing_report.csv
pca_scores.csv
pca_loadings.csv
pca_explained_variance.csv
cluster_quality_by_k.csv
cluster_profiles.csv
cluster_representatives.csv
dendrogram.png
pca_scatter_clusters.png
metric_space_umap_clusters.png
cluster_profile_heatmap.png
summary.md
summary.json
```

When all three spaces are run with the same main cluster column, comparison
outputs are written under:

```text
outputs/analysis/metric_clustering/comparison/
```

Comparison files include adjusted Rand index, normalized mutual information,
contingency tables for each pair of spaces, an agreement heatmap, and Markdown
and JSON summaries.

## Interpretation Notes

Start with the preprocessing report before interpreting clusters. A cluster
solution based on many dropped metrics or many imputed values should be treated
as fragile.

Read cluster profiles as standardized median differences from the global
metric distribution. Representatives are subfields closest to each cluster
centroid in the PCA score space; they are examples, not labels.

Low agreement between `embedding_only` and `umap_only` does not automatically
mean a problem. The two spaces measure related but different aspects of the
same corpus: high-dimensional semantic structure and projected landscape
morphology.
