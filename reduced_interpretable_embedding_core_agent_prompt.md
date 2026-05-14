# Agent Prompt — Build the Reduced Interpretable Embedding Core Metrics Step

You are working in the local repository:

```text
C:/Users/Z0058EYW/Workspace/TFM
```

The project studies the morphology of scientific subfields using SPECTER2 embeddings. The previous step computed a broad embedding-space metric table with many columns, and a first "clean core" heatmap showed that the metric set is conceptually better but still statistically redundant.

Your task is to create a **separate analysis step** that defines and evaluates a more reduced, interpretable embedding-space core. Do **not** modify the metric computation logic in script 12. Do **not** delete any columns from the existing metric table. This should be a new downstream step that selects a final reduced core for analysis.

---

## Main Goal

Create a new script that defines a reduced, interpretable embedding-space core metric set and produces diagnostics for it:

```text
scripts/16b_build_reduced_interpretable_embedding_core.py
```

or, if naming conventions in the repo suggest another number/name, choose the most consistent unused script name. The important point is that this is a **new step after the clean-core script**, not a replacement for script 12.

The reduced core should be treated as the likely final embedding-space feature set for the main thesis analysis.

---

## Input

Use the existing full embedding-space metric table produced by script 12:

```text
data/processed/subfield_embedding_space_metrics.parquet
```

Optionally support a CLI argument:

```bash
--input data/processed/subfield_embedding_space_metrics.parquet
```

The script should filter to rows with:

```text
metric_status in {"completed", "completed_with_warnings"}
```

unless the existing project utilities already use a different but equivalent filtering convention.

---

## Reduced Interpretable Core Metrics

Define the following exact reduced core list:

```python
REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS = [
    # Global semantic dispersion
    "embedding_distance_to_centroid_median",
    "embedding_distance_to_centroid_iqr",
    "embedding_distance_to_centroid_p90",

    # Local semantic density / hubness
    "embedding_knn_median_distance",
    "embedding_knn_distance_cv",
    "embedding_knn_indegree_gini",

    # Intrinsic dimensionality
    "embedding_pca_dim_80",
    "embedding_pca_spectral_entropy",

    # Temporal semantic movement
    "embedding_centroid_drift_early_late",
    "embedding_radial_expansion_slope",
    "embedding_recent_novelty_score",
]
```

These 11 metrics intentionally remove the following variables from the previous clean core because they were highly redundant:

```text
embedding_centroid_norm
embedding_knn_p90_distance
embedding_pca_top5_variance_share
```

Do not delete those columns from the full dataset. Just exclude them from this reduced-core analysis.

---

## Conceptual Rationale to Document

The reduced core has four interpretable blocks:

1. **Global semantic dispersion**
   - `embedding_distance_to_centroid_median`: typical distance of papers to the subfield semantic centroid.
   - `embedding_distance_to_centroid_iqr`: robust spread of distances to the centroid.
   - `embedding_distance_to_centroid_p90`: upper-tail distance to the centroid.

2. **Local semantic density and hubness**
   - `embedding_knn_median_distance`: typical local semantic spacing.
   - `embedding_knn_distance_cv`: inequality/heterogeneity of local density.
   - `embedding_knn_indegree_gini`: hubness, measuring whether a small set of papers acts as semantic hubs in the kNN structure.

3. **Intrinsic dimensionality**
   - `embedding_pca_dim_80`: number of principal components needed to explain 80% of variance.
   - `embedding_pca_spectral_entropy`: how evenly the variance is distributed across dimensions.

4. **Temporal semantic movement**
   - `embedding_centroid_drift_early_late`: semantic movement from the early period to the late period.
   - `embedding_radial_expansion_slope`: whether the subfield expands or contracts in semantic dispersion over time.
   - `embedding_recent_novelty_score`: whether recent papers are far from the historical semantic core.

The goal is not to prove that these are the only possible metrics, but to create a compact, interpretable, lower-redundancy feature set for the main thesis analysis.

---

## Outputs

Create a new output folder:

```text
outputs/analysis/reduced_interpretable_embedding_core/
```

Write at least the following outputs:

```text
outputs/analysis/reduced_interpretable_embedding_core/reduced_interpretable_core_metrics.csv
outputs/analysis/reduced_interpretable_embedding_core/reduced_interpretable_core_metrics.parquet

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

The reduced metric table should include useful identifier/control columns if present, for example:

```text
subfield_id
subfield_display_name
subfield_label_unique
subfield_label_short
field_id
field_display_name
domain_id
domain_display_name
n_available
n_used
year_min
year_max
metric_status
metric_warning_message
```

plus the 11 reduced core metrics.

---

## Heatmap Requirements

Produce both Spearman and Pearson heatmaps for the 11 metrics.

Requirements:

- Use readable labels, ideally without the `embedding_` prefix.
- Use a diverging correlation scale from -1 to 1.
- Make the figure large enough that axis labels are readable.
- Add vertical/horizontal separators or clear grouping if easy:
  - global dispersion
  - local density / hubness
  - intrinsic dimensionality
  - temporal movement
- Save PNGs at high enough DPI for thesis use.

The Spearman heatmap is the more important one for interpretation because several metrics are skewed or outlier-prone.

---

## Histogram Requirements

Create a grid of histograms for the 11 reduced core metrics:

```text
reduced_core_histograms.png
```

Requirements:

- One subplot per metric.
- Readable titles.
- Avoid plotting non-core controls.
- Use the same general style as the existing metric diagnostic plots.

---

## Distribution Diagnostics

Create a CSV with one row per reduced core metric containing, at minimum:

```text
metric
n
missing_count
missing_share
unique_count
mean
std
median
iqr
min
p01
p05
p25
p75
p95
p99
max
skewness
```

If the project already has utilities for this in `src/metric_distribution_diagnostics.py`, reuse them instead of duplicating logic.

Also add simple flags such as:

```text
near_constant
high_missing
high_outlier_risk
```

Use sensible thresholds consistent with the existing project, or define them clearly in the summary.

---

## Top Correlation Pairs

Create two CSVs:

```text
reduced_core_top_absolute_spearman_pairs.csv
reduced_core_top_absolute_pearson_pairs.csv
```

Each should contain columns similar to:

```text
metric_a
metric_b
correlation
abs_correlation
```

Sort descending by absolute correlation, excluding self-correlations.

This is important because the purpose of this step is to verify that the reduced core is less redundant than the previous 14-metric clean core.

---

## Summary Markdown

Write a concise but useful:

```text
outputs/analysis/reduced_interpretable_embedding_core/summary.md
```

It should include:

1. Purpose of this step.
2. Input table used.
3. Number of eligible subfields.
4. The 11 reduced core metrics grouped by conceptual block.
5. Why the three previous metrics were excluded:
   - `embedding_centroid_norm`: nearly inverse of centroid-distance metrics.
   - `embedding_knn_p90_distance`: highly redundant with kNN median distance.
   - `embedding_pca_top5_variance_share`: highly redundant/anti-correlated with spectral entropy and dim80.
6. Top absolute Spearman correlations among the reduced core metrics.
7. Any metrics still showing high skewness/outlier risk.
8. Interpretation note: this reduced core is intended for the main downstream analysis, while the full metric table is retained for sensitivity checks.

---

## JSON Summary

Write a machine-readable:

```text
outputs/analysis/reduced_interpretable_embedding_core/summary.json
```

Include at least:

```json
{
  "input_path": "...",
  "output_dir": "...",
  "n_rows_input": 0,
  "n_rows_eligible": 0,
  "core_metric_count": 11,
  "core_metrics": [],
  "excluded_from_clean_core_v1": [],
  "top_abs_spearman_pairs": [],
  "top_abs_pearson_pairs": [],
  "created_outputs": []
}
```

---

## Documentation Update

Add or update a short documentation file, preferably:

```text
docs/reduced_interpretable_embedding_core.md
```

or update the existing embedding metrics documentation if that is more consistent.

The documentation should explain:

- This step does not recompute embeddings or metrics.
- It selects an interpretable reduced core from the full script-12 metric table.
- It is intended to reduce redundancy before PCA, clustering, classification, or disciplinary comparison.
- The full metric table remains available for sensitivity analysis.

Do not overclaim. Do not say these are objectively the “true” metrics. Say they are selected for interpretability, distributional stability, and reduced redundancy.

---

## CLI

The script should be runnable as:

```powershell
.\.venv\Scripts\python.exe scripts\16b_build_reduced_interpretable_embedding_core.py --overwrite
```

Also support optional arguments:

```bash
--input
--output-dir
--overwrite
```

If `--overwrite` is not provided and outputs already exist, fail clearly or skip safely according to the existing project convention.

---

## Tests / Verification

Add or update tests if appropriate. At minimum, verify:

1. The reduced metric list has exactly 11 metrics.
2. All required reduced metrics exist in the current full metric table.
3. The script produces the expected output files in a temporary directory on a small sample input.
4. The correlation matrices are square 11x11 with matching row/column labels.
5. The top-pairs CSV excludes self-correlations.

Run the relevant tests. If the full test suite is not too expensive, run it. Otherwise run the focused tests and report what passed.

---

## Important Constraints

- Do not modify script 12 metric computation.
- Do not delete existing columns from `subfield_embedding_space_metrics.parquet`.
- Do not change the current full metric table unless explicitly needed.
- This is a downstream selection/diagnostic step.
- Keep the code consistent with the existing repository style.
- Keep outputs deterministic.
- Make paths configurable but use the default paths above.
- Keep the methodology conservative and defensible.

---

## Final Response Expected From You

When finished, report:

1. Files changed/created.
2. Exact command to run.
3. Output files generated.
4. Tests/verification performed.
5. Any warning about remaining correlations or distribution issues.

Do not just say it is done. Be precise.
