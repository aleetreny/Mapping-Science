# Temporary field-level morphology probe

You are working in the repo:

```text
C:\Users\Z0058EYW\Workspace\TFM
```

This is a **dirty exploratory probe**, not a production pipeline stage. The goal is to quickly test whether doing the same map/morphology/prediction workflow at the **OpenAlex field level** is more promising than at the subfield level.

Important: do not modify the production Stage 10–16 outputs unless absolutely necessary. Put all new outputs in a temporary folder at the repo root.

Use:

```text
tmp_field_level_probe/
```

Generated outputs must stay ignored by Git. If needed, add only this folder to `.gitignore`:

```gitignore
tmp_field_level_probe/
```

Do not commit PNGs, parquet files, CSV outputs, or large artifacts.

---

## Goal

Create a quick field-level version of the current pipeline:

1. Build one UMAP map per OpenAlex **field**, not per subfield.
2. Save one PNG per field with:
   - scatter panel,
   - density/KDE panel using `viridis`.
3. Compute morphology metrics for each field map.
4. Build field-level growth targets from existing field-year count data.
5. Join field morphology + field growth.
6. Re-run a lightweight predictive smoke test using the same logic as Stage 15/16.
7. Produce an honest summary of whether field-level maps look more promising.

This is exploratory. Do not over-engineer.

---

## Data to use

Use the existing analysis matrix and index:

```text
data/processed/analysis_embedding_index.parquet
embeddings/specter2_v1/analysis/main_embeddings.float16.npy
```

The index should already contain:

```text
field_id
field_display_name
domain_id
domain_display_name
subfield_id
subfield_display_name
publication_year
```

Inspect the actual column names before assuming. Fail clearly if required field/domain/year columns are missing.

Use the same morphology input window as before:

```text
2010-2019
```

Do **not** use 2020-2025 embeddings for the maps or morphology metrics.

For growth targets, use full OpenAlex grouped counts, not sampled UMAP papers. Prefer:

```text
data/interim/field_year_counts.parquet
```

using the count column:

```text
n_works_article_preprint
```

If this file is unavailable, fail with a clear message. Do not call the OpenAlex API unless explicitly necessary.

---

## Important methodological note

At field level there will only be around 25–30 units. This is **too small for reliable supervised prediction**.

So the predictive part should be framed as a **smoke test / sanity check only**, not evidence.

The output summary must explicitly say:

```text
Field-level N is small; predictive scores are exploratory and unstable.
```

Still run the smoke tests because we want to see whether there is any obvious signal.

---

## Temporary folder structure

Create something like:

```text
tmp_field_level_probe/
  maps/
    coordinates/
      <field_id>__<safe_field_name>.parquet
    figures/
      <field_id>__<safe_field_name>.png
    field_umap_manifest.parquet
    field_umap_summary.json

  metrics/
    field_morphology_metrics.parquet
    field_morphology_metrics.csv
    field_morphology_metrics_summary.json
    field_morphology_metrics_dictionary.csv

  growth/
    field_growth_targets.parquet
    field_growth_targets.csv
    field_growth_targets_summary.json
    field_year_counts_panel.parquet
    field_year_counts_panel.csv

  modeling/
    field_morphology_growth_dataset.parquet
    field_morphology_growth_dataset.csv
    field_modeling_summary.json
    field_prediction_smoke_results.csv
    field_target_definitions.csv
    figures/
      target_distributions.png
      family_scores_by_target.png
      model_comparison.png
      morphology_pca_or_family_scatter.png

  README_field_probe.md
```

---

## Part A — Field-level UMAP maps

Add a temporary script, for example:

```text
tmp_field_level_probe/run_field_level_probe.py
```

or, if cleaner:

```text
scripts/tmp_field_level_probe.py
```

But outputs must go to `tmp_field_level_probe/`.

The script should:

1. Load `analysis_embedding_index.parquet`.
2. Filter to `2010 <= publication_year <= 2019`.
3. Group rows by `field_id`.
4. For each field:
   - select corresponding embedding rows from the memory-mapped matrix,
   - convert only that subset to `float32`,
   - fit one UMAP per field,
   - save coordinates,
   - save a two-panel PNG: scatter + density.

Reuse existing Stage 10 plotting logic if convenient.

Suggested UMAP defaults:

```text
n_neighbors = 30
min_dist = 0.05
metric = cosine
random_state = 42
```

Because fields may have very different numbers of papers, do **not** force equal sample sizes by default. But include safety options:

```text
--max-papers-per-field 30000
--min-papers 500
--limit-fields
--field-id
--overwrite
```

Default behavior for this probe:

```text
Use all available 2010-2019 papers per field unless above max-papers-per-field.
If a field exceeds max-papers-per-field, sample deterministically.
```

The manifest must record:

```text
field_id
field_display_name
domain_id
domain_display_name
n_available
n_used
sampled
year_min
year_max
coordinate_path
figure_path
status
error_message
```

PNG titles should be unambiguous:

```text
<field_id> | <domain_display_name> / <field_display_name>
n=<n_used>, years 2010-2019
```

---

## Part B — Field-level morphology metrics

Compute the same 25 core morphology metrics as the current subfield layer wherever possible.

Reuse `src/morphology_metrics.py` as much as possible. If the current functions assume subfield labels, wrap them or generalize lightly, but avoid breaking Stage 11.

The field-level metric table should include:

```text
field_id
field_display_name
domain_id
domain_display_name
field_label_unique
n_points
year_min
year_max
metric_status
metric_error_message
```

Then include all core metrics:

```text
radial_tail_index
radial_iqr_index
knn_median_distance
knn_distance_cv
density_entropy
peak_dominance
effective_area_90
core_periphery_ratio
density_peak_count
peak_mass_entropy
dense_component_count
largest_component_mass_share
component_separation_index
mst_gap_index
anisotropy_ratio
support_solidity
boundary_complexity
max_normalized_radius
centroid_drift_early_late
annual_centroid_path_length
directionality_ratio
radial_expansion_slope
annual_centroid_step_cv
radial_expansion_r2
density_entropy_slope_by_year
```

Also compute the same diagnostics if easy:

```text
density_gini
effective_area_50
support_circularity
hole_count
outlier_share_r_gt_1
outlier_share_r_gt_1_5
outlier_share_outside_density_extent
density_entropy_slope_r2
```

Also compute field-level family scores:

```text
diffuseness_score
concentration_score
fragmentation_score
elongation_score
temporal_dynamism_score
directional_change_score
expansion_score
diversification_score
```

Use the same definitions as Stage 12 so results are comparable.

---

## Part C — Field-level growth targets

Use:

```text
data/interim/field_year_counts.parquet
```

Build field-level growth targets using the same simple annualized definition as Stage 13:

```text
annual_rate_2010_2019 = papers_2010_2019 / 10
annual_rate_2020_2025 = papers_2020_2025 / 6
annualized_log_growth = log(1 + annual_rate_2020_2025) - log(1 + annual_rate_2010_2019)
```

Create at least:

```text
growth_above_median
domain_adjusted_annualized_log_growth
growth_above_domain_median
growth_top_25pct
growth_top_20pct
growth_top_10pct
domain_adjusted_growth_top_25pct
domain_adjusted_growth_top_20pct
domain_adjusted_growth_top_10pct
```

But be careful: at field level, some labels may have too few positives. If a target has too few positives/negatives, skip it and record why.

Use strict checks:

```text
one row per field_id
no missing 2010-2025 field-year counts unless explicitly handled
no zero input-count fields unless documented
no zero target-count fields unless documented
```

---

## Part D — Join field morphology + growth

Join strictly by:

```text
field_id
```

Do not join by field name.

Create:

```text
tmp_field_level_probe/modeling/field_morphology_growth_dataset.parquet
tmp_field_level_probe/modeling/field_morphology_growth_dataset.csv
```

The joined table should have:

```text
one row per field
field metadata
growth targets
growth controls
core morphology metrics
diagnostic metrics
family scores
```

Write an audit file with:

```text
n_field_morphology_rows
n_field_growth_rows
n_joined_rows
n_unmatched_morphology_rows
n_unmatched_growth_rows
missing_metric_values
missing_growth_values
```

---

## Part E — Predictive smoke tests

Run a lightweight model comparison only. This is exploratory because N is small.

Evaluate these target types if eligible:

```text
domain_display_name
growth_above_median
growth_above_domain_median
growth_top_25pct
growth_top_20pct
growth_top_10pct
domain_adjusted_growth_top_25pct
domain_adjusted_growth_top_20pct
domain_adjusted_growth_top_10pct
```

Skip targets with too few positive/negative examples.

Suggested minimum:

```text
binary target: at least 5 positives and 5 negatives
multiclass target: at least 4 examples per class
```

Use simple models only:

```text
logistic_l2
logistic_l1_or_elasticnet if stable
random_forest_or_extra_trees
ridge_regression for continuous annualized_log_growth only if easy
```

Feature sets:

```text
baseline_size_only:
  log_papers_2010_2019

baseline_domain_only:
  domain_display_name
  # not allowed when target is domain_display_name

baseline_size_domain:
  log_papers_2010_2019
  domain_display_name
  # not allowed when target is domain_display_name

family_scores:
  8 field-level family scores

core_morphology_metrics:
  25 field-level core metrics

combined_primary:
  log_papers_2010_2019
  domain_display_name
  8 family scores
  # remove domain_display_name when target is domain_display_name
```

Leakage rules:

- Never use target columns as predictors.
- Never use `papers_2020_2025`, `annual_rate_2020_2025`, ranks, percentiles, or any future-count columns as predictors.
- For domain prediction, remove all domain fields from features.
- For field prediction, do not include field labels or IDs.
- Use only morphology + allowed baseline controls.

Validation:

Because N is tiny, use one of:

```text
LeaveOneOut CV
or RepeatedStratifiedKFold with safe n_splits based on smallest class
```

Do not force 5 folds if a class is too small.

Metrics:

For binary targets:

```text
ROC AUC
average precision
balanced accuracy
F1
precision
recall
Brier score if probabilities available
```

For multiclass domain:

```text
macro F1
balanced accuracy
macro ROC AUC OvR if possible
confusion matrix
```

Also report:

```text
n_samples
n_classes
class_counts
cv_strategy_used
```

---

## Part F — Visual outputs

Generate easy-to-read figures:

1. Field-level map gallery/contact sheet:
   - small thumbnails of all field PNGs if feasible.
2. Target distributions.
3. Growth by domain boxplot.
4. Family scores by target label.
5. Model comparison barplot.
6. Field-level PCA or UMAP of morphology metrics colored by:
   - domain,
   - top-growth label,
   - archetype if clustering is included.
7. Top/bottom fields by:
   - annualized growth,
   - domain-adjusted growth,
   - fragmentation,
   - concentration,
   - diffuseness,
   - temporal dynamism.

Keep figures simple and readable.

---

## Part G — Summary report

Write:

```text
tmp_field_level_probe/README_field_probe.md
tmp_field_level_probe/modeling/field_level_probe_summary.json
```

The markdown report should answer:

1. How many fields were mapped?
2. How many papers were used per field?
3. Which fields produced visually interesting maps?
4. Which field-level morphology metrics vary the most?
5. Which prediction targets were eligible?
6. Which targets were skipped and why?
7. Which model/feature set performed best for each target?
8. Did morphology-only features beat simple baselines?
9. Are field-level maps more promising than subfield-level maps?
10. What should we try next?

The report must include an honest caveat:

```text
This is a temporary field-level smoke test with a small number of field units. Results are exploratory and should not be interpreted as final evidence.
```

---

## Acceptance criteria

This task is complete when:

1. All outputs are written under `tmp_field_level_probe/`.
2. No production Stage 10–16 outputs are overwritten.
3. One PNG per eligible field is generated.
4. Field-level coordinate parquets are generated.
5. Field-level morphology metrics are generated.
6. Field-level growth targets are generated.
7. Field-level morphology-growth dataset is generated.
8. Predictive smoke tests run for all eligible targets.
9. Skipped targets are explicitly recorded.
10. A readable markdown summary is created.
11. Generated artifacts are ignored by Git.
12. The final response reports:
    - files changed,
    - exact commands run,
    - number of fields mapped,
    - output folder,
    - best targets/models,
    - whether morphology-only features beat baselines,
    - any warnings or assumptions.

Remember: this is intentionally quick and dirty. Prefer a clear, working exploratory probe over a perfect production refactor.
