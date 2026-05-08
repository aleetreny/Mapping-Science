# Morphology-Growth Dataset

Stage 14 joins the final morphology feature layer with the annualized growth
targets. It creates a clean, audited, model-ready table at the OpenAlex
subfield level. It does not fit classifiers, regressions, or any other
prediction model; modelling belongs to Stage 15.

## Inputs

Default inputs:

```text
outputs/metrics/morphology_analysis/tables/curated_model_features.parquet
data/processed/subfield_morphology_metrics.parquet
outputs/metrics/morphology_analysis/tables/family_scores.csv
outputs/metrics/morphology_analysis/tables/metric_pca_scores.csv
data/processed/subfield_growth_targets.parquet
```

The curated morphology table supplies identifiers, taxonomy labels, 25 core
morphology metrics, and family scores. The raw morphology table supplies
diagnostic metrics such as `outlier_share_r_gt_1_5`. The growth table supplies
the Stage 13 annualized targets and publication-count controls.

The join key is only:

```text
subfield_id
```

Display names and labels are deliberately not used for joining because some
OpenAlex display names are duplicated across fields or domains.

## Targets

The joined dataset includes:

```text
annualized_log_growth
growth_above_median
domain_adjusted_annualized_log_growth
growth_above_domain_median
growth_label_global_int
growth_label_domain_int
```

The boolean labels are also stored as stable 0/1 integer columns for Stage 15.

## Feature Groups

Stage 14 writes feature-group metadata to:

```text
outputs/modeling/stage14_feature_groups.json
```

The key groups are:

```text
metadata_columns
growth_target_columns
growth_control_columns
core_morphology_metric_columns
diagnostic_morphology_metric_columns
family_score_columns
pca_score_columns
recommended_primary_feature_columns
recommended_family_feature_columns
recommended_core_metric_feature_columns
```

Recommended Stage 15 feature sets include:

```text
baseline_size_only
baseline_domain_only
baseline_size_domain
family_scores
core_morphology_metrics
combined_primary
```

Categorical columns such as `domain_display_name` remain human-readable here.
Stage 15 should decide how to encode them. Growth targets, future-count
columns, and target-like columns are explicitly excluded from predictor groups.

## Outputs

Primary joined dataset:

```text
data/processed/subfield_morphology_growth_dataset.parquet
data/processed/subfield_morphology_growth_dataset.csv
```

Audit and metadata outputs:

```text
outputs/modeling/stage14_morphology_growth_dataset_summary.json
outputs/modeling/stage14_morphology_growth_dataset_dictionary.csv
outputs/modeling/stage14_feature_groups.json
outputs/modeling/stage14_join_audit.csv
```

Exploratory tables:

```text
outputs/modeling/stage14_target_feature_correlations.csv
outputs/modeling/stage14_family_score_target_summary.csv
outputs/modeling/stage14_domain_target_summary.csv
outputs/modeling/stage14_growth_rankings.csv
outputs/modeling/stage14_family_difference_rankings.csv
```

Figures:

```text
outputs/modeling/stage14_figures/
```

These figures are descriptive only: target distributions, domain boxplots,
family-score relationships, correlation summaries, and a PCA/family-score
scatter. They should not be read as prediction performance.

## Command

Run:

```powershell
.\.venv\Scripts\python.exe scripts\14_build_morphology_growth_dataset.py --overwrite
```

With explicit inputs:

```powershell
.\.venv\Scripts\python.exe scripts\14_build_morphology_growth_dataset.py `
  --morphology-path outputs/metrics/morphology_analysis/tables/curated_model_features.parquet `
  --raw-morphology-path data/processed/subfield_morphology_metrics.parquet `
  --growth-path data/processed/subfield_growth_targets.parquet `
  --overwrite
```

## Audit Interpretation

The summary JSON reports morphology row count, growth row count, joined row
count, unmatched IDs, class balance, duplicate-name rows, feature-group sizes,
and validation checks.

The join audit should contain 240 `matched` rows in the full main analysis. Any
`missing_growth` or `missing_morphology` row indicates the Stage 14 dataset is
not safe to use for modelling.

The dictionary CSV marks each column as metadata, target, control, core
morphology metric, family score, diagnostic morphology metric, or exploratory
PCA score. Use this file and `stage14_feature_groups.json` when selecting
predictors for Stage 15.
