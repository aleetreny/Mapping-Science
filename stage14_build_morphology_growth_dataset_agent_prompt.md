# Stage 14 Agent Task: Build the morphology–growth modelling dataset

You are working in the repository:

```text
aleetreny/TFM
```

Implement **Stage 14** of the TFM pipeline: join the final morphology feature layer with the validated growth targets.

The purpose of this stage is to create a clean, audited, model-ready tabular dataset at the **OpenAlex subfield** level. Do not train prediction models yet. Stage 15 will handle modelling.

You have full freedom to inspect the current repo structure and adapt details where sensible. The constraints below define the required methodological behaviour and outputs.

---

## Context

Previous stages already exist:

### Morphology side

Stage 10 generated per-subfield UMAP maps.

Stage 11 computed morphology metrics.

Stage 12 analyzed morphology metrics and generated curated feature tables.

Relevant likely files:

```text
data/processed/subfield_morphology_metrics.parquet
data/processed/subfield_morphology_metrics.csv

outputs/metrics/morphology_analysis/curated_model_features.parquet
outputs/metrics/morphology_analysis/curated_model_features.csv
outputs/metrics/morphology_analysis/family_scores.csv
outputs/metrics/morphology_analysis/metric_pca_scores.csv
outputs/metrics/morphology_analysis/morphology_analysis_summary.json
outputs/metrics/duplicate_subfield_names_report.csv
```

The final morphology layer has:

- 240 subfields.
- 25 core morphology metrics.
- diagnostic metrics.
- family scores.
- unambiguous labels:
  - `subfield_label_unique`
  - `subfield_label_short`
  - `subfield_display_name_is_duplicated`

Important: morphology is computed only from the input window:

```text
2010-2019
```

### Growth side

Stage 13 generated validated annualized growth targets.

Relevant likely files:

```text
data/processed/subfield_growth_targets.parquet
data/processed/subfield_growth_targets.csv

outputs/growth/subfield_growth_targets_summary.json
outputs/growth/growth_rankings.csv
outputs/growth/domain_growth_summary.csv
outputs/growth/subfield_year_counts_panel.parquet
outputs/growth/count_validation/growth_count_extremes_validation.json
```

The main growth target is:

```text
annualized_log_growth =
log(1 + annual_rate_2020_2025) - log(1 + annual_rate_2010_2019)
```

Main global classification label:

```text
growth_above_median
```

Domain-adjusted continuous target:

```text
domain_adjusted_annualized_log_growth
```

Domain-adjusted classification label:

```text
growth_above_domain_median
```

The target counts are from full OpenAlex grouped yearly counts, not the sampled morphology corpus.

---

## Main goal

Create a single joined dataset with one row per subfield:

```text
data/processed/subfield_morphology_growth_dataset.parquet
data/processed/subfield_morphology_growth_dataset.csv
```

This dataset must include:

1. Stable identifiers and labels.
2. Domain/field/subfield metadata.
3. Growth targets from Stage 13.
4. Morphology core metrics from Stage 11/12.
5. Family scores from Stage 12.
6. Optional diagnostic metrics and PCA scores, clearly marked as diagnostic/exploratory.
7. Feature-group metadata so Stage 15 can select feature sets safely.

No prediction model should be fitted in Stage 14.

---

## Critical methodological requirements

### 1. Join only by `subfield_id`

Never join by:

```text
subfield_display_name
subfield_label_unique
subfield_label_short
```

The display names are not unique. This is known and expected.

The join must be one-to-one:

```text
240 morphology rows
240 growth rows
240 joined rows
```

If the counts differ, fail loudly with a clear error.

### 2. Avoid target leakage

The joined dataset must make clear that:

```text
morphology features use 2010-2019 only
growth targets use 2020-2025, annualized against 2010-2019
```

Do not create features from 2020-2025.

Do not use growth columns as predictors.

Do not derive any new morphology feature from growth outcomes.

### 3. Preserve feature groups

Stage 15 will need different modelling feature sets. Create explicit feature groups and save them as JSON.

At minimum:

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

Suggested modelling sets:

```text
baseline_size_only:
  - log_papers_2010_2019 or log_annual_rate_2010_2019

baseline_domain_only:
  - domain_display_name one-hot later in Stage 15, not encoded here unless useful

baseline_size_domain:
  - log_papers_2010_2019
  - domain_display_name

family_scores:
  - diffuseness_score
  - concentration_score
  - fragmentation_score
  - elongation_score
  - temporal_dynamism_score
  - directional_change_score
  - expansion_score
  - diversification_score

core_morphology_metrics:
  all 25 core morphology metrics from Stage 12

combined_primary:
  - log_papers_2010_2019
  - family scores
  - domain_display_name as categorical metadata for Stage 15
```

Do not one-hot encode everything unless you decide it is useful to also save a separate fully numeric modelling matrix. The primary joined dataset should remain human-readable.

### 4. Add simple useful derived controls

Add these derived columns if not already present:

```text
log_papers_2010_2019 = log1p(papers_2010_2019)
log_papers_2020_2025 = log1p(papers_2020_2025)
log_annual_rate_2010_2019 = log1p(annual_rate_2010_2019)
log_annual_rate_2020_2025 = log1p(annual_rate_2020_2025)
```

These are controls, not morphology metrics.

Optional but useful:

```text
growth_label_global_int = 0/1 version of growth_above_median
growth_label_domain_int = 0/1 version of growth_above_domain_median
```

---

## Implementation request

Add a new script:

```text
scripts/14_build_morphology_growth_dataset.py
```

Optionally add a supporting module if it keeps the code clean:

```text
src/morphology_growth_dataset.py
```

The script should support CLI arguments like:

```text
--morphology-path outputs/metrics/morphology_analysis/curated_model_features.parquet
--raw-morphology-path data/processed/subfield_morphology_metrics.parquet
--family-scores-path outputs/metrics/morphology_analysis/family_scores.csv
--pca-scores-path outputs/metrics/morphology_analysis/metric_pca_scores.csv
--growth-path data/processed/subfield_growth_targets.parquet
--output-path data/processed/subfield_morphology_growth_dataset.parquet
--output-csv data/processed/subfield_morphology_growth_dataset.csv
--summary-path outputs/modeling/stage14_morphology_growth_dataset_summary.json
--dictionary-path outputs/modeling/stage14_morphology_growth_dataset_dictionary.csv
--feature-groups-path outputs/modeling/stage14_feature_groups.json
--figures-dir outputs/modeling/stage14_figures
--expected-rows 240
--overwrite
```

You may adjust defaults if the repo has a better output convention, but keep the stage number clear.

---

## Output files

Required:

```text
data/processed/subfield_morphology_growth_dataset.parquet
data/processed/subfield_morphology_growth_dataset.csv

outputs/modeling/stage14_morphology_growth_dataset_summary.json
outputs/modeling/stage14_morphology_growth_dataset_dictionary.csv
outputs/modeling/stage14_feature_groups.json
```

Recommended exploratory outputs:

```text
outputs/modeling/stage14_join_audit.csv
outputs/modeling/stage14_target_feature_correlations.csv
outputs/modeling/stage14_family_score_target_summary.csv
outputs/modeling/stage14_domain_target_summary.csv
```

Recommended figures:

```text
outputs/modeling/stage14_figures/target_distribution.png
outputs/modeling/stage14_figures/domain_adjusted_target_distribution.png
outputs/modeling/stage14_figures/growth_by_domain_boxplot.png
outputs/modeling/stage14_figures/family_scores_vs_growth.png
outputs/modeling/stage14_figures/family_scores_by_growth_label.png
outputs/modeling/stage14_figures/core_metric_target_correlation_barplot.png
outputs/modeling/stage14_figures/morphology_growth_pca_or_family_scatter.png
```

Keep figures simple, readable and directly useful. Avoid decorative plots.

---

## Audits and validations

The script must validate and report:

1. Morphology row count.
2. Growth target row count.
3. Joined row count.
4. Missing `subfield_id` in either table.
5. Duplicate `subfield_id` in either table.
6. Unmatched morphology rows.
7. Unmatched growth rows.
8. Missing values in core morphology metrics.
9. Missing values in growth target columns.
10. Class balance for:
   - `growth_above_median`
   - `growth_above_domain_median`
11. Number of duplicated display-name rows.
12. Whether the joined file contains all expected family scores.
13. Whether the joined file contains `density_entropy_slope_by_year`.
14. Whether `outlier_share_r_gt_1_5` is present only as diagnostic, not as a core feature if feature groups are created.
15. Whether any target-like columns accidentally appear in predictor groups.

If validation fails, raise clear errors rather than silently continuing.

---

## Analysis outputs for intuition

Stage 14 is not modelling, but it should help us understand the dataset before Stage 15.

Create intuitive summaries such as:

### Target distribution

- histogram of `annualized_log_growth`;
- histogram of `domain_adjusted_annualized_log_growth`;
- counts of high/low labels.

### Domain-level target summaries

For each domain:

```text
n_subfields
mean annualized_log_growth
median annualized_log_growth
share growth_above_median
mean domain_adjusted_annualized_log_growth
share growth_above_domain_median
```

### Family score vs growth summaries

For each family score:

```text
mean score among high-growth subfields
mean score among low-growth subfields
difference
correlation with annualized_log_growth
correlation with domain_adjusted_annualized_log_growth
```

### Correlation tables

Compute Pearson and Spearman correlations between:

- core morphology metrics and `annualized_log_growth`;
- core morphology metrics and `domain_adjusted_annualized_log_growth`;
- family scores and both targets.

This is exploratory only. Do not claim prediction performance.

### Rankings

Create top/bottom tables for:

- annualized growth;
- domain-adjusted growth;
- largest positive morphology-family differences between high and low growth.

---

## Data dictionary

Create a dictionary CSV with:

```text
column_name
column_group
description
is_predictor_candidate
is_target
is_control
is_diagnostic
```

Make it good enough that a reader can understand the final joined table.

---

## Tests

Add tests using synthetic data. Do not depend on full repo data.

Test at least:

1. One-to-one join by `subfield_id`.
2. Duplicate subfield IDs raise errors.
3. Missing morphology rows raise errors.
4. Missing growth rows raise errors.
5. Derived log controls are computed correctly.
6. Feature groups do not contain target columns.
7. Boolean labels are converted to stable 0/1 columns.
8. Duplicate display names do not collapse rows.
9. The CLI runs on tiny synthetic morphology/growth files.
10. Expected output files are written.

Run:

```powershell
.\.venv\Scripts\python.exe -m py_compile ...
.\.venv\Scripts\python.exe -m pytest tests/test_morphology_growth_dataset.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

---

## Documentation

Add or update:

```text
docs/morphology_growth_dataset.md
README.md
```

The documentation should explain:

- what Stage 14 does;
- inputs;
- outputs;
- why the join is only by `subfield_id`;
- why this stage does not train prediction models;
- which targets are available;
- which feature groups are recommended for Stage 15;
- how to run the script;
- how to interpret the audit outputs.

---

## Git hygiene

Generated outputs must stay ignored by Git.

Do not commit:

```text
data/processed/subfield_morphology_growth_dataset.parquet
data/processed/subfield_morphology_growth_dataset.csv
outputs/modeling/
```

If `.gitignore` already ignores these, do not change it unnecessarily.

Do not commit secrets, `.env`, API keys, local absolute paths, or generated large artifacts.

---

## Suggested run commands

After implementation, run:

```powershell
.\.venv\Scripts\python.exe -m py_compile src\morphology_growth_dataset.py scripts\14_build_morphology_growth_dataset.py

.\.venv\Scripts\python.exe -m pytest tests\test_morphology_growth_dataset.py -q

.\.venv\Scripts\python.exe -m pytest -q

.\.venv\Scripts\python.exe scripts\14_build_morphology_growth_dataset.py --overwrite
```

Then report:

```text
Files changed
Commands run
Test results
Number of morphology rows
Number of growth rows
Number of joined rows
Class balance for both labels
Feature groups created
Any warnings or assumptions
Top exploratory findings from Stage 14 summaries
```

---

## Acceptance criteria

Stage 14 is complete only if:

1. A new Stage 14 script exists.
2. It creates a joined morphology-growth dataset with exactly 240 rows.
3. The join is strictly by `subfield_id`.
4. It validates one-to-one row matching.
5. It includes the global annualized log growth target.
6. It includes the global median classification label.
7. It includes the domain-adjusted annualized log growth target.
8. It includes the domain median classification label.
9. It includes morphology core metrics and family scores.
10. It saves feature-group metadata for Stage 15.
11. It creates human-readable audit tables and summary JSON.
12. It creates simple exploratory figures.
13. It does not train prediction models.
14. Tests are added and pass.
15. Docs are updated.
16. Generated outputs are not committed.

Remember: be practical and use your judgement. The goal is not to over-engineer; the goal is to create a clean, trusted, model-ready dataset and enough intuitive summaries to understand the morphology-growth relationship before prediction.
