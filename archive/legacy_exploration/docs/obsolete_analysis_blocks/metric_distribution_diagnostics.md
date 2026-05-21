# Metric Distribution Diagnostics

This diagnostic summarizes metric distributions so low-information or fragile
features are easy to spot before interpretation.

## Command

```bash
python scripts/14_summarize_metric_distributions.py --overwrite
```

The script requires both metric tables to already exist:

```text
data/processed/subfield_morphology_metrics.parquet
data/processed/subfield_embedding_space_metrics.parquet
```

## Outputs

```text
outputs/analysis/metric_distributions/metric_distribution_summary.csv
outputs/analysis/metric_distributions/low_information_metrics.csv
outputs/analysis/metric_distributions/metric_distribution_summary.md
outputs/analysis/metric_distributions/umap_metric_histograms.png
outputs/analysis/metric_distributions/embedding_metric_histograms.png
outputs/analysis/metric_distributions/all_metric_boxplots_zscore.png
```

For embedding-space metrics, this cross-family summary now uses only the core
substantive metric list. The script 12-specific folder
`outputs/analysis/embedding_space_metric_diagnostics/` contains the broader
role-labelled embedding diagnostics, including controls and non-core retained
diagnostic columns.

## Summary Fields

For each metric, the summary includes:

```text
missing_share
n_unique
unique_share
zero_share
mean
std
min
p01
p05
p25
median
p75
p95
p99
max
iqr
is_constant
is_low_unique
is_high_missing
is_zero_dominated
recommended_review_flag
```

Default flags:

```text
is_constant: n_unique <= 1
is_low_unique: n_unique <= 5 or unique_share <= 0.05
is_high_missing: missing_share >= 0.30
is_zero_dominated: zero_share >= 0.80
```

Flags are review prompts, not automatic deletion rules. A metric may be
theoretically useful even if it needs careful handling.
