# Subfield Growth Targets

Stage 13 builds the standalone outcome table for later prediction work. It
does not train models and does not join targets back into the morphology
analysis stage.

## Design

The unit of analysis is the OpenAlex subfield identified by `subfield_id`.
The target table is restricted to the same 240 main-analysis subfields used by
the morphology layer.

The windows are explicit and inclusive:

```text
Morphology input window: 2010-2019
Growth target window:    2020-2025
```

Raw totals are not directly comparable because the input window has 10 years
and the target window has 6 years. Stage 13 therefore computes annual
publication rates first:

```text
annual_rate_2010_2019 = papers_2010_2019 / 10
annual_rate_2020_2025 = papers_2020_2025 / 6
```

The main continuous target is:

```text
annualized_log_growth =
  log1p(annual_rate_2020_2025) - log1p(annual_rate_2010_2019)
```

The main binary target is:

```text
growth_above_median = annualized_log_growth > median(annualized_log_growth)
```

The threshold is strict. Values exactly equal to the global median are labelled
`False` and counted as ties in the summary JSON.

Stage 13 also computes a domain-adjusted target:

```text
domain_adjusted_annualized_log_growth =
  annualized_log_growth - median(annualized_log_growth within domain_id)

growth_above_domain_median =
  domain_adjusted_annualized_log_growth > 0
```

This is useful because publication growth levels differ across broad OpenAlex
domains. The domain-adjusted version asks whether a subfield grew faster than
the typical subfield in its own domain.

## Count Source

Growth counts must come from full OpenAlex grouped yearly counts, not from the
sampled 3,000-paper-per-subfield morphology corpus. The default local source is:

```text
data/interim/subfield_year_counts.parquet
```

That file is produced by:

```powershell
.\.venv\Scripts\python.exe scripts\01_build_counts.py
```

The default count column is `n_works_article_preprint`, which counts full
OpenAlex article/preprint production after excluding retracted and paratext
works. It does not require English text or abstract availability.

If the default count table is missing, the Stage 13 script can read a supplied
count table, use the DuckDB `subfield_year_counts` table, reuse a raw OpenAlex
count cache, or fetch sparse OpenAlex group-by counts with
`--fetch-counts-from-openalex`.

## Command

Full local run:

```powershell
.\.venv\Scripts\python.exe scripts\13_build_subfield_growth_targets.py --overwrite
```

With explicit inputs:

```powershell
.\.venv\Scripts\python.exe scripts\13_build_subfield_growth_targets.py `
  --subfields-path data/processed/subfield_morphology_metrics.parquet `
  --counts-path data/interim/subfield_year_counts.parquet `
  --overwrite
```

If fetching counts directly from OpenAlex is needed:

```powershell
.\.venv\Scripts\python.exe scripts\13_build_subfield_growth_targets.py `
  --fetch-counts-from-openalex `
  --raw-counts-cache data/interim/openalex_subfield_year_counts_2010_2025.parquet `
  --overwrite
```

## Outputs

Primary target table:

```text
data/processed/subfield_growth_targets.parquet
data/processed/subfield_growth_targets.csv
```

Balanced yearly count panel:

```text
outputs/growth/subfield_year_counts_panel.parquet
outputs/growth/subfield_year_counts_panel.csv
```

Diagnostics:

```text
outputs/growth/subfield_growth_targets_summary.json
outputs/growth/growth_rankings.csv
outputs/growth/domain_growth_summary.csv
outputs/growth/figures/
```

The figures include histograms, annual-rate scatter plots, domain boxplots,
top/bottom barplots, yearly count examples, and size-vs-growth checks.

## Extreme Count Validation

Stage 13b is a small sanity check for unusually large or influential growth
counts. It does not modify the Stage 13 target table. Instead, it compares the
local `data/interim/subfield_year_counts.parquet` value in
`n_works_article_preprint` with direct OpenAlex `/works` counts for selected
subfield-year pairs.

Run:

```powershell
.\.venv\Scripts\python.exe scripts\13b_validate_growth_count_extremes.py --overwrite
```

The direct OpenAlex query uses the same count filters as Stage 01:

```text
primary_topic.subfield.id:<subfield_id>
publication_year:<year>
type:article|preprint
is_retracted:false
is_paratext:false
```

Default checks include `2202` Aerospace Engineering, `2200` General
Engineering, `3500` General Dentistry, `2611` Modeling and Simulation, `2725`
Infectious Diseases, and `3307` Human Factors and Ergonomics for 2010, 2019,
2020, and 2025.

Outputs:

```text
outputs/growth/count_validation/growth_count_extremes_validation.csv
outputs/growth/count_validation/growth_count_extremes_validation.json
```

The CSV records local counts, API counts, differences, exact-match status,
query filters, timestamps, and any API error message. The JSON summary counts
matches, mismatches, missing local rows, API failures, and suspicious cases.

## Interpretation Caveats

The target is an OpenAlex snapshot. Late 2025 records may still reflect
indexing lag depending on when counts were retrieved, so the summary JSON
records source timestamps and coverage warnings.

Missing subfield-year count rows are only filled with zero when the source is
known to be sparse, such as an OpenAlex group-by fetch. Complete local count
tables are expected to contain every subfield-year combination.

Duplicate OpenAlex display names are handled by joining on `subfield_id` and
using `subfield_label_unique` or `subfield_label_short` in human-facing tables
and plots.
