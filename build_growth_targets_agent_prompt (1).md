# Task: Build annualized growth targets for subfield-level prediction

You are working in the repository:

```text
aleetreny/TFM
```

Implement the next pipeline stage for the TFM: build simple, transparent, and robust **growth targets** for the 240 OpenAlex main-analysis subfields.

This stage comes **after** the morphology feature layer and **before** any prediction model. Do not train prediction models here.

---

## Goal

Create a reproducible script that builds a subfield-level growth target table using full OpenAlex yearly counts, not the sampled embedding corpus.

The thesis design is:

```text
Morphology input window: 2010-2019
Growth target window:    2020-2025
Unit of analysis:        OpenAlex subfield
```

The main growth metric must correct for the different number of years in the two windows.

Therefore, do **not** compare raw totals directly.

Use annualized publication rates:

```text
annual_rate_2010_2019 = papers_2010_2019 / 10
annual_rate_2020_2025 = papers_2020_2025 / 6
```

Then compute:

```text
annualized_log_growth = log(1 + annual_rate_2020_2025) - log(1 + annual_rate_2010_2019)
```

This should be the main continuous growth target.

Also compute:

```text
growth_above_median = annualized_log_growth > median(annualized_log_growth across the 240 main-analysis subfields)
```

And a domain-adjusted target:

```text
domain_adjusted_annualized_log_growth = annualized_log_growth - median(annualized_log_growth within the same domain)

growth_above_domain_median = domain_adjusted_annualized_log_growth > 0
```

These are the two central targets we need:

1. global annualized log growth, with a global median label;
2. domain-adjusted annualized log growth, with a within-domain median label.

---

## Important methodological constraints

### 1. Do not use the sampled 3,000-paper-per-subfield corpus for counts

The morphology maps were based on a sampled/capped corpus. That sample is **not** valid for counting real growth.

Do **not** count papers from:

```text
outputs/maps/per_subfield_umap/coordinates/
embeddings/specter2_v1/analysis/main_embeddings.float16.npy
sampled embedding subsets
```

Those files are for morphology only.

Growth counts must come from a full-count source, such as:

- existing OpenAlex count tables in the repo, if present;
- a full local database/table with subfield/year counts, if present;
- or a new OpenAlex count-fetching step if no full count table exists.

Inspect the repo before choosing the source. Reuse existing count/downloader utilities if they already exist.

### 2. Use the same 240 main-analysis subfields

Restrict the target table to the exact same main-analysis unit set used by the morphology layer.

Use one of these sources to identify the 240 subfields, in order of preference:

1. `data/processed/subfield_morphology_metrics.parquet`, if present;
2. `outputs/metrics/morphology_analysis/curated_model_features.parquet`, if present;
3. `data/processed/analysis_embedding_index.parquet`, filtering `main_analysis_eligible_2500 == True` and taking unique subfields.

The final growth target table must have exactly one row per main-analysis subfield.

Join and identify rows using:

```text
subfield_id
```

Never merge using only `subfield_display_name`, because some display names are duplicated across fields/domains.

Keep these label/context columns when available:

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
```

If label columns are missing, reuse `src/subfield_labels.py` to create them.

### 3. Year windows must be explicit and inclusive

Use:

```text
input_year_min = 2010
input_year_max = 2019
input_years = 10

target_year_min = 2020
target_year_max = 2025
target_years = 6
```

Do not silently change these windows.

Expose CLI arguments for flexibility, but defaults must be exactly the above.

### 4. Treat missing yearly count rows carefully

If a subfield-year count is missing from the count table, treat it as zero **only if** the count source is known to return sparse rows where missing combinations mean zero.

If the count source is not known to be sparse, fail clearly or document the assumption in the summary.

The final script should create a complete subfield-year panel for all 240 subfields and all years 2010-2025.

### 5. Do not build prediction models here

This stage only builds and analyzes targets.

Do not fit classifiers, regressions, random forests, XGBoost, neural networks, or any predictive model.

---

## Required scripts

Create a new script:

```text
scripts/13_build_subfield_growth_targets.py
```

Optionally create a supporting module if needed:

```text
src/growth_targets.py
```

The script should:

1. Load or fetch full yearly publication counts by `subfield_id` and `publication_year` for 2010-2025.
2. Restrict to the 240 main-analysis subfields.
3. Build a complete balanced subfield-year panel.
4. Aggregate counts over 2010-2019 and 2020-2025.
5. Compute annualized rates.
6. Compute global annualized log growth.
7. Compute global median label.
8. Compute domain-adjusted annualized log growth.
9. Compute within-domain median label.
10. Save machine-readable target tables.
11. Save intuitive diagnostics, rankings, and plots.

---

## Input discovery logic

The agent should inspect the repo and decide the safest source of count data.

Potential existing inputs may include files or tables created by earlier stages, such as:

```text
data/raw/
data/interim/
data/processed/
warehouse/*.duckdb
scripts/01_build_counts.py
scripts/02_build_corpus_plan.py
config.yaml
```

The script must not assume a specific count file exists without checking.

If a full count table already exists, use it.

If no full count table exists, implement a reusable count-fetching path that queries OpenAlex counts by subfield and year. The fetching code should:

- use existing OpenAlex utilities if present;
- respect `.env` settings and configured polite email if already used in the repo;
- cache raw/intermediate count results;
- be restartable;
- avoid downloading full work records when only counts are needed;
- include rate limiting/retry logic if the repo already uses it.

The preferred count shape is:

```text
subfield_id
subfield_display_name
field_id
field_display_name
domain_id
domain_display_name
publication_year
works_count
count_source
retrieved_at
```

If OpenAlex returns concept URLs or IDs in a different format, normalize them to match the `subfield_id` format used by the morphology table. Check the actual repo convention before transforming IDs.

---

## Output files

Write outputs under:

```text
data/processed/subfield_growth_targets.parquet
data/processed/subfield_growth_targets.csv
outputs/growth/subfield_growth_targets_summary.json
outputs/growth/subfield_year_counts_panel.parquet
outputs/growth/subfield_year_counts_panel.csv
outputs/growth/growth_rankings.csv
outputs/growth/domain_growth_summary.csv
outputs/growth/figures/
```

Generated outputs must remain ignored by Git.

Update `.gitignore` if needed:

```text
outputs/growth/
data/processed/subfield_growth_targets.parquet
data/processed/subfield_growth_targets.csv
```

---

## Final target table columns

The main target table must include at least:

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

papers_2010_2019
papers_2020_2025
annual_rate_2010_2019
annual_rate_2020_2025

annualized_growth_abs
annualized_growth_ratio
annualized_growth_pct
annualized_log_growth

global_growth_median
growth_above_median
growth_rank
growth_percentile

domain_growth_median
domain_adjusted_annualized_log_growth
growth_above_domain_median
domain_growth_rank
domain_growth_percentile

input_year_min
input_year_max
target_year_min
target_year_max
input_years
target_years

n_years_with_counts_2010_2019
n_years_with_counts_2020_2025
has_zero_input_count
has_zero_target_count
count_source
```

Optional but useful:

```text
papers_by_year_2010
papers_by_year_2011
...
papers_by_year_2025
```

or keep yearly counts only in the panel file.

---

## Exact metric definitions

Let:

```text
x = papers_2010_2019
y = papers_2020_2025
rx = x / 10
ry = y / 6
```

Compute:

```text
annualized_growth_abs = ry - rx
annualized_growth_ratio = (ry + 1) / (rx + 1)
annualized_growth_pct = annualized_growth_ratio - 1
annualized_log_growth = log1p(ry) - log1p(rx)
```

Use `np.log1p`, not `np.log(1 + value)` manually.

Global label:

```text
global_growth_median = median(annualized_log_growth over the 240 subfields)
growth_above_median = annualized_log_growth > global_growth_median
```

Use a strict `>` threshold. If a value is exactly equal to the median, label it as `False` and count/report ties in the summary.

Domain-adjusted metric:

```text
domain_growth_median = median(annualized_log_growth within domain_id)
domain_adjusted_annualized_log_growth = annualized_log_growth - domain_growth_median
growth_above_domain_median = domain_adjusted_annualized_log_growth > 0
```

Again use strict `>` and report exact ties.

Percentiles:

```text
growth_percentile = percentile rank of annualized_log_growth across all subfields

domain_growth_percentile = percentile rank of annualized_log_growth within the same domain
```

Use a deterministic pandas ranking method and document it. Suggested:

```python
rank(pct=True, method="average")
```

Growth ranks:

```text
growth_rank = rank 1 for highest annualized_log_growth globally
domain_growth_rank = rank 1 for highest annualized_log_growth within domain
```

Use deterministic tie handling.

---

## Diagnostics and plots

Create intuitive outputs for human inspection. The goal is to immediately understand what the target is doing.

### Tables

Create:

```text
outputs/growth/growth_rankings.csv
```

Include:

- top 20 by `annualized_log_growth`;
- bottom 20 by `annualized_log_growth`;
- top 20 by `domain_adjusted_annualized_log_growth`;
- bottom 20 by `domain_adjusted_annualized_log_growth`;
- top and bottom within each domain if feasible.

Create:

```text
outputs/growth/domain_growth_summary.csv
```

Include, by domain:

```text
domain_id
domain_display_name
n_subfields
total_papers_2010_2019
total_papers_2020_2025
mean_annual_rate_2010_2019
mean_annual_rate_2020_2025
median_annualized_log_growth
mean_annualized_log_growth
std_annualized_log_growth
share_growth_above_global_median
```

### Figures

Create these figures under:

```text
outputs/growth/figures/
```

Use matplotlib. Keep plots simple, readable, and thesis-friendly.

Required figures:

1. `annualized_log_growth_histogram.png`
   - Histogram of `annualized_log_growth`.
   - Vertical line for global median.
   - Optional line for zero growth.

2. `annual_rates_scatter.png`
   - x-axis: `annual_rate_2010_2019`.
   - y-axis: `annual_rate_2020_2025`.
   - Use log scale if needed.
   - Add diagonal `y=x` reference line.
   - Color or facet by domain if readable.

3. `growth_by_domain_boxplot.png`
   - Boxplot of `annualized_log_growth` by domain.
   - Show global median as reference if possible.

4. `domain_adjusted_growth_histogram.png`
   - Histogram of `domain_adjusted_annualized_log_growth`.
   - Vertical line at zero.

5. `top_bottom_growth_barplots.png`
   - Barplot for top 15 and bottom 15 subfields by `annualized_log_growth`.
   - Use `subfield_label_short` or `subfield_label_unique` to avoid duplicate-name confusion.

6. `top_bottom_domain_adjusted_growth_barplots.png`
   - Same but for `domain_adjusted_annualized_log_growth`.

7. `yearly_counts_examples.png`
   - Select a few interpretable examples:
     - highest growth;
     - lowest growth;
     - highest domain-adjusted growth;
     - lowest domain-adjusted growth;
     - one or two large stable fields.
   - Plot yearly counts from 2010 to 2025.
   - Add a vertical separator between 2019 and 2020.

Optional but useful:

8. `growth_vs_initial_size.png`
   - x-axis: `annual_rate_2010_2019`.
   - y-axis: `annualized_log_growth`.
   - Shows whether growth is mostly a size effect.

9. `growth_percentile_by_domain.png`
   - Dot or strip plot of growth percentiles by domain.

---

## Summary JSON

Write:

```text
outputs/growth/subfield_growth_targets_summary.json
```

Include:

```text
created_at
input_paths
output_paths
count_source
n_subfields_expected
n_subfields_output
input_year_min
input_year_max
target_year_min
target_year_max
input_years
target_years
global_growth_median
n_growth_above_median
n_growth_below_or_equal_median
n_global_median_ties
n_growth_above_domain_median
n_growth_below_or_equal_domain_median
n_domain_median_ties
n_domains
domain_counts
n_subfields_with_zero_input_count
n_subfields_with_zero_target_count
annualized_log_growth_summary
domain_adjusted_annualized_log_growth_summary
warnings
assumptions
```

`annualized_log_growth_summary` should include:

```text
min
p05
p25
median
mean
p75
p95
max
std
```

Same for `domain_adjusted_annualized_log_growth_summary`.

---

## Robustness and edge cases

Handle these explicitly:

1. **Different window lengths**
   - Must use annual rates, not raw total ratios.

2. **Zero counts**
   - Use `log1p` rates.
   - Do not divide by zero.
   - Report zero-count cases.

3. **Duplicate display names**
   - Use `subfield_id` for all joins.
   - Labels must be unique in plots/tables.

4. **Missing years**
   - Complete the panel to all years 2010-2025.
   - Fill missing sparse count rows with zero only when justified.
   - Report assumption.

5. **Incomplete 2025 issue**
   - Since 2025 may depend on OpenAlex indexing completeness, add a summary warning if the count source was retrieved before the end of 2025 or if metadata indicates partial coverage.
   - The default target still uses 2020-2025, but the summary must document the retrieval date and source.

6. **Main-analysis subfield mismatch**
   - If count data is missing for any of the 240 subfields, fail clearly unless the missing count can be safely interpreted as zero.
   - Report any missing subfields.

7. **Domain medians with small groups**
   - Compute domain-adjusted growth even for small domains, but include `n_subfields` per domain in the summary.

8. **No prediction leakage issue in this stage**
   - It is expected that this target uses 2020-2025, because this is the future outcome.
   - Do not join the target back into morphology analysis scripts yet unless creating a separate join stage later.

---

## Tests

Add tests for the growth-target logic. Use synthetic data only.

Suggested test file:

```text
tests/test_growth_targets.py
```

Test at least:

1. Annualized rates correctly handle 10-year vs 6-year windows.
2. Equal annual rates produce `annualized_log_growth == 0`.
3. Higher target annual rate produces positive growth.
4. Lower target annual rate produces negative growth.
5. `growth_above_median` uses strict `>`.
6. Domain-adjusted growth subtracts the correct within-domain median.
7. Duplicate display names do not collapse distinct `subfield_id` rows.
8. Missing subfield-year combinations are completed in the panel.
9. The CLI runs on a tiny synthetic count table and writes parquet/csv/json/figures.
10. The output has exactly one row per input subfield.

Run:

```powershell
.\.venv\Scripts\python.exe -m py_compile src/growth_targets.py scripts/13_build_subfield_growth_targets.py
.\.venv\Scripts\python.exe -m pytest tests/test_growth_targets.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

---

## Documentation

Create or update:

```text
docs/subfield_growth_targets.md
README.md
```

The documentation should explain:

- why raw totals are not comparable because 2010-2019 has 10 years and 2020-2025 has 6 years;
- why annual rates solve this;
- the exact formula for `annualized_log_growth`;
- why `growth_above_median` is the main classification target;
- why domain-adjusted growth is useful;
- what data source was used for counts;
- that the sampled morphology corpus is not used for growth counts;
- command examples;
- output file descriptions;
- interpretation caveats.

---

## CLI design

The script should expose CLI arguments such as:

```text
--subfields-path data/processed/subfield_morphology_metrics.parquet
--counts-path <optional existing count table>
--output-path data/processed/subfield_growth_targets.parquet
--output-csv data/processed/subfield_growth_targets.csv
--yearly-panel-path outputs/growth/subfield_year_counts_panel.parquet
--summary-path outputs/growth/subfield_growth_targets_summary.json
--rankings-path outputs/growth/growth_rankings.csv
--domain-summary-path outputs/growth/domain_growth_summary.csv
--figures-dir outputs/growth/figures
--input-year-min 2010
--input-year-max 2019
--target-year-min 2020
--target-year-max 2025
--overwrite
```

If implementing OpenAlex fetching, include relevant arguments:

```text
--fetch-counts-from-openalex
--raw-counts-cache data/interim/openalex_subfield_year_counts_2010_2025.parquet
--mailto <optional or from env/config>
--sleep-seconds <optional>
```

Do not make fetching mandatory if an existing count table is available.

---

## Acceptance criteria

The task is complete only if:

1. A new Stage 13 script exists and runs from CLI.
2. Growth counts come from a full-count source, not the sampled morphology corpus.
3. The output has exactly one row per main-analysis subfield.
4. The target table includes `annualized_log_growth`.
5. The target table includes `growth_above_median`.
6. The target table includes `domain_adjusted_annualized_log_growth`.
7. The target table includes `growth_above_domain_median`.
8. Counts are annualized to correct the 10-year vs 6-year window difference.
9. Duplicate display names are handled using unique labels and `subfield_id` joins.
10. Missing years and zero counts are handled explicitly.
11. A yearly count panel is saved.
12. Summary JSON is saved and documents assumptions/warnings.
13. Growth rankings and domain summaries are saved.
14. Required figures are generated and readable.
15. Tests are added and pass.
16. Docs are updated.
17. Generated outputs are ignored by Git.
18. No prediction model is trained in this stage.

After implementation, report:

- files changed;
- data source used for full yearly counts;
- exact commands run;
- test results;
- number of subfields in output;
- global median growth value;
- number above/below global median;
- number above/below domain median;
- top 10 and bottom 10 by annualized log growth;
- top 10 and bottom 10 by domain-adjusted growth;
- warnings or assumptions, especially about missing years, zero counts, and 2025 coverage.
