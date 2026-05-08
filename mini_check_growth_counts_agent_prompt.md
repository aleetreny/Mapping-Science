# Mini-check: validate extreme OpenAlex growth counts

You are working in the repo `aleetreny/TFM`.

We need a short validation check for Stage 13 growth targets. The growth script looks logically correct, but some yearly counts look suspiciously large, especially `2202 | Aerospace Engineering`. Before moving to prediction, verify whether the local count table matches direct OpenAlex API counts for a few extreme subfields.

## Goal

Create a small validation script:

```text
scripts/13b_validate_growth_count_extremes.py
```

It should compare, for selected `subfield_id` and `publication_year` pairs:

1. the local count in `data/interim/subfield_year_counts.parquet`, using column:
   ```text
   n_works_article_preprint
   ```
2. the direct OpenAlex `/works` count using the same filters as Stage 01:
   ```text
   primary_topic.subfield.id:<subfield_id>
   publication_year:<year>
   type:article|preprint
   is_retracted:false
   is_paratext:false
   ```

Use the existing `OpenAlexClient`, `build_filter`, and/or existing query utilities where possible.

## Subfields to check

Validate at least these subfields:

```text
2202 Aerospace Engineering
2200 General Engineering
3500 General Dentistry
2611 Modeling and Simulation
2725 Infectious Diseases
3307 Human Factors and Ergonomics
```

Check these years:

```text
2010
2019
2020
2025
```

If 2025 is affected by indexing lag, still include it but flag it in the output.

## Outputs

Write:

```text
outputs/growth/count_validation/growth_count_extremes_validation.csv
outputs/growth/count_validation/growth_count_extremes_validation.json
```

The CSV should include:

```text
subfield_id
subfield_label_unique
publication_year
local_count
openalex_api_count
absolute_difference
relative_difference
matches_exactly
status
query_filter
checked_at
```

`status` should be:

```text
matched
mismatch
api_failed
missing_local_count
```

The JSON summary should include:

```text
n_checks
n_matched
n_mismatch
n_api_failed
max_absolute_difference
max_relative_difference
suspicious_cases
notes
```

## CLI

Support:

```powershell
.\.venv\Scripts\python.exe scripts\13b_validate_growth_count_extremes.py --overwrite
```

Optional arguments:

```text
--counts-path data/interim/subfield_year_counts.parquet
--growth-targets-path data/processed/subfield_growth_targets.parquet
--output-dir outputs/growth/count_validation
--years 2010 2019 2020 2025
--subfield-id 2202
--sleep-seconds 0.1
--overwrite
```

## Important requirements

- Do not change Stage 13 outputs.
- Do not modify growth target definitions.
- Do not use sampled UMAP/morphology papers for this check.
- Do not silently pass if OpenAlex requests fail.
- Print a readable terminal report showing local vs API counts.
- Add lightweight tests using mocked/synthetic data where possible.
- Update `docs/subfield_growth_targets.md` with a short section explaining this validation check.

## Acceptance criteria

The task is complete only if:

1. The script runs from CLI.
2. It compares local counts to direct OpenAlex counts for the selected extreme subfields.
3. It writes CSV and JSON outputs.
4. It clearly reports mismatches.
5. It does not overwrite existing outputs unless `--overwrite` is passed.
6. Tests pass.
7. The agent reports:
   - files changed,
   - commands run,
   - validation result,
   - whether `2202 Aerospace Engineering` local counts match direct OpenAlex API counts.
