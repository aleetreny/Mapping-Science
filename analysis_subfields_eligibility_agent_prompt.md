# Agent Prompt — Add analysis eligibility flags after corpus download

We have completed the OpenAlex corpus download.

Current validation summary:

- Planned works: 744,402
- Downloaded valid works: 733,419
- Eligible subfields downloaded: 251
- Subfields with 3,000 valid works: 225
- Subfields with 2,500-2,999 valid works: 15
- Subfields with 500-2,499 valid works: 10
- Subfields below 500 valid works: 1
- Duplicate work IDs: 0
- Missing titles: 0
- Missing abstracts: 0
- Missing primary_topic_id: 0
- Missing topics_json: 0

The corpus is good. Do **not** redownload data. Do **not** retry shortfalls. Do **not** change the API pipeline.

## Goal

Add a clean analysis eligibility layer.

The full corpus should remain untouched, but the main analysis should use only subfields with at least **2,500 valid works**.

This should make later morphology metrics more stable.

## Required outputs

Create:

```text
data/processed/analysis_subfields.parquet
```

Also write a DuckDB table:

```text
analysis_subfields
```

Each row should represent one subfield.

Required columns:

```text
subfield_id
subfield_display_name
field_id
field_display_name
domain_id
domain_display_name
n_valid_works
planned_works
shortfall
main_analysis_eligible_2500
robustness_eligible_500
is_low_sample
exclusion_reason
```

Rules:

```text
main_analysis_eligible_2500 = n_valid_works >= 2500
robustness_eligible_500 = n_valid_works >= 500
is_low_sample = n_valid_works < 2500
```

`exclusion_reason`:

```text
"main_analysis_included" if n_valid_works >= 2500
"below_2500_valid_works" if 500 <= n_valid_works < 2500
"below_500_valid_works" if n_valid_works < 500
```

## Add script

Create:

```text
scripts/06_build_analysis_subfields.py
```

The script should:

1. Read `data/processed/works_text.parquet`.
2. Read `data/interim/sample_plan.parquet` or `data/interim/corpus_plan.parquet` to get planned works and metadata.
3. Aggregate valid works by subfield.
4. Build `analysis_subfields.parquet`.
5. Write the DuckDB table.
6. Print a short summary:
   - total subfields
   - main analysis subfields `>=2500`
   - robustness subfields `>=500`
   - excluded from main analysis
   - excluded from robustness
   - total works in main analysis
   - total works excluded from main analysis

Keep it simple.

## Update validation

Update:

```text
scripts/05_validate_database.py
```

so that if `analysis_subfields.parquet` or DuckDB table `analysis_subfields` exists, it reports:

```text
main_analysis_subfields_2500
robustness_subfields_500
main_analysis_works
excluded_from_main_analysis_subfields
excluded_from_main_analysis_works
excluded_from_robustness_subfields
excluded_from_robustness_works
```

Do not make validation fail if the file does not exist.

## Update docs

Update:

```text
README.md
docs/data_model.md
docs/full_download_runbook.md
```

Explain:

- The full corpus remains available.
- The main morphology analysis uses subfields with `n_valid_works >= 2500`.
- The robustness analysis can use subfields with `n_valid_works >= 500`.
- The 2,500 threshold is chosen to avoid unstable morphology metrics in very small semantic clouds.
- We are not deleting any downloaded papers.

## Embedding decision

Do **not** filter the embedding input yet.

For now, embeddings should still be generated for the full `works_text.parquet`.

Later morphology scripts will use `analysis_subfields.parquet` to select:

```text
main_analysis_eligible_2500 == true
```

This preserves flexibility for robustness checks.

## Tests

Add tests for the eligibility logic:

- 3,000 works → main eligible
- 2,500 works → main eligible
- 2,499 works → not main eligible but robustness eligible
- 500 works → robustness eligible
- 499 works → excluded from robustness

Run:

```bash
pytest
```

## Acceptance criteria

The task is complete when:

1. `scripts/06_build_analysis_subfields.py` exists.
2. It creates `data/processed/analysis_subfields.parquet`.
3. It writes DuckDB table `analysis_subfields`.
4. Validation includes the new eligibility counts when available.
5. README/docs explain the 2,500 main-analysis threshold.
6. No data is deleted.
7. No API code or download logic is changed.
8. No embeddings or morphology metrics are implemented yet.
9. `pytest` passes.

Keep the update small and surgical.
