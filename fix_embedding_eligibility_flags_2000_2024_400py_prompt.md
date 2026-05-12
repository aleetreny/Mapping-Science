# Fix embedding validation and analysis matrix eligibility flags for `2000_2024_400py`

We have successfully generated, downloaded, and locally stored the new SPECTER2 embeddings for dataset version:

```text
2000_2024_400py
```

Current embedding artifact folder:

```text
embeddings/specter2_v1_2000_2024_400py/
```

Current versioned corpus files:

```text
data/processed/works_text_2000_2024_400py.parquet
data/processed/analysis_subfields_2000_2024_400py.parquet
```

Canonical files have also been copied to:

```text
data/processed/works_text.parquet
data/processed/analysis_subfields.parquet
```

The new `analysis_subfields` file has these eligibility columns:

```text
eligible_10000_full_period
eligible_min_5000_full_period
eligible_for_temporal_5year_exploration
```

But the current embedding validation / analysis matrix pipeline still expects the old flag names:

```text
main_analysis_eligible_2500
robustness_eligible_500
```

This caused `scripts/07_validate_embeddings.py` to fail even though the embeddings themselves are complete and aligned.

Current validation facts:

```text
Expected shards: 119
Embedding shards: 119
Metadata shards: 119
Summary files: 119
Total embedding rows: 2378036
Total metadata rows: 2378036
Works text rows: 2378036
Embedding index rows: 2378036
Embedding dimension: 768
Embedding dtype: float16
Duplicate metadata work IDs: 0
Embedded IDs missing from works_text: 0
works_text IDs missing from embeddings: 0
Subfield mismatch rows: 0
DuckDB table written: True
```

But validation failed with:

```text
2378036 embedding_index rows could not join analysis flags
analysis_subfields is missing eligibility flag columns
```

Then `scripts/08_prepare_analysis_matrix.py` produced an empty matrix:

```text
n_rows: 0
```

because the main-analysis flag was missing.

## Task

Fix the repo properly. Do not solve this with an invisible one-off manual patch only.

The goal is to repair the eligibility-flag integration between:

```text
analysis_subfields
embedding_index
analysis_embedding_index
main_embeddings.float16.npy
```

Do not regenerate embeddings. Do not touch the `.npy` shard files. The embedding artifacts are complete.

## Required design

Add a simple, explicit compatibility layer for eligibility flags.

For the new dataset version `2000_2024_400py`, define the internal/canonical analysis flags as:

```text
main_analysis_eligible = eligible_for_temporal_5year_exploration
robustness_eligible = eligible_min_5000_full_period
strict_full_period_eligible = eligible_10000_full_period
```

Reasoning:

- `eligible_for_temporal_5year_exploration` keeps the broad comparative morphology analysis around the 241 usable subfields.
- `eligible_10000_full_period` is stricter and cleaner but would reduce coverage to roughly 176 subfields, which is too restrictive for the main comparative analysis.
- `eligible_min_5000_full_period` is a reasonable robustness/general inclusion flag.

Preserve backward compatibility with the old dataset where possible:

```text
main_analysis_eligible_2500
robustness_eligible_500
```

The old flag names can remain as aliases, but downstream code should not depend blindly on only those names.

## Requirements

### 1. Update the relevant code

Review and update, as needed:

```text
scripts/07_validate_embeddings.py
scripts/08_prepare_analysis_matrix.py
src/embeddings.py
src/analysis_matrix.py
```

Also search the repo for direct dependencies on:

```text
main_analysis_eligible_2500
robustness_eligible_500
```

and make sure downstream scripts still work.

Likely affected downstream stages include:

```text
scripts/09_build_first_umap_maps.py
scripts/10_build_per_subfield_umap_maps.py
scripts/10b_build_per_field_umap_maps.py
scripts/10c_build_per_domain_umap_maps.py
scripts/11_compute_subfield_morphology_metrics.py
scripts/12_compute_subfield_embedding_space_metrics.py
scripts/13_compare_metric_families.py
scripts/14_summarize_metric_distributions.py
scripts/15_cluster_metric_spaces.py
```

Do not overengineer. A small helper function that normalizes eligibility flags is enough if that keeps things clean.

### 2. Make the pipeline fail loudly if matrix rows are zero

`scripts/08_prepare_analysis_matrix.py` must not silently produce:

```text
n_rows: 0
```

If the selected main-analysis index is empty, raise a clear error unless there is an explicit debug/testing override flag.

The error message should tell the user that eligibility flags could not be resolved or selected zero rows.

### 3. Keep output columns understandable

After validation, `data/processed/embedding_index.parquet` should contain clear eligibility columns.

Preferred internal names:

```text
main_analysis_eligible
robustness_eligible
strict_full_period_eligible
```

Backward-compatible aliases may also be written:

```text
main_analysis_eligible_2500
robustness_eligible_500
```

if this reduces downstream disruption.

But document clearly which one is the canonical name going forward.

### 4. Update tests

Add or update tests covering:

1. Old-style `analysis_subfields` with:
   ```text
   main_analysis_eligible_2500
   robustness_eligible_500
   ```

2. New-style `analysis_subfields` with:
   ```text
   eligible_10000_full_period
   eligible_min_5000_full_period
   eligible_for_temporal_5year_exploration
   ```

3. Missing eligibility columns should produce a clear error.

4. `08_prepare_analysis_matrix.py` / `prepare_analysis_embedding_index` should not accept a zero-row selected matrix silently.

### 5. Update docs

Update relevant docs, especially:

```text
docs/embedding_data_model.md
docs/data_model.md
docs/analysis_matrix_and_first_umap.md
```

Explain the new dataset version and eligibility mapping simply:

```text
For 2000_2024_400py:
- main_analysis_eligible = eligible_for_temporal_5year_exploration
- robustness_eligible = eligible_min_5000_full_period
- strict_full_period_eligible = eligible_10000_full_period
```

Also update any old references that say the embedding artifact set contains 37 shards when referring to the new version. The new version contains 119 shards.

## Commands that should work after the fix

After implementing the fix, these commands should pass:

```powershell
.\.venv\Scripts\python.exe scripts\07_validate_embeddings.py `
  --embedding-dir embeddings/specter2_v1_2000_2024_400py `
  --expected-shards 119
```

Expected:

```text
Embedding validation completed successfully.
```

Then:

```powershell
.\.venv\Scripts\python.exe scripts\08_prepare_analysis_matrix.py `
  --embedding-dir embeddings/specter2_v1_2000_2024_400py `
  --force
```

Expected:

```text
data/processed/analysis_embedding_index.parquet
embeddings/specter2_v1_2000_2024_400py/analysis/main_embeddings.float16.npy
embeddings/specter2_v1_2000_2024_400py/analysis/main_work_ids.parquet
embeddings/specter2_v1_2000_2024_400py/analysis/main_matrix_summary.json
```

The resulting `main_matrix_summary.json` must have:

```text
n_rows > 0
```

and the number of main-analysis subfields should match the selected main flag, likely around 241 for:

```text
eligible_for_temporal_5year_exploration
```

## Verification

Run:

```powershell
.\.venv\Scripts\python.exe -m compileall scripts src
.\.venv\Scripts\python.exe -m pytest
```

Also provide a short final summary with:

```text
- Files changed
- Eligibility mapping implemented
- Commands run
- Test results
- Any remaining assumptions or risks
```

## Important constraints

- Do not regenerate embeddings.
- Do not delete `embeddings/specter2_v1_2000_2024_400py/`.
- Do not delete `data/processed/works_text_2000_2024_400py.parquet`.
- Do not delete `data/processed/analysis_subfields_2000_2024_400py.parquet`.
- Do not make the repo depend only on a manual alias patch.
- Prefer simplicity and clarity over abstraction.
