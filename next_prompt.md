# Agent Prompt — Production-Scale OpenAlex Corpus Download

You are working in the private GitHub repository:

```text
aleetreny/TFM
```

The project is:

```text
OpenAlex Subfield Morphology and Scientific Growth Prediction
```

The repository already has a working six-step data pipeline:

```text
scripts/00_fetch_taxonomy.py
scripts/01_build_counts.py
scripts/02_build_corpus_plan.py
scripts/03_build_sample_plan.py
scripts/04_download_sampled_corpus.py
scripts/05_validate_database.py
```

The previous limited test worked:

```text
Planned: 15,000 works for 5 subfields
Kept after local validation: 12,124 works
Duplicates: 0
Missing titles/abstracts: 0
pytest: 15 passed
```

This is good, but before launching the full corpus download we need one production hardening step.

The goal of this task is:

1. Improve the sampled download so that we get as close as possible to **3,000 valid works per eligible subfield**.
2. Add oversampling and backfill.
3. Make the full download resumable.
4. Add a download manifest.
5. Keep the pipeline simple and ready for future expansion.
6. Then provide commands to run the full download safely.

Do **not** implement embeddings, UMAP, clustering, morphology metrics, models, plots, dashboards, or ML infrastructure yet.

---

## Main problem to solve

The current sample plan asks OpenAlex for exactly the planned sample size.

Example:

```text
planned_sample_size = 300
sample=300
```

But after local validation, some works are discarded because of:

- title too short
- abstract too short
- unexpected year mismatch
- unexpected primary subfield mismatch
- other local filters

Therefore, a plan of 15,000 works produced only 12,124 valid works in the limited test.

For the production download, we want to get much closer to:

```text
3,000 valid works per eligible subfield
```

when enough valid works exist.

---

## Required strategy

Use:

```text
sample_plan → oversampled API requests → local validation → backfill attempts → final capped corpus
```

Do not download all candidate works first.

For large cells, still use OpenAlex `sample` + `seed`.

For small cells, use cursor pagination and download all available works.

---

## Update config

Update `config.yaml` sampling section:

```yaml
sampling:
  target_sample_per_subfield: 3000
  max_sample_per_subfield: 3000
  min_valid_texts_per_subfield: 500
  stratify_by_year: true
  target_per_year_per_subfield: 300
  random_seed: 42
  use_openalex_sample_api: true

  # Production download controls
  oversample_factor: 1.75
  max_backfill_rounds: 4
  backfill_seed_step: 100000
  write_every_n_subfields: 5
  resume_existing_download: true
  save_download_manifest: true
```

Meaning:

- `oversample_factor`: if we need 300 valid works, initially request up to 525 raw works when available.
- `max_backfill_rounds`: if local validation still leaves a shortfall, repeat with a different seed.
- `backfill_seed_step`: seed increment for each backfill round.
- `write_every_n_subfields`: periodically write partial results to disk.
- `resume_existing_download`: avoid re-downloading rows already completed.
- `save_download_manifest`: save a detailed audit table.

You may adjust implementation details if needed, but preserve the behavior.

---

## Update sample plan

Keep `scripts/03_build_sample_plan.py`, but extend the output.

Current required columns remain:

```text
subfield_id
subfield_display_name
field_id
field_display_name
domain_id
domain_display_name
publication_year
available_valid_works
planned_sample_size
sampling_method
seed
```

Add:

```text
api_initial_sample_size
expected_shortfall_risk
```

Rules:

For each subfield-year cell:

```python
api_initial_sample_size = min(
    available_valid_works,
    ceil(planned_sample_size * oversample_factor)
)
```

But:

- if `planned_sample_size = 0`, `api_initial_sample_size = 0`
- if `sampling_method = download_all_available`, `api_initial_sample_size = planned_sample_size`
- if `sampling_method = sample_api`, `api_initial_sample_size >= planned_sample_size` when possible
- never exceed `available_valid_works`
- never exceed OpenAlex `sample` limits

Set:

```text
expected_shortfall_risk = true
```

when:

```text
api_initial_sample_size == planned_sample_size
```

because there is no oversampling buffer.

---

## Update OpenAlex query helper

In `src/openalex.py`, keep the current sampled query helper, but ensure it can request:

```text
sample=<api_initial_sample_size>
```

not only the planned kept size.

The query must still use:

```text
primary_topic.subfield.id:<subfield_id>
publication_year:<year>
has_abstract:true
language:en
type:article|preprint
is_retracted:false
is_paratext:false
select=...
sample=...
seed=...
```

Never use `topics.subfield.id` for the main corpus.

Never require PDF/fulltext/OA filters.

Do not add:

```text
is_oa:true
has_pdf_url:true
has_fulltext:true
has_content.pdf:true
```

---

## Update download script

Modify:

```text
scripts/04_download_sampled_corpus.py
```

The script should still support:

```bash
python scripts/04_download_sampled_corpus.py --dry-run
python scripts/04_download_sampled_corpus.py --limit-subfields 5
python scripts/04_download_sampled_corpus.py
```

Add optional arguments:

```bash
--resume
--force
--limit-subfields N
--only-subfield SUBFIELD_ID
--max-backfill-rounds N
```

Behavior:

### Default production behavior

When run without `--limit-subfields`, it should process all eligible subfields in `sample_plan.parquet`.

It must be safe to interrupt and resume.

### Resume behavior

If `data/processed/works_text.parquet` already exists and `--resume` is active/default, load it and skip already completed subfield-year cells where the manifest says the target was met or no more available works could be found.

If `--force` is passed, ignore previous results and rebuild the output.

### Download per subfield-year

For each row in `sample_plan`:

- `target_keep = planned_sample_size`
- `available = available_valid_works`
- `method = sampling_method`

If `target_keep = 0`, skip.

If `method = download_all_available`:

- use cursor pagination
- download all available works up to `planned_sample_size`
- validate locally
- keep all valid works

If `method = sample_api`:

1. Round 0:
   - request `api_initial_sample_size`
   - use original `seed`
2. Validate locally.
3. Deduplicate against:
   - works already collected in this run
   - works already in existing `works_text.parquet` when resuming
4. Keep up to `target_keep` valid works for this subfield-year.
5. If fewer than `target_keep`, run backfill rounds:
   - new seed = seed + backfill_round * backfill_seed_step
   - request enough extra raw works to plausibly fill the shortfall
   - validate and deduplicate again
6. Stop when:
   - target_keep is reached, or
   - max_backfill_rounds is reached, or
   - no new valid works are found in a round, or
   - available works are exhausted.

Important:

Do not let any single subfield exceed:

```text
max_sample_per_subfield = 3000
```

If global deduplication causes a subfield to exceed 3,000, keep the earliest deterministic records and drop the excess.

---

## Backfill sizing

Use a simple rule.

If the shortfall is:

```text
shortfall = target_keep - valid_kept
```

then request:

```python
extra_sample_size = min(
    available,
    ceil(shortfall * oversample_factor * 1.5)
)
```

You can choose a simple implementation, but the key is:

- request more than the shortfall
- use a new seed
- deduplicate
- keep only enough to reach the planned target

---

## Add download manifest

Create and update:

```text
data/interim/download_manifest.parquet
```

Also write DuckDB table:

```text
download_manifest
```

Each row should represent one subfield-year cell and include:

```text
subfield_id
subfield_display_name
field_id
field_display_name
domain_id
domain_display_name
publication_year
sampling_method
available_valid_works
planned_sample_size
api_initial_sample_size
raw_returned_works
valid_after_local_filter
kept_works
duplicate_or_already_seen
discarded_local_validation
shortfall
backfill_rounds_used
seeds_used
status
error_message
```

Status values:

```text
completed_target_met
completed_shortfall
skipped_no_available_works
failed
```

The manifest is essential. It will let us understand why some subfields do not reach 3,000.

---

## Periodic checkpointing

The full run may take time.

The script must periodically write:

```text
data/processed/works_text.parquet
data/interim/download_manifest.parquet
```

at least every `write_every_n_subfields` subfields.

Also write a final version at the end.

This makes the download resumable.

Do not keep everything only in memory until the end if avoidable.

Keep the implementation simple: loading/writing full Parquet files periodically is acceptable for now.

---

## Validation improvements

Update:

```text
scripts/05_validate_database.py
```

It should report:

- total eligible subfields
- total planned works
- total downloaded valid works
- downloaded valid works / planned works
- subfields with 3,000 valid works
- subfields with 2,500-2,999 valid works
- subfields with 500-2,499 valid works
- subfields below 500 valid works
- planned vs kept per subfield
- worst shortfalls
- planned works by sampling method
- kept works by sampling method
- manifest status counts
- average local validation retention rate
- duplicate count
- missing titles
- missing abstracts
- works per year
- works per subfield
- estimated future embedding size for 768-dimensional vectors:
  - float32
  - float16

Save:

```text
data/interim/validation_report.md
data/interim/validation_summary.json
```

---

## Add a full-run checklist doc

Create:

```text
docs/full_download_runbook.md
```

Include:

```bash
pip install -r requirements.txt

python scripts/00_fetch_taxonomy.py
python scripts/01_build_counts.py --dry-run
python scripts/01_build_counts.py
python scripts/02_build_corpus_plan.py
python scripts/03_build_sample_plan.py

python scripts/04_download_sampled_corpus.py --dry-run --limit-subfields 5
python scripts/04_download_sampled_corpus.py --limit-subfields 5
python scripts/05_validate_database.py

# Full production run
python scripts/04_download_sampled_corpus.py --resume
python scripts/05_validate_database.py
```

Also include:

- how to resume after interruption
- how to force a clean redownload
- how to download only one subfield for debugging
- where outputs are saved
- what level of shortfall is acceptable
- what to inspect if many subfields fail to reach 3,000

---

## Expected output sizes

The target is approximately:

```text
~254 subfields × 3,000 works ≈ 762,000 works
```

But final valid works may be lower if some subfields do not have enough valid title+abstract works.

The validation report should make this explicit.

Expected embedding storage later:

```text
762,000 × 768 × float32 ≈ 2.34 GB
762,000 × 768 × float16 ≈ 1.17 GB
```

Do not compute embeddings now.

---

## Robust future expansion

Design this so we can later expand the base easily.

The download pipeline should make it easy to change:

- morphology years
- growth years
- target sample per subfield
- oversample factor
- minimum abstract length
- sample strategy
- language filter
- inclusion/exclusion of preprints

Do not hardcode these values inside logic if they already exist in `config.yaml`.

---

## Tests

Add or update tests for:

- `api_initial_sample_size` is greater than planned size when oversampling is possible
- sample allocation never exceeds 3,000 per subfield
- sample allocation never exceeds available works
- backfill seed increments are deterministic
- sampled queries include `sample`, `seed`, and `select`
- resume logic can identify already completed cells from a manifest
- local validation still rejects short abstracts/titles
- query builder still uses `primary_topic.subfield.id`
- query builder does not use `topics.subfield.id`
- API key/email are redacted in logs

Run:

```bash
pytest
```

All tests must pass.

---

## Acceptance criteria

The task is complete when:

1. The full download script is production-safe and resumable.
2. The script uses oversampling for `sample_api` rows.
3. The script uses backfill rounds when local validation leaves shortfalls.
4. The script writes `download_manifest.parquet`.
5. The script periodically checkpoints `works_text.parquet`.
6. Validation clearly reports planned vs kept works.
7. Running the 5-subfield test improves the retained count compared with 12,124/15,000, unless the shortfall is genuinely due to insufficient available works.
8. The full-run command is documented in `docs/full_download_runbook.md`.
9. No data, DuckDB files, `.env`, or generated large outputs are committed.
10. No embeddings, UMAP, clustering, morphology metrics, plots, or ML code are added.

Keep the code boring, explicit, and easy to inspect.
