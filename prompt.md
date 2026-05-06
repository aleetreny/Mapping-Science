# Agent Prompt — Add API-Based Sampling to the OpenAlex Subfield Pipeline

You are working in the private GitHub repo:

```text
aleetreny/TFM
```

The repo has already been refactored toward the new thesis direction:

```text
OpenAlex Subfield Morphology and Scientific Growth Prediction
```

The current pipeline is good as a first structure. Do **not** rewrite everything from scratch.

Your task is narrower:

1. Add an explicit `sample_plan` step.
2. Change corpus download so it requests samples directly from the OpenAlex API when possible.
3. Load OpenAlex credentials from the existing `.env`.
4. Keep the repo simple and readable.
5. Do **not** add embeddings, UMAP, clustering, morphology metrics, models, or plots.

---

## Current repo state

The repo already has:

```text
README.md
config.yaml
src/openalex.py
src/abstracts.py
src/storage.py
src/sampling.py
src/growth.py
scripts/00_fetch_taxonomy.py
scripts/01_build_counts.py
scripts/02_build_corpus_plan.py
scripts/03_download_corpus.py
scripts/04_validate_database.py
```

The current README correctly describes:

- unit: OpenAlex subfield
- morphology window: 2010-2019
- growth window: 2020-2025
- excluded year: 2026
- sample target: 3,000 papers per eligible subfield
- storage: DuckDB
- no embeddings/models yet

Keep that direction.

---

## Main issue to fix

At the moment, `scripts/03_download_corpus.py` still fetches candidates through cursor pagination and then performs local stratified sampling.

That is not what we want for the full corpus.

We do **not** want to download all candidate works and then sample locally.

We want:

```text
count → corpus_plan → sample_plan → API sampled download
```

For large subfield-year cells, use OpenAlex:

```text
sample=<planned_sample_size>
seed=<stable_seed>
```

For small cells, where available works are less than or equal to the planned sample, downloading all available works is fine.

---

## Keep existing scripts, but rename/insert one step

Change the numbered scripts to:

```text
scripts/00_fetch_taxonomy.py
scripts/01_build_counts.py
scripts/02_build_corpus_plan.py
scripts/03_build_sample_plan.py
scripts/04_download_sampled_corpus.py
scripts/05_validate_database.py
```

You can rename the current `03_download_corpus.py` into `04_download_sampled_corpus.py` and adapt it.

Do not keep two active download scripts that do almost the same thing.

Update README commands accordingly.

---

## Load `.env`

The repo already has a local `.env` with:

```bash
OPENALEX_EMAIL=...
OPENALEX_API_KEY=...
```

Add `python-dotenv` to `requirements.txt`.

In `src/openalex.py`, load `.env` from the repo root before reading environment variables.

Do not hardcode credentials.

Do not print the API key.

The current `safe_url_for_logging` should continue redacting `api_key`, `mailto`, and email-like parameters.

---

## Update config

Update `config.yaml` sampling section to include:

```yaml
sampling:
  target_sample_per_subfield: 3000
  max_sample_per_subfield: 3000
  min_valid_texts_per_subfield: 500
  stratify_by_year: true
  target_per_year_per_subfield: 300
  random_seed: 42
  use_openalex_sample_api: true
```

Keep the current windows:

```yaml
morphology_start_year: 2010
morphology_end_year: 2019
growth_start_year: 2020
growth_end_year: 2025
exclude_years: [2026]
```

---

## Add `scripts/03_build_sample_plan.py`

This script should read:

```text
data/interim/corpus_plan.parquet
data/interim/subfield_year_counts.parquet
```

For each eligible subfield and each year 2010-2019, use:

```text
n_works_article_preprint_en_with_abstract
```

as the initial available count.

Build:

```text
data/interim/sample_plan.parquet
```

and write DuckDB table:

```text
sample_plan
```

Required columns:

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

`sampling_method` must be one of:

```text
sample_api
download_all_available
skip_no_available_works
```

Sampling logic:

- Target total per subfield: 3,000.
- Initial target per year: 300 for each year 2010-2019.
- If `available_valid_works >= 300`, initially plan 300 and use `sample_api`.
- If `0 < available_valid_works < 300`, plan all available and use `download_all_available`.
- If `available_valid_works = 0`, plan 0 and use `skip_no_available_works`.
- Redistribute missing capacity across years with available capacity above 300.
- Never exceed available works for a year.
- Never exceed 3,000 works for a subfield.
- Use deterministic seeds.

A stable seed can be:

```python
seed = random_seed + publication_year + numeric_part_of_subfield_id
```

or another deterministic function.

Print a short summary:

```text
eligible subfields
planned total works
subfields with planned_sample_size < 3000
estimated API requests
```

---

## Update `src/openalex.py`

Add support for sampled requests.

Suggested helper:

```python
def build_sampled_text_query_params(
    subfield_id: str,
    year: int,
    sample_size: int,
    seed: int,
    work_types: Iterable[str],
    language: str = "en",
    select_fields: Iterable[str] | None = None,
) -> dict[str, Any]:
    ...
```

It should build a works query with:

```text
primary_topic.subfield.id:<subfield_id>
publication_year:<year>
has_abstract:true
language:en
type:article|preprint
is_retracted:false
is_paratext:false
sample=<sample_size>
seed=<seed>
select=<needed fields>
```

Keep using `primary_topic.subfield.id`.

Do not use `topics.subfield.id` for the main corpus.

Keep `select=`.

---

## Replace download behavior

Create/adapt:

```text
scripts/04_download_sampled_corpus.py
```

It should read:

```text
data/interim/sample_plan.parquet
```

For each row:

- if `planned_sample_size = 0`, skip.
- if `sampling_method = sample_api`, call OpenAlex with `sample` and `seed`.
- if `sampling_method = download_all_available`, use cursor pagination.
- validate locally anyway.

Always filter:

```text
primary_topic.subfield.id:<subfield_id>
publication_year:<year>
has_abstract:true
language:en
type:article|preprint
is_retracted:false
is_paratext:false
```

Always fetch only:

```text
id
doi
title
display_name
abstract_inverted_index
publication_year
publication_date
type
language
primary_topic
cited_by_count
referenced_works_count
is_retracted
is_paratext
```

Reconstruct abstract with existing `src/abstracts.py`.

Final output:

```text
data/processed/works_text.parquet
```

DuckDB table:

```text
works_text
```

Required final columns:

```text
work_id
doi
title
abstract
text_for_embedding
publication_year
publication_date
type
language
subfield_id
subfield_display_name
field_id
field_display_name
domain_id
domain_display_name
cited_by_count
referenced_works_count
```

where:

```text
text_for_embedding = title + "\n\n" + abstract
```

Keep support for:

```bash
python scripts/04_download_sampled_corpus.py --dry-run
python scripts/04_download_sampled_corpus.py --limit-subfields 5
python scripts/04_download_sampled_corpus.py
```

Dry-run should not call the full API download. It should show planned works, methods, estimated requests, and a few example query params.

---

## Update validation

Rename/adapt validation to:

```text
scripts/05_validate_database.py
```

Make it include `sample_plan` in the report:

- total planned works
- planned works by sampling method
- subfields planned below 3,000
- downloaded works vs planned works
- duplicate work IDs
- works per year
- works per subfield
- estimated future embedding size for 768 dimensions with float32 and float16

Outputs remain:

```text
data/interim/validation_report.md
data/interim/validation_summary.json
```

---

## Update docs and README

Update:

```text
README.md
docs/data_model.md
docs/openalex_queries.md
docs/growth_definition.md
```

The README run section should become:

```bash
pip install -r requirements.txt

python scripts/00_fetch_taxonomy.py
python scripts/01_build_counts.py --dry-run
python scripts/01_build_counts.py
python scripts/02_build_corpus_plan.py
python scripts/03_build_sample_plan.py
python scripts/04_download_sampled_corpus.py --dry-run
python scripts/04_download_sampled_corpus.py --limit-subfields 5
python scripts/05_validate_database.py
```

Explain clearly that the corpus is sampled through the OpenAlex API when possible, instead of downloading all candidates first.

---

## Tests

Add or update minimal tests for:

- `.env` loading does not expose credentials
- query builder uses `primary_topic.subfield.id`
- query builder does not use `topics.subfield.id`
- sampled query includes `sample`, `seed`, and `select`
- sample plan redistribution never exceeds 3,000 per subfield
- abstract reconstruction still works
- growth formula still works

Run:

```bash
pytest
```

---

## Acceptance criteria

The task is complete when:

1. `scripts/03_build_sample_plan.py` exists.
2. `scripts/04_download_sampled_corpus.py` uses `sample` and `seed` for large cells.
3. The old local-candidate-download-then-sample approach is no longer the main full-corpus strategy.
4. `.env` is loaded correctly.
5. `python-dotenv` is in `requirements.txt`.
6. README and docs match the new numbered scripts.
7. Dry-run works.
8. `--limit-subfields 5` works.
9. No embeddings, ML code, plots, or dashboards are added.
10. Data and secrets remain ignored by Git.

Keep the code boring, explicit, and easy to inspect.
