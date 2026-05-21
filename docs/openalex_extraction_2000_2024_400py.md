# OpenAlex Extraction 2000-2024 400py

This is a versioned OpenAlex text-corpus extraction for later embedding work.
It does not overwrite the canonical checkpoint files and does not generate
embeddings, UMAP maps, metrics, or clusters.

```text
dataset_version: 2000_2024_400py
years: 2000-2024 inclusive
target: 400 valid papers per year per OpenAlex subfield
full-period target: 10,000 papers per subfield
natural future windows: 2000-2004, 2005-2009, 2010-2014, 2015-2019, 2020-2024
```

The extraction remains annual. It writes `publication_year` and does not create
or depend on a temporal block column.

## Commands

Dry-run counts:

```bash
python scripts/01_build_counts.py --dataset-version 2000_2024_400py --dry-run
```

Build counts:

```bash
python scripts/01_build_counts.py --dataset-version 2000_2024_400py
```

Build corpus and sample plans:

```bash
python scripts/02_build_corpus_plan.py --dataset-version 2000_2024_400py
python scripts/03_build_sample_plan.py --dataset-version 2000_2024_400py
```

Dry-run download:

```bash
python scripts/04_download_sampled_corpus.py --dataset-version 2000_2024_400py --dry-run
```

Actual download:

```bash
python scripts/04_download_sampled_corpus.py --dataset-version 2000_2024_400py --write-every-n-subfields 1
```

For long runs, the downloader handles OpenAlex 429 responses by honoring
`Retry-After` when OpenAlex sends it, otherwise sleeping for the configured
cooldown. You can tune the rate-limit policy explicitly:

```bash
python scripts/04_download_sampled_corpus.py --dataset-version 2000_2024_400py --write-every-n-subfields 1 --cooldown-after-429 180 --max-consecutive-429 10 --resume
```

Cells that still exceed the consecutive-429 limit are written with
`status = rate_limited`, not treated as completed, so a later `--resume` run
will retry them. The final downloader summary includes failed cells by error
type, including `rate_limit_429`.

Validate and build analysis-subfield summary:

```bash
python scripts/05_validate_database.py --dataset-version 2000_2024_400py
python scripts/06_build_analysis_subfields.py --dataset-version 2000_2024_400py
```

## Outputs

Counts:

```text
data/interim/subfield_year_counts_2000_2024_400py.parquet
data/interim/field_year_counts_2000_2024_400py.parquet
data/interim/domain_year_counts_2000_2024_400py.parquet
```

Plans and manifest:

```text
data/interim/corpus_plan_2000_2024_400py.parquet
data/interim/sample_plan_2000_2024_400py.parquet
data/interim/download_manifest_2000_2024_400py.parquet
```

Processed tables:

```text
data/processed/works_text_2000_2024_400py.parquet
data/processed/analysis_subfields_2000_2024_400py.parquet
```

Validation and diagnostics:

```text
outputs/01_corpus_construction/validation/validation_report_2000_2024_400py.md
outputs/01_corpus_construction/validation/validation_summary_2000_2024_400py.json
outputs/01_corpus_construction/diagnostics/openalex_extraction_2000_2024_400py/subfield_year_download_coverage.csv
outputs/01_corpus_construction/diagnostics/openalex_extraction_2000_2024_400py/subfield_coverage_summary.csv
outputs/01_corpus_construction/diagnostics/openalex_extraction_2000_2024_400py/year_coverage_summary.csv
outputs/01_corpus_construction/diagnostics/openalex_extraction_2000_2024_400py/extraction_summary.md
```

DuckDB tables use the same version suffix, for example
`works_text_2000_2024_400py` and `sample_plan_2000_2024_400py`.

## Filters

The version keeps the same text-corpus filters as the existing extraction:

```text
unit: subfield
use_primary_topic_subfield_only: true
language: en
work_types: article, preprint
require_abstract_for_text_corpus: true
require_title_for_text_corpus: true
min_title_tokens: 5
min_abstract_tokens: 80
exclude_retracted: true
exclude_paratext: true
```

The downloader still validates works locally after OpenAlex returns them. A
work is kept only if its primary-topic subfield matches the planned subfield,
its publication year matches the planned annual cell, and the text/type/language
filters pass.

## Selected Fields

OpenAlex queries keep the existing `DEFAULT_SELECT_FIELDS`:

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
topics
cited_by_count
referenced_works_count
is_retracted
is_paratext
```

The normalized output keeps the existing `WORKS_TEXT_COLUMNS`:

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
primary_topic_id
primary_topic_display_name
topics_json
title_token_count
abstract_token_count
text_token_count
downloaded_at
cited_by_count
referenced_works_count
```

## Sampling

For this version:

```text
target_per_year_per_subfield: 400
target_sample_per_subfield: 10000
max_sample_per_subfield: 10000
min_valid_texts_per_subfield: 500
oversample_factor: 1.75
max_backfill_rounds: 4
cooldown_after_429: 180
max_consecutive_429: 10
write_every_n_subfields: 1
```

Each annual cell plans:

```text
planned_sample_size = min(available_valid_works, 400)
```

The version does not redistribute unused capacity from sparse years into other
years. If early years have fewer than 400 valid papers, the plan records the
shortfall instead of failing the extraction.

## Drive Handoff

After the actual download and validation succeed, upload the versioned corpus
for later Kaggle embeddings:

```powershell
rclone mkdir gdrive:TFM/openalex_subfields/data_2000_2024_400py

rclone copy data/processed/works_text_2000_2024_400py.parquet `
  gdrive:TFM/openalex_subfields/data_2000_2024_400py/ `
  -P
```

Recommended later Kaggle input:

```text
gdrive:TFM/openalex_subfields/data_2000_2024_400py/works_text_2000_2024_400py.parquet
```

Recommended later embedding output:

```text
gdrive:TFM/openalex_subfields/embeddings/specter2_v1_2000_2024_400py/
```

Recommended local future embedding path:

```text
embeddings/specter2_v1_2000_2024_400py/
```

## Risks

Older years may have weaker abstract coverage, especially before the 2010s.
The coverage diagnostics are the first place to check whether 400 valid
papers/year/subfield is realistic. API runtime and request volume will also be
substantially larger than the earlier checkpoint because the target is 25 years
times 400 papers per year.
