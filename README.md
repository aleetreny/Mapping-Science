# OpenAlex Subfield Morphology and Scientific Growth Prediction

The full thesis context is summarized in [research/README.md](research/README.md). This codebase is currently implementing the data collection and database organization phase only.

The project builds a clean OpenAlex database for later testing whether the title and abstract structure of a subfield before 2020 helps predict whether that subfield grows during 2020-2025.

No embeddings, dimensionality reduction, clustering, morphology metrics, regression models, prediction models, maps, dashboards, or ML infrastructure are implemented in this phase.

## Current Data Design

- Unit of analysis: OpenAlex subfield
- Morphology window: 2010-2019
- Growth window: 2020-2025
- Excluded year: 2026, because it is the current incomplete year
- Text source: title plus abstract
- OpenAlex classification: `primary_topic.subfield.id`
- Sample target: 3,000 papers per eligible subfield
- Storage: one DuckDB database at `warehouse/tfm_openalex.duckdb`

## Why Subfields, Not Topics

OpenAlex topics are too numerous for the first stable thesis pipeline. Subfields are broad enough to form meaningful semantic spaces and give roughly a few hundred units, which is more realistic for a master thesis.

The pipeline therefore uses `primary_topic.subfield.id` as the main classification. It does not use `topics.subfield.id` as the active unit of analysis.

## Why Not Download Everything

Downloading all OpenAlex works for all subfields would be too large and unnecessary. A capped, stratified sample gives comparable morphology inputs across subfields while keeping storage and later embedding work manageable.

The corpus download uses OpenAlex's `sample` and `seed` parameters for large subfield-year cells. Small cells are downloaded in full. This avoids downloading all candidates first and then sampling locally.

Growth targets are based primarily on article and preprint counts, not only abstract-available works, because growth should measure scientific production rather than abstract availability.

## Repository Layout

```text
src/        small reusable helpers
scripts/    numbered pipeline steps
docs/       data model and query documentation
research/   whole-project thesis context for future agents
data/       ignored local data folders
warehouse/  ignored DuckDB database location
tests/      basic unit tests
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Set `OPENALEX_EMAIL` in `.env` if possible. `OPENALEX_API_KEY` may be left empty.

## Run

```bash
python scripts/00_fetch_taxonomy.py
python scripts/01_build_counts.py --dry-run
python scripts/01_build_counts.py
python scripts/02_build_corpus_plan.py
python scripts/03_build_sample_plan.py
python scripts/04_download_sampled_corpus.py --dry-run
python scripts/04_download_sampled_corpus.py --limit-subfields 5
python scripts/05_validate_database.py
```

The full corpus download may take time. Test first with `--limit-subfields 5`.

## Outputs

The organized local outputs are:

```text
warehouse/tfm_openalex.duckdb
data/interim/domains.parquet
data/interim/fields.parquet
data/interim/subfields.parquet
data/interim/subfield_year_counts.parquet
data/interim/field_year_counts.parquet
data/interim/domain_year_counts.parquet
data/interim/corpus_plan.parquet
data/interim/sample_plan.parquet
data/processed/works_text.parquet
data/interim/validation_report.md
data/interim/validation_summary.json
```

Data files, secrets, and large artifacts are ignored by Git.
