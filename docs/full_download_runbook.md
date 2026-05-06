# Full Download Runbook

This runbook is for the production-scale OpenAlex title and abstract corpus download.

The target is roughly:

```text
~254 subfields x 3,000 works = ~762,000 works
```

The final valid count may be lower if some subfields do not have enough locally valid title and abstract works.

## Setup

```bash
pip install -r requirements.txt
```

Make sure `.env` exists locally and contains:

```bash
OPENALEX_EMAIL=your_email@example.com
OPENALEX_API_KEY=
```

The API key may be empty. Do not commit `.env`.

## Build Plans

```bash
python scripts/00_fetch_taxonomy.py
python scripts/01_build_counts.py --dry-run
python scripts/01_build_counts.py
python scripts/02_build_corpus_plan.py
python scripts/03_build_sample_plan.py
```

## Test Download

```bash
python scripts/04_download_sampled_corpus.py --dry-run --limit-subfields 5
python scripts/04_download_sampled_corpus.py --limit-subfields 5
python scripts/05_validate_database.py
```

The five-subfield run should land close to 15,000 valid works. Some shortfall is acceptable when local validation rejects short titles or abstracts.

Use `--force --limit-subfields 5` when you want the test to ignore an existing partial corpus.

## Full Production Run

```bash
python scripts/04_download_sampled_corpus.py --resume
python scripts/05_validate_database.py
```

The downloader checkpoints periodically to:

```text
data/processed/works_text.parquet
data/interim/download_manifest.parquet
```

It also writes DuckDB tables:

```text
works_text
download_manifest
```

## Resume After Interruption

Run the same command again:

```bash
python scripts/04_download_sampled_corpus.py --resume
```

Completed cells in `download_manifest.parquet` are skipped.

## Force A Clean Redownload

Use:

```bash
python scripts/04_download_sampled_corpus.py --force
```

For a clean five-subfield test:

```bash
python scripts/04_download_sampled_corpus.py --force --limit-subfields 5
```

## Debug One Subfield

```bash
python scripts/04_download_sampled_corpus.py --force --only-subfield 2613
python scripts/05_validate_database.py
```

## Outputs

- `data/interim/sample_plan.parquet`
- `data/interim/download_manifest.parquet`
- `data/processed/works_text.parquet`
- `warehouse/tfm_openalex.duckdb`
- `data/interim/validation_report.md`
- `data/interim/validation_summary.json`

## Shortfall Guidance

Small shortfalls are acceptable. Inspect the validation report if many subfields fall below 3,000 valid works.

Look first at:

- manifest status counts
- worst shortfalls
- average local validation retention rate
- `discarded_local_validation`
- `duplicate_or_already_seen`
- cells where `expected_shortfall_risk` is true in `sample_plan`

If many subfields are below 2,500 valid works, consider increasing `oversample_factor` or `max_backfill_rounds` in `config.yaml`.

## Later Embedding Size Estimate

Do not compute embeddings now, but the expected storage later is approximately:

```text
762,000 x 768 x float32 ~= 2.34 GB
762,000 x 768 x float16 ~= 1.17 GB
```
