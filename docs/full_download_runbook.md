# Full Download Runbook

This runbook is for the production-scale OpenAlex title and abstract corpus download.

The target is roughly:

```text
~254 subfields x 3,000 works = ~762,000 works
```

The final valid count may be lower if some subfields do not have enough locally valid title and abstract works.

The downloader stores compact OpenAlex topic metadata in `works_text` for later interpretation of subfield morphology. The unit of analysis remains `primary_topic.subfield.id`; topics are not used as the main unit.

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
python scripts/06_build_analysis_subfields.py
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

## Build Analysis Eligibility

After the corpus download is complete, build the subfield eligibility layer:

```bash
python scripts/06_build_analysis_subfields.py
python scripts/05_validate_database.py
```

This creates:

```text
data/processed/analysis_subfields.parquet
```

and the DuckDB table:

```text
analysis_subfields
```

The full corpus remains in `works_text`. No downloaded papers are deleted. Main morphology analysis uses subfields with `n_valid_works >= 2500`; robustness checks can use subfields with `n_valid_works >= 500`. The 2,500 threshold avoids unstable morphology metrics in very small semantic clouds while keeping the full `works_text.parquet` available for embeddings and later sensitivity checks.

## Embedding Artifacts

SPECTER2 embeddings were generated externally and should be downloaded from Drive, not regenerated as part of this repo workflow.

PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_embeddings_from_drive.ps1
python scripts/07_validate_embeddings.py
```

For the `2000_2024_400py` embedding artifacts:

```powershell
$env:LOCAL_EMBEDDINGS_DIR = "embeddings/specter2_v1_2000_2024_400py"
.\.venv\Scripts\python.exe scripts\07_validate_embeddings.py --expected-shards 119
```

Bash:

```bash
bash scripts/download_embeddings_from_drive.sh
python scripts/07_validate_embeddings.py
```

The remote folder is:

```text
gdrive:TFM/openalex_subfields/embeddings/specter2_v1_2000_2024_400py
```

The local folder is:

```text
embeddings/specter2_v1_2000_2024_400py/
```

The validator checks the expected embedding shards, metadata shards, summary
files, float16 dtype, 768 dimensions, row-count consistency, work ID coverage
against `works_text`, and eligibility joins against `analysis_subfields`. The
legacy artifact set uses 37 shards; `2000_2024_400py` uses 119 shards. It
writes `data/processed/embedding_index.parquet`, the DuckDB `embedding_index`
table, and validation files inside the configured embedding directory.

## Prepare Analysis Matrix And First Map

After embedding validation succeeds, prepare the main-analysis matrix and first sampled UMAP map:

```powershell
$env:LOCAL_EMBEDDINGS_DIR = "embeddings/specter2_v1_2000_2024_400py"
.\.venv\Scripts\python.exe scripts\08_prepare_analysis_matrix.py --force
.\.venv\Scripts\python.exe scripts\09_build_global_umap_visualization.py `
  --sample-per-subfield 500 `
  --year-min 2000 `
  --year-max 2024 `
  --force
```

The matrix uses only rows where `main_analysis_eligible == true`. The first map uses a balanced sample per subfield for visual inspection.

Downstream matrix consumers also accept explicit embedding directories:

```powershell
.\.venv\Scripts\python.exe scripts\10_build_subfield_umap_visualizations.py `
  --embedding-dir embeddings/specter2_v1_2000_2024_400py `
  --year-min 2000 `
  --year-max 2024 `
  --limit-subfields 3 `
  --overwrite
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
- `data/processed/analysis_subfields.parquet`
- `data/processed/embedding_index.parquet`
- `data/processed/analysis_embedding_index.parquet`
- `warehouse/tfm_openalex.duckdb`
- `data/interim/validation_report.md`
- `data/interim/validation_summary.json`
- `outputs/08_visualization/global_umap/umap_global_sample.parquet`
- `outputs/08_visualization/global_umap/umap_global_sample.png`
- `outputs/08_visualization/global_umap/umap_global_sample_summary.json`

## Shortfall Guidance

Small shortfalls are acceptable. Inspect the validation report if many subfields fall below 3,000 valid works.

Look first at:

- manifest status counts
- worst shortfalls
- average local validation retention rate
- `discarded_local_validation`
- `duplicate_or_already_seen`
- cells where `expected_shortfall_risk` is true in `sample_plan`

For the completed production corpus, do not retry shortfalls just to force every subfield above 2,500 works. Use `analysis_subfields` to keep the main analysis stable and preserve lower-sample subfields for robustness checks when they have at least 500 valid works.

## Embedding Size

The local SPECTER2 artifact set is about 1 GiB. For reference, dense storage is approximately:

```text
762,000 x 768 x float32 ~= 2.34 GB
762,000 x 768 x float16 ~= 1.17 GB
```
