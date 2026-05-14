# OpenAlex Subfield Morphology Metrics

The full thesis context is summarized in [research/README.md](research/README.md).

This repository builds a reproducible pipeline for measuring and comparing the
morphology of OpenAlex scientific subfields. It uses title plus abstract text,
SPECTER2 embeddings, per-subfield UMAP landscapes, and direct embedding-space
structure metrics.

The active project is descriptive and comparative. It does not build growth
targets, regression datasets, classifiers, dashboards, or predictive models.

## Current Data Design

- Unit of analysis: OpenAlex subfield
- Current versioned analysis period: 2000-2024
- Legacy supported analysis period: 2010-2025, excluding incomplete/current 2026
- Text source: title plus abstract
- Embedding model: SPECTER2
- OpenAlex classification: `primary_topic.subfield.id`
- Topic metadata: stored for interpretation, not used as the unit of analysis
- Sample target: 3,000 papers per eligible subfield
- Main analysis threshold: `n_valid_works >= 2500`
- Robustness threshold: `n_valid_works >= 500`
- Storage: one DuckDB database at `warehouse/tfm_openalex.duckdb`

## Why Subfields, Not Topics

OpenAlex topics are too numerous for the first stable thesis pipeline.
Subfields are broad enough to form meaningful semantic spaces and give a few
hundred units, which is realistic for a master thesis.

The pipeline therefore uses `primary_topic.subfield.id` as the main
classification. Downloaded works also store compact topic metadata so semantic
regions can be interpreted later without re-querying OpenAlex.

## Why Not Download Everything

Downloading all OpenAlex works for all subfields would be too large for this
project. A capped, stratified sample gives comparable morphology inputs across
subfields while keeping storage and embedding work manageable.

The corpus downloader uses OpenAlex's `sample` and `seed` parameters for large
subfield-year cells. Small cells are downloaded in full. The production
downloader oversamples raw API results, backfills shortfalls with new seeds,
writes a manifest, and can resume after interruption.

## Extended OpenAlex Extraction: 2000-2024, 400/year/subfield

A new versioned extraction target is available as `2000_2024_400py`. It is a
clean annual-resolution OpenAlex text corpus for 2000-2024 inclusive, targeting
400 valid title-plus-abstract papers per year per subfield. The full-period
target is 10,000 papers per subfield and the period naturally supports later
5-year temporal morphology checks.

This version does not overwrite the old checkpoint files, does not activate
canonical `works_text.parquet`, and does not include embeddings yet. Run the
versioned extraction before starting a new Kaggle embedding job:

```bash
python scripts/01_build_counts.py --dataset-version 2000_2024_400py --dry-run
python scripts/01_build_counts.py --dataset-version 2000_2024_400py
python scripts/02_build_corpus_plan.py --dataset-version 2000_2024_400py
python scripts/03_build_sample_plan.py --dataset-version 2000_2024_400py
python scripts/04_download_sampled_corpus.py --dataset-version 2000_2024_400py --dry-run
python scripts/04_download_sampled_corpus.py --dataset-version 2000_2024_400py --write-every-n-subfields 1
python scripts/05_validate_database.py --dataset-version 2000_2024_400py
python scripts/06_build_analysis_subfields.py --dataset-version 2000_2024_400py
```

Main versioned outputs include:

```text
data/interim/subfield_year_counts_2000_2024_400py.parquet
data/interim/corpus_plan_2000_2024_400py.parquet
data/interim/sample_plan_2000_2024_400py.parquet
data/interim/download_manifest_2000_2024_400py.parquet
data/processed/works_text_2000_2024_400py.parquet
data/processed/analysis_subfields_2000_2024_400py.parquet
outputs/validation/validation_report_2000_2024_400py.md
```

See
[docs/openalex_extraction_2000_2024_400py.md](docs/openalex_extraction_2000_2024_400py.md)
for the full runbook, preserved filters, 429 cooldown/resume guidance, output
paths, and Drive handoff.

## Analysis Eligibility

The full downloaded corpus remains available in
`data/processed/works_text.parquet` and in the DuckDB `works_text` table. No
downloaded papers are deleted by the analysis eligibility step.

The eligibility layer is stored in
`data/processed/analysis_subfields.parquet` and the DuckDB
`analysis_subfields` table. Main morphology analysis uses subfields where
`n_valid_works >= 2500`; robustness checks can use subfields where
`n_valid_works >= 500`.

## Embedding Artifacts

SPECTER2 embeddings live outside Git in `embeddings/specter2_v1/` and on Drive
at `gdrive:TFM/openalex_subfields/embeddings/specter2_v1`. The local folder
contains float16, 768-dimensional embedding shards plus matching metadata and
summary files.

Download and validate on Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_embeddings_from_drive.ps1
python scripts/07_validate_embeddings.py
```

For the current `2000_2024_400py` embedding artifacts, point the downstream
matrix stages at the versioned directory:

```powershell
$env:LOCAL_EMBEDDINGS_DIR = "embeddings/specter2_v1_2000_2024_400py"
.\.venv\Scripts\python.exe scripts\07_validate_embeddings.py --expected-shards 119
```

Or with Bash:

```bash
bash scripts/download_embeddings_from_drive.sh
python scripts/07_validate_embeddings.py
```

Validation writes `data/processed/embedding_index.parquet` and the DuckDB
`embedding_index` table. See
[docs/embedding_data_model.md](docs/embedding_data_model.md).

## Active Pipeline

Prepare the row-aligned analysis matrix, UMAP maps, projected morphology
metrics, embedding-space metrics, diagnostics, and exploratory clusters with:

```powershell
$env:LOCAL_EMBEDDINGS_DIR = "embeddings/specter2_v1_2000_2024_400py"
.\.venv\Scripts\python.exe scripts\07_validate_embeddings.py --expected-shards 119
.\.venv\Scripts\python.exe scripts\08_prepare_analysis_matrix.py --force

.\.venv\Scripts\python.exe scripts\09_build_first_umap_maps.py --sample-per-subfield 500 --year-min 2000 --year-max 2024 --force
.\.venv\Scripts\python.exe scripts\10_build_per_subfield_umap_maps.py --year-min 2000 --year-max 2024 --overwrite
.\.venv\Scripts\python.exe scripts\10b_build_per_field_umap_maps.py --year-min 2000 --year-max 2024 --overwrite
.\.venv\Scripts\python.exe scripts\10c_build_per_domain_umap_maps.py --year-min 2000 --year-max 2024 --overwrite

.\.venv\Scripts\python.exe scripts\11_compute_subfield_morphology_metrics.py --year-min 2000 --year-max 2024 --overwrite
.\.venv\Scripts\python.exe scripts\12_compute_subfield_embedding_space_metrics.py --year-min 2000 --year-max 2024 --overwrite
.\.venv\Scripts\python.exe scripts\13_compare_metric_families.py --overwrite
.\.venv\Scripts\python.exe scripts\14_summarize_metric_distributions.py --overwrite
.\.venv\Scripts\python.exe scripts\16_build_clean_embedding_core_diagnostics.py --overwrite
.\.venv\Scripts\python.exe scripts\16b_build_reduced_interpretable_embedding_core.py --overwrite
.\.venv\Scripts\python.exe scripts\15_cluster_metric_spaces.py --default-k 5 --overwrite
```

The per-subfield UMAP stage fits one separate 2D UMAP model per OpenAlex
subfield. It may cap papers per subfield for runtime, but records
`n_available`, `n_used`, and `sampling_applied` in the manifest.
Field and domain UMAPs are supporting inspection outputs only; subfields remain
the main analysis unit.

The active analysis produces two complementary metric tables:

- Projected morphology metrics, computed from normalized 2D per-subfield UMAP
  coordinates.
- Embedding-space structure metrics, computed directly from L2-normalized
  SPECTER2 vectors.

The projected UMAP table exposes 25 curated core metrics; the direct
embedding-space table now exposes 26 curated core metrics after adding
spectral entropy, kNN hubness, and recent novelty while moving low-information
graph connectivity checks out of core. This is an interpretation choice, not a
claim of equal statistical weight. Later comparative analysis should
standardize features and may use block weighting if needed.

## Metric Outputs

Projected UMAP morphology metrics:

```text
data/processed/subfield_morphology_metrics.parquet
data/processed/subfield_morphology_metrics.csv
outputs/metrics/subfield_morphology_metrics_summary.json
outputs/metrics/subfield_morphology_metrics_dictionary.csv
```

Embedding-space structure metrics:

```text
data/processed/subfield_embedding_space_metrics.parquet
data/processed/subfield_embedding_space_metrics.csv
outputs/metrics/subfield_embedding_space_metrics_summary.json
outputs/metrics/subfield_embedding_space_metrics_dictionary.csv
outputs/analysis/embedding_space_metric_diagnostics/*
```

See [docs/subfield_morphology_metrics.md](docs/subfield_morphology_metrics.md)
and
[docs/subfield_embedding_space_metrics.md](docs/subfield_embedding_space_metrics.md).
Metric-family comparison and distribution diagnostics are documented in
[docs/metric_family_comparison.md](docs/metric_family_comparison.md) and
[docs/metric_distribution_diagnostics.md](docs/metric_distribution_diagnostics.md).
Exploratory metric-space clustering is documented in
[docs/metric_clustering.md](docs/metric_clustering.md).

## Repository Layout

```text
src/        reusable helpers
scripts/    numbered pipeline steps
docs/       data model and stage documentation
research/   thesis framing for future agents/readers
data/       ignored local data folders
warehouse/  ignored DuckDB database location
tests/      unit and smoke tests
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Set `OPENALEX_EMAIL` in `.env` if possible. `OPENALEX_API_KEY` may be left
empty.

## Full Run

```powershell
python scripts/00_fetch_taxonomy.py
python scripts/01_build_counts.py --dry-run
python scripts/01_build_counts.py
python scripts/02_build_corpus_plan.py
python scripts/03_build_sample_plan.py
python scripts/04_download_sampled_corpus.py --dry-run
python scripts/04_download_sampled_corpus.py --limit-subfields 5
python scripts/06_build_analysis_subfields.py
python scripts/05_validate_database.py
$env:LOCAL_EMBEDDINGS_DIR = "embeddings/specter2_v1_2000_2024_400py"
.\.venv\Scripts\python.exe scripts\07_validate_embeddings.py --expected-shards 119
.\.venv\Scripts\python.exe scripts\08_prepare_analysis_matrix.py --force
.\.venv\Scripts\python.exe scripts\09_build_first_umap_maps.py --sample-per-subfield 500 --year-min 2000 --year-max 2024 --force
.\.venv\Scripts\python.exe scripts\10_build_per_subfield_umap_maps.py --limit-subfields 3 --year-min 2000 --year-max 2024 --max-papers-per-subfield 2000 --overwrite
.\.venv\Scripts\python.exe scripts\10b_build_per_field_umap_maps.py --limit-fields 3 --year-min 2000 --year-max 2024 --max-papers-per-group 2000 --overwrite
.\.venv\Scripts\python.exe scripts\10c_build_per_domain_umap_maps.py --limit-domains 2 --year-min 2000 --year-max 2024 --max-papers-per-group 2000 --overwrite
.\.venv\Scripts\python.exe scripts\11_compute_subfield_morphology_metrics.py --limit-subfields 3 --year-min 2000 --year-max 2024 --overwrite
.\.venv\Scripts\python.exe scripts\12_compute_subfield_embedding_space_metrics.py --limit-subfields 3 --year-min 2000 --year-max 2024 --overwrite
python scripts/13_compare_metric_families.py --overwrite
python scripts/14_summarize_metric_distributions.py --overwrite
python scripts/15_cluster_metric_spaces.py --default-k 5 --overwrite
```

The full corpus download may take time. Test first with `--limit-subfields 5`,
then use the production runbook in
[docs/full_download_runbook.md](docs/full_download_runbook.md).

## Key Outputs

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
data/interim/download_manifest.parquet
data/processed/works_text.parquet
data/processed/analysis_subfields.parquet
data/processed/embedding_index.parquet
data/processed/analysis_embedding_index.parquet
outputs/maps/umap_global_sample.parquet
outputs/maps/umap_global_sample.png
outputs/maps/umap_global_sample_summary.json
outputs/maps/per_subfield_umap/coordinates/*.parquet
outputs/maps/per_subfield_umap/figures/*.png
outputs/maps/per_subfield_umap/per_subfield_umap_manifest.parquet
outputs/maps/per_subfield_umap/per_subfield_umap_summary.json
outputs/maps/per_field_umap/coordinates/*.parquet
outputs/maps/per_field_umap/figures/*.png
outputs/maps/per_field_umap/per_field_umap_manifest.parquet
outputs/maps/per_field_umap/per_field_umap_summary.json
outputs/maps/per_domain_umap/coordinates/*.parquet
outputs/maps/per_domain_umap/figures/*.png
outputs/maps/per_domain_umap/per_domain_umap_manifest.parquet
outputs/maps/per_domain_umap/per_domain_umap_summary.json
data/processed/subfield_morphology_metrics.parquet
data/processed/subfield_morphology_metrics.csv
data/processed/subfield_embedding_space_metrics.parquet
data/processed/subfield_embedding_space_metrics.csv
outputs/metrics/subfield_morphology_metrics_summary.json
outputs/metrics/subfield_morphology_metrics_dictionary.csv
outputs/metrics/subfield_embedding_space_metrics_summary.json
outputs/metrics/subfield_embedding_space_metrics_dictionary.csv
outputs/metrics/duplicate_subfield_names_report.csv
outputs/analysis/metric_family_comparison/*
outputs/analysis/embedding_space_metric_diagnostics/*
outputs/analysis/clean_embedding_core_metrics/*
outputs/analysis/reduced_interpretable_embedding_core/*
outputs/analysis/metric_distributions/*
outputs/analysis/metric_clustering/*
```

Data files, secrets, and large artifacts are ignored by Git.
