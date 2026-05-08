# OpenAlex Subfield Morphology and Scientific Growth Prediction

The full thesis context is summarized in [research/README.md](research/README.md). This codebase is currently implementing the data collection and database organization phase only.

The project builds a clean OpenAlex database for later testing whether the title and abstract structure of a subfield before 2020 helps predict whether that subfield grows during 2020-2025.

The current pipeline prepares SPECTER2 embedding artifacts, UMAP visual inspection maps, a tabular morphology-metrics dataset, and pre-prediction morphology diagnostics. It still does not add paper clustering, regression models, prediction models, growth-target joins, dashboards, or ML infrastructure.

## Current Data Design

- Unit of analysis: OpenAlex subfield
- Morphology window: 2010-2019
- Growth window: 2020-2025
- Excluded year: 2026, because it is the current incomplete year
- Text source: title plus abstract
- OpenAlex classification: `primary_topic.subfield.id`
- Topic metadata: stored only for later interpretation, not as the unit of analysis
- Sample target: 3,000 papers per eligible subfield
- Main analysis threshold: `n_valid_works >= 2500`
- Robustness threshold: `n_valid_works >= 500`
- Storage: one DuckDB database at `warehouse/tfm_openalex.duckdb`

## Why Subfields, Not Topics

OpenAlex topics are too numerous for the first stable thesis pipeline. Subfields are broad enough to form meaningful semantic spaces and give roughly a few hundred units, which is more realistic for a master thesis.

The pipeline therefore uses `primary_topic.subfield.id` as the main classification. It does not use `topics.subfield.id` as the active unit of analysis.

The downloaded corpus also stores compact topic metadata from OpenAlex so future morphology work can interpret semantic regions without re-querying hundreds of thousands of works. Topics remain contextual metadata, not the main unit.

## Why Not Download Everything

Downloading all OpenAlex works for all subfields would be too large and unnecessary. A capped, stratified sample gives comparable morphology inputs across subfields while keeping storage and later embedding work manageable.

The corpus download uses OpenAlex's `sample` and `seed` parameters for large subfield-year cells. Small cells are downloaded in full. The production downloader oversamples raw API results, backfills shortfalls with new seeds, writes a manifest, and can resume after interruption. This avoids downloading all candidates first and then sampling locally.

Growth targets are based primarily on article and preprint counts, not only abstract-available works, because growth should measure scientific production rather than abstract availability.

## Analysis Eligibility

The full downloaded corpus remains available in `data/processed/works_text.parquet` and in the DuckDB `works_text` table. No downloaded papers are deleted by the analysis eligibility step.

Main morphology analysis should use subfields where `n_valid_works >= 2500`. Robustness checks can also use subfields where `n_valid_works >= 500`. The 2,500-work threshold is intended to avoid unstable morphology metrics in very small semantic clouds while preserving the full corpus for embeddings, diagnostics, and later sensitivity checks.

The eligibility layer is stored in `data/processed/analysis_subfields.parquet` and the DuckDB `analysis_subfields` table. Embedding artifacts are kept for the full `works_text.parquet`; later morphology scripts can join against `analysis_subfields` and select `main_analysis_eligible_2500 == true`.

## Embedding Artifacts

SPECTER2 embeddings live outside Git in `embeddings/specter2_v1/` and on Drive at `gdrive:TFM/openalex_subfields/embeddings/specter2_v1`. The local folder contains 37 float16, 768-dimensional embedding shards plus matching metadata and summary files.

Download and validate on Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_embeddings_from_drive.ps1
python scripts/07_validate_embeddings.py
```

Or with Bash:

```bash
bash scripts/download_embeddings_from_drive.sh
python scripts/07_validate_embeddings.py
```

Validation writes `data/processed/embedding_index.parquet` and the DuckDB `embedding_index` table. The index maps each `work_id` to a shard and row position, then joins subfield metadata and analysis flags. It does not store embedding vectors.

See [docs/embedding_data_model.md](docs/embedding_data_model.md) for shard loading and join examples.

## Analysis Matrix And First UMAP

Prepare the main-analysis matrix, first sampled UMAP map, per-subfield visual
inspection maps, morphology metrics, and metric diagnostics with:

```bash
python scripts/08_prepare_analysis_matrix.py
python scripts/09_build_first_umap_maps.py --sample-per-subfield 500
python scripts/10_build_per_subfield_umap_maps.py --limit-subfields 3 --max-papers-per-subfield 2000 --overwrite
python scripts/11_compute_subfield_morphology_metrics.py --limit-subfields 3 --overwrite
python scripts/12_analyze_morphology_metrics.py --limit-subfields 20 --overwrite
```

The matrix uses only `main_analysis_eligible_2500 == true` rows and preserves deterministic order by `subfield_id`, `publication_year`, and `work_id`. The UMAP script uses a balanced per-subfield sample for first visual inspection only.

The per-subfield UMAP stage fits one separate UMAP model per OpenAlex subfield,
using only the morphology input window by default (`2010 <= publication_year <= 2019`).
It writes one coordinate parquet and one scatter+density PNG per attempted
subfield, plus a manifest and summary. It does not add PCA, clustering, or
predictive models.

See [docs/analysis_matrix_and_first_umap.md](docs/analysis_matrix_and_first_umap.md)
and [docs/per_subfield_umap_maps.md](docs/per_subfield_umap_maps.md) for UMAP
outputs and options.

The morphology stage consumes the per-subfield coordinate parquets and writes
one row per completed subfield map with 25 curated core v2 normalized-coordinate
metrics plus diagnostic metrics. OpenAlex `subfield_display_name` values are not
unique, so outputs include `subfield_label_unique` and use `subfield_id` as the
primary key. The core v2 set includes `density_entropy_slope_by_year`, a
2010-2019-only temporal semantic-diversification metric; sparse outlier shares
such as `outlier_share_r_gt_1_5` remain diagnostics rather than modeling
features. See
[docs/subfield_morphology_metrics.md](docs/subfield_morphology_metrics.md).

The morphology analysis stage writes distributions, correlations, low-variance
checks, duplicate-name reports, family scores, domain/field profiles,
rankings, rule-based archetypes, case-study candidates, a Markdown report, a
figure index, and exploratory metric-table PCA diagnostics. Its canonical
outputs are organized under
`outputs/metrics/morphology_analysis/tables/` and
`outputs/metrics/morphology_analysis/figures/`, with figure subfolders for
quality, distributions, correlations, family scores, domain/field profiles,
PCA, rankings, and the case-study atlas. This PCA is applied only to the final
metric table, not to SPECTER2 embeddings before UMAP. Stage 12 is descriptive:
it does not join growth targets and does not estimate prediction models.

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
python scripts/06_build_analysis_subfields.py
python scripts/05_validate_database.py
python scripts/07_validate_embeddings.py
python scripts/08_prepare_analysis_matrix.py
python scripts/09_build_first_umap_maps.py --sample-per-subfield 500
python scripts/10_build_per_subfield_umap_maps.py --limit-subfields 3 --max-papers-per-subfield 2000 --overwrite
python scripts/11_compute_subfield_morphology_metrics.py --limit-subfields 3 --overwrite
python scripts/12_analyze_morphology_metrics.py --limit-subfields 20 --overwrite
```

The full corpus download may take time. Test first with `--limit-subfields 5`, then use the production runbook in [docs/full_download_runbook.md](docs/full_download_runbook.md).

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
data/interim/download_manifest.parquet
data/processed/works_text.parquet
data/processed/analysis_subfields.parquet
data/processed/embedding_index.parquet
data/processed/analysis_embedding_index.parquet
data/interim/validation_report.md
data/interim/validation_summary.json
outputs/maps/umap_global_sample.parquet
outputs/maps/umap_global_sample.png
outputs/maps/umap_global_sample_summary.json
outputs/maps/per_subfield_umap/coordinates/*.parquet
outputs/maps/per_subfield_umap/figures/*.png
outputs/maps/per_subfield_umap/per_subfield_umap_manifest.parquet
outputs/maps/per_subfield_umap/per_subfield_umap_summary.json
data/processed/subfield_morphology_metrics.parquet
data/processed/subfield_morphology_metrics.csv
outputs/metrics/subfield_morphology_metrics_summary.json
outputs/metrics/subfield_morphology_metrics_dictionary.csv
outputs/metrics/duplicate_subfield_names_report.csv
outputs/metrics/morphology_analysis/
outputs/metrics/morphology_analysis/tables/
outputs/metrics/morphology_analysis/figures/
outputs/metrics/morphology_analysis/morphology_analysis_report.md
outputs/metrics/morphology_analysis/morphology_analysis_figure_index.csv
```

Data files, secrets, and large artifacts are ignored by Git.
