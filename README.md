# TFM: Scientific Discipline Morphology in SPECTER2 Space

This repository builds a reproducible, descriptive pipeline for comparing the
morphology and evolution of OpenAlex scientific disciplines. The main geometry
is the original SPECTER2 embedding space built from title plus abstract text.

UMAP remains in the project only as a visualization layer. UMAP-derived
morphology metrics, growth prediction, classification, dashboards, and mixed
UMAP-plus-embedding clustering are archived legacy work, not active thesis
evidence.

## Current Thesis Direction

The active question is:

> How do scientific subfields, fields, and domains differ in their semantic
> structure and evolution when measured directly in SPECTER2 embedding space?

The active pipeline is:

```text
OpenAlex corpus
-> title + abstract text
-> SPECTER2 embeddings
-> row-aligned analysis matrix
-> embedding-space metrics
-> reduced 11-metric interpretable core
-> static discipline comparison
-> temporal evolution
-> morphological convergence/divergence
-> visualizations
```

## Data Design

- Unit of analysis: OpenAlex subfields, with aggregation to fields and domains.
- Current extraction target: `2000_2024_400py`.
- Period: 2000-2024 inclusive.
- Sampling: balanced annual target of 400 valid works per year per subfield.
- Active corpus snapshot: 2,378,036 retained works across 252 subfields.
- Main analysis subset: 2,344,927 embedded works across 241 analysis subfields.
- Text source: title plus abstract.
- Embedding model: SPECTER2.
- Storage: DuckDB and Parquet under ignored local data folders.
- Main local embedding directory: `embeddings/specter2_v1_2000_2024_400py`.

The obsolete unversioned first SPECTER2 embedding set is not part of the active
thesis path. When present locally, it is archived under
`archive/legacy_embeddings/specter2_v1_legacy_700k/`.

## Reduced Metric Core

The final thesis metric set is the reduced 11-metric embedding-space core:

```text
embedding_distance_to_centroid_median
embedding_distance_to_centroid_iqr
embedding_distance_to_centroid_p90
embedding_knn_median_distance
embedding_knn_distance_cv
embedding_knn_indegree_gini
embedding_pca_dim_80
embedding_pca_spectral_entropy
embedding_centroid_drift_early_late
embedding_radial_expansion_slope
embedding_recent_novelty_score
```

These metrics cover global semantic dispersion, local density and hubness,
intrinsic dimensionality, and temporal semantic movement.

## Active Pipeline Commands

Corpus construction:

```powershell
python scripts/00_fetch_taxonomy.py
python scripts/01_build_counts.py --dataset-version 2000_2024_400py
python scripts/02_build_corpus_plan.py --dataset-version 2000_2024_400py
python scripts/03_build_sample_plan.py --dataset-version 2000_2024_400py
python scripts/04_download_sampled_corpus.py --dataset-version 2000_2024_400py
python scripts/05_validate_database.py --dataset-version 2000_2024_400py
python scripts/06_build_analysis_subfields.py --dataset-version 2000_2024_400py
```

Embedding matrix and metrics:

```powershell
$env:LOCAL_EMBEDDINGS_DIR = "embeddings/specter2_v1_2000_2024_400py"
.\.venv\Scripts\python.exe scripts\07_validate_embeddings.py --expected-shards 119
.\.venv\Scripts\python.exe scripts\08_prepare_analysis_matrix.py --force
.\.venv\Scripts\python.exe scripts\11_compute_embedding_space_metrics.py --year-min 2000 --year-max 2024 --overwrite
.\.venv\Scripts\python.exe scripts\12_build_reduced_metric_core.py --overwrite
```

Downstream comparison:

```powershell
.\.venv\Scripts\python.exe scripts\13_analyze_static_discipline_profiles.py --overwrite
.\.venv\Scripts\python.exe scripts\14_compute_temporal_metric_evolution.py --overwrite
.\.venv\Scripts\python.exe scripts\15_compute_morphological_similarity_evolution.py --overwrite
```

Auxiliary visualization:

```powershell
.\.venv\Scripts\python.exe scripts\09_build_global_umap_visualization.py --sample-per-subfield 500 --year-min 2000 --year-max 2024 --force
.\.venv\Scripts\python.exe scripts\10_build_subfield_umap_visualizations.py --year-min 2000 --year-max 2024 --overwrite
.\.venv\Scripts\python.exe scripts\10b_build_field_umap_visualizations.py --year-min 2000 --year-max 2024 --overwrite
.\.venv\Scripts\python.exe scripts\10c_build_domain_umap_visualizations.py --year-min 2000 --year-max 2024 --overwrite
.\.venv\Scripts\python.exe scripts\16_build_temporal_centroid_visualizations.py --overwrite
.\.venv\Scripts\python.exe scripts\17_build_temporal_umap_visualizations.py --overwrite
```

## Output Organization

Generated artifacts are ignored by Git and organized by active pipeline stage:

```text
outputs/
  01_corpus_construction/
  02_embedding_matrix/
  03_embedding_metrics/
  04_reduced_metric_core/
  05_static_comparison/
  06_temporal_evolution/
  07_morphological_similarity/
  08_visualization/
    per_subfield_umap_smooth_density/
    per_field_umap_smooth_density/
    per_domain_umap_smooth_density/
  archive_or_legacy/
```

`outputs/02_embedding_matrix/` contains only lightweight summary and row alignment diagnostics (`.json`), not the large matrix itself. Core data tables remain in `data/processed/`, including the analysis embedding
index, full embedding metric table, reduced temporal tables, and pairwise
morphological distance tables.

## Documentation

- [Corpus Construction](docs/data_corpus.md)
- [Embedding Pipeline](docs/embedding_pipeline.md)
- [Embedding Metrics](docs/embedding_metrics.md)
- [Reduced Metric Core](docs/reduced_metric_core.md)
- [Static Comparison](docs/static_comparison.md)
- [Temporal Evolution](docs/temporal_evolution.md)
- [Morphological Similarity](docs/morphological_similarity.md)
- [Visualization](docs/visualization.md)
- [Output Organization](docs/output_organization.md)

Historical exploratory work is preserved in `archive/legacy_exploration/`.

## Setup

```bash
pip install -r requirements.txt
```

Set `OPENALEX_EMAIL` in `.env` for OpenAlex politeness-pool access. SPECTER2
embedding shards live outside Git under `embeddings/`.

## Validation

Run the active unit tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

For a faster smoke check after documentation-only edits:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_reduced_interpretable_embedding_core.py tests/test_static_discipline_profiles.py tests/test_temporal_morphology_scripts.py
```
