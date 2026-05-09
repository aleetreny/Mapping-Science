# Thesis Context

## Current Thesis Question

How can the morphology of scientific subfields be measured and compared using
scientific text embeddings, projected UMAP landscapes, and original
embedding-space structure?

The project studies OpenAlex subfields as semantic spaces. Subfields may differ
in concentration, dispersion, fragmentation, local density, dimensionality,
support shape, and temporal movement. The active pipeline is designed to make
those differences measurable and interpretable.

## Current Scope

- Unit of analysis: OpenAlex subfield
- Analysis period: 2010-2025
- Excluded year: 2026, because it is incomplete/current
- Text source: title plus abstract
- Embedding model: SPECTER2
- Main analysis threshold: at least 2,500 valid downloaded works
- Robustness threshold: at least 500 valid downloaded works

The active project is not a prediction pipeline. It does not define future
targets or train regression/classification models.

## Metric Families

The active metric design has two complementary blocks:

1. Projected UMAP morphology metrics.
   These are computed from separately fitted 2D UMAP coordinates for each
   subfield and describe visible semantic landscape shape.

2. Embedding-space structure metrics.
   These are computed directly from L2-normalized SPECTER2 vectors and describe
   high-dimensional semantic structure.

Each family exposes 25 curated core metrics. This creates an interpretable,
balanced design for comparison. It does not mean the two families automatically
have equal statistical weight; later analysis should standardize features and
may use block weighting.

## Pipeline Phases

Phase 1: OpenAlex corpus and counts.

- fetch taxonomy
- build yearly count tables for 2010-2025
- build a corpus plan for the active analysis period
- build and download a sampled title-plus-abstract corpus
- validate the local DuckDB and Parquet data

Phase 2: Embeddings.

- validate SPECTER2 shards and metadata
- build a row-aligned main-analysis embedding matrix

Phase 3: Projected morphology.

- build a first global balanced-sample UMAP map
- build one per-subfield UMAP map for the 2010-2025 analysis period
- compute the 25 projected morphology metrics

Phase 4: Embedding-space structure.

- compute the 25 direct embedding-space metrics from the same analysis matrix
- keep controls and diagnostics separate from core metrics

Phase 5, later: interpretation and comparison.

- compare projected and embedding-space metric families
- inspect robust patterns by field/domain
- run exploratory metric-space clustering once both metric tables are stable
- decide whether feature standardization, block weighting, or sensitivity checks
  are needed

Do not treat exploratory clusters as final thesis claims before the two metric
tables, diagnostics, and sensitivity checks are stable and documented.
