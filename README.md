# Scientific Morphology of Research Fields

This repository contains the final thesis snapshot for a project on the
semantic morphology of scientific disciplines. It asks a simple question:
when papers are represented as SPECTER2 document embeddings, do fields of
science differ only by topic, or also by shape?

Explore the interactive project viewer:

https://aleetreny.github.io/Mapping-Science/

## What Is Here

- `memory/`: LaTeX source, final thesis figures, tables, and appendices.
- `scripts/`: core data, embedding, metric, visualization, and typology scripts.
- `src/`: reusable analysis code used by the pipeline.
- `docs/`: compact methodological notes from the project development.

Large local artifacts such as raw data, processed matrices, embeddings, and
generated output folders are intentionally kept out of Git.

## Thesis Snapshot

The analysis uses a balanced OpenAlex title-abstract corpus from 2000 to 2024,
represented with SPECTER2 embeddings. Morphology is measured in the original
embedding space through interpretable metrics for dispersion, local density,
hubness, spectral structure, temporal movement, and field similarity.

UMAP appears only as a visualization layer. Quantitative claims are based on
the original embedding-space metrics, not on low-dimensional projections.

## Build

From `memory/`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error memoria.tex
```

Run twice after structural edits so references, figure lists, and appendix
letters settle.

## Data Note

The repository is designed as a readable final project snapshot, not as a
packaged data release. The corpus, embedding shards, DuckDB files, Parquet
tables, and generated outputs are excluded because they are large local
artifacts.
