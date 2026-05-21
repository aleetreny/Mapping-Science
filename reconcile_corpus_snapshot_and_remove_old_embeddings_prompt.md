# Agent Prompt — Reconcile Corpus Snapshot and Remove Obsolete Embeddings

Reconcile the corpus snapshot before thesis writing and remove all traces of the old first embedding version.

## Current issue

There is a mismatch between corpus validation outputs and the current processed Parquet files:

- validation-backed outputs report `2,358,036` works and `250` retained subfields;
- current `data/processed/works_text_2000_2024_400py.parquet` reports `2,378,036` works and `252` subfields.

There is also an obsolete first embedding version from an earlier experiment with around 700k papers, probably under:

```text
embeddings/specter2_v1/
```

The active thesis version is the 2000–2024 corpus with more than 2 million papers, under something like:

```text
embeddings/specter2_v1_2000_2024_400py/
```

## Required tasks

### 1. Determine the authoritative corpus snapshot

Determine which corpus snapshot was actually used to generate:

- `data/processed/analysis_embedding_index.parquet`;
- the active SPECTER2 embedding matrix;
- `outputs/03_embedding_metrics/`;
- `outputs/04_reduced_metric_core/`;
- `outputs/05_static_comparison/`;
- `outputs/06_temporal_evolution/`;
- `outputs/07_morphological_similarity/`.

Choose the correct thesis snapshot:

- If downstream embeddings/metrics use `2,378,036` works and `252` subfields, rerun or update corpus validation summaries and thesis tables to match this snapshot.
- If downstream embeddings/metrics use `2,358,036` works and `250` subfields, regenerate or filter processed Parquet files so they match validation.
- Do **not** leave both snapshots active.

### 2. Synchronize corpus, embeddings, metrics, and thesis summaries

After deciding the authoritative snapshot, ensure that the following are mutually consistent:

- processed corpus files;
- analysis embedding index;
- active embedding matrix;
- embedding metrics;
- reduced metric core;
- static comparison outputs;
- temporal evolution outputs;
- morphological similarity outputs;
- thesis-ready corpus summary and tables.

Update as needed:

```text
research/final_outputs/corpus_summary_for_thesis.md
memory/tables/tab_03_corpus_summary.tex
memory/tables/tab_03_domain_distribution.tex
```

Also update or remove any validation warning text that is no longer true after synchronization.

### 3. Remove or archive the obsolete first embedding version

Identify the obsolete old embedding version, probably:

```text
embeddings/specter2_v1/
```

This corresponds to the earlier experimental corpus of around 700k papers and is no longer part of the thesis.

Required actions:

1. Confirm that `embeddings/specter2_v1/` or any equivalent old 700k-paper embedding directory is not referenced by active scripts, docs, configs, or outputs.
2. Remove it locally if safe, or move it to a clearly marked legacy/archive location outside the active thesis path.
3. Do **not** delete or modify the active embedding directory:

```text
embeddings/specter2_v1_2000_2024_400py/
```

### 4. Remove old references from the repository

Search the whole repo for old or obsolete references, including:

```text
specter2_v1
700k
old embedding
first embedding
2010_2025
old corpus
legacy embedding
```

Also search for outdated embedding paths or documentation that suggests the first embedding version is still active.

Update all active documentation so that only the current thesis version is presented as active:

- active corpus: `2000_2024_400py`;
- active embedding path: `embeddings/specter2_v1_2000_2024_400py/`;
- active corpus size: whichever synchronized snapshot is authoritative after reconciliation.

Old corpus or embedding versions should be either removed or clearly marked as legacy.

### 5. Update documentation

Update, at minimum, if relevant:

```text
README.md
docs/embedding_pipeline.md
docs/data_corpus.md
docs/output_organization.md
research/final_outputs/corpus_summary_for_thesis.md
memory/tables/tab_03_corpus_summary.tex
memory/tables/tab_03_domain_distribution.tex
```

Also update any other docs that still point to old paths, old corpus versions, or obsolete corpus sizes.

### 6. Run checks

After cleanup, run relevant validation/import/smoke checks.

At minimum:

- verify active scripts still import;
- verify the active embedding path exists;
- verify row counts match between corpus, analysis index, embedding matrix, and metrics;
- verify thesis summary tables match the authoritative snapshot;
- verify no active documentation points to the old 700k embedding version.

## Final output

Report clearly:

1. Which corpus snapshot is authoritative for the thesis.
2. Why that snapshot was selected.
3. Whether the active embedding matrix matches that snapshot.
4. Whether downstream metrics and summaries match that snapshot.
5. What happened to the obsolete `embeddings/specter2_v1/` directory.
6. Which old references were removed or archived.
7. Which files were regenerated or updated.
8. Confirmation that corpus, embeddings, metrics, outputs, and thesis tables are synchronized.

Do not proceed to writing the thesis chapter until the snapshot mismatch and old embedding-version traces are resolved.
