# Corpus Construction Summary for Thesis

This summary uses the synchronized validation outputs for dataset version `2000_2024_400py`. The active thesis corpus snapshot contains 2,378,036 text-eligible OpenAlex works across 252 subfields. The downstream analysis matrix is the eligible analysis subset of this same snapshot: 2,344,927 embedded works across 241 analysis subfields.

## Headline Corpus Counts

| Measure | Value |
|---|---:|
| Dataset version | `2000_2024_400py` |
| Years covered | 2000-2024 inclusive |
| Retained works in corpus snapshot | 2,378,036 |
| Retained subfields in corpus snapshot | 252 |
| Fields represented | 26 |
| Domains represented | 4 |
| Target annual sample | up to 400 works per subfield-year |
| Maximum full-period target | 10,000 works per subfield |
| Planned works | 2,431,303 |
| Total shortfall | 53,267 |
| Validation failed | false |

## Downstream Analysis Snapshot

| Artifact | Works or rows | Subfields | Notes |
|---|---:|---:|---|
| Processed text corpus | 2,378,036 | 252 | `data/processed/works_text_2000_2024_400py.parquet` |
| Embedding index | 2,378,036 | 252 | Full embedded corpus index |
| Active SPECTER2 shard set | 2,378,036 | 252 | `embeddings/specter2_v1_2000_2024_400py/` |
| Analysis embedding index | 2,344,927 | 241 | Main metric-analysis subset |
| Active analysis matrix | 2,344,927 | 241 | `analysis/main_embeddings.float16.npy` |
| Subfield metric outputs | 241 | 241 | One metric row per analysis subfield |

## Target Sampling Design

The corpus is organized by OpenAlex primary-topic subfield. For each subfield-year cell, the target sample is up to 400 eligible works. Over 25 years, this implies a maximum of 10,000 works per subfield. The active extraction uses English `article` and `preprint` records, requires both title and abstract text, requires at least 5 title tokens and 80 abstract tokens, and excludes retracted and paratextual records. The sampling run used an oversample factor of 1.75 and up to 4 backfill rounds.

## Works by Domain

| Domain | Fields | Retained subfields | Retained works | Share of retained works |
|---|---:|---:|---:|---:|
| Life Sciences | 5 | 42 | 397,808 | 16.73% |
| Social Sciences | 6 | 58 | 542,774 | 22.82% |
| Physical Sciences | 10 | 89 | 844,382 | 35.51% |
| Health Sciences | 5 | 63 | 593,072 | 24.94% |
| **Total** | **26** | **252** | **2,378,036** | **100.00%** |

## Works by Year

| Year | Retained works | Cells below target |
|---:|---:|---:|
| 2000 | 88,474 | 69 |
| 2001 | 89,385 | 66 |
| 2002 | 90,484 | 56 |
| 2003 | 91,072 | 60 |
| 2004 | 91,778 | 51 |
| 2005 | 92,423 | 52 |
| 2006 | 93,157 | 41 |
| 2007 | 93,606 | 39 |
| 2008 | 94,310 | 39 |
| 2009 | 95,128 | 34 |
| 2010 | 95,511 | 33 |
| 2011 | 95,846 | 29 |
| 2012 | 96,225 | 27 |
| 2013 | 96,520 | 27 |
| 2014 | 96,716 | 25 |
| 2015 | 96,941 | 23 |
| 2016 | 96,877 | 18 |
| 2017 | 97,082 | 17 |
| 2018 | 97,318 | 18 |
| 2019 | 97,860 | 17 |
| 2020 | 98,180 | 15 |
| 2021 | 98,226 | 13 |
| 2022 | 97,958 | 15 |
| 2023 | 98,293 | 13 |
| 2024 | 98,666 | 10 |

## Subfield-Year Coverage Summary

| Coverage measure | Value |
|---|---:|
| Diagnostic subfield-year rows | 6,300 |
| Cells with planned sample greater than zero | 6,298 |
| Cells reaching annual target of 400 works | 5,493 |
| Cells below annual target of 400 works | 807 |
| Cells with downloaded works below planned sample | 805 |
| Cells with zero downloaded works | 2 |
| Total work shortfall | 53,267 |

## Cells Below Target

| Below-target category | Cells | Planned works | Retained works | Shortfall |
|---|---:|---:|---:|---:|
| Full 400-work target with local/download shortfall | 330 | 132,000 | 113,570 | 18,430 |
| Sparse planned cell with fewer than 400 available works | 475 | 102,103 | 67,266 | 34,837 |
| No eligible works available | 2 | 0 | 0 | 0 |
| **Total cells below target** | **807** | **234,103** | **180,836** | **53,267** |

## Validation Warnings

- The validation report did not fail.
- Critical checks are clean: zero duplicate work IDs, zero rows outside 2000-2024, zero rows in 2025 or 2026, zero missing subfield/field/domain IDs, zero missing primary topic IDs, zero missing title or abstract fields, zero short titles or abstracts, zero non-English rows, and zero disallowed work-type rows.
- The remaining sampling limitation is coverage, not validation failure: 807 subfield-year cells are below the 400-work annual target, mostly because some cells contain fewer than 400 eligible works or because local filtering/backfill did not fully recover the target.

## Local Files Read or Reconciled

- `outputs/01_corpus_construction/validation/validation_summary_2000_2024_400py.json`
- `outputs/01_corpus_construction/validation/validation_report_2000_2024_400py.md`
- `outputs/01_corpus_construction/diagnostics/openalex_extraction_2000_2024_400py/extraction_summary.md`
- `outputs/01_corpus_construction/diagnostics/openalex_extraction_2000_2024_400py/subfield_coverage_summary.csv`
- `outputs/01_corpus_construction/diagnostics/openalex_extraction_2000_2024_400py/subfield_year_download_coverage.csv`
- `outputs/01_corpus_construction/diagnostics/openalex_extraction_2000_2024_400py/year_coverage_summary.csv`
- `data/processed/works_text_2000_2024_400py.parquet`
- `data/processed/embedding_index.parquet`
- `data/processed/analysis_embedding_index.parquet`
- `data/processed/analysis_subfields_2000_2024_400py.parquet`
- `data/processed/subfield_embedding_space_metrics.parquet`
- `data/processed/temporal/`
- `data/interim/corpus_plan_2000_2024_400py.parquet`
- `data/interim/sample_plan_2000_2024_400py.parquet`
- `data/interim/download_manifest_2000_2024_400py.parquet`
- `embeddings/specter2_v1_2000_2024_400py/`
- `outputs/03_embedding_metrics/`
- `outputs/04_reduced_metric_core/`
- `outputs/05_static_comparison/`
- `outputs/06_temporal_evolution/`
- `outputs/07_morphological_similarity/`
- `warehouse/tfm_openalex.duckdb`
