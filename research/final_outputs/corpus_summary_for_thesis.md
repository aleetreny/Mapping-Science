# Corpus Construction Summary for Thesis

This summary uses the validation-backed corpus-construction outputs for dataset version `2000_2024_400py`. The headline figures below are taken from `outputs/01_corpus_construction/validation/validation_summary_2000_2024_400py.json` and the accompanying diagnostic CSV files.

Important reliability note: the validation outputs and the current Parquet copies under `data/processed/` and `data/interim/` are not fully synchronized. The validation-backed corpus contains 2,358,036 retained works and 250 non-empty subfields. The current `data/processed/works_text_2000_2024_400py.parquet` contains 2,378,036 works and 252 subfields, 20,000 works more. Treat the validation-backed figures as the thesis snapshot until the Parquet files are regenerated or validation is rerun.

## Headline Corpus Counts

| Measure | Value |
|---|---:|
| Dataset version | `2000_2024_400py` |
| Years covered | 2000-2024 inclusive |
| Retained works | 2,358,036 |
| Retained non-empty subfields | 250 |
| Planned taxonomy subfields | 252 |
| Fields represented | 26 |
| Domains represented | 4 |
| Target annual sample | up to 400 works per subfield-year |
| Maximum full-period target | 10,000 works per subfield |
| Planned works | 2,431,303 |
| Total shortfall | 73,267 |
| Validation failed | false |

## Target Sampling Design

The corpus is organized by OpenAlex primary-topic subfield. For each subfield-year cell, the target sample is up to 400 eligible works. Over 25 years, this implies a maximum of 10,000 works per subfield. The active extraction uses English `article` and `preprint` records, requires both title and abstract text, requires at least 5 title tokens and 80 abstract tokens, and excludes retracted and paratextual records. The sampling run used an oversample factor of 1.75 and up to 4 backfill rounds.

## Works by Domain

| Domain | Fields | Retained subfields | Retained works | Share of retained works |
|---|---:|---:|---:|---:|
| Life Sciences | 5 | 42 | 397,808 | 16.87% |
| Social Sciences | 6 | 58 | 542,774 | 23.02% |
| Physical Sciences | 10 | 89 | 844,382 | 35.81% |
| Health Sciences | 5 | 61 | 573,072 | 24.30% |
| **Total** | **26** | **250** | **2,358,036** | **100.00%** |

## Works by Year

| Year | Retained works | Cells below target |
|---:|---:|---:|
| 2000 | 87,674 | 71 |
| 2001 | 88,585 | 68 |
| 2002 | 89,684 | 58 |
| 2003 | 90,272 | 62 |
| 2004 | 90,978 | 53 |
| 2005 | 91,623 | 54 |
| 2006 | 92,357 | 43 |
| 2007 | 92,806 | 41 |
| 2008 | 93,510 | 41 |
| 2009 | 94,328 | 36 |
| 2010 | 94,711 | 35 |
| 2011 | 95,046 | 31 |
| 2012 | 95,425 | 29 |
| 2013 | 95,720 | 29 |
| 2014 | 95,916 | 27 |
| 2015 | 96,141 | 25 |
| 2016 | 96,077 | 20 |
| 2017 | 96,282 | 19 |
| 2018 | 96,518 | 20 |
| 2019 | 97,060 | 19 |
| 2020 | 97,380 | 17 |
| 2021 | 97,426 | 15 |
| 2022 | 97,158 | 17 |
| 2023 | 97,493 | 15 |
| 2024 | 97,866 | 12 |

## Subfield-Year Coverage Summary

| Coverage measure | Value |
|---|---:|
| Diagnostic subfield-year rows | 6,300 |
| Cells with planned sample greater than zero | 6,298 |
| Cells reaching annual target of 400 works | 5,443 |
| Cells below annual target of 400 works | 857 |
| Retained-subfield cells below target | 807 |
| Cells with downloaded works below planned sample | 855 |
| Cells with zero downloaded works | 52 |
| Total work shortfall | 73,267 |

## Cells Below Target

| Below-target category | Cells | Planned works | Retained works | Shortfall |
|---|---:|---:|---:|---:|
| Full 400-work target with local/download shortfall | 330 | 132,000 | 113,570 | 18,430 |
| Sparse planned cell with fewer than 400 available works | 475 | 102,103 | 67,266 | 34,837 |
| Planned subfield not retained in validation snapshot | 50 | 20,000 | 0 | 20,000 |
| No eligible works available | 2 | 0 | 0 | 0 |
| **Total cells below target** | **857** | **254,103** | **180,836** | **73,267** |

The 50 planned cells not retained in the validation snapshot correspond to two planned Health Professions subfields: Radiological and Ultrasound Technology, and Speech and Hearing.

## Validation Warnings

- The validation report itself did not fail. It reports zero duplicate work IDs, zero rows outside 2000-2024, zero rows in 2025 or 2026, zero missing subfield/field/domain IDs, zero missing primary topic IDs, zero missing title or abstract fields, zero short titles or abstracts, zero non-English rows, and zero disallowed work-type rows.
- Two planned subfields have zero downloaded works in the validation diagnostics: Radiological and Ultrasound Technology, and Speech and Hearing. Together these account for 50 below-target cells and 20,000 works of shortfall.
- The validation-backed outputs and current Parquet files disagree. `data/processed/works_text_2000_2024_400py.parquet` currently contains 2,378,036 works and 252 subfields, while the validation summary reports 2,358,036 works and 250 subfields. `data/interim/download_manifest_2000_2024_400py.parquet` contains 6,300 rows and includes the two Health Professions subfields, while the validated manifest table used by the report contains 6,250 rows.
- Because of this mismatch, Chapter 3 prose should not be updated with final counts until the corpus files and validation outputs are synchronized.

## Local Files Read

- `outputs/01_corpus_construction/validation/validation_summary_2000_2024_400py.json`
- `outputs/01_corpus_construction/validation/validation_report_2000_2024_400py.md`
- `outputs/01_corpus_construction/diagnostics/openalex_extraction_2000_2024_400py/extraction_summary.md`
- `outputs/01_corpus_construction/diagnostics/openalex_extraction_2000_2024_400py/subfield_coverage_summary.csv`
- `outputs/01_corpus_construction/diagnostics/openalex_extraction_2000_2024_400py/subfield_year_download_coverage.csv`
- `outputs/01_corpus_construction/diagnostics/openalex_extraction_2000_2024_400py/year_coverage_summary.csv`
- `data/processed/works_text_2000_2024_400py.parquet`
- `data/processed/analysis_subfields_2000_2024_400py.parquet`
- `data/interim/corpus_plan_2000_2024_400py.parquet`
- `data/interim/sample_plan_2000_2024_400py.parquet`
- `data/interim/download_manifest_2000_2024_400py.parquet`
- `data/interim/domains.parquet`
- `data/interim/fields.parquet`
- `data/interim/subfields.parquet`
- `data/interim/domain_year_counts_2000_2024_400py.parquet`
- `data/interim/field_year_counts_2000_2024_400py.parquet`
- `data/interim/subfield_year_counts_2000_2024_400py.parquet`
- `warehouse/tfm_openalex.duckdb` (read-only reconciliation check of versioned corpus tables)
