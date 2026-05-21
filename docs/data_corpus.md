# Corpus Construction

The active corpus is the OpenAlex `2000_2024_400py` extraction.

Design:

- Unit: OpenAlex subfield.
- Aggregation: fields and domains for downstream summaries.
- Period: 2000-2024.
- Target: 400 valid title-plus-abstract works per year per subfield.
- Synchronized corpus snapshot: 2,378,036 retained works across 252 subfields.
- Downstream analysis subset: 2,344,927 embedded works across 241 analysis subfields.
- Filters: English, valid title and abstract, supported work types, no
  retracted/paratype records.
- Storage: DuckDB plus Parquet files under ignored `data/` folders.

Run:

```powershell
python scripts/00_fetch_taxonomy.py
python scripts/01_build_counts.py --dataset-version 2000_2024_400py
python scripts/02_build_corpus_plan.py --dataset-version 2000_2024_400py
python scripts/03_build_sample_plan.py --dataset-version 2000_2024_400py
python scripts/04_download_sampled_corpus.py --dataset-version 2000_2024_400py
python scripts/05_validate_database.py --dataset-version 2000_2024_400py
python scripts/06_build_analysis_subfields.py --dataset-version 2000_2024_400py
```

Corpus validation outputs now live under:

```text
outputs/01_corpus_construction/
```
