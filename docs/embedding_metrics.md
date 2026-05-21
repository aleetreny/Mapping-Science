# Embedding Metrics

`scripts/11_compute_embedding_space_metrics.py` computes structural metrics
directly from the row-aligned SPECTER2 matrix.

The active metric snapshot uses 2,344,927 analysis-matrix rows across 241
eligible subfields, derived from the synchronized 2,378,036-work corpus.

Inputs:

```text
data/processed/analysis_embedding_index.parquet
embeddings/specter2_v1_2000_2024_400py/analysis/main_embeddings.float16.npy
```

Main outputs:

```text
data/processed/subfield_embedding_space_metrics.parquet
data/processed/subfield_embedding_space_metrics.csv
outputs/03_embedding_metrics/subfield_embedding_space_metrics_summary.json
outputs/03_embedding_metrics/subfield_embedding_space_metrics_dictionary.csv
outputs/03_embedding_metrics/diagnostics/
```

The full metric table is intentionally broader than the thesis core. It keeps
diagnostics and candidate metrics so the reduced 11-metric set can be audited
and reproduced. The main downstream scripts use `docs/reduced_metric_core.md`
as the thesis metric definition.

Run:

```powershell
$env:LOCAL_EMBEDDINGS_DIR = "embeddings/specter2_v1_2000_2024_400py"
.\.venv\Scripts\python.exe scripts\11_compute_embedding_space_metrics.py --year-min 2000 --year-max 2024 --overwrite
```

UMAP coordinates are not used to compute these metrics.
