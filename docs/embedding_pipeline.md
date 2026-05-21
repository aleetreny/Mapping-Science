# Embedding Pipeline

The embedding pipeline validates SPECTER2 shards and builds the row-aligned
analysis matrix used by all active metric scripts.

Run:

```powershell
$env:LOCAL_EMBEDDINGS_DIR = "embeddings/specter2_v1_2000_2024_400py"
.\.venv\Scripts\python.exe scripts\07_validate_embeddings.py --expected-shards 119
.\.venv\Scripts\python.exe scripts\08_prepare_analysis_matrix.py --force
```

Key outputs:

```text
data/processed/embedding_index.parquet
data/processed/analysis_embedding_index.parquet
embeddings/specter2_v1_2000_2024_400py/analysis/main_embeddings.float16.npy
embeddings/specter2_v1_2000_2024_400py/analysis/main_work_ids.parquet
embeddings/specter2_v1_2000_2024_400py/analysis/main_matrix_summary.json
outputs/02_embedding_matrix/embedding_matrix_summary.json
outputs/02_embedding_matrix/row_alignment_summary.json
```

The new `outputs/02_embedding_matrix/` folder contains only lightweight summaries and row-alignment validation diagnostics (`.json`), not the large embedding matrix itself.

The mapping between `analysis_row_id`, `work_id`, metadata, and matrix row is the contract for every downstream metric stage.
