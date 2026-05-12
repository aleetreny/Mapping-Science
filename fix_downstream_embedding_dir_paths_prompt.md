# Mini-fix: make downstream scripts use the versioned embedding directory

The embedding eligibility fix worked, but `scripts/10_build_per_subfield_umap_maps.py` still defaults to the old matrix path:

```text
embeddings/specter2_v1/analysis/main_embeddings.float16.npy
```

This caused a mismatch:

```text
analysis_row_id range 0-2344926, matrix rows 717845
```

because `data/processed/analysis_embedding_index.parquet` now belongs to the new dataset, while the default matrix path points to the old dataset.

## Task

Please update downstream scripts so they consistently use the versioned embedding directory, ideally via `LOCAL_EMBEDDINGS_DIR`, just like scripts 07/08.

### Requirements

1. Update `scripts/10_build_per_subfield_umap_maps.py` so it accepts either:
   - `--embedding-dir embeddings/specter2_v1_2000_2024_400py`, or
   - uses `LOCAL_EMBEDDINGS_DIR` from `.env` / environment.

2. The default `--embeddings-path` should be derived from:

```text
<embedding_dir>/analysis/main_embeddings.float16.npy
```

not hardcoded to:

```text
embeddings/specter2_v1/analysis/main_embeddings.float16.npy
```

3. Apply the same fix to any downstream script that reads the main matrix directly, especially:
   - per-subfield UMAP maps
   - per-field/domain UMAP maps
   - morphology metrics
   - embedding-space metrics
   - metric comparison/clustering scripts if relevant

4. Add a defensive validation: if `analysis_embedding_index.parquet` has `max(analysis_row_id) >= matrix.shape[0]`, print both paths and fail with a clear message saying the index and matrix belong to different embedding versions.

5. Update docs/runbook commands to use:

```powershell
$env:LOCAL_EMBEDDINGS_DIR = "embeddings/specter2_v1_2000_2024_400py"
```

or explicit:

```powershell
--embedding-dir embeddings/specter2_v1_2000_2024_400py
```

6. Verify this command works:

```powershell
.\.venv\Scripts\python.exe scripts\10_build_per_subfield_umap_maps.py `
  --embedding-dir embeddings/specter2_v1_2000_2024_400py `
  --year-min 2000 `
  --year-max 2024 `
  --limit-subfields 3 `
  --overwrite
```

7. Run:

```powershell
.\.venv\Scripts\python.exe -m compileall scripts src
.\.venv\Scripts\python.exe -m pytest
```

Do not regenerate embeddings or modify `.npy` shard files. This is only a path/config consistency fix.
