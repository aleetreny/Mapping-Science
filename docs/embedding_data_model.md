# Embedding Data Model

SPECTER2 embeddings are stored as local generated artifacts, not as Git-tracked source files.

## Locations

Remote Drive folder:

```text
gdrive:TFM/openalex_subfields/embeddings/specter2_v1
```

Local folder:

```text
embeddings/specter2_v1/
```

Current `2000_2024_400py` embedding artifact folder:

```text
embeddings/specter2_v1_2000_2024_400py/
```

Download with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_embeddings_from_drive.ps1
```

or:

```bash
bash scripts/download_embeddings_from_drive.sh
```

Then validate and build the local index:

```bash
python scripts/07_validate_embeddings.py
```

The defaults can be overridden in `.env`:

```text
RCLONE_REMOTE=gdrive
DRIVE_EMBEDDINGS_PATH=TFM/openalex_subfields/embeddings/specter2_v1
LOCAL_EMBEDDINGS_DIR=embeddings/specter2_v1
```

For `2000_2024_400py`, set the embedding directory before validation and
downstream matrix consumers:

```powershell
$env:LOCAL_EMBEDDINGS_DIR = "embeddings/specter2_v1_2000_2024_400py"
```

## Shard Format

The legacy SPECTER2 artifact set contains 37 shards:

- `shard_0000_embeddings.npy`
- `shard_0000_metadata.parquet`
- `shard_0000_summary.json`
- ...
- `shard_0036_embeddings.npy`
- `shard_0036_metadata.parquet`
- `shard_0036_summary.json`

The `2000_2024_400py` artifact set contains 119 shards:

- `shard_0000_embeddings.npy`
- `shard_0000_metadata.parquet`
- `shard_0000_summary.json`
- ...
- `shard_0118_embeddings.npy`
- `shard_0118_metadata.parquet`
- `shard_0118_summary.json`

Embeddings were generated with:

- model base: `allenai/specter2_base`
- adapter: `allenai/specter2`
- embedding version: `specter2_v1`
- dtype: `float16`
- dimension: 768

The `.npy` files contain vectors only. The matching metadata parquet gives row order and work IDs. Do not concatenate all vectors just to inspect them; load shards with memory mapping.

```python
import numpy as np

embeddings = np.load(
    "embeddings/specter2_v1/shard_0000_embeddings.npy",
    mmap_mode="r",
)
print(embeddings.shape, embeddings.dtype)
```

## Why Embeddings Are Not Committed

The shard set is about 1 GiB and is reproducible from the external Kaggle/Drive workflow. Git tracks the scripts, validation logic, and lightweight index only. The `embeddings/` folder is ignored.

## `embedding_index.parquet`

Validation writes:

```text
data/processed/embedding_index.parquet
```

and the DuckDB table:

```text
embedding_index
```

The index links each `work_id` to its shard and zero-based row position:

- `embedding_shard_id`
- `embedding_shard_file`
- `embedding_row_in_shard`
- `metadata_shard_file`

It also carries subfield, field, domain, primary-topic, publication-year, and analysis eligibility flags. It does not contain embedding vectors.

Canonical eligibility columns written to `embedding_index.parquet` are:

- `main_analysis_eligible`
- `robustness_eligible`
- `strict_full_period_eligible`

For `2000_2024_400py`, these are mapped from `analysis_subfields` as:

- `main_analysis_eligible = eligible_for_temporal_5year_exploration`
- `robustness_eligible = eligible_min_5000_full_period`
- `strict_full_period_eligible = eligible_10000_full_period`

Legacy aliases are still written for compatibility:

- `main_analysis_eligible_2500`
- `robustness_eligible_500`

## Selecting Analysis Rows

Main morphology analysis should use:

```python
import pandas as pd

index = pd.read_parquet("data/processed/embedding_index.parquet")
main = index[index["main_analysis_eligible"]]
robustness = index[index["robustness_eligible"]]
```

The full `works_text.parquet` and full embedding shard set remain available.
The eligibility flags select rows for analysis without deleting downloaded
papers or embedding vectors. Active morphology scripts then filter those rows
to `2010 <= publication_year <= 2025`; 2026 is excluded because it is
incomplete/current.

## Main-Analysis Matrix

Build the row-aligned main-analysis matrix with:

```powershell
.\.venv\Scripts\python.exe scripts\08_prepare_analysis_matrix.py `
  --embedding-dir embeddings/specter2_v1_2000_2024_400py
```

Outputs:

```text
data/processed/analysis_embedding_index.parquet
<embedding-dir>/analysis/main_embeddings.float16.npy
<embedding-dir>/analysis/main_work_ids.parquet
<embedding-dir>/analysis/main_matrix_summary.json
```

`analysis_embedding_index.parquet` adds `analysis_row_id`, which points to rows in `main_embeddings.float16.npy`.

## First UMAP Map

Build the first balanced-sample visual inspection map with:

```powershell
.\.venv\Scripts\python.exe scripts\09_build_first_umap_maps.py `
  --embedding-dir embeddings/specter2_v1_2000_2024_400py `
  --sample-per-subfield 500 `
  --year-min 2000 `
  --year-max 2024
```

Outputs:

```text
outputs/maps/umap_global_sample.parquet
outputs/maps/umap_global_sample.png
outputs/maps/umap_global_sample_summary.json
```

See [analysis_matrix_and_first_umap.md](analysis_matrix_and_first_umap.md) for details.

## Embedding-Space Metrics

Compute direct high-dimensional structure metrics from the row-aligned matrix:

```powershell
.\.venv\Scripts\python.exe scripts\12_compute_subfield_embedding_space_metrics.py `
  --embedding-dir embeddings/specter2_v1_2000_2024_400py `
  --year-min 2000 `
  --year-max 2024 `
  --overwrite
```

This stage memory-maps `main_embeddings.float16.npy`, reads only each
subfield's selected rows, converts the subfield slice to `float32`, and
L2-normalizes rows before cosine-based metrics. See
[subfield_embedding_space_metrics.md](subfield_embedding_space_metrics.md).

## Joining Shard Metadata

Shard metadata can be joined back to the lightweight index by `work_id`:

```python
import pandas as pd

metadata = pd.read_parquet(
    "embeddings/specter2_v1/shard_0000_metadata.parquet"
)
index = pd.read_parquet("data/processed/embedding_index.parquet")

joined = metadata.merge(
    index[
        [
            "work_id",
            "embedding_shard_id",
            "embedding_row_in_shard",
            "main_analysis_eligible",
            "robustness_eligible",
            "strict_full_period_eligible",
        ]
    ],
    on="work_id",
    how="left",
)
```
