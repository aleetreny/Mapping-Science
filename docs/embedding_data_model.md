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

## Shard Format

The SPECTER2 artifact set contains 37 shards:

- `shard_0000_embeddings.npy`
- `shard_0000_metadata.parquet`
- `shard_0000_summary.json`
- ...
- `shard_0036_embeddings.npy`
- `shard_0036_metadata.parquet`
- `shard_0036_summary.json`

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

## Selecting Analysis Rows

Main morphology analysis should use:

```python
import pandas as pd

index = pd.read_parquet("data/processed/embedding_index.parquet")
main = index[index["main_analysis_eligible_2500"]]
robustness = index[index["robustness_eligible_500"]]
```

The full `works_text.parquet` and full embedding shard set remain available. The eligibility flags select rows for analysis without deleting downloaded papers or embedding vectors.

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
            "main_analysis_eligible_2500",
            "robustness_eligible_500",
        ]
    ],
    on="work_id",
    how="left",
)
```
