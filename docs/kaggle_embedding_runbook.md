# Kaggle SPECTER2 Embedding Runbook

The production Kaggle script is archived at:

```text
scripts/embed_specter2_kaggle.py
```

It is kept for reproducibility of the existing SPECTER2 artifacts. Do not rerun it unless the embedding artifact set needs to be intentionally regenerated.

## Existing Artifact Set

The current Drive artifact set lives at:

```text
gdrive:TFM/openalex_subfields/embeddings/specter2_v1
```

It contains:

- 37 `shard_*_embeddings.npy` files
- 37 `shard_*_metadata.parquet` files
- 37 `shard_*_summary.json` files
- `embedding_config.json`
- `embedding_run_manifest.csv`

The embeddings were generated with `allenai/specter2_base` plus the `allenai/specter2` adapter, active adapter verified, 768 dimensions, and float16 output.

## Kaggle Setup

The script expects `rclone` to access Drive. On Kaggle, store the rclone config as a Kaggle secret, usually named:

```text
RCLONE_CONF
```

Then run a remote check:

```bash
python scripts/embed_specter2_kaggle.py \
  --setup-rclone-from-secret RCLONE_CONF \
  --check-remote-only
```

## Embedding Command

The production-style command is:

```bash
python scripts/embed_specter2_kaggle.py \
  --setup-rclone-from-secret RCLONE_CONF \
  --resume
```

Useful bounded test:

```bash
python scripts/embed_specter2_kaggle.py \
  --setup-rclone-from-secret RCLONE_CONF \
  --limit-rows 1000 \
  --end-shard 1 \
  --no-upload
```

The default remote input is:

```text
gdrive:TFM/openalex_subfields/data/works_text.parquet
```

The default remote output is:

```text
gdrive:TFM/openalex_subfields/embeddings/specter2_v1
```

## Local Follow-Up

After downloading the existing artifact set locally, validate it and build the lightweight index:

```bash
python scripts/07_validate_embeddings.py
```

Then prepare the main-analysis matrix and first sampled UMAP map:

```bash
python scripts/08_prepare_analysis_matrix.py
python scripts/09_build_first_umap_maps.py --sample-per-subfield 500
```
