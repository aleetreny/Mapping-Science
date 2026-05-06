# Analysis Matrix And First UMAP

This layer prepares the first local visual inspection inputs from the validated SPECTER2 shards.

It does not add clustering, morphology metrics, prediction models, or dashboards.

## Prepare Main-Analysis Matrix

Run:

```bash
python scripts/07_validate_embeddings.py
python scripts/08_prepare_analysis_matrix.py
```

The matrix script reads:

```text
data/processed/embedding_index.parquet
data/processed/analysis_subfields.parquet
embeddings/specter2_v1/shard_*_embeddings.npy
```

It filters to:

```text
main_analysis_eligible_2500 == true
```

and writes:

```text
data/processed/analysis_embedding_index.parquet
embeddings/specter2_v1/analysis/main_embeddings.float16.npy
embeddings/specter2_v1/analysis/main_work_ids.parquet
embeddings/specter2_v1/analysis/main_matrix_summary.json
```

The row order is deterministic:

```text
subfield_id, publication_year, work_id
```

`analysis_embedding_index.parquet` has the same row order as the matrix and adds `analysis_row_id`, a zero-based row pointer into `main_embeddings.float16.npy`.

If outputs already exist, use:

```bash
python scripts/08_prepare_analysis_matrix.py --force
```

## Build First UMAP Map

Run:

```bash
python scripts/09_build_first_umap_maps.py --sample-per-subfield 500
```

The script samples within each `subfield_id`, keeping at most 500 rows per subfield by default. Subfields with fewer rows keep all available rows. The sample is deterministic for a fixed random seed.

Outputs:

```text
outputs/maps/umap_global_sample.parquet
outputs/maps/umap_global_sample.png
outputs/maps/umap_global_sample_summary.json
```

The parquet output includes:

- `work_id`
- `analysis_row_id`
- subfield, field, domain, and primary-topic metadata
- `publication_year`
- `umap_x`
- `umap_y`

Use a smaller sample for quick checks:

```bash
python scripts/09_build_first_umap_maps.py --sample-per-subfield 100 --force
```

Use a different deterministic sample:

```bash
python scripts/09_build_first_umap_maps.py --sample-per-subfield 500 --random-seed 7 --force
```
