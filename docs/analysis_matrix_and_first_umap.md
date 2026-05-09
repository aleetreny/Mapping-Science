# Analysis Matrix And First UMAP

This layer prepares the row-aligned SPECTER2 matrix used by the active
2010-2025 morphology pipeline. It does not add clustering, dashboards, or
predictive models.

## Prepare Main-Analysis Matrix

Run:

```bash
python scripts/07_validate_embeddings.py
python scripts/08_prepare_analysis_matrix.py --force
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

`analysis_embedding_index.parquet` has the same row order as the matrix and
adds `analysis_row_id`, a zero-based row pointer into
`main_embeddings.float16.npy`.

## Build First UMAP Map

Run:

```bash
python scripts/09_build_first_umap_maps.py --sample-per-subfield 500 --year-min 2010 --year-max 2025 --force
```

The first UMAP is a global visual inspection map. It uses a balanced sample
within each `subfield_id`, keeping at most 500 rows per subfield by default.
Subfields with fewer rows keep all available rows. The sample is deterministic
for a fixed random seed.

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
python scripts/09_build_first_umap_maps.py --sample-per-subfield 100 --year-min 2010 --year-max 2025 --force
```

## Build Per-Subfield UMAP Maps

After the main matrix exists, build one separate map per OpenAlex subfield:

```bash
python scripts/10_build_per_subfield_umap_maps.py --limit-subfields 3 --year-min 2010 --year-max 2025 --max-papers-per-subfield 2000 --overwrite
```

The per-subfield stage defaults to the active analysis period
`2010 <= publication_year <= 2025`, uses the SPECTER2 matrix directly with no
PCA, and writes scatter plus density PNGs under:

```text
outputs/maps/per_subfield_umap/
```

See [per_subfield_umap_maps.md](per_subfield_umap_maps.md) for the full CLI,
manifest schema, and runtime notes.

## Compute Metric Tables

Projected UMAP morphology metrics:

```bash
python scripts/11_compute_subfield_morphology_metrics.py --limit-subfields 3 --year-min 2010 --year-max 2025 --overwrite
```

Embedding-space structure metrics:

```bash
python scripts/12_compute_subfield_embedding_space_metrics.py --limit-subfields 3 --year-min 2010 --year-max 2025 --overwrite
```

The projected morphology stage reads completed per-subfield coordinate
parquets, normalizes each subfield's 2D UMAP coordinates independently, and
writes one row per subfield. The embedding-space stage reads
`analysis_embedding_index.parquet` plus the memory-mapped SPECTER2 matrix and
computes one row per subfield directly in the original embedding space.

See [subfield_morphology_metrics.md](subfield_morphology_metrics.md) and
[subfield_embedding_space_metrics.md](subfield_embedding_space_metrics.md).
