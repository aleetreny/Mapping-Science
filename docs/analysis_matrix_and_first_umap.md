# Analysis Matrix And First UMAP

This layer prepares the row-aligned SPECTER2 matrix used by the active
`2000_2024_400py` morphology pipeline. This layer itself does not add
dashboards or predictive models.

## Prepare Main-Analysis Matrix

Run:

```powershell
$env:LOCAL_EMBEDDINGS_DIR = "embeddings/specter2_v1_2000_2024_400py"
.\.venv\Scripts\python.exe scripts\07_validate_embeddings.py --expected-shards 119
.\.venv\Scripts\python.exe scripts\08_prepare_analysis_matrix.py --force
```

The matrix script reads:

```text
data/processed/embedding_index.parquet
data/processed/analysis_subfields.parquet
<embedding-dir>/shard_*_embeddings.npy
```

When using `2000_2024_400py`, the embedding directory is:

```text
embeddings/specter2_v1_2000_2024_400py/
```

It filters to:

```text
main_analysis_eligible == true
```

For `2000_2024_400py`, validation maps the versioned
`analysis_subfields` flags to canonical matrix flags as:

- `main_analysis_eligible = eligible_for_temporal_5year_exploration`
- `robustness_eligible = eligible_min_5000_full_period`
- `strict_full_period_eligible = eligible_10000_full_period`

The legacy aliases `main_analysis_eligible_2500` and
`robustness_eligible_500` remain in the index for compatibility, but new
analysis code should use the canonical names.

and writes:

```text
data/processed/analysis_embedding_index.parquet
<embedding-dir>/analysis/main_embeddings.float16.npy
<embedding-dir>/analysis/main_work_ids.parquet
<embedding-dir>/analysis/main_matrix_summary.json
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

```powershell
.\.venv\Scripts\python.exe scripts\09_build_first_umap_maps.py `
  --embedding-dir embeddings/specter2_v1_2000_2024_400py `
  --sample-per-subfield 500 `
  --year-min 2000 `
  --year-max 2024 `
  --color-by domain `
  --force
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

```powershell
.\.venv\Scripts\python.exe scripts\09_build_first_umap_maps.py `
  --embedding-dir embeddings/specter2_v1_2000_2024_400py `
  --sample-per-subfield 100 `
  --year-min 2000 `
  --year-max 2024 `
  --force
```

The PNG can be colored by `domain`, `field`, or `subfield` with `--color-by`.
Large legends are skipped automatically.

## Build Per-Subfield UMAP Maps

After the main matrix exists, build one separate map per OpenAlex subfield:

```powershell
.\.venv\Scripts\python.exe scripts\10_build_per_subfield_umap_maps.py `
  --embedding-dir embeddings/specter2_v1_2000_2024_400py `
  --limit-subfields 3 `
  --year-min 2000 `
  --year-max 2024 `
  --max-papers-per-subfield 2000 `
  --overwrite
```

The per-subfield stage uses the requested publication-year window, uses the
SPECTER2 matrix directly with no PCA, and writes scatter plus density PNGs
under:

```text
outputs/maps/per_subfield_umap/
```

See [per_subfield_umap_maps.md](per_subfield_umap_maps.md) for the full CLI,
manifest schema, and runtime notes.

## Build Supporting Field And Domain Maps

Subfields remain the main analysis unit. Field and domain maps are optional
supporting inspection outputs:

```powershell
.\.venv\Scripts\python.exe scripts\10b_build_per_field_umap_maps.py `
  --embedding-dir embeddings/specter2_v1_2000_2024_400py `
  --year-min 2000 `
  --year-max 2024 `
  --overwrite
.\.venv\Scripts\python.exe scripts\10c_build_per_domain_umap_maps.py `
  --embedding-dir embeddings/specter2_v1_2000_2024_400py `
  --year-min 2000 `
  --year-max 2024 `
  --overwrite
```

See [higher_level_umap_maps.md](higher_level_umap_maps.md).

## Compute Metric Tables

Projected UMAP morphology metrics:

```powershell
.\.venv\Scripts\python.exe scripts\11_compute_subfield_morphology_metrics.py `
  --limit-subfields 3 `
  --year-min 2000 `
  --year-max 2024 `
  --overwrite
```

Embedding-space structure metrics:

```powershell
.\.venv\Scripts\python.exe scripts\12_compute_subfield_embedding_space_metrics.py `
  --embedding-dir embeddings/specter2_v1_2000_2024_400py `
  --limit-subfields 3 `
  --year-min 2000 `
  --year-max 2024 `
  --overwrite
```

The projected morphology stage reads completed per-subfield coordinate
parquets, normalizes each subfield's 2D UMAP coordinates independently, and
writes one row per subfield. The embedding-space stage reads
`analysis_embedding_index.parquet` plus the memory-mapped SPECTER2 matrix and
computes one row per subfield directly in the original embedding space.

See [subfield_morphology_metrics.md](subfield_morphology_metrics.md) and
[subfield_embedding_space_metrics.md](subfield_embedding_space_metrics.md).

## Compare And Diagnose Metrics

After both metric tables exist:

```bash
python scripts/13_compare_metric_families.py --overwrite
python scripts/14_summarize_metric_distributions.py --overwrite
python scripts/15_cluster_metric_spaces.py --default-k 5 --overwrite
```

These scripts write readable CSV, Markdown, and PNG diagnostics under
`outputs/analysis/`. The clustering stage is documented in
[metric_clustering.md](metric_clustering.md).
