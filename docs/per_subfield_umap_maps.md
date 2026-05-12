# Per-Subfield UMAP Maps

This stage builds one internal 2D UMAP landscape for each OpenAlex subfield in
the main-analysis set. The maps are projected semantic landscapes used for
visual inspection and projected morphology metrics. This stage does not add
paper clustering, PCA, dashboards, or predictive models.

## Purpose

The legacy/default analysis period is:

```text
Analysis period: 2010-2025
Excluded year: 2026, because it is incomplete/current
Unit of analysis: OpenAlex subfield
```

By default, per-subfield maps use works with:

```text
2010 <= publication_year <= 2025
```

For `2000_2024_400py`, pass `--year-min 2000 --year-max 2024` so the map
window matches the versioned corpus.

## Inputs

The script reads the main-analysis embedding index and the SPECTER2 matrix:

```text
data/processed/analysis_embedding_index.parquet
<embedding-dir>/analysis/main_embeddings.float16.npy
```

The `.npy` matrix is loaded with `np.load(..., mmap_mode="r")`, and each
subfield subset is converted to `float32` only immediately before fitting UMAP.
The script expects `analysis_embedding_index.parquet` to contain
`publication_year`; if it is missing, the run fails clearly.

`--embedding-dir` defaults to `LOCAL_EMBEDDINGS_DIR` from `.env` or the
environment. For `2000_2024_400py`, use:

```powershell
$env:LOCAL_EMBEDDINGS_DIR = "embeddings/specter2_v1_2000_2024_400py"
```

or pass `--embedding-dir embeddings/specter2_v1_2000_2024_400py`.

The index columns used for grouping are:

```text
subfield_id
subfield_display_name
field_id
field_display_name
domain_id
domain_display_name
main_analysis_eligible
publication_year
analysis_row_id
work_id
```

`subfield_id` is the primary key. `subfield_display_name` is not guaranteed to
be unique in OpenAlex, so the script adds:

```text
subfield_label_unique = "{subfield_id} | {domain_display_name} / {field_display_name} / {subfield_display_name}"
subfield_label_short = "{subfield_id} | {field_display_name} / {subfield_display_name}"
subfield_display_name_is_duplicated
```

## Outputs

By default, outputs are written under:

```text
outputs/maps/per_subfield_umap/
  coordinates/
    <safe_subfield_id>__<safe_subfield_name>.parquet
  figures/
    <safe_subfield_id>__<safe_subfield_name>.png
  per_subfield_umap_manifest.parquet
  per_subfield_umap_summary.json
```

Each PNG contains two panels:

- Panel A: scatter plot of 2D UMAP coordinates.
- Panel B: KDE density view when feasible, with a `viridis` fallback density
  renderer for larger or numerically difficult point clouds.

The coordinate parquet stores work metadata plus `umap_x` and `umap_y`.
The manifest records one row per attempted subfield, including status,
available papers, used papers, output paths, UMAP settings, deterministic
sampling fields, and any error message.

Important manifest controls:

```text
n_available
n_used
sampling_applied
max_papers_per_subfield
year_min
year_max
random_state
```

## Commands

Small test run:

```powershell
.\.venv\Scripts\python.exe scripts\10_build_per_subfield_umap_maps.py `
  --embedding-dir embeddings/specter2_v1_2000_2024_400py `
  --limit-subfields 3 `
  --year-min 2000 `
  --year-max 2024 `
  --max-papers-per-subfield 2000 `
  --overwrite
```

Single subfield:

```powershell
.\.venv\Scripts\python.exe scripts\10_build_per_subfield_umap_maps.py `
  --embedding-dir embeddings/specter2_v1_2000_2024_400py `
  --subfield-id 1100 `
  --year-min 2000 `
  --year-max 2024 `
  --overwrite
```

Full run:

```powershell
.\.venv\Scripts\python.exe scripts\10_build_per_subfield_umap_maps.py `
  --embedding-dir embeddings/specter2_v1_2000_2024_400py `
  --year-min 2000 `
  --year-max 2024 `
  --overwrite
```

Debugging a narrower input year window is possible:

```powershell
.\.venv\Scripts\python.exe scripts\10_build_per_subfield_umap_maps.py `
  --embedding-dir embeddings/specter2_v1_2000_2024_400py `
  --year-min 2012 `
  --year-max 2020 `
  --limit-subfields 3 `
  --overwrite
```

## Runtime And RAM

The stage is sequential by default and does not hold all subfield matrices or
all UMAP coordinates in memory. For each attempted subfield, it:

1. Selects row ids for that subfield and year window.
2. Samples deterministically if more than `--max-papers-per-subfield` papers
   are available.
3. Copies only that subfield's embeddings from the memory-mapped matrix into a
   `float32` array.
4. Fits one UMAP model for that subfield.
5. Writes the coordinate parquet and PNG immediately.

The default cap is 10,000 papers per subfield and the default minimum is 250
papers. The cap is a runtime control, not a conceptual exclusion from the active
2010-2025 period.

## Supporting Higher Levels

Fields and domains can also be mapped for inspection with:

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

These outputs are backup/inspection maps. They do not change the main unit of
analysis, which remains OpenAlex subfield. See
[higher_level_umap_maps.md](higher_level_umap_maps.md).
