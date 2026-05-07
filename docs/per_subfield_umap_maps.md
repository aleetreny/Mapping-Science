# Per-Subfield UMAP Maps

This stage builds one internal semantic map for each OpenAlex subfield in the
main-analysis set. Each map is for visual inspection only: it does not add
clustering, morphology metrics, regressions, prediction models, dashboards, or
PCA.

## Purpose

The thesis design separates the morphology input window from the later growth
target window:

```text
Input morphology window: 2010-2019
Target growth window: 2020-2025
Unit of analysis: OpenAlex subfield
```

For that reason, the default per-subfield maps use only works with
`2010 <= publication_year <= 2019`. Works from 2020-2025 are excluded because
those years define the target growth period and should not leak into morphology
inputs.

## Inputs

The script reads the main-analysis embedding index and the SPECTER2 matrix:

```text
data/processed/analysis_embedding_index.parquet
embeddings/specter2_v1/analysis/main_embeddings.float16.npy
```

The `.npy` matrix is loaded with `np.load(..., mmap_mode="r")`, and each
subfield subset is converted to `float32` only immediately before fitting UMAP.
The script expects `analysis_embedding_index.parquet` to contain
`publication_year`; if it is missing, the run fails clearly instead of silently
using all years.

The index columns used for grouping are the existing OpenAlex subfield columns:

```text
subfield_id
subfield_display_name
field_id
field_display_name
domain_id
domain_display_name
main_analysis_eligible_2500
publication_year
analysis_row_id
work_id
```

`subfield_id` is the primary key. `subfield_display_name` is not guaranteed to
be unique in OpenAlex: for example, two different subfields can both be called
`Biochemistry` under different fields or domains. For human-readable outputs,
the script therefore adds:

```text
subfield_label_unique = "{subfield_id} | {domain_display_name} / {field_display_name} / {subfield_display_name}"
subfield_label_short = "{subfield_id} | {field_display_name} / {subfield_display_name}"
subfield_display_name_is_duplicated
```

The short label is used in PNG titles so ambiguous names are visible without
having to inspect filenames.

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
available papers, used papers, output paths, UMAP settings, and any error
message. Label columns are included in both coordinate and manifest outputs for
human-readable inspection, but downstream joins should still use `subfield_id`.

## Commands

Small test run:

```bash
python scripts/10_build_per_subfield_umap_maps.py --limit-subfields 3 --max-papers-per-subfield 2000 --overwrite
```

Single subfield:

```bash
python scripts/10_build_per_subfield_umap_maps.py --subfield-id 1100 --overwrite
```

Full run:

```bash
python scripts/10_build_per_subfield_umap_maps.py --overwrite
```

Debugging a different input year window is possible, but the thesis default is
2010-2019:

```bash
python scripts/10_build_per_subfield_umap_maps.py --year-min 2012 --year-max 2018 --limit-subfields 3 --overwrite
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

Default sampling keeps at most 10,000 papers per subfield and skips subfields
with fewer than 250 papers in the selected input window.
