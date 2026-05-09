# Higher-Level UMAP Maps

Subfields remain the main unit of analysis. Field and domain UMAP maps are
supporting visual inspection outputs that help check broad structure and
communicate context.

## Commands

Global balanced sample:

```bash
python scripts/09_build_first_umap_maps.py --sample-per-subfield 500 --year-min 2010 --year-max 2025 --color-by domain --force
```

Main unit, one map per subfield:

```bash
python scripts/10_build_per_subfield_umap_maps.py --year-min 2010 --year-max 2025 --overwrite
```

Supporting field maps:

```bash
python scripts/10b_build_per_category_umap_maps.py --level field --year-min 2010 --year-max 2025 --overwrite
```

Supporting domain maps:

```bash
python scripts/10b_build_per_category_umap_maps.py --level domain --year-min 2010 --year-max 2025 --overwrite
```

## Inputs

The grouped field/domain script reads:

```text
data/processed/analysis_embedding_index.parquet
embeddings/specter2_v1/analysis/main_embeddings.float16.npy
```

It memory-maps the embedding matrix with `np.load(..., mmap_mode="r")`, filters
to the active year window, and fits one separate UMAP per field or domain.

## Outputs

Field outputs:

```text
outputs/maps/per_field_umap/coordinates/*.parquet
outputs/maps/per_field_umap/figures/*.png
outputs/maps/per_field_umap/per_field_umap_manifest.parquet
outputs/maps/per_field_umap/per_field_umap_summary.json
```

Domain outputs:

```text
outputs/maps/per_domain_umap/coordinates/*.parquet
outputs/maps/per_domain_umap/figures/*.png
outputs/maps/per_domain_umap/per_domain_umap_manifest.parquet
outputs/maps/per_domain_umap/per_domain_umap_summary.json
```

Each PNG uses the same simple two-panel format:

- A. Scatter
- B. Density

Field maps color by subfield display name only when the legend is readable.
Domain maps color by field display name only when the legend is readable. If
there are too many categories, the script uses a neutral scatter and records
that no legend was included.

## Important Runtime Controls

```text
--min-papers
--max-papers-per-group
--group-id
--limit-groups
--random-state
--n-neighbors
--min-dist
--metric
--dpi
```

Field and especially domain groups may contain many papers. If UMAP fitting is
too slow or memory-heavy, lower `--max-papers-per-group`. Sampling is
deterministic and recorded in the manifest with `n_available`, `n_used`, and
`sampling_applied`.
