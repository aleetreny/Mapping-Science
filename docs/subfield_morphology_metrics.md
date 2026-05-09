# Projected Subfield Morphology Metrics

This stage converts per-subfield UMAP coordinate files into a tabular dataset
with one row per OpenAlex subfield. The metrics summarize the visible
morphology of each projected 2D semantic landscape.

The table contains exactly 25 curated core projected metrics plus diagnostic
metrics and controls. It does not build growth targets, regression datasets,
classifiers, paper clusters, or dashboards.

## Inputs

The script consumes the per-subfield UMAP manifest:

```text
outputs/maps/per_subfield_umap/per_subfield_umap_manifest.parquet
```

Only rows with `status == "completed"` are attempted. For each attempted
subfield, the script reads the `coordinate_path` parquet produced by:

```text
scripts/10_build_per_subfield_umap_maps.py
```

The coordinate parquet is expected to include subfield, field, domain,
publication year, and UMAP coordinate columns:

```text
subfield_id
subfield_display_name
field_id
field_display_name
domain_id
domain_display_name
publication_year
umap_x
umap_y
```

`subfield_id` is the primary key. `subfield_display_name` is not unique in
OpenAlex, so outputs also contain:

```text
subfield_label_unique
subfield_label_short
subfield_display_name_is_duplicated
```

## Outputs

```text
data/processed/subfield_morphology_metrics.parquet
data/processed/subfield_morphology_metrics.csv
outputs/metrics/subfield_morphology_metrics_summary.json
outputs/metrics/subfield_morphology_metrics_dictionary.csv
outputs/metrics/duplicate_subfield_names_report.csv
```

The parquet file is the machine-readable table. The CSV is for quick
inspection. The dictionary CSV explains the metric and important control
columns.

## Coordinate Normalization

Each subfield UMAP is fitted separately, so raw UMAP positions, orientation, and
scale are not comparable across subfields. Before computing the 25 core
metrics, each subfield is normalized independently:

1. Keep finite `(umap_x, umap_y)` points in the selected year window.
2. Center points by the coordinate-wise median.
3. Compute raw radial distances from that center.
4. Compute `raw_r95`, the 95th percentile of raw radial distances.
5. Divide centered coordinates by `raw_r95`.

All core and diagnostic projected morphology metrics are computed from these
normalized coordinates. Raw UMAP bounds and `raw_r95` are saved as diagnostics,
not as morphology features.

## Fixed Density Grid And Outliers

Density and support metrics are computed on a fixed normalized grid:

```text
[-1.5, 1.5] x [-1.5, 1.5]
```

Points outside this rectangle do not stretch the density/support grid. They
are recorded with explicit control or diagnostic columns:

```text
max_normalized_radius
outlier_share_r_gt_1
outlier_share_r_gt_1_5
outlier_share_outside_density_extent
density_entropy_slope_r2
n_density_entropy_years
density_x_min
density_x_max
density_y_min
density_y_max
```

## Core Projected Metric Families

`CORE_METRIC_COLUMNS_V2` contains exactly these 25 curated projected metrics:

- Radial dispersion and local granularity:
  `radial_tail_index`, `radial_iqr_index`, `knn_median_distance`,
  `knn_distance_cv`.
- Density concentration:
  `density_entropy`, `peak_dominance`, `effective_area_90`,
  `core_periphery_ratio`.
- Multimodality and fragmentation:
  `density_peak_count`, `peak_mass_entropy`, `dense_component_count`,
  `largest_component_mass_share`, `component_separation_index`,
  `mst_gap_index`.
- Shape, support and outlier morphology:
  `anisotropy_ratio`, `support_solidity`, `boundary_complexity`,
  `max_normalized_radius`.
- Temporal morphology across 2010-2025:
  `centroid_drift_early_late`, `annual_centroid_path_length`,
  `directionality_ratio`, `radial_expansion_slope`,
  `annual_centroid_step_cv`, `radial_expansion_r2`,
  `density_entropy_slope_by_year`.

Temporal metrics use years inside the active analysis period. The fixed early
window is 2010-2012 and the fixed late window is 2023-2025. If a subfield has
too few papers in an early, late, or annual window, the affected temporal metric
is returned as `NaN` and the row is marked `completed_with_warnings`.

`density_entropy_slope_by_year` computes annual density entropy on the
normalized fixed grid where enough papers are available, then fits a linear
trend of yearly density entropy on publication year. Positive values mean the
projected landscape became spatially more diversified; negative values mean it
became more concentrated.

Diagnostic metrics:

```text
density_gini
effective_area_50
support_circularity
hole_count
outlier_share_r_gt_1
outlier_share_r_gt_1_5
outlier_share_outside_density_extent
density_entropy_slope_r2
```

The MST gap metric can be expensive, so it deterministically samples at most
`--mst-max-points` normalized coordinates per subfield. This sampling is
recorded in the output controls.

## Commands

Smoke test:

```bash
python scripts/11_compute_subfield_morphology_metrics.py --limit-subfields 3 --year-min 2010 --year-max 2025 --overwrite
```

Single subfield:

```bash
python scripts/11_compute_subfield_morphology_metrics.py --subfield-id 1100 --year-min 2010 --year-max 2025 --overwrite
```

Full run:

```bash
python scripts/11_compute_subfield_morphology_metrics.py --year-min 2010 --year-max 2025 --overwrite
```

Useful runtime knobs:

```bash
python scripts/11_compute_subfield_morphology_metrics.py --grid-size 120 --mst-max-points 2000 --year-min 2010 --year-max 2025 --overwrite
```

## Interpretation Caveats

These metrics summarize internal geometry of separately fitted UMAP maps. The
normalization removes raw translation and scale, but UMAP is still a nonlinear
2D projection. Treat these metrics as projected morphology descriptors and
compare them with the direct embedding-space metrics rather than as physical
distances in the original SPECTER2 space.

After both projected and embedding-space tables exist, run:

```bash
python scripts/13_compare_metric_families.py --overwrite
python scripts/14_summarize_metric_distributions.py --overwrite
```

These scripts are diagnostics only. They do not cluster subfields or make
scientific claims.
