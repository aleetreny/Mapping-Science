# Subfield Morphology Metrics

This stage converts the per-subfield UMAP coordinate files into a tabular
dataset with one row per OpenAlex subfield. It writes a curated set of 25 core
v2 morphology metrics for later modeling plus diagnostic metrics and controls.
It does not build prediction models, add growth targets, run paper clustering,
or create dashboards.

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
publication year, and UMAP coordinate columns, especially:

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
OpenAlex, so the output table also contains:

```text
subfield_label_unique
subfield_label_short
subfield_display_name_is_duplicated
```

`subfield_label_unique` has the form
`{subfield_id} | {domain_display_name} / {field_display_name} / {subfield_display_name}`.
Duplicate display names are reported, not treated as data errors.

## Outputs

Primary outputs:

```text
data/processed/subfield_morphology_metrics.parquet
data/processed/subfield_morphology_metrics.csv
outputs/metrics/subfield_morphology_metrics_summary.json
outputs/metrics/subfield_morphology_metrics_dictionary.csv
outputs/metrics/duplicate_subfield_names_report.csv
```

The parquet file is the machine-readable table. The CSV is for quick
inspection. The dictionary CSV explains the metric and important control
columns for thesis documentation.

## Coordinate Normalization

Each subfield UMAP is fitted separately, so raw UMAP positions, orientation, and
scale are not comparable across subfields. Before computing the 25 core
metrics, each subfield is normalized independently:

1. Keep only finite `(umap_x, umap_y)` points in the manifest year window.
2. Center points by the coordinate-wise median.
3. Compute raw radial distances from that center.
4. Compute `raw_r95`, the 95th percentile of raw radial distances.
5. Divide centered coordinates by `raw_r95`.

All core and diagnostic morphology metrics are computed from these normalized
coordinates. Raw UMAP bounds and `raw_r95` are saved only as diagnostics, not as
morphology features.

## Fixed Density Grid And Outliers

Density and support metrics are computed on a fixed normalized grid:

```text
[-1.5, 1.5] x [-1.5, 1.5]
```

The fixed extent is used because coordinates have already been median-centered
and divided by `raw_r95`. Most semantic mass should lie within radius about 1,
so using the same grid for every subfield keeps density cell areas comparable.
It also prevents one extreme UMAP outlier from stretching the KDE or support
grid and distorting area, solidity, circularity, component, or hole metrics.

Points outside this rectangle are not used to stretch the density/support grid.
They are instead recorded with explicit control columns:

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

`max_normalized_radius` is retained as a core v2 outlier morphology feature.
`outlier_share_r_gt_1`, `outlier_share_r_gt_1_5`, and
`outlier_share_outside_density_extent` are diagnostic only because they are
sparse, low-variance, partly mechanical after `r95` normalization, or tied to
the fixed grid.

## Core V2 Metric Families

`CORE_METRIC_COLUMNS_V2` contains exactly these 25 curated metrics recommended
for later modeling:

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
- Temporal morphology inside the morphology window:
  `centroid_drift_early_late`, `annual_centroid_path_length`,
  `directionality_ratio`, `radial_expansion_slope`,
  `annual_centroid_step_cv`, `radial_expansion_r2`,
  `density_entropy_slope_by_year`.

`density_entropy_slope_by_year` is a temporal semantic-diversification metric:
for each publication year inside 2010-2019, the script computes density entropy
on the normalized fixed grid where enough papers are available, then fits a
linear trend of yearly density entropy on year. Positive values mean the
subfield became spatially more diversified within the morphology window;
negative values mean it became more concentrated. It never reads 2020-2025
works.

The output table also keeps diagnostic metrics:

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

These were demoted because they are redundant with core metrics or weakly
varying in the first full run. `density_gini` and `effective_area_50` strongly
track density entropy/effective area, `support_circularity` is the inverse view
of `boundary_complexity`, `hole_count` has little variation, and
`outlier_share_r_gt_1` is nearly mechanical because `r95` normalization places
about five percent of points beyond radius 1. `outlier_share_r_gt_1_5` was also
demoted because the first analysis showed it was sparse and low-variance.
`density_entropy_slope_r2` is retained as diagnostic trend-fit strength because
R-squared has no direction by itself.

Density metrics use the fixed normalized KDE-like grid described above. The
preferred density estimator is `scipy.stats.gaussian_kde`; if KDE is
numerically unstable or the point count is high, the code falls back to a
smoothed 2D histogram.

The MST gap metric can be expensive, so it deterministically samples at most
`--mst-max-points` normalized coordinates per subfield. At this checkpoint the
pipeline stops after writing the morphology metric table; downstream analysis
and prediction scripts have been removed so the next direction can be rebuilt
cleanly.

## Temporal Leakage Guardrail

Temporal metrics use only the `publication_year` values present in the
coordinate parquet and inside the manifest year window, normally 2010-2019.
The early centroid is based on 2010-2012 and the late centroid on 2017-2019.
The stage does not read 2020-2025 works and does not join growth targets.

## Commands

Smoke test:

```powershell
.\.venv\Scripts\python.exe scripts\11_compute_subfield_morphology_metrics.py --limit-subfields 3 --overwrite
```

Single subfield:

```powershell
.\.venv\Scripts\python.exe scripts\11_compute_subfield_morphology_metrics.py --subfield-id 1100 --overwrite
```

Full run:

```powershell
.\.venv\Scripts\python.exe scripts\11_compute_subfield_morphology_metrics.py --overwrite
```

Useful runtime knobs:

```powershell
.\.venv\Scripts\python.exe scripts\11_compute_subfield_morphology_metrics.py --grid-size 120 --mst-max-points 2000 --overwrite
```

## Interpretation Caveats

These metrics summarize internal geometry of separately fitted UMAP maps. The
normalization removes raw translation and scale, but UMAP is still a nonlinear
visual embedding. The metrics are best treated as structured morphology
descriptors for later analysis, not as direct physical distances in the
original SPECTER2 embedding space.

Rows can have `metric_status == "completed_with_warnings"` when core metrics
were computed but a partially unavailable metric, usually temporal, returned
`NaN`. Rows with unrecoverable coordinate or normalization problems are kept
with `metric_status == "failed"` and a `metric_error_message`.
