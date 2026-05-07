# Subfield Morphology Metrics

This stage converts the per-subfield UMAP coordinate files into a tabular
dataset with one row per OpenAlex subfield and 25 morphology metrics. It does
not build prediction models, add growth targets, run clustering, use PCA, or
create dashboards.

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

## Outputs

Primary outputs:

```text
data/processed/subfield_morphology_metrics.parquet
data/processed/subfield_morphology_metrics.csv
outputs/metrics/subfield_morphology_metrics_summary.json
outputs/metrics/subfield_morphology_metrics_dictionary.csv
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

All 25 morphology metrics are computed from these normalized coordinates. Raw
UMAP bounds and `raw_r95` are saved only as diagnostics, not as morphology
features.

## Metric Families

The table contains exactly these 25 metric columns:

- Radial dispersion and local granularity:
  `radial_tail_index`, `radial_iqr_index`, `knn_median_distance`,
  `knn_distance_cv`.
- Density concentration:
  `density_entropy`, `density_gini`, `peak_dominance`,
  `effective_area_50`, `effective_area_90`, `core_periphery_ratio`.
- Multimodality and fragmentation:
  `density_peak_count`, `peak_mass_entropy`, `dense_component_count`,
  `largest_component_mass_share`, `component_separation_index`,
  `mst_gap_index`.
- Shape and topology of occupied support:
  `anisotropy_ratio`, `support_solidity`, `support_circularity`,
  `boundary_complexity`, `hole_count`.
- Temporal morphology inside the morphology window:
  `centroid_drift_early_late`, `annual_centroid_path_length`,
  `directionality_ratio`, `radial_expansion_slope`.

Density metrics use a normalized KDE-like grid. The preferred density estimator
is `scipy.stats.gaussian_kde`; if KDE is numerically unstable or the point count
is high, the code falls back to a smoothed 2D histogram.

The MST gap metric can be expensive, so it deterministically samples at most
`--mst-max-points` normalized coordinates per subfield.

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
