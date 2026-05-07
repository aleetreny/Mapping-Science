# Task: Implement subfield morphology metrics from per-subfield UMAP coordinates

You are working in the repo:

```text
aleetreny/TFM
```

Act as a strict technical research assistant. Implement the next stage of the TFM pipeline: a tabular morphology-metrics dataset with one row per OpenAlex subfield.

## Current repo state

The repo now has a per-subfield UMAP stage:

```text
scripts/10_build_per_subfield_umap_maps.py
src/per_subfield_umap_maps.py
docs/per_subfield_umap_maps.md
tests/test_per_subfield_umap_maps.py
```

The script generates one UMAP per subfield using only the morphology input window by default:

```text
2010-2019
```

and saves:

```text
outputs/maps/per_subfield_umap/
  coordinates/
    <safe_subfield_id>__<safe_subfield_name>.parquet
  figures/
    <safe_subfield_id>__<safe_subfield_name>.png
  per_subfield_umap_manifest.parquet
  per_subfield_umap_summary.json
```

Each coordinate parquet should contain at least:

```text
work_id
analysis_row_id
subfield_id
subfield_display_name
field_id
field_display_name
domain_id
domain_display_name
primary_topic_id
primary_topic_display_name
publication_year
umap_x
umap_y
```

The new stage must consume these coordinate parquet files. Do **not** rerun UMAP in this task.

---

## Goal

Create a reproducible pipeline that computes a final tabular dataset of morphology metrics for every completed subfield map.

The final output must be a table with:

```text
one row per subfield
metadata columns for the subfield/field/domain
25 morphology metrics
quality/control columns
```

Primary outputs:

```text
data/processed/subfield_morphology_metrics.parquet
data/processed/subfield_morphology_metrics.csv
outputs/metrics/subfield_morphology_metrics_summary.json
outputs/metrics/subfield_morphology_metrics_dictionary.csv
```

The `.parquet` file is the main machine-readable table.
The `.csv` file is for quick inspection.
The dictionary CSV must explain each metric in plain English.

Do **not** add prediction models yet.
Do **not** add growth targets yet.
Do **not** add dashboards.
Do **not** add clustering methods such as Leiden/HDBSCAN/KMeans.
Do **not** use PCA.
Do **not** use 2020-2025 data.

---

## Methodological constraint: coordinate comparability

Each subfield UMAP was fitted separately. Therefore raw coordinates, raw orientation and raw scale are not directly comparable across subfields.

Before computing the 25 core morphology metrics, normalize each subfield's coordinates as follows:

1. Take finite `(umap_x, umap_y)` points only.
2. Center by the coordinate-wise median:
   ```python
   center = np.nanmedian(coords, axis=0)
   ```
3. Compute radial distances from that center.
4. Compute `r95 = np.percentile(radial_distance, 95)`.
5. If `r95 <= 0` or non-finite, fail or mark metrics as failed for that subfield.
6. Use:
   ```python
   coords_norm = (coords - center) / r95
   ```

All 25 final morphology metrics must be computed from `coords_norm`, not raw coordinates.

Also save raw diagnostic columns separately where useful, e.g.:

```text
raw_umap_x_min
raw_umap_x_max
raw_umap_y_min
raw_umap_y_max
raw_r95
```

but do **not** treat raw coordinate values as main morphology features.

The 25 metric definitions below should be invariant to translation, rotation and scale as far as possible.

---

## Add new module

Add a new module, for example:

```text
src/morphology_metrics.py
```

It should contain pure, testable functions for:

- coordinate validation,
- coordinate normalization,
- KDE grid construction,
- density concentration metrics,
- local KNN metrics,
- peak/component metrics,
- support/shape metrics,
- temporal metrics,
- row assembly for one subfield,
- schema/dictionary definitions.

Avoid placing all logic inside the CLI script.

---

## Add new script

Add a script:

```text
scripts/11_compute_subfield_morphology_metrics.py
```

The script should:

1. Read the per-subfield UMAP manifest:

   ```text
   outputs/maps/per_subfield_umap/per_subfield_umap_manifest.parquet
   ```

2. Keep only rows with:

   ```text
   status == "completed"
   ```

3. For each completed subfield:
   - read the coordinate parquet from `coordinate_path`,
   - validate required columns,
   - normalize coordinates,
   - compute the 25 metrics,
   - append one result row.

4. Write:

   ```text
   data/processed/subfield_morphology_metrics.parquet
   data/processed/subfield_morphology_metrics.csv
   outputs/metrics/subfield_morphology_metrics_summary.json
   outputs/metrics/subfield_morphology_metrics_dictionary.csv
   ```

5. Print progress in the terminal in a useful format:

   ```text
   [12/240 | 5.0%] 1602 - Analytical Chemistry: completed
   ```

6. Fail loudly for global input problems, but for one bad subfield:
   - record `metric_status = "failed"`,
   - write `metric_error_message`,
   - continue with the next subfield.

7. Include CLI options:

   ```text
   --manifest-path outputs/maps/per_subfield_umap/per_subfield_umap_manifest.parquet
   --output-parquet data/processed/subfield_morphology_metrics.parquet
   --output-csv data/processed/subfield_morphology_metrics.csv
   --summary-path outputs/metrics/subfield_morphology_metrics_summary.json
   --dictionary-path outputs/metrics/subfield_morphology_metrics_dictionary.csv
   --limit-subfields <optional int>
   --subfield-id <optional id>
   --grid-size 160
   --k-neighbors 15
   --mst-max-points 3000
   --random-state 42
   --overwrite
   ```

Default behavior:
- process every completed subfield in the manifest;
- use all rows in each coordinate parquet;
- deterministic subsampling only for expensive MST computation if needed.

Example command for full run:

```powershell
.\.venv\Scripts\python.exe scripts\11_compute_subfield_morphology_metrics.py --overwrite
```

Example smoke test:

```powershell
.\.venv\Scripts\python.exe scripts\11_compute_subfield_morphology_metrics.py --limit-subfields 3 --overwrite
```

---

## The 25 final morphology metrics

Implement exactly these 25 metric columns.

### A. Radial dispersion and local granularity

#### 1. `radial_tail_index`

Definition:

```text
r95 / r50
```

where `rXX` is the XXth percentile of normalized radial distance from the robust center.

Interpretation:
- high value = long semantic periphery / tails;
- low value = compact field.

Because coordinates are divided by raw `r95`, normalized `r95` should be close to 1, but this ratio remains informative because it compares tail to median radius.

#### 2. `radial_iqr_index`

Definition:

```text
(r75 - r25) / r50
```

Interpretation:
- high value = broad internal radial spread;
- low value = homogeneous radial spread.

#### 3. `knn_median_distance`

Use `sklearn.neighbors.NearestNeighbors`.

Definition:
- fit on `coords_norm`;
- use `k_neighbors + 1` because each point is its own nearest neighbour;
- remove the self-neighbour;
- take the distance to the `k_neighbors`-th neighbour or the mean over neighbours;
- use one consistent definition and document it in the dictionary.

Recommended:
```text
median distance to the k-th nearest neighbour, k=15 by default
```

Interpretation:
- high = locally sparse field;
- low = locally compact field.

#### 4. `knn_distance_cv`

Definition:

```text
std(kNN distances) / mean(kNN distances)
```

Use the same per-point kNN distance used in metric 3.

Interpretation:
- high = heterogeneous local density;
- low = even local density.

---

### B. Density concentration

Build a KDE-like 2D density grid from `coords_norm`.

Preferred:
- use `scipy.stats.gaussian_kde` when possible.
- evaluate on a fixed square grid covering the normalized coordinates with padding.

Fallback:
- use `numpy.histogram2d` plus `scipy.ndimage.gaussian_filter`.

The density grid must be non-negative and normalized to sum to 1 for entropy/mass metrics.

Use a fixed `grid_size` default of 160.

#### 5. `density_entropy`

Definition:

```text
Shannon entropy of positive density grid cells, normalized by log(number_positive_cells)
```

Range should be roughly `[0, 1]`.

Interpretation:
- high = mass spread across the map;
- low = mass concentrated in few cells.

#### 6. `density_gini`

Gini coefficient of density values over occupied/positive grid cells.

Interpretation:
- high = density dominated by few cells;
- low = density evenly distributed.

#### 7. `peak_dominance`

Definition:

```text
max_density / mean_positive_density
```

Interpretation:
- high = one extremely dominant density peak;
- low = no dominant peak.

#### 8. `effective_area_50`

Definition:
- sort density cells descending;
- find the minimum number of grid cells needed to accumulate 50% of total density mass;
- multiply by normalized cell area.

Interpretation:
- area of the high-density core.

#### 9. `effective_area_90`

Same as above, but for 90% of total density mass.

Interpretation:
- effective semantic footprint of the field.

#### 10. `core_periphery_ratio`

Definition:

```text
effective_area_50 / effective_area_90
```

Interpretation:
- low = small dense core and broad periphery;
- high = core occupies a large share of the functional area.

---

### C. Multimodality and fragmentation

Use the normalized KDE grid.

Define:
- `support_mask`: cells with density above a low threshold.
- `core_mask`: cells in the top-density region that accumulates a fixed mass or above a high threshold.

Recommended robust thresholds:
- support: cells required to accumulate 95% of mass, converted to a binary mask.
- core/dense region: cells required to accumulate 50% or 75% of mass.

Document the exact implementation.

#### 11. `density_peak_count`

Detect local maxima on the density grid.

Implementation guidance:
- use `scipy.ndimage.maximum_filter`;
- a peak is a cell equal to the local maximum in its neighbourhood;
- ignore tiny peaks below a relative threshold, e.g. `density >= 0.10 * max_density`;
- optionally require a minimum separation in grid cells;
- return integer count.

Interpretation:
- high = many semantic foci.

#### 12. `peak_mass_entropy`

Estimate the relative mass around detected peaks.

Implementation guidance:
- assign each high-density/core cell to nearest detected peak in grid coordinates;
- sum density mass per peak;
- compute normalized Shannon entropy of peak masses.

Fallback:
- if only one peak exists, return `0.0`;
- if no peak can be detected, return `NaN` and explain in summary.

Interpretation:
- high = several balanced peaks;
- low = one peak dominates.

#### 13. `dense_component_count`

Number of connected components in the dense/core region.

Implementation:
- use `scipy.ndimage.label`;
- 8-connectivity is acceptable;
- define the dense region consistently, preferably the cells accumulating 50% of KDE mass.

Interpretation:
- high = multiple dense islands.

#### 14. `largest_component_mass_share`

Definition:
- for the same dense/core region, compute density mass per connected component;
- return largest component mass / total dense-region mass.

Interpretation:
- high = one dominant dense island;
- low = several balanced dense islands.

#### 15. `component_separation_index`

Definition:
- identify dense/core connected components;
- for the two largest components by mass:
  - compute their density-weighted centroids in normalized coordinate units;
  - compute distance between centroids;
  - divide by average within-component radius or by square root of average component area.
- If fewer than two components exist, return `0.0`.

Interpretation:
- high = separated semantic islands;
- low = one component or weak separation.

#### 16. `mst_gap_index`

Use a minimum spanning tree on normalized point coordinates.

Implementation:
- if `n_points > --mst-max-points`, deterministically sample `mst_max_points` points using `random_state` and `subfield_id`;
- compute pairwise Euclidean distances only on that MST subset;
- build MST using `scipy.sparse.csgraph.minimum_spanning_tree`;
- extract MST edge lengths;
- compute:

```text
p95_edge_length / median_edge_length
```

or:

```text
max_edge_length / median_edge_length
```

Recommended: use `p95 / median` to reduce single-outlier sensitivity.

Interpretation:
- high = large gaps between parts of the map;
- low = continuous map.

---

### D. Shape and topology of occupied support

Use the normalized KDE support mask, not raw convex hull alone.

Do not add obscure dependencies such as `alphashape`. Prefer robust rasterized support metrics using `scipy.ndimage` and, if needed, `scipy.spatial.ConvexHull`.

#### 17. `anisotropy_ratio`

Compute covariance of `coords_norm`.

Definition:

```text
sqrt(lambda_max / lambda_min)
```

where `lambda_*` are covariance eigenvalues.

Interpretation:
- high = elongated field / gradient;
- near 1 = rounder field.

Handle zero eigenvalues with epsilon.

#### 18. `support_solidity`

Definition:

```text
occupied_support_area / convex_hull_area_of_support_cells
```

Use support mask cells converted to coordinate points, or use original point convex hull as fallback.

Interpretation:
- high = compact filled shape;
- low = concave, fragmented or holey shape.

#### 19. `support_circularity`

Definition:

```text
4 * pi * support_area / support_perimeter^2
```

Area and perimeter are computed from the binary support mask.

Interpretation:
- high = compact round support;
- low = elongated/irregular support.

#### 20. `boundary_complexity`

Definition:

```text
support_perimeter^2 / (4 * pi * support_area)
```

This is the inverse of circularity.

Interpretation:
- high = irregular or ramified boundary.

#### 21. `hole_count`

Count holes in the support mask.

Implementation:
- use connected-component logic on the complement of the support mask;
- ignore the outside background component touching the grid border;
- count remaining enclosed background components.

Interpretation:
- high = occupied region contains internal voids.

---

### E. Temporal morphology inside 2010-2019 only

Use the `publication_year` column from the coordinate parquet. Never use 2020-2025.

Use normalized coordinates.

#### 22. `centroid_drift_early_late`

Definition:
- early period: 2010-2012;
- late period: 2017-2019;
- compute robust centroids as coordinate medians for each period;
- return Euclidean distance between early and late centroids.

If either period has too few papers, return `NaN` and record a warning flag.

Interpretation:
- high = net semantic displacement before 2020.

#### 23. `annual_centroid_path_length`

Definition:
- compute annual median centroid for each year in 2010-2019 where enough papers exist;
- sort by year;
- sum Euclidean distances between consecutive available annual centroids.

Interpretation:
- high = internally dynamic or unstable field.

#### 24. `directionality_ratio`

Definition:

```text
net early-to-late centroid distance / annual_centroid_path_length
```

If path length is zero or unavailable, return `NaN`.

Interpretation:
- high = movement is coherent in one direction;
- low = movement is wandering/back-and-forth.

#### 25. `radial_expansion_slope`

Definition:
- for each year, compute median radial distance from the global robust center of the normalized map;
- regress annual median radius on year using simple least squares;
- return the slope.

Do not use 2020-2025.

Interpretation:
- positive = field was semantically expanding during 2010-2019;
- negative = field was contracting.

---

## Required metadata and control columns

The final table must include at least:

```text
subfield_id
subfield_display_name
field_id
field_display_name
domain_id
domain_display_name
n_points
year_min
year_max
metric_status
metric_error_message
coordinate_path
```

Add useful controls:

```text
raw_umap_x_min
raw_umap_x_max
raw_umap_y_min
raw_umap_y_max
raw_r95
normalized_center_x
normalized_center_y
grid_size
k_neighbors
mst_points_used
mst_sampling_applied
n_years_available
n_early_points
n_late_points
```

The final metric columns must be exactly the 25 names above:

```text
radial_tail_index
radial_iqr_index
knn_median_distance
knn_distance_cv
density_entropy
density_gini
peak_dominance
effective_area_50
effective_area_90
core_periphery_ratio
density_peak_count
peak_mass_entropy
dense_component_count
largest_component_mass_share
component_separation_index
mst_gap_index
anisotropy_ratio
support_solidity
support_circularity
boundary_complexity
hole_count
centroid_drift_early_late
annual_centroid_path_length
directionality_ratio
radial_expansion_slope
```

Use `NaN` for metrics that are not computable for a specific subfield, but preserve the row.

---

## Metric dictionary

Create:

```text
outputs/metrics/subfield_morphology_metrics_dictionary.csv
```

with columns:

```text
metric_name
family
definition
interpretation
higher_means
computed_on
notes
```

It must include the 25 metrics plus the most important control columns.

This dictionary matters because the TFM needs to explain the table academically.

---

## Summary JSON

Create:

```text
outputs/metrics/subfield_morphology_metrics_summary.json
```

Include:

```text
created_at
manifest_path
output_parquet
output_csv
dictionary_path
n_subfields_in_manifest
n_subfields_attempted
n_completed
n_failed
metric_columns
control_columns
grid_size
k_neighbors
mst_max_points
random_state
status_counts
failed_subfields
warnings
```

---

## Tests

Add tests, for example:

```text
tests/test_morphology_metrics.py
```

Do not require the real 240 subfield maps in tests.

Use synthetic coordinate frames.

Test at least:

1. Coordinate normalization is translation- and scale-invariant.
2. `radial_tail_index` and `radial_iqr_index` are finite on a simple cloud.
3. Density grid sums to approximately 1.
4. Density entropy is higher for a spread-out/uniform-ish cloud than for a concentrated cloud.
5. Peak count detects at least two peaks in a two-blob synthetic cloud.
6. Dense component count is higher for separated blobs than for one blob.
7. Anisotropy is higher for an elongated cloud than for a round cloud.
8. Temporal drift is positive when synthetic yearly centroids move steadily.
9. The CLI can run on a tiny synthetic manifest + coordinate parquet directory.
10. The final output table has exactly one row per completed manifest row and includes all 25 metric columns.

Keep tests lightweight and deterministic.

---

## Documentation

Create or update:

```text
docs/subfield_morphology_metrics.md
```

Explain:

- purpose of the stage;
- inputs and outputs;
- why coordinates are normalized per subfield;
- why raw UMAP scale/orientation are not used as core features;
- the 25 metric families;
- how temporal metrics avoid 2020-2025 leakage;
- CLI commands;
- interpretation caveats;
- that this stage does not build prediction models yet.

Also update any pipeline index/readme if the repo has one.

---

## Git hygiene

Generated outputs must remain ignored by Git.

Ensure `.gitignore` ignores:

```text
outputs/metrics/
data/processed/subfield_morphology_metrics.parquet
data/processed/subfield_morphology_metrics.csv
```

Do not commit generated metric tables.
Do not commit coordinate parquets.
Do not commit PNGs.
Do not commit secrets/tokens or local machine paths.

---

## Implementation guidance

### Use only current dependencies where possible

`requirements.txt` already includes:

```text
numpy
pandas
pyarrow
scikit-learn
matplotlib
umap-learn
pytest
```

UMAP/scikit environments often include `scipy` transitively, but if you import `scipy` directly, add `scipy` explicitly to `requirements.txt`.

Avoid adding heavy or obscure dependencies unless absolutely necessary.

### Be careful with O(n^2)

The MST metric can be O(n^2). That is acceptable for ~3,000 points per subfield, but guard it with:

```text
--mst-max-points 3000
```

If a subfield has more points, sample deterministically.

### Avoid silent failure

Do not silently fill failed metrics with zero unless zero is meaningful.

Preferred:

```text
metric_status = "completed"
```

or

```text
metric_status = "failed"
metric_error_message = "..."
```

For partially unavailable metrics, keep `metric_status = "completed_with_warnings"` only if you decide to support that status and document it.

### Avoid leakage

All metrics must come only from per-subfield coordinate files generated from the 2010-2019 UMAP stage.

Do not read 2020-2025 works.
Do not compute growth targets.
Do not join future publication counts.

---

## Acceptance criteria

The task is complete only if:

1. `scripts/11_compute_subfield_morphology_metrics.py` exists and runs from CLI.
2. `src/morphology_metrics.py` contains testable metric functions.
3. The script consumes `outputs/maps/per_subfield_umap/per_subfield_umap_manifest.parquet`.
4. It reads per-subfield coordinate parquets and does not rerun UMAP.
5. It computes exactly the 25 morphology metric columns listed above.
6. It includes subfield, field and domain metadata in the final table.
7. It writes:
   - `data/processed/subfield_morphology_metrics.parquet`
   - `data/processed/subfield_morphology_metrics.csv`
   - `outputs/metrics/subfield_morphology_metrics_summary.json`
   - `outputs/metrics/subfield_morphology_metrics_dictionary.csv`
8. It prints terminal progress with counts and percentages.
9. It supports `--limit-subfields`, `--subfield-id`, and `--overwrite`.
10. It normalizes coordinates per subfield before computing metrics.
11. It avoids using raw UMAP orientation/scale as morphology features.
12. It avoids 2020-2025 leakage.
13. It does not add prediction models, growth targets, clustering or PCA.
14. Tests are added and pass.
15. Docs are updated.
16. Generated metric outputs are ignored by Git.

After implementation, report:

- files changed;
- exact commands run;
- test results;
- example smoke-test command;
- example full-run command;
- number of metric rows generated in the smoke test;
- any assumptions made about coordinate parquet columns;
- any failed or warning subfields.
