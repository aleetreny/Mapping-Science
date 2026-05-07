# Task: Build per-subfield UMAP maps with scatter + KDE density panels

You are working in the repo:

```text
aleetreny/TFM
```

Implement the next pipeline stage for the TFM: generating one internal semantic UMAP map for each OpenAlex subfield in the main-analysis set.

## Goal

Create a reproducible pipeline that generates, for each eligible OpenAlex subfield, one PNG with two panels:

1. **Panel A — Scatter**
   - 2D UMAP coordinates of papers in that subfield.
   - Normal point cloud visualization.

2. **Panel B — Density**
   - Density view of the same 2D coordinates.
   - Use kernel density estimation or an equivalent smooth density estimator.
   - Render with the `viridis` colormap.

The purpose is visual inspection of the internal semantic morphology of each discipline/subfield.

Do **not** add morphology metrics yet.
Do **not** add clustering.
Do **not** add predictive models.
Do **not** add dashboards.
Do **not** use PCA.

---

## Existing repo context

The repo already has:

```text
data/processed/analysis_subfields.parquet
data/processed/analysis_embedding_index.parquet
embeddings/specter2_v1/analysis/main_embeddings.float16.npy
embeddings/specter2_v1/analysis/main_work_ids.parquet
embeddings/specter2_v1/analysis/main_matrix_summary.json
```

The matrix has around:

```text
717,845 rows
240 main-analysis subfields
768-dimensional SPECTER2 embeddings
float16 dtype
```

The row order of `main_embeddings.float16.npy` corresponds to `analysis_embedding_index.parquet`.

There is already a global UMAP pipeline:

```text
scripts/09_build_first_umap_maps.py
src/umap_maps.py
docs/analysis_matrix_and_first_umap.md
```

Reuse existing utilities where sensible, but do not break the global UMAP script.

---

## Methodological constraints

This is important.

The thesis design is:

```text
Input morphology window: 2010-2019
Target growth window: 2020-2025
Unit of analysis: OpenAlex subfield
```

Therefore:

- The per-subfield maps should use only the morphology input window by default: `2010 <= publication_year <= 2019`.
- Do **not** use 2020-2025 text/embeddings as input for these maps.
- If `publication_year` is not available in the existing `analysis_embedding_index.parquet`, fail loudly with a clear error message instead of silently using all years.
- Add CLI options to override the year window only for debugging/robustness, but defaults must be 2010-2019.

Use SPECTER2 embeddings directly.
Do not run PCA before UMAP.

---

## Expected implementation

Add a new script, for example:

```text
scripts/10_build_per_subfield_umap_maps.py
```

The script should:

1. Load the analysis index:

```text
data/processed/analysis_embedding_index.parquet
```

2. Load the main embedding matrix with memory mapping:

```python
np.load(path, mmap_mode="r")
```

Do not load the full matrix into RAM unnecessarily.

3. Filter to:

```text
main-analysis subfields
2010-2019 publication years by default
```

4. Group works by subfield.

Important: inspect the actual columns in `analysis_embedding_index.parquet`. Use the correct subfield identifier/name columns already present in the repo. Do not invent column names without checking.

5. For each subfield:
   - Select the row indices corresponding to that subfield.
   - Extract only that subfield’s embeddings.
   - Convert only that subfield subset to `float32` before UMAP.
   - Fit a separate UMAP model for that subfield.
   - Save a `.parquet` file with coordinates and metadata.
   - Save a `.png` file with the two-panel visualization.

6. Use deterministic settings:
   - Fixed `random_state`.
   - Stable ordering.
   - Deterministic output filenames.

7. Add CLI options such as:

```text
--index-path data/processed/analysis_embedding_index.parquet
--embeddings-path embeddings/specter2_v1/analysis/main_embeddings.float16.npy
--output-dir outputs/maps/per_subfield_umap
--year-min 2010
--year-max 2019
--min-papers 250
--max-papers-per-subfield 10000
--subfield-id <optional single subfield id>
--limit-subfields <optional integer for test runs>
--random-state 42
--n-neighbors 30
--min-dist 0.05
--metric cosine
--dpi 180
--overwrite
```

Default behavior:
- Run all main-analysis subfields.
- Use years 2010-2019.
- Use at most 10,000 papers per subfield for runtime safety.
- If a subfield has more than `max_papers_per_subfield`, sample deterministically.
- If a subfield has fewer than `min_papers`, skip it and record this in the summary.

8. Produce outputs like:

```text
outputs/maps/per_subfield_umap/
  coordinates/
    <safe_subfield_id>__<safe_subfield_name>.parquet
  figures/
    <safe_subfield_id>__<safe_subfield_name>.png
  per_subfield_umap_summary.json
  per_subfield_umap_manifest.parquet
```

The manifest should contain one row per attempted subfield, including:

```text
subfield_id
subfield_name
n_available
n_used
year_min
year_max
status: completed/skipped/failed
coordinate_path
figure_path
error_message
umap_n_neighbors
umap_min_dist
umap_metric
random_state
```

The summary JSON should include:
- run timestamp,
- input paths,
- output paths,
- year window,
- number of subfields attempted,
- number completed,
- number skipped,
- number failed,
- total papers used,
- UMAP parameters,
- sampling parameters.

---

## Visualization requirements

Each PNG must have two panels in the same figure.

### Panel A: Scatter

- Use matplotlib.
- Show points with small size and alpha.
- Avoid huge markers.
- No legend unless useful.
- Title should include:
  - subfield name,
  - number of papers used,
  - year window.

### Panel B: Density

Preferred implementation:
- Use `scipy.stats.gaussian_kde` if available and not too slow.
- Otherwise use a robust fallback such as `matplotlib.pyplot.hexbin` or smoothed 2D histogram.
- The panel must use the `viridis` colormap.
- Add a colorbar only if it does not make the figure too cluttered.
- Use the same coordinate limits as the scatter panel.

Important:
- The density panel should be based on the UMAP coordinates, not on the original 768-dimensional embeddings.
- Keep axis labels simple: `UMAP 1`, `UMAP 2`.
- Use `tight_layout()` or equivalent.
- Close figures after saving to avoid memory leaks.

---

## Runtime and RAM constraints

Be careful. This may run over ~240 subfields.

Requirements:

- Use memory mapping for the main `.npy`.
- Process subfields sequentially by default.
- Do not create a giant list of all subfield embedding matrices.
- Do not store all UMAP coordinates in one huge in-memory object.
- Save each subfield’s coordinates and figure immediately.
- Add useful progress logging.
- Allow small test runs:

```bash
python scripts/10_build_per_subfield_umap_maps.py --limit-subfields 3 --max-papers-per-subfield 2000 --overwrite
```

Also allow running a single subfield:

```bash
python scripts/10_build_per_subfield_umap_maps.py --subfield-id <ID> --overwrite
```

---

## Tests

Add tests for the new functionality.

Do not write tests that require running full UMAP on the full dataset.

Add lightweight tests for:

1. Safe filename generation.
2. Deterministic sampling.
3. Manifest row creation for completed/skipped/failed subfields.
4. Year-window filtering.
5. Validation that the script fails clearly if required columns are missing.
6. Density plotting helper works on a small synthetic coordinate array.
7. The script can run on a tiny synthetic embedding matrix and index.

Use synthetic data in tests.

Run the full test suite and make sure it passes.

---

## Documentation

Update or create a doc, for example:

```text
docs/per_subfield_umap_maps.md
```

The doc should explain:

- Purpose of this stage.
- Inputs.
- Outputs.
- Default year window: 2010-2019.
- Why 2020-2025 is excluded from map inputs.
- CLI examples:
  - test run,
  - single subfield,
  - full run.
- Notes on runtime and RAM.
- Explanation that these are visual inspection maps only; morphology metrics are not implemented yet.

Also update any pipeline index/readme if the repo has one.

---

## Git hygiene

Generated outputs must remain ignored by Git.

Check `.gitignore` and make sure these are ignored:

```text
outputs/maps/per_subfield_umap/
```

Do not commit PNGs, parquet coordinate outputs, summaries, or large generated artifacts.

Do not commit secrets, tokens, Drive credentials, Kaggle credentials, or local machine paths.

---

## Acceptance criteria

The task is complete only if:

1. A per-subfield UMAP script exists and runs from CLI.
2. It uses the existing SPECTER2 main matrix directly.
3. It filters to 2010-2019 by default.
4. It does not use PCA.
5. It does not use clustering.
6. It does not compute morphology metrics.
7. It creates one PNG per subfield with:
   - scatter panel,
   - density/KDE panel using viridis.
8. It saves per-subfield coordinate parquet files.
9. It writes a manifest and summary JSON.
10. It supports small test runs with `--limit-subfields`.
11. It supports a single-subfield run with `--subfield-id`.
12. It uses memory mapping and avoids loading the full matrix unnecessarily.
13. Docs are updated.
14. Tests are added and pass.
15. Generated artifacts are ignored by Git.

After implementation, report:

- files changed,
- exact commands run,
- test results,
- example command for a 3-subfield smoke test,
- any assumptions made about index column names,
- any skipped or failed subfields in the smoke test.
