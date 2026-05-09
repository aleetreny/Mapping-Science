# Agent Prompt — Fix Field/Domain Density Plots and Split 10b/10c Scripts

## Context

You are working inside the `aleetreny/TFM` repository.

The previous extension added a generic script:

```text
scripts/10b_build_per_category_umap_maps.py
```

with:

```bash
--level field
--level domain
```

It also added field/domain UMAP outputs.

The user has now inspected a generated **domain-level** PNG and the density panel looks wrong for the intended visual style: it falls back to a discrete hexbin / honeycomb-like map with many white holes. The desired density panel should look like the smoother density panels used elsewhere: continuous `viridis` density traces / smooth color field, not hexagonal bins.

The user also wants the field and domain scripts separated for easier use:

```text
scripts/10b_build_per_field_umap_maps.py
scripts/10c_build_per_domain_umap_maps.py
```

with clear names and simple commands.

Do not run full expensive map generation. Implement scripts/helpers/docs/tests only.

---

## Main requests

### 1. Fix density plots for field/domain maps

The right panel should be a continuous density visualization, not a discrete hexbin.

Current bad behavior:

```text
large group → KDE skipped because too many points → fallback to hexbin
```

Desired behavior:

```text
large group → use smoothed 2D histogram or efficient KDE-like raster
→ display with imshow/contourf using viridis
→ continuous-looking density panel
```

Do not rely on full `scipy.stats.gaussian_kde` for large groups, because domains/fields can have many points and KDE can be slow. Instead, implement an efficient continuous fallback:

```text
np.histogram2d → scipy.ndimage.gaussian_filter → normalized/smoothed density → imshow
```

This should be used when point count is above a safe KDE threshold, or even as the default for field/domain density panels.

The density panel should:

- use `viridis`;
- use a continuous raster/image, not hexbin;
- have stable axis limits equal to the scatter panel;
- use a colorbar;
- avoid white holes caused by empty hex bins;
- be visually comparable across field/domain maps;
- work for large groups without becoming too slow.

Recommended helper name:

```python
plot_continuous_density_panel(...)
```

or similar.

It can support:

```text
method = "auto" | "kde" | "smooth_hist"
```

Recommended default for field/domain:

```text
density_method = "smooth_hist"
```

Recommended default for subfield can remain `"auto"` if existing subfield plots already look good, but it is acceptable to move all UMAP map density panels to the same continuous helper if it improves consistency.

---

## 2. Keep density metrics separate from plotting

Do not confuse this plotting change with the metric computation in:

```text
src/morphology_metrics.py
scripts/11_compute_subfield_morphology_metrics.py
```

The metric computation can stay as it is unless there is a clear bug.

This task is about the **PNG density panel**, not changing the scientific metric definitions.

---

## 3. Split field and domain scripts

Replace the active user-facing generic command:

```bash
python scripts/10b_build_per_category_umap_maps.py --level field ...
python scripts/10b_build_per_category_umap_maps.py --level domain ...
```

with two clearer scripts:

```bash
python scripts/10b_build_per_field_umap_maps.py --year-min 2010 --year-max 2025 --overwrite
python scripts/10c_build_per_domain_umap_maps.py --year-min 2010 --year-max 2025 --overwrite
```

### Recommended implementation

Keep reusable logic in:

```text
src/per_category_umap_maps.py
```

or refactor it into a clearer shared module if needed.

Then make:

```text
scripts/10b_build_per_field_umap_maps.py
scripts/10c_build_per_domain_umap_maps.py
```

small wrappers around the shared implementation.

This avoids duplicated code but gives the user clear commands.

### What to do with the existing generic script

Choose the cleanest option:

#### Preferred

Keep `scripts/10b_build_per_category_umap_maps.py` only if needed internally or for backwards compatibility, but remove it from the active README pipeline. If keeping it, add a warning in its help/description that the clearer active scripts are now:

```text
10b_build_per_field_umap_maps.py
10c_build_per_domain_umap_maps.py
```

#### Alternative

Rename/replace it entirely if that is cleaner and tests/docs are updated.

Do not leave README/docs telling the user to run the old generic script as the main path.

---

## 4. Expected outputs remain the same

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

Do not change output folder names unless absolutely necessary.

---

## 5. CLI arguments

Both new scripts should expose the same useful arguments as the generic script, but without requiring `--level`.

### Field script

```bash
python scripts/10b_build_per_field_umap_maps.py \
  --year-min 2010 \
  --year-max 2025 \
  --max-papers-per-group 10000 \
  --overwrite
```

Optional:

```text
--field-id
--limit-fields
--min-papers
--random-state
--n-neighbors
--min-dist
--metric
--dpi
--density-method auto|kde|smooth_hist
--density-grid-size
--density-sigma
```

If you prefer generic names internally, still expose user-friendly aliases like `--field-id` and `--limit-fields`.

### Domain script

```bash
python scripts/10c_build_per_domain_umap_maps.py \
  --year-min 2010 \
  --year-max 2025 \
  --max-papers-per-group 10000 \
  --overwrite
```

Optional:

```text
--domain-id
--limit-domains
--min-papers
--random-state
--n-neighbors
--min-dist
--metric
--dpi
--density-method auto|kde|smooth_hist
--density-grid-size
--density-sigma
```

If the shared implementation expects `group_id`, map the user-facing argument cleanly.

---

## 6. Update README and docs

Update the active pipeline commands.

Old:

```bash
python scripts/10b_build_per_category_umap_maps.py --level field --year-min 2010 --year-max 2025 --overwrite
python scripts/10b_build_per_category_umap_maps.py --level domain --year-min 2010 --year-max 2025 --overwrite
```

New:

```bash
python scripts/10b_build_per_field_umap_maps.py --year-min 2010 --year-max 2025 --overwrite
python scripts/10c_build_per_domain_umap_maps.py --year-min 2010 --year-max 2025 --overwrite
```

Also update:

```text
docs/higher_level_umap_maps.md
docs/per_subfield_umap_maps.md
README.md
```

If relevant, mention:

> Field and domain density panels use a smoothed 2D histogram for large groups to avoid slow KDE and avoid discrete hexbin artifacts.

---

## 7. Tests

Update or add tests.

Relevant test file likely exists:

```text
tests/test_per_category_umap_maps.py
```

Update it or add new tests such as:

```text
tests/test_higher_level_umap_plotting.py
tests/test_per_field_domain_scripts.py
```

Test without running heavy UMAP.

Minimum tests:

1. The continuous density helper returns/creates an image artist and does not use hexbin.
2. `smooth_hist` works on a synthetic cloud with 10,000 points.
3. The field wrapper passes `level="field"` or equivalent shared config.
4. The domain wrapper passes `level="domain"` or equivalent shared config.
5. Output paths for field/domain remain correct.
6. Existing tests still pass.

Run:

```bash
python -m pytest
python -m compileall scripts src
```

Do not leave failing tests.

---

## 8. Important plotting guidance

The bad domain image had density panel title:

```text
B. Density
```

but the visual was hexagonal/discrete. The new density should look more like:

```text
smooth continuous heatmap
soft viridis gradients
no honeycomb cells
no sparse white holes inside the support caused by hexbin discretization
```

Implementation suggestion:

```python
hist, x_edges, y_edges = np.histogram2d(
    x,
    y,
    bins=grid_size,
    range=[[xlim[0], xlim[1]], [ylim[0], ylim[1]]],
)
density = scipy.ndimage.gaussian_filter(hist.T, sigma=density_sigma)
density = np.nan_to_num(density)
ax.imshow(
    density,
    origin="lower",
    extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
    aspect="auto",
    cmap="viridis",
)
```

Make sure orientation is correct. The transpose may be needed depending on how `histogram2d` returns axes. Verify with a small synthetic test if possible.

Do not mask low-density cells too aggressively, because that can reintroduce holes.

---

## 9. Final response requested from the agent

When done, summarize:

1. Files changed.
2. New/renamed scripts.
3. Whether the generic `10b_build_per_category_umap_maps.py` was kept, deprecated, or removed.
4. How the density plot fallback changed.
5. The new commands to run field/domain maps.
6. Tests run and result.
7. Whether full expensive map generation was skipped.

Do not run full field/domain/domain-level UMAP generation unless explicitly requested.
