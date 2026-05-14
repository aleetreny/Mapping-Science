# Prompt for workspace agent — dimensionality-reduction comparison across metric families

## Objective
Build the next comparison stage of the subfield-morphology analysis.

We have already completed the clustering stage based on **UMAP-derived morphology metrics + embedding-space metrics**, and we already have a plot plus some conclusions about visual separation. The next task is to extend that analysis by comparing **six dimensionality-reduction techniques** on three feature sets:

1. **UMAP-only metrics**
2. **Embedding-only metrics**
3. **Combined metrics** (UMAP metrics + embedding metrics)

The main goal is to visually compare how the different methods organize subfields in 2D, and whether any of them reveal clearer neighborhoods or more interpretable structure than the current UMAP projection.

---

## Dimensionality-reduction methods to include
Use these **six** methods:

1. **PCA**
2. **MDS**
3. **t-SNE**
4. **UMAP**
5. **Isomap**
6. **PHATE**

If a method requires an extra dependency (especially PHATE), feel free to add it in the most sensible/reproducible way for this repository and document that clearly.

---

## High-level task
Please inspect the current repository state and reuse the existing pipeline, naming conventions, data products, plotting style, and clustering outputs as much as possible.

Then implement the same style of visualization for the six methods above, for each of the three feature sets:

- **UMAP-only**
- **Embedding-only**
- **Combined**

I want the output to make it easy to compare the methods side by side.

---

## Main deliverables
### A. Core visual outputs (highest priority)
Produce **three PNG figures**, one per feature set, each arranged as a **2x3 grid** with the six methods:

- **2x3 grid for UMAP-only metrics**
- **2x3 grid for embedding-only metrics**
- **2x3 grid for combined metrics**

Each grid should contain:

- PCA
- MDS
- t-SNE
- UMAP
- Isomap
- PHATE

Prefer a layout that is easy to scan visually, for example:

- Row 1: PCA, MDS, t-SNE
- Row 2: UMAP, Isomap, PHATE

These PNGs are the most important artifacts.

### B. Supporting outputs
Also generate the supporting outputs needed to make the analysis reusable and interpretable:

1. **A script or pipeline step** integrated into the repo that reproduces the figures.
2. **Saved 2D coordinates** for each projection/method/feature-set (CSV, parquet, or the project’s standard format).
3. **A concise markdown summary** explaining what was done and highlighting the main visual differences across methods.

If it fits the existing repo structure, also save any metadata/config used (random seeds, hyperparameters, standardized feature sets, etc.).

---

## Desired visualization style
Please make the new figures as consistent as possible with the current plot style already used in the repository.

### Reuse / preserve if already present
If the current visualization already uses some or all of the following, preserve them unless there is a strong reason not to:

- colors for clusters
- marker shapes for domains
- a few labels for representative subfields
- legend style
- plot titles / subtitles
- naming conventions

### Flexibility
You have freedom to adapt details based on what already exists in the repo. For example:

- If the current cluster coloring logic should be reused, reuse it.
- If separate feature-set-specific cluster labels are more appropriate, you may do that, but be explicit in the summary.
- If label density is too high, label a carefully chosen subset of representative or outlying subfields.

The priority is to keep the visuals clean, consistent, and comparable.

---

## Data / feature-set expectations
Please infer the exact feature definitions from the repository context, but conceptually the three matrices should be:

### 1. UMAP-only metrics
Only the metrics derived from the 2D UMAP paper maps (shape, spread, density, fragmentation, geometry, etc.).

### 2. Embedding-only metrics
Only the metrics derived from the original embedding space / semantic structure.

### 3. Combined metrics
The union of both groups.

Please standardize / preprocess each feature matrix appropriately before applying the dimensionality-reduction methods, following best practice and the project’s existing conventions.

---

## Methodological guidance
Use best judgment here; you have more repository context than I do. Still, please keep the following in mind:

1. **Comparability matters.** Use sensible, consistent preprocessing across methods and feature sets.
2. **Reproducibility matters.** Fix random seeds where appropriate.
3. **Avoid overfitting aesthetics.** The goal is not to tune each method until it “looks nice”, but to obtain a fair comparison.
4. **Parameter choices should be reasonable and documented.**
5. **If some methods need special handling** (e.g. MDS distance computation, PHATE parameters, t-SNE perplexity, Isomap neighbors), document the chosen defaults.

You do **not** need to spend forever optimizing every projection. A solid, fair comparison is better than endless tuning.

---

## Optional but valuable additions
If straightforward, include some lightweight quantitative comparison or notes, for example:

- trustworthiness
- neighborhood preservation
- qualitative cluster/domain coherence
- whether the space looks continuous vs clearly separated

This is optional. Do it only if it fits naturally and does not bloat the task.

---

## Expected interpretation focus
In the brief summary, please comment on things like:

- whether any method shows clearer local neighborhoods
- whether any method shows stronger apparent separation between clusters/domains
- whether the structure looks continuous rather than sharply partitioned
- whether UMAP-only, embedding-only, or combined metrics appear more informative visually
- whether different methods largely agree or tell different stories

The summary should be concise and decision-oriented, not a huge essay.

---

## Suggested output naming (adapt if repo conventions differ)
Please use the repository’s preferred folder structure and naming scheme. If there is no strong convention, something like this is fine:

### Figures
- `dr_comparison_umap_only_grid.png`
- `dr_comparison_embedding_only_grid.png`
- `dr_comparison_combined_grid.png`

### Coordinates
- `dr_coords_umap_only.csv`
- `dr_coords_embedding_only.csv`
- `dr_coords_combined.csv`

### Summary
- `dr_comparison_summary.md`

### Script
- a new script in the project’s scripts folder, e.g. something like:
  - `scripts/.../build_dimensionality_reduction_comparison.py`

---

## Minimum acceptance criteria
The task is successful if, at minimum:

1. The repo contains a reproducible script/pipeline step for the comparison.
2. There are **three final PNGs**, each one a **2x3 grid** comparing the six dimensionality-reduction methods.
3. The three PNGs correspond to:
   - UMAP-only metrics
   - embedding-only metrics
   - combined metrics
4. A short markdown summary explains what was done and the main visual takeaways.

---

## Final note
Please use initiative and adapt to the repository’s current state. You already have the workspace context, so prioritize consistency with the existing analysis rather than blindly following generic assumptions. If you need to make a choice that is not fully determined from this prompt, make the most sensible one and document it briefly.
