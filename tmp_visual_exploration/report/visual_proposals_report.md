# Visual Proposals Report: Selected TFM Conceptual Figures

This report presents a structured analysis and description of the **three premium conceptual figures** selected for integration into the Master Thesis *"Measuring the Shape of Science: Morphological Indicators and Evolution of Research Fields"*. 

All deliverables are generated using a dedicated Python script and are saved in the isolated workspace: `tmp_visual_exploration/`.

---

## 1. Selected Visual Figures

### Figure 1: Citation Networks vs. Semantic Continuous Spaces
- **File**: [fig_exp_11_citation_vs_semantic.png](file:///c:/Users/Z0058EYW/Workspace/TFM/tmp_visual_exploration/figures/fig_exp_11_citation_vs_semantic.png)
- **Suggested Placement**: Chapter 2 (Background and Related Work), Section 2.2 (The Paradigm Shift)
- **What it Shows**: A side-by-side comparative layout:
  - *Panel A (Discrete Citation Graph)*: Visualizes papers as nodes connected by citation links, color-coded into discrete communities (navy vs. red). Illustrates boundary-bound topology.
  - *Panel B (Continuous Semantic Space)*: Visualizes a continuous density contour field of SPECTER2 embeddings (navy-blue gradients), demonstrating that distance represents meaning and papers occupy a continuous spectrum without hard link boundaries.
- **Why it Adds Value**: Essential for pedagogical clarity. Visually anchors the core theoretical paradigm shift of the thesis—moving from traditional citation graphs to dense vector representation of text.

---

### Figure 2: Trajectory Evolution Modes (Refined Contrast & Red Contraction)
- **File**: [fig_exp_16_trajectory_modes.png](file:///c:/Users/Z0058EYW/Workspace/TFM/tmp_visual_exploration/figures/fig_exp_16_trajectory_modes.png)
- **Suggested Placement**: Chapter 7 (Temporal Evolution of Scientific Morphology), Section 7.1 (Theoretical Trajectories)
- **What it Shows**: A 3-panel conceptual layout comparing:
  - *Panel A (Centroid Drift)*: Centroid moves from Epoch 1 to Epoch 2; dispersion remains constant.
  - *Panel B (Area Expansion)*: Point cloud expands outward; Epoch 1 is slate gray, Epoch 2 is coral-red.
  - *Panel C (Contraction / Densification)*: Point cloud contracts inward; Epoch 1 is broad slate gray, **Epoch 2 is contracted red (matching the color coding of Panels A and B)**.
- **Why it Adds Value**: Outstanding pedagogical visual for Chapter 7. Highlights the three distinct ways a research field can evolve over epoch windows. The updated uniform red color for Epoch 2 across all panels creates a cohesive, professional aesthetic that is instantly readable.

---

### Figure 3: Corpus Construction / Pipeline Flow Funnel (Real Live Database Numbers)
- **File**: [fig_req_09_pipeline_flow.png](file:///c:/Users/Z0058EYW/Workspace/TFM/tmp_visual_exploration/figures/fig_req_09_pipeline_flow.png)
- **Suggested Placement**: Chapter 3 (Data and Corpus Construction), Section 3.2 (Filtering Pipeline)
- **What it Shows**: A gorgeous stepped horizontal funnel displaying the systematic reduction from the global OpenAlex database down to the TFM analysis subset, using **exact live database numbers** (with TFM snapshot counts in millions with two decimals):
  - **OpenAlex works** (current global Works index): **314.9M**
  - **2000-2024** (publication-date window): **205.1M** (65% retained)
  - **article or preprint** (document-type filter): **150.8M** (74% retained)
  - **English records** (language filter): **106.6M** (71% retained)
  - **abstract, not retracted** (broad API text pool): **71.8M** (67% retained)
  - **planned sample** (252 subfields; <=400/year): **2.43M** (3.38% of total text pool)
  - **validated corpus** (local text and metadata checks): **2.38M** (3.31% of total text pool)
  - **analysis subset** (row-aligned SPECTER2 matrix): **2.34M** (3.26% of total text pool)
  - **241 analysis subfields ➔ 11 morphology metrics**
- **Why it Adds Value**: Outstanding visual polish that mirrors the TFM aesthetics. The funnel blocks are log-scaled to remain visually balanced and represent the exact steps of data construction, divided into three distinct color-coded semantic groups (API Counts in Navy, TFM Snapshot in Coral/Red, and metrics in Green). The TFM snapshot counts are expressed in millions with two decimal places (e.g. `2.43M`, `2.38M`, `2.34M`), which provides the exact granular distinctions while matching the `M` unit notation of the API box. The percentage values under the blocks are rendered as clean, standalone percentage numbers. The TFM snapshot steps are calculated as percentages of the parent input total (71.8M broad API text pool) using two decimal places (e.g. `3.38%`, `3.31%`, and `3.26%`), ensuring consistent reference, precise scaling, and academic excellence across the entire pipeline.
