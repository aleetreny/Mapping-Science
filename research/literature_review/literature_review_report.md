# Literature Review Report: Measuring the Shape of Science

**Thesis Title:** Measuring the Shape of Science: Morphological Indicators and Evolution of Research Fields  
**Date:** May 2026  
**Author:** Alejandro Treny Ortega  
**Workspace:** `c:\Users\Z0058EYW\Workspace\TFM`  

---

## 1. Executive Summary

This report serves as the foundation for the literature review of the Master's Thesis, which explores the **semantic morphology** of scientific disciplines. The thesis investigates how fields are structured, how they evolve, and how they converge or diverge using a rigorous empirical pipeline:

$$\text{OpenAlex Corpus} \longrightarrow \text{Metadata (Title + Abstract)} \longrightarrow \text{SPECTER2 Embeddings (768-D)} \longrightarrow \text{Reduced 11-Metric Core} \longrightarrow \text{Static \& Temporal Analyses}$$

This report strictly separates our **active implemented methodology** from **supporting background literature** and **non-active future extensions**. This ensures that the thesis defense is perfectly aligned with the codebase, does not overstate empirical claims, and protects the methodology from criticisms.

A core methodological tenet of this work is that **all morphological measurements are performed in the original 768-dimensional SPECTER2 embedding space**. Non-linear dimensionality reduction via UMAP is restricted strictly to qualitative, auxiliary visualization. Embedding-space distances are treated as model-dependent semantic approximations, not objective measures of scientific reality. Projected coordinates are **not used as quantitative evidence because nonlinear projections may distort distances, densities, and neighborhood relations**.

Furthermore, clustering is treated exclusively as an **exploratory typology** based on the 11 metric profiles. This chapter is planned as a downstream exploratory analysis and must be implemented only over the reduced 11-metric profiles. No active clustering script is currently in production.

---

## 2. Strict Methodological and Bibliographical Separation

To maintain perfect alignment with our active codebase, we categorize every reference into one of three classes:
1.  **`Core Active-Methodology`**: Directly supports our active computational pipeline, key metric justifications, and structural constraints.
2.  **`Background`**: Provides necessary historical context and conceptual framing but is not directly computed or implemented.
3.  **`Future-Work`**: Refers to non-active methodologies, alternative mathematical formulations, or advanced models reserved strictly for future work.

We similarly classify every computational method using one of four precise labels:
1.  **`implemented`**: Only for methods actually computed by the current codebase.
2.  **`methodological justification`**: For theoretical or cautionary sources that justify a design choice but are not themselves implemented.
3.  **`background`**: For contextual framing literature.
4.  **`future work`**: For methods not implemented.

---

## 3. Thematic Areas Analysis

### 3.1 Active Thesis Methodology

This represents the active mathematical engine of the thesis:
*   **OpenAlex Corpus and Taxonomy:**
    *   *Methodology Status:* `implemented`
    *   *References:* **Priem et al. (2022)** (`Core Active-Methodology`). This validates the completeness and coverage of OpenAlex as our data source.
    *   *References:* **Martín-Martín et al. (2021)** (`Core Active-Methodology` | `methodological justification`). This validates that open citation databases provide a highly comprehensive basis for large-scale science mapping. **Culbert et al. (2025)** (`Core Active-Methodology` | `methodological justification`) show that OpenAlex reference coverage is broadly comparable to Web of Science and Scopus in a cleaned shared-publication comparison, while also documenting metadata limitations, including lower abstract coverage.
*   **SPECTER2 Document Embeddings:**
    *   *Methodology Status:* `implemented`
    *   *References:* **Singh et al. (2023)** (`Core Active-Methodology`). This validates that SPECTER2 embeddings represent cohesive, citation-informed topical semantic spaces.
    *   *References:* **Cohan et al. (2020)** (`Core Active-Methodology` | `methodological justification`). This establishes the contrastive learning triplet loss that underpins these embeddings.
*   **Original 768-D Space Analysis:**
    *   *Methodology Status:* `implemented`
    *   *References:* **Mikolov et al. (2013)** (`Core Active-Methodology` | `methodological justification`). This supports the semantic properties of continuous dense vector geometries, treated as model-dependent approximations rather than objective reality.
*   **Reduced 11-Metric Morphological Core:**
    *   *Methodology Status:* `implemented`
    *   *References:* **Radovanović et al. (2010)** (`Core Active-Methodology` | `methodological justification`). This provides the mathematical basis for high-dimensional nearest-neighbor concentration ("hubness"), justifying our local Gini coefficient of nearest-neighbor indegrees (`embedding_knn_indegree_gini`) to characterize subfields.
*   **Exploratory Profile Clustering (Morphological Typologies):**
    *   *Methodology Status:* `methodological justification` / `implemented` (conceptual boundary)
    *   *Status Statement:* **This chapter is planned as a downstream exploratory analysis and must be implemented only over the reduced 11-metric profiles.**
    *   *References:* **Arthur & Vassilvitskii (2007)** (`Core Active-Methodology` | `implemented` for standard KMeans++ initialization seeding), **Kleinberg (2002)** (`Core Active-Methodology` | `methodological justification`). Kleinberg's theorem mathematically proves that no clustering algorithm represents a unique, perfect division of data, establishing our clustering as strictly exploratory.
*   **UMAP Restricted to Qualitative Visualization:**
    *   *Methodology Status:* `implemented` (strictly constrained to auxiliary qualitative visualization)
    *   *References:* **McInnes et al. (2018)** (`Core Active-Methodology` | `implemented` strictly for 2D plotting), **Chari, Banerjee, & Pachter (2023)** (`Core Active-Methodology` | `methodological justification`), **Wattenberg et al. (2016)** (`Core Active-Methodology` | `methodological justification`). Chari et al. (2023) mathematically proves that UMAP distorts global and local distance metrics, justifying our restriction of UMAP to qualitative visualization.

### 3.2 Supporting Background Literature

These references frame the historical and conceptual boundaries of our work but do not have active lines of code:
*   **Classical Science Mapping & Bibliometrics:**
    *   *Methodology Status:* `background`
    *   *References:* **Börner et al. (2003)** (`Background`), **Van Eck & Waltman (2010)** (`Background`), **Leydesdorff et al. (2013)** (`Background`). These establish the traditions of mapping scientific domains and distance-based visualization which our semantic space method builds upon.
*   **Interdisciplinarity & Cognitive Distance:**
    *   *Methodology Status:* `background`
    *   *References:* **Stirling (2007)** (`Background` — variety/balance/disparity framework), **Nooteboom et al. (2007)** (`Background` — inverted-U optimal cognitive distance). These form the conceptual backing for understanding interdisciplinary proximity in Chapter 8.
*   **Scientific Field Evolution:**
    *   *Methodology Status:* `background`
    *   *References:* **Chavalarias & Cointet (2013)** (`Background` — phylomemetic patterns), **Uzzi et al. (2013)** (`Background` — atypical novelty definitions). These provide the conceptual vocabulary (drifting, splitting, merging) for describing our temporal drift findings.

### 3.3 Non-Active Methodology (Future Work)

These methods and references are **excluded from the active codebase** and are discussed solely in the temporal and discussion chapters as advanced future work:
*   **Leiden Clustering on Paper-Level Graphs:**
    *   *Methodology Status:* `future work`
    *   *References:* **Traag et al. (2019)** (`Future-Work`). We do not build paper-level citation or kNN graphs to perform topological community detection. This is a potential future step to scale from our current subfield-level profile clustering.
*   **Mathematical Hubness Reduction:**
    *   *Methodology Status:* `future work`
    *   *References:* **Feldbauer et al. (2019)** (`Future-Work`). While we measure local hubness using Gini coefficients, we do not implement Mutual Proximity or Local Scaling transformations to correct high-dimensional distances.
*   **Maximum Likelihood Local Intrinsic Dimension (LID):**
    *   *Methodology Status:* `future work`
    *   *References:* **Levina & Bickel (2004)** (`Future-Work`). Our active pipeline measures dimension using PCA-based metrics (`embedding_pca_dim_80`) rather than nearest-neighbor LID equations.
*   **Stirling-Rafols Disparity Indicators:**
    *   *Methodology Status:* `future work`
    *   *References:* **Rafols & Meyer (2010)** (`Future-Work`). We do not compute categories-based Shannon diversity matrices.
*   **Dynamic Topic Models:**
    *   *Methodology Status:* `future work`
    *   *References:* **Blei & Lafferty (2006)** (`Future-Work`). We do not run probabilistic topic models over time, relying instead on continuous vector centroid tracking.

---

## 4. Claims We Can Defend

To protect the thesis from critique during the defense, we list the primary claims that are **empirically and theoretically defendable**:

1.  **UMAP is strictly visual:** We defend that UMAP introduces metric distortions using the proofs of Chari & Pachter (2023). All our 11 morphological metrics are calculated in 768-D to preserve high-dimensional relationships as model-dependent semantic approximations, not objective measures of scientific reality.
2.  **SPECTER2 is a citation-informed semantic vector space:** The representations carry deep semantic and structural citation signals, outperforming standard language models for scholarly document mapping.
3.  **Local subfield profiles suffer from varying hubness:** We can measure this using `embedding_knn_indegree_gini`, showing that some subfields have highly dominant "central papers" (hubs) while others are topologically flat.
4.  **Morphological typologies are strictly exploratory:** We present our planned KMeans groupings of the 11 metrics as qualitative, heuristic typologies, backed by Kleinberg's (2002) proof that no clustering is absolute.
5.  **OpenAlex coverage is statistically sound for macro-level mapping:** OpenAlex reference coverage is broadly comparable to Web of Science and Scopus in a cleaned shared-publication comparison, as shown by Culbert et al. (2025), and is highly comprehensive, as shown by Martín-Martín et al. (2021).

---

## 5. Claims We Should Avoid

We must **explicitly avoid** making these claims to prevent methodological vulnerabilities:

1.  **"UMAP 2D plots preserve absolute cognitive distance or cluster shapes":** Do not claim that visual coordinates represent exact cognitive distances.
2.  **"Our KMeans clustering reveals the natural taxonomy of science":** Do not claim our typologies represent absolute, objective divisions.
3.  **"We implement global distance corrections (Mutual Proximity) or graph-level Leiden algorithms":** Do not present these extensions as active pipeline components; they are strictly future work.
4.  **"SPECTER2 vector coordinates are free from publishing, citation, or database bias":** Acknowledge database and selection biases.
5.  **"Our pipeline measures daily or weekly micro-evolution":** Acknowledge that our 5-year time windows are designed to capture macro-level structural trends.

---

## 6. Verification and Bibliography Link

The full verified BibTeX database is saved in:
[candidate_references.bib](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib)

Key references organized by chapter are located in:
[key_references_by_chapter.md](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/key_references_by_chapter.md)

Detailed claims mappings, active methodology sources, and future work extension sources are located in:
*   [claims_to_sources.md](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/claims_to_sources.md)
*   [active_methodology_sources.md](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/active_methodology_sources.md)
*   [future_work_sources.md](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/future_work_sources.md)
*   [sources_to_verify.md](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/sources_to_verify.md)
