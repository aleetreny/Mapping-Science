# Literature Review Report: Measuring the Shape of Science (Refined)

**Thesis Title:** Measuring the Shape of Science: Morphological Indicators and Evolution of Research Fields  
**Date:** May 2026  
**Author:** Candidate for Master of Science  
**Workspace:** `c:\Users\Z0058EYW\Workspace\TFM`  

---

## 1. Executive Summary

This report serves as the foundation for the literature review of the Master's Thesis, which explores the **semantic morphology** of scientific disciplines. The thesis investigates how fields are structured, how they evolve, and how they converge or diverge using a rigorous empirical pipeline:

$$\text{OpenAlex Corpus} \longrightarrow \text{Metadata (Title + Abstract)} \longrightarrow \text{SPECTER2 Embeddings (768-D)} \longrightarrow \text{Reduced 11-Metric Core} \longrightarrow \text{Static \& Temporal Analyses}$$

This refined report strictly separates our **active implemented methodology** from **supporting background literature** and **non-active future extensions**. This ensures that the thesis defense is perfectly aligned with the codebase, does not overstate empirical claims, and protects the methodology from criticisms.

A core methodological tenet of this work is that **all morphological measurements are performed in the original 768-dimensional SPECTER2 embedding space**. Non-linear dimensionality reduction via UMAP is restricted strictly to qualitative, auxiliary visualization. Furthermore, clustering is treated exclusively as an **exploratory typology** based on the 11 metric profiles, never as a natural, absolute division of scientific knowledge.

---

## 2. Strict Methodological and Bibliographical Separation

To maintain perfect alignment with our active codebase, we categorize every reference into one of three classes:
1.  **`core`**: Directly supports our active, implemented mathematical pipeline and data architecture.
2.  **`background`**: Provides necessary historical context and conceptual framing but is not directly computed or implemented.
3.  **`extension`**: Refers to non-active methodologies, alternative mathematical formulations, or advanced models reserved strictly for future work.

We similarly classify every computational method as:
1.  **`implemented`**: Active code running in the pipeline.
2.  **`context only`**: Conceptual framing used only to discuss findings qualitatively.
3.  **`future work`**: Advanced mathematical extensions not currently in the pipeline.

---

## 3. Thematic Areas Analysis

### 3.1 Active Thesis Methodology (Implemented)

This represents the active mathematical engine of the thesis:
*   **OpenAlex Corpus and Taxonomy:**
    *   *Implemented Method:* We extract balanced corpora of subfields, fields, and domains over the 2000–2024 period using a target sample of 400 valid works per year.
    *   *Core References:* **Priem et al. (2022)** (`core`), **Visser et al. (2020)** (`core`), **Scheidsteger & Lindner (2025)** (`core`). These validate the completeness and coverage of OpenAlex as our data source.
*   **SPECTER2 Document Embeddings:**
    *   *Implemented Method:* Title and abstract texts are converted into 768-dimensional dense vectors using the citation-trained SPECTER2 model.
    *   *Core References:* **Singh et al. (2023)** (`core`), **Cohan et al. (2020)** (`core`). These validate that SPECTER2 embeddings represent cohesive, citation-informed topical semantic spaces.
*   **Original 768-D Space Analysis (No High-Dimensional Dimensionality Reduction):**
    *   *Implemented Method:* All 11 morphological metrics are calculated in the raw, unreduced 768-dimensional space.
    *   *Core References:* **Mikolov et al. (2013)** (`core`), **Pennington et al. (2014)** (`core`). These support the semantic properties of continuous dense vector geometries.
*   **Reduced 11-Metric Morphological Core:**
    *   *Implemented Method:* We compute a reduced core of 11 interpretable metrics covering:
        1.  *Global Dispersion:* Centroid distances (`embedding_distance_to_centroid_median`, `embedding_distance_to_centroid_iqr`, `embedding_distance_to_centroid_p90`).
        2.  *Local Density:* KNN distances (`embedding_knn_median_distance`, `embedding_knn_distance_cv`).
        3.  *Local Hubness:* KNN indegree concentration (`embedding_knn_indegree_gini`, supported by **Radovanović et al., 2010** [`core`]).
        4.  *Intrinsic Dimensionality:* PCA-based component count and entropy (`embedding_pca_dim_80`, `embedding_pca_spectral_entropy`).
        5.  *Temporal Shifts & Evolution:* Centroid drift (`embedding_centroid_drift_early_late`) and radial expansion (`embedding_radial_expansion_slope`).
        6.  *Novelty:* Outlier scores (`embedding_recent_novelty_score`).
*   **Exploratory Profile Clustering (Morphological Typologies):**
    *   *Implemented Method:* We perform exploratory KMeans and hierarchical Ward clustering **over the 11 metric profiles of subfields** to identify structural typologies of disciplines.
    *   *Core References:* **Arthur & Vassilvitskii (2007)** (`core`, KMeans++ optimization), **Kleinberg (2002)** (`core`, Impossibility Theorem for Clustering). Kleinberg's theorem mathematically proves that no clustering algorithm represents a unique, perfect division of data, establishing our clustering as strictly exploratory.
*   **UMAP Restricted to Qualitative Visualization:**
    *   *Implemented Method:* UMAP is used **only** as a qualitative visualization layer to project embedding point clouds and temporal trajectories into 2D smooth densities for plotting.
    *   *Core References:* **McInnes et al. (2018)** (`core`), **Chari, Banerjee, & Pachter (2023)** (`core` - UMAP critique), **Wattenberg et al. (2016)** (`core`). Chari et al. (2023) mathematically proves that UMAP distorts global and local distance metrics, justifying our restriction of UMAP to qualitative visualization.

### 3.2 Supporting Background Literature (Context Only)

These references frame the historical and conceptual boundaries of our work but do not have active lines of code:
*   **Classical Science Mapping & Bibliometrics:**
    *   *Status:* `context only`
    *   *References:* **Börner et al. (2003)** (`background`), **Van Eck & Waltman (2010)** (`background`), **Leydesdorff et al. (2013)** (`background`), **Klavans & Boyack (2009)** (`background`), **Boyack & Klavans (2014)** (`background`). These establish the traditions of mapping scientific domains and distance-based visualization which our semantic space method builds upon.
*   **Interdisciplinarity & Cognitive Distance:**
    *   *Status:* `context only`
    *   *References:* **Stirling (2007)** (`background` - variety/balance/disparity framework), **Nooteboom et al. (2007)** (`background` - inverted-U optimal cognitive distance), **Wagner et al. (2011)** (`background` - measuring interdisciplinarity), **Newman (2001)** (`background` - collaboration networks), **Salton et al. (1983)** (`background` - vector space history). These form the conceptual backing for understanding interdisciplinary proximity in Chapter 8.
*   **Scientific Field Evolution:**
    *   *Status:* `context only`
    *   *References:* **Chavalarias & Cointet (2013)** (`background` - phylomemetic patterns), **Uzzi et al. (2013)** (`background` - atypical novelty definitions). These provide the conceptual vocabulary (drifting, splitting, merging) for describing our temporal drift findings.

### 3.3 Possible Extensions & Non-Active Methodology (Future Work)

These methods and references are **excluded from the active codebase** and are discussed solely in the temporal and discussion chapters as advanced future work:
*   **Leiden Clustering on Paper-Level Graphs:**
    *   *Status:* `future work`
    *   *References:* **Traag et al. (2019)** (`extension`), **Blondel et al. (2008)** (`extension`). We do not build paper-level citation or kNN graphs to perform topological community detection. This is a potential future step to scale from our current subfield-level profile clustering.
*   **Mathematical Hubness Reduction:**
    *   *Status:* `future work`
    *   *References:* **Feldbauer et al. (2019)** (`extension`), **Dinu et al. (2014)** (`extension`), **Flexer & Feldbauer (2016)** (`extension`). While we measure local hubness using Gini coefficients, we do not implement Mutual Proximity or Local Scaling transformations to correct high-dimensional distances.
*   **Maximum Likelihood Local Intrinsic Dimension (LID):**
    *   *Status:* `future work`
    *   *References:* **Levina & Bickel (2004)** (`extension`). Our active pipeline measures dimension using PCA-based metrics (`embedding_pca_dim_80`) rather than nearest-neighbor LID equations.
*   **Stirling-Rafols Disparity Indicators:**
    *   *Status:* `future work`
    *   *References:* **Rafols & Meyer (2010)** (`extension`), **Stirling (1998)** (`background` / `extension`). We do not compute categories-based Shannon diversity matrices.
*   **Dynamic Topic Models:**
    *   *Status:* `future work`
    *   *References:* **Blei & Lafferty (2006)** (`extension`). We do not run probabilistic topic models over time, relying instead on continuous vector centroid tracking.

---

## 4. Claims We Can Defend

To protect the thesis from critique during the defense, we list the primary claims that are **empirically and theoretically defendable**:

1.  **UMAP is strictly visual:** We defend that UMAP introduces metric distortions using the proofs of Chari & Pachter (2023). All our 11 morphological metrics are calculated in 768-D to preserve true geometry.
2.  **SPECTER2 is a citation-informed semantic vector space:** The representations carry deep semantic and structural citation signals, outperforming standard language models for scholarly document mapping.
3.  **Local subfield profiles suffer from varying hubness:** We can measure this using `embedding_knn_indegree_gini`, showing that some subfields have highly dominant "central papers" (hubs) while others are topologically flat.
4.  **Morphological typologies are strictly exploratory:** We present our KMeans groupings of the 11 metrics as qualitative, heuristic typologies, backed by Kleinberg's (2002) proof that no clustering is absolute.
5.  **OpenAlex coverage is statistically sound for macro-level mapping:** Its coverage matches Scopus/WoS as proven by Visser et al. (2020) and Scheidsteger & Lindner (2025).

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
