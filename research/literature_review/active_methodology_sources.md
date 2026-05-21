# Active Methodology Sources

This document lists and details the **essential academic references** that directly support the active empirical pipeline of the Master's Thesis. 

All methods listed here are **fully implemented** in the active codebase. References are marked as `core` and their associated methodologies as `implemented`.

---

## 1. Summary of Active Methodology (Implemented)
The thesis explores the semantic morphology of scientific disciplines through a concrete, reproducible data pipeline:
$$\text{OpenAlex Balanced Corpus} \longrightarrow \text{SPECTER2 Embeddings (768-D)} \longrightarrow \text{Reduced 11-Metric Embedding-Space Core} \longrightarrow \text{Static, Temporal, \& Convergence Analyses}$$

*   **No High-Dimensional Dimensionality Reduction for Calculations:** All 11 morphological metrics are calculated in the original 768-dimensional SPECTER2 space.
*   **UMAP Status:** Strictly constrained to **auxiliary qualitative visualization** (`scripts/09_build_global_umap_visualization.py`, `scripts/10_build_subfield_umap_visualizations.py`, etc.). It is never used as quantitative evidence.
*   **Clustering Status:** Strictly limited to **exploratory morphological typologies** based on KMeans and hierarchical Ward clustering over the **reduced 11-metric profiles** of subfields (`scripts/15_cluster_metric_spaces.py` from legacy or downstream typology analysis), **not** as a natural, absolute categorization of scientific knowledge, and **not** performed on the raw paper-level graphs.

---

## 2. Core Implemented References (Essential Bibliography)

### 2.1 Corpus & Taxonomy Construction
*   **[priem2022openalex](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib#L1-L10)** (Priem et al., 2022)
    *   *Reference Category:* `core`
    *   *Methodology Status:* `implemented`
    *   *Role in Pipeline:* Establishes the validity and hierarchical structure of the OpenAlex scientific taxonomy (domains, fields, and subfields) used as our primary units of analysis.
*   **[visser2020google](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib#L210-L221)** (Visser et al., 2020)
    *   *Reference Category:* `core`
    *   *Methodology Status:* `implemented`
    *   *Role in Pipeline:* Provides statistical proof that open bibliographic metadata databases derived from Microsoft Academic Graph (which underlies OpenAlex) have coverage quality equivalent to or exceeding proprietary databases (Scopus, WoS) for science mapping.
*   **[scheidsteger2025reference](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib#L198-L209)** (Scheidsteger & Lindner, 2025)
    *   *Reference Category:* `core`
    *   *Methodology Status:* `implemented`
    *   *Role in Pipeline:* The latest peer-reviewed audit confirming that OpenAlex's citation reference list coverage is complete and reliable for large-scale structural analysis.

### 2.2 Semantic Vector Representations
*   **[singh2023scirepeval](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib#L46-L57)** (Singh et al., 2023)
    *   *Reference Category:* `core`
    *   *Methodology Status:* `implemented`
    *   *Role in Pipeline:* Canonical citation for the **SPECTER2** model series. Validates that contrastive, citation-informed scientific text embeddings (768-D) represent cohesive topical proximities and generalize across disciplines.
*   **[cohan2020specter](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib#L34-L45)** (Cohan et al., 2020)
    *   *Reference Category:* `core`
    *   *Methodology Status:* `implemented`
    *   *Role in Pipeline:* Establishes the fundamental contrastive triplet pretraining framework (pulling citing/cited papers closer and pushing non-cited ones apart) which underpins SPECTER2's semantic properties.

### 2.3 original Embedding-Space Geometry & Morphological Metrics
*   **[radovanovic2010hubness](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib#L80-L90)** (Radovanović et al., 2010)
    *   *Reference Category:* `core`
    *   *Methodology Status:* `implemented` (theory and localized KNN metrics)
    *   *Role in Pipeline:* Provides the mathematical proof for "hubness"—the tendency of high-dimensional spaces to generate popular nearest neighbors. Directly justifies our inclusion of **`embedding_knn_indegree_gini`** in our reduced 11-metric core to measure the local hubness profile of each subfield.

### 2.4 Mathematical Critique of Projection Distortions (UMAP Constraints)
*   **[chari2023specious](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib#L186-L197)** (Chari, Banerjee, & Pachter, 2023)
    *   *Reference Category:* `core`
    *   *Methodology Status:* `implemented` (as constraint justification)
    *   *Role in Pipeline:* Proves mathematically that non-linear dimensionality reduction algorithms (UMAP and t-SNE) distort global and local distances, making them completely unsuitable for quantitative physical or biological calculations. Directly supports our core methodological design: restricting UMAP to visual representation and computing all quantitative metrics in 768-D.
*   **[wattenberg2016tsne](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib#L232-L241)** (Wattenberg et al., 2016)
    *   *Reference Category:* `core`
    *   *Methodology Status:* `implemented` (as constraint justification)
    *   *Role in Pipeline:* Conceptually explains how non-linear distance projections distort densities and inter-cluster distances, supporting the qualitative-only visual status of UMAP.
*   **[mcinnes2018umap](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib#L217-L226)** (McInnes et al., 2018)
    *   *Reference Category:* `core`
    *   *Methodology Status:* `implemented` (solely for auxiliary qualitative visualization)
    *   *Role in Pipeline:* Establishes the mathematical formulation of UMAP. Used to generate the 2D projected coordinate spaces solely for qualitative plots, never for quantitative measurements.

### 2.5 Exploratory Typologies & Clustering Limits
*   **[arthur2007kmeans](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib#L341-L350)** (Arthur & Vassilvitskii, 2007)
    *   *Reference Category:* `core`
    *   *Methodology Status:* `implemented`
    *   *Role in Pipeline:* Formulates the k-means++ seeding technique. Directly supports our implemented exploratory clustering of the 11-metric profiles to identify morphology typologies.
*   **[kleinberg2002clustering](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib#L164-L174)** (Kleinberg, 2002)
    *   *Reference Category:* `core`
    *   *Methodology Status:* `implemented` (as conceptual boundary)
    *   *Role in Pipeline:* Proves mathematically that no single clustering algorithm can satisfy scale-invariance, richness, and consistency simultaneously. Directly justifies why our morphological typologies are presented as **exploratory, qualitative characterizations** rather than objective "natural categories" of science.
