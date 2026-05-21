# Active Methodology Sources

This document lists and details the **essential academic references** that directly support the active empirical pipeline of the Master's Thesis. 

All references listed here are categorized as core to the methodology. Their associated statuses are marked as either `core-active` (for methods actually computed by our active code) or `core-justification` (for theoretical or cautionary sources that justify a design choice but are not themselves computational components).

---

## 1. Summary of Active Methodology

The thesis explores the semantic morphology of scientific disciplines through a concrete, reproducible data pipeline:
$$\text{OpenAlex Balanced Corpus} \longrightarrow \text{SPECTER2 Embeddings (768-D)} \longrightarrow \text{Reduced 11-Metric Embedding-Space Core} \longrightarrow \text{Static, Temporal, \& Convergence Analyses}$$

*   **No High-Dimensional Dimensionality Reduction for Calculations:** All 11 morphological metrics are calculated in the original 768-dimensional SPECTER2 space. Embedding-space distances are treated as model-dependent semantic approximations, not objective measures of scientific reality.
*   **UMAP Status:** Strictly constrained to **auxiliary qualitative visualization** (`scripts/09_build_global_umap_visualization.py`, `scripts/10_build_subfield_umap_visualizations.py`, etc.). It is not used as quantitative evidence because nonlinear projections may distort distances, densities, and neighborhood relations.
*   **Clustering Status:** This chapter is planned as a downstream exploratory analysis and must be implemented only over the reduced 11-metric profiles. It focuses on exploratory morphological typologies of subfield-level metric profiles, not as natural, absolute categories of science, and is not performed on raw paper-level graphs.

---

## 2. Core Implemented and Justifying References

### 2.1 Corpus & Taxonomy Construction
*   **[priem2022openalex](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib)** (Priem et al., 2022)
    *   *Reference Category:* `core-active`
    *   *Role in Pipeline:* Establishes the validity and hierarchical structure of the OpenAlex scientific taxonomy (domains, fields, and subfields) used as our primary units of analysis in corpus extraction code.
*   **[martinmartin2021google](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib)** (Martín-Martín et al., 2021)
    *   *Reference Category:* `core-justification`
    *   *Role in Pipeline:* Provides statistical evidence that open bibliographic metadata databases (such as Google Scholar and Microsoft Academic Graph, which underlies OpenAlex) have citation coverage and quality equivalent to or exceeding proprietary databases (Scopus, WoS) for large-scale science mapping, justifying our database choice.
*   **[culbert2025reference](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib)** (Culbert et al., 2025)
    *   *Reference Category:* `core-justification`
    *   *Role in Pipeline:* Provides a recent peer-reviewed audit confirming that OpenAlex's citation reference list coverage is complete and reliable for large-scale structural analysis compared to Web of Science and Scopus, justifying its use in structural modeling.

### 2.2 Semantic Vector Representations
*   **[singh2023scirepeval](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib)** (Singh et al., 2023)
    *   *Reference Category:* `core-active`
    *   *Role in Pipeline:* Canonical citation for the **SPECTER2** model series. Used in our embedding extraction scripts to generate citation-informed semantic document representations (768-D vectors) that generalize robustly across disciplines.
*   **[cohan2020specter](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib)** (Cohan et al., 2020)
    *   *Reference Category:* `core-justification`
    *   *Role in Pipeline:* Establishes the foundational contrastive triplet pretraining framework (pulling citing/cited papers closer and pushing non-cited ones apart) which underpins the semantic properties of SPECTER2 embeddings.

### 2.3 Original Embedding-Space Geometry & Morphological Metrics
*   **[radovanovic2010hubness](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib)** (Radovanović et al., 2010)
    *   *Reference Category:* `core-justification`
    *   *Role in Pipeline:* Provides the mathematical proof for "hubness"—the tendency of high-dimensional spaces to generate popular nearest neighbors. Directly justifies our inclusion of **`embedding_knn_indegree_gini`** in our reduced 11-metric core to measure and describe the local hubness profile of each subfield.

### 2.4 Mathematical Critique of Projection Distortions (UMAP Constraints)
*   **[chari2023specious](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib)** (Chari, Banerjee, & Pachter, 2023)
    *   *Reference Category:* `core-justification`
    *   *Role in Pipeline:* Provides mathematical evidence that non-linear dimensionality reduction algorithms (UMAP and t-SNE) introduce metric distortions in global and local distances. This serves as the primary methodological justification for restricting UMAP strictly to qualitative visualization and computing all quantitative indicators in the original 768-D space.
*   **[wattenberg2016tsne](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib)** (Wattenberg et al., 2016)
    *   *Reference Category:* `core-justification`
    *   *Role in Pipeline:* Conceptually explains how non-linear distance projections distort density, area, and inter-cluster distances, supporting our qualitative-only visual framing of projected spaces.
*   **[mcinnes2018umap](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib)** (McInnes et al., 2018)
    *   *Reference Category:* `core-active` (restricted strictly to auxiliary qualitative visualization)
    *   *Role in Pipeline:* Establishes the mathematical formulation of UMAP. Implemented in our plotting scripts to project embedding point clouds and temporal trajectories into 2D smooth densities for qualitative visual representation only.

### 2.5 Exploratory Typologies & Clustering Limits
*   **[arthur2007kmeans](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib)** (Arthur & Vassilvitskii, 2007)
    *   *Reference Category:* `core-active`
    *   *Role in Pipeline:* Formulates the k-means++ seeding technique. Implemented in our profile analysis to optimize the initial centroids for our planned exploratory KMeans typologies of subfield metrics.
*   **[kleinberg2002clustering](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib)** (Kleinberg, 2002)
    *   *Reference Category:* `core-justification`
    *   *Role in Pipeline:* Proves mathematically that no single clustering algorithm can satisfy scale-invariance, richness, and consistency simultaneously. Directly justifies why our planned morphological typologies are presented strictly as **exploratory, qualitative characterizations** rather than objective "natural categories" of science.
