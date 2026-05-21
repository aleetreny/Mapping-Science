# Future Work & Non-Active Methodology Sources

This document details the academic references and methodologies that are **not actively implemented** in the current thesis pipeline, but represent valuable **theoretical context**, **potential extensions**, or **avenues for future work**.

All items listed here are explicitly categorized using our precise systematic taxonomy as either `future-work` (for methods or algorithms not implemented in our active pipeline) or `background` (for supporting contextual literature).

---

## 1. Summary of Non-Active Methodologies

To maintain a highly focused and rigorous defense, we have excluded the following methods from our active pipeline. They are relegated to future extensions or contextual discussions:
*   **Paper-Level Network Graph Clustering (Leiden/Louvain):** We do not build paper-level citation or kNN graphs to perform community detection. Instead, our active methodology utilizes OpenAlex's pre-defined subfields and maps their high-level **reduced 11-metric profiles** using standard KMeans/Ward. Paper-level network models are relegated strictly to future work.
*   **Mathematical Hubness Reduction (Mutual Proximity / Local Scaling):** While we measure local hubness as a descriptive topology metric (`embedding_knn_indegree_gini`), we do not implement distance-correction transformations (such as Mutual Proximity or Local Scaling).
*   **Maximum Likelihood Local Intrinsic Dimension (LID):** We do not compute the continuous, localized Levina-Bickel nearest-neighbor LID. Instead, our active methodology measures intrinsic dimension using **`embedding_pca_dim_80`** and **`embedding_pca_spectral_entropy`**.
*   **Stirling-Rafols Disparity Equations:** We do not compute categories-based diversity equations using Shannon entropy combined with taxonomic distance matrices. Instead, our active methodology relies on raw continuous embedding-space dispersion and density metrics.
*   **Dynamic Topic Modeling (DTM):** We do not train dynamic probabilistic LDA topic models over sequential document corpora. Instead, our active methodology tracks continuous semantic drift directly through sequential centroid movement and radial dispersion in 5-year time windows.

---

## 2. Extension & Contextual References

### 2.1 Network Graph & Paper-Level Clustering (Louvain / Leiden)
*   **[traag2019leiden](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib)** (Traag et al., 2019)
    *   *Reference Category:* `future-work`
    *   *Methodology Status:* `future work`
    *   *Role/Context:* Proposes the Leiden algorithm to correct the Louvain algorithm's tendency to find disconnected or arbitrary communities. This is a valuable community detection model for future projects scaling from subfield-level metric clustering to a complete paper-level topological classification.
*   **[blondel2008louvain](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib)** (Blondel et al., 2008)
    *   *Reference Category:* `future-work`
    *   *Methodology Status:* `future work`
    *   *Role/Context:* The canonical reference for the Louvain heuristic for modularity optimization. It provides historical context for topological community detection in network-based science mapping.

### 2.2 Mathematical Hubness Reduction
*   **[feldbauer2019skhubness](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib)** (Feldbauer et al., 2019)
    *   *Reference Category:* `future-work`
    *   *Methodology Status:* `future work`
    *   *Role/Context:* Presents the `skhubness` package which implements Mutual Proximity (MP), Local Scaling (LS), and DisSim Local (DSL) to correct distance concentration in high-dimensional nearest-neighbor queries. This represents a concrete future step to refine our local Gini hubness calculations.
*   **[dinu2014hubness](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib)** (Dinu et al., 2014)
    *   *Reference Category:* `future-work`
    *   *Methodology Status:* `background`
    *   *Role/Context:* Demonstrates hubness reduction in zero-shot language translation models, providing context for the impact of hubness on semantic alignment.
*   **[flexer2016hubness](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib)** (Flexer & Feldbauer, 2016)
    *   *Reference Category:* `future-work`
    *   *Methodology Status:* `background`
    *   *Role/Context:* Reviews distance-based hubness reduction methods, outlining alternative mathematical approaches to nearest-neighbor calculations.

### 2.3 Intrinsic Dimensionality and Advanced Diversity Math
*   **[levina2004intrinsic](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib)** (Levina & Bickel, 2004)
    *   *Reference Category:* `future-work`
    *   *Methodology Status:* `future work`
    *   *Role/Context:* Proposes the Maximum Likelihood Estimator for local intrinsic dimension. Represents a useful methodological extension to compare against our implemented PCA-based dimensionality metrics (`embedding_pca_dim_80`).
*   **[rafols2010diversity](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib)** (Rafols & Meyer, 2010)
    *   *Reference Category:* `future-work`
    *   *Methodology Status:* `background`
    *   *Role/Context:* Integrates cognitive distances into Stirling’s diversity indicators (using Shannon entropy and distance matrices over discrete categories). This is useful background for conceptualizing interdisciplinarity, but its discrete category mathematics are not computed in our continuous embedding point cloud.

### 2.4 Dynamic Generative Models
*   **[blei2006dynamic](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib)** (Blei & Lafferty, 2006)
    *   *Reference Category:* `future-work`
    *   *Methodology Status:* `future work`
    *   *Role/Context:* Introduces Dynamic Topic Models. Provides an alternative, generative approach to track the temporal drift of scientific topics in discrete corpora, which contrasts with our continuous vector-space centroid tracking.
