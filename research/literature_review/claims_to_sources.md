# Claims-to-Sources Cross-Reference

This document provides a precise mapping between the specific claims, arguments, and methodological choices in the thesis and the peer-reviewed scientific literature that supports them. 

Every claim has been scoped conservatively to **avoid overstating what the empirical pipeline can prove**, and each reference is marked with its precise methodology status (`implemented`, `methodological justification`, `background`, `future work`).

---

## 1. Dimensionality Reduction & Visualization Constraints

### Claim 1: "UMAP is used strictly for qualitative, auxiliary visualization. Its projected coordinates are not used as quantitative evidence because nonlinear projections may distort distances, densities, and neighborhood relations. Embedding-space distances are treated as model-dependent semantic approximations, not objective measures of scientific reality."
*   **Supporting Literature:**
    *   **[chari2023specious](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib#L186-L197)** (Chari et al., 2023)  
        *   *Methodology Status:* `methodological justification`  
        *   *Thesis Support:* Provides the definitive mathematical proof that non-linear projections (UMAP/t-SNE) introduce high metric distortions and fail to preserve physical distance metrics.
    *   **[wattenberg2016tsne](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib#L232-L241)** (Wattenberg et al., 2016)  
        *   *Methodology Status:* `methodological justification`  
        *   *Thesis Support:* Demonstrates that projected cluster sizes, densities, and inter-cluster distances in 2D do not correspond to the high-dimensional shapes.
    *   **[coenen2019umap](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib#L242-L249)** (Coenen & Pearce, 2019)  
        *   *Methodology Status:* `methodological justification`  
        *   *Thesis Support:* Visually details how UMAP prioritizes local neighbors over global distances, justifying our strict visual-only status.
*   **Empirical Limit:** The thesis *does not* claim UMAP plots represent exact coordinates; all statistical findings are derived from the raw 768-D space.

---

## 2. Original Embedding-Space Geometry & Hubness

### Claim 2: "High-dimensional spaces inherently suffer from 'hubness' (popular nearest neighbors). We measure this structural artifact locally in our eight-metric structural core to characterize subfields, but we do not apply global distance-reduction corrections."
*   **Supporting Literature:**
    *   **[radovanovic2010hubness](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib#L80-L90)** (Radovanović et al., 2010)  
        *   *Methodology Status:* `methodological justification`  
        *   *Thesis Support:* Proves that hubness is an intrinsic geometric property of high-dimensional spaces. This supports our use of the KNN graph indegree Gini coefficient (**`embedding_knn_indegree_gini`**) in the eight-metric structural core to measure the local hubness profile of each subfield.
    *   **[feldbauer2019skhubness](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib#L91-L101)** (Feldbauer et al., 2019)  
        *   *Methodology Status:* `future work`  
        *   *Thesis Support:* Introduces software and algorithms (Mutual Proximity, Local Scaling) to correct hubness bias. Cited in the discussion as a concrete future step to refine high-dimensional nearest-neighbor measurements.
*   **Empirical Limit:** The thesis measures local hubness as a morphological descriptor but *does not* apply distance corrections (Mutual Proximity) to the active pipeline, leaving this as future work.

---

## 3. Data & Representation Validity

### Claim 3: "OpenAlex provides a highly complete and robust index of global scientific metadata, making it a reliable data source for large-scale science mapping."
*   **Supporting Literature:**
    *   **[priem2022openalex](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib#L1-L10)** (Priem et al., 2022)  
        *   *Methodology Status:* `implemented`  
        *   *Thesis Support:* The primary citation validating OpenAlex's metadata architecture and hierarchical taxonomic classification (subfield, field, domain).
    *   **[martinmartin2021google](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib)** (Martín-Martín et al., 2021)  
        *   *Methodology Status:* `methodological justification`  
        *   *Thesis Support:* Statistical comparison showing that open citation data has equivalent or superior coverage to Scopus and Web of Science.
    *   **[culbert2025reference](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib)** (Culbert et al., 2025)  
        *   *Methodology Status:* `methodological justification`  
        *   *Thesis Support:* Culbert et al. (2025) show that OpenAlex reference coverage is broadly comparable to Web of Science and Scopus in a cleaned shared-publication comparison, while also documenting metadata limitations, including lower abstract coverage.

### Claim 4: "SPECTER2 embeddings effectively capture both semantic text context and citation-informed relatedness, providing a dense vector coordinate space for document-level science mapping. High-dimensional relationships are treated as model-dependent semantic approximations, not objective measures of scientific reality."
*   **Supporting Literature:**
    *   **[singh2023scirepeval](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib#L46-L57)** (Singh et al., 2023)  
        *   *Methodology Status:* `implemented`  
        *   *Thesis Support:* The canonical SciRepEval evaluation paper demonstrating that SPECTER2's multi-task scientific adapters produce cohesive semantic document vectors.
    *   **[cohan2020specter](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib#L34-L45)** (Cohan et al., 2020)  
        *   *Methodology Status:* `methodological justification`  
        *   *Thesis Support:* Formulates the contrastive citation-informed triplet loss that underpins the semantic alignment of document point clouds.
*   **Empirical Limit:** The vector representations reflect the consensus of published abstracts and citation links; they *do not* represent absolute, bias-free human knowledge.

---

## 4. Science Mapping & Bibliometric Background

### Claim 5: "Science mapping has evolved from network visualization to high-dimensional continuous semantic spaces."
*   **Supporting Literature:**
    *   **[borner2003visualizing](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib#L21-L33)** (Börner et al., 2003)  
        *   *Methodology Status:* `background`  
        *   *Thesis Support:* Frames the historical evolution of domain visualization, providing context for moving from discrete network nodes to continuous coordinate-space morphology.
    *   **[vaneck2010vosviewer](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib#L11-L20)** (Van Eck & Waltman, 2010)  
        *   *Methodology Status:* `background`  
        *   *Thesis Support:* Details the principles of distance-based mapping, showing how physical proximity on maps represents semantic similarity.
    *   **[leydesdorff2013global](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib#L269-L279)** (Leydesdorff et al., 2013)  
        *   *Methodology Status:* `background`  
        *   *Thesis Support:* Documents global science overlays using journal classification schemes, serving as background for our subfield-level analysis.

---

## 5. Structural Dynamics & Future Extensions

### Claim 6: "The diversity of a scientific discipline is conceptually defined by its variety, balance, and cognitive distance, which we operationalize through embedding dispersion metrics."
*   **Supporting Literature:**
    *   **[stirling2007diversity](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib#L113-L122)** (Stirling, 2007)  
        *   *Methodology Status:* `background`  
        *   *Thesis Support:* Formulates the variety-balance-disparity framework. Used for conceptual framing of diversity in Chapter 6.
    *   **[rafols2010diversity](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib#L143-L153)** (Rafols & Meyer, 2010)  
        *   *Methodology Status:* `future work`  
        *   *Thesis Support:* Proposes equations combining Shannon entropy and distance matrices. Relegated to future work as we use raw embedding-space dispersion and density metrics (`embedding_distance_to_centroid_median`, etc.) instead of discrete category math.

### Claim 7: "The evolution of scientific fields involves semantic drifting, expansion, and contraction over time."
*   **Supporting Literature:**
    *   **[chavalarias2013phylomemetic](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib#L132-L142)** (Chavalarias & Cointet, 2013)  
        *   *Methodology Status:* `background`  
        *   *Thesis Support:* Conceptually describes phylogenetic splits, merges, and drifts of scientific subfields, providing the descriptive vocabulary for our temporal discussion in Chapter 7.
    *   **[blei2006dynamic](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib#L123-L131)** (Blei & Lafferty, 2006)  
        *   *Methodology Status:* `future work`  
        *   *Thesis Support:* Establishes Dynamic Topic Models. Cited in the temporal chapter as a probabilistic baseline to contrast against our direct vector-space centroid drift tracking (**`embedding_centroid_drift_early_late`**).

---

## 6. Exploratory Typologies & Clustering Limits

### Claim 8: "Clustering is strictly an exploratory tool to identify morphological typologies. No clustering algorithm represents a 'natural, perfect partition' of scientific knowledge. The active typology is implemented only over the eight-metric structural profiles."
*   **Supporting Literature:**
    *   **[kleinberg2002clustering](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib#L164-L174)** (Kleinberg, 2002)  
        *   *Methodology Status:* `methodological justification`  
        *   *Thesis Support:* Proves the mathematical impossibility of a perfect clustering algorithm, justifying why our morphological typologies are strictly heuristic.
    *   **[arthur2007kmeans](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib#L341-L350)** (Arthur & Vassilvitskii, 2007)  
        *   *Methodology Status:* `implemented`  
        *   *Thesis Support:* Formulates k-means++ seeding, which is implemented in our profile analysis to optimize initial centroids for the planned KMeans typologies.
    *   **[traag2019leiden](file:///c:/Users/Z0058EYW/Workspace/TFM/research/literature_review/candidate_references.bib#L175-L185)** (Traag et al., 2019)  
        *   *Methodology Status:* `future work`  
        *   *Thesis Support:* Details Leiden graph clustering. Relegated to future work because we do not construct or cluster a paper-level network graph in our active pipeline.
