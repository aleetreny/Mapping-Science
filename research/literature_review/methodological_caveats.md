# Methodological Caveats and Mathematical Limitations

This document serves as a critical **Risk and Limitation Register** for the empirical pipeline of the Master's Thesis. It details the structural, mathematical, and database limitations that must be acknowledged to defend the thesis successfully.

---

## 1. Dimensionality Reduction: UMAP Distance Distortions

### The Caveat:
Non-linear dimensionality reduction algorithms like UMAP (Uniform Manifold Approximation and Projection) and t-SNE (t-Distributed Stochastic Neighbor Embedding) are designed to preserve local neighborhood structure. They **do not** preserve global distances, volumes, or true density ratios when projecting from 768-D to 2D.

### Mathematical & Conceptual Risks:
1. **Distance Distortion:** Under non-linear mapping, the relative distance between clusters in a 2D projection does not represent their true cognitive distance in the 768-D embedding space. Two clusters that appear close together on a UMAP plot may actually be further apart than two clusters that appear far apart.
2. **Topological Tearing:** UMAP assumes that the high-dimensional data lies on a local Riemannian manifold that is locally connected. In scientific document embeddings, data can be highly sparse and fragmented, causing UMAP to "tear" continuous semantic structures into arbitrary 2D clusters.
3. **Volume Inflation/Deflation:** The apparent density or size of a 2D cluster on a plot is highly sensitive to UMAP's hyperparameters (`n_neighbors`, `min_dist`) and does not correspond to the actual volume of the point cloud in 768-D.

### Academic Defense (How We Mitigate It):
*   We cite **Chari, Banerjee, & Pachter (2023)** ("The Specious Art of Single-Cell Genomics") as our primary mathematical backing. They proved that UMAP distortions make it fundamentally unsuitable for quantitative morphological measurements.
*   **Methodological Decision:** We perform **all** 11 morphological calculations (e.g., dispersion, volume, intrinsic dimension, cognitive distance, centroid drift) in the **raw 768-dimensional SPECTER2 space**. UMAP is used **exclusively** for qualitative, auxiliary visualization.

---

## 2. High-Dimensional Geometry: The Hubness Phenomenon

### The Caveat:
As the dimensionality of a vector space increases, the distribution of distances between points concentrates: the difference between the distance to the nearest neighbor and the distance to the furthest neighbor approaches zero. In this high-dimensional regime, an artifact called **hubness** emerges.

### Mathematical & Conceptual Risks:
1. **Emergence of Hubs:** A small number of points (hubs) naturally become the nearest neighbors of an exceptionally large number of other points. In document spaces, a "hub paper" appears as the semantic neighbor of almost all other papers, regardless of their actual topical focus.
2. **Orphans:** Conversely, a large number of points (orphans) become the nearest neighbors of zero points, despite not being spatial outliers.
3. **Density Distortion:** Nearest-neighbor statistics can be dominated by these hubs, potentially introducing structural bias into raw nearest-neighbor measurements if left unchecked.

### Academic Defense (How We Mitigate It):
*   We cite **Radovanović, Nanopoulos, & Ivanović (2010)** ("Hubs in Space") to establish that hubness is an inherent property of high-dimensional spaces.
*   **Methodological Decision:** In our active pipeline, we measure local hubness as a descriptive morphological metric using the Gini coefficient of nearest-neighbor indegrees (`embedding_knn_indegree_gini`). This helps us characterize the structural topology of subfields.
*   **Non-Active Extension (Future Work):** We do **not** apply global distance-reduction corrections (such as Mutual Proximity or Local Scaling) in our active empirical pipeline. We discuss these methods—and the use of libraries like `skhubness` (**Feldbauer et al., 2019**)—strictly as **future work extensions** to mitigate high-dimensional neighborhood bias.

---

## 3. Clustering Axioms: Kleinberg's Impossibility Theorem

### The Caveat:
Clustering is a fundamental step in grouping subfields into Morphological Typologies. However, there is no mathematically "perfect" clustering algorithm.

### Mathematical & Conceptual Risks:
*   In **Kleinberg (2002)**, it was proven that no single clustering algorithm can simultaneously satisfy three intuitive, basic axioms:
    1.  **Scale-Invariance:** Scaling all distances by a constant factor does not change the resulting clustering.
    2.  **Richness:** The algorithm can produce any arbitrary partition of the data depending on the distance matrix.
    3.  **Consistency:** Shrinking distances within clusters or expanding distances between clusters does not change the clustering.
*   Therefore, any taxonomy of scientific disciplines produced by clustering is a **heuristic compromise** shaped by the specific algorithm's mathematical biases.

### Academic Defense (How We Mitigate It):
*   We cite **Kleinberg (2002)** to provide a rigorous theoretical backing for treating our morphological typologies as **exploratory, qualitative heuristics** rather than objective, absolute taxonomies of knowledge.
*   **Methodological Decision:** Our active pipeline groups disciplines strictly by running standard KMeans and hierarchical Ward clustering **over the reduced 11 embedding-space metric profiles** at the subfield level.
*   **Non-Active Extension (Future Work):** We do **not** build paper-level citation or kNN graphs, nor do we run graph-level community detection such as **Leiden community detection** (**Traag et al., 2019**). These paper-level graph models are explicitly relegated to **future work extensions**.

---

## 4. Database Coverage: OpenAlex Metadata and Classification Biases

### The Caveat:
Our empirical pipeline relies on OpenAlex. While OpenAlex is an exceptional open database, it is not free of metadata skew and coverage biases.

### Mathematical & Conceptual Risks:
1. **Text Dependency:** SPECTER2 generates representations based on titles and abstracts. In OpenAlex, papers that are missing abstract metadata will yield lower-quality representations.
2. **Taxonomy Imperfections:** OpenAlex assigns concepts to papers using an automated machine learning classifier. Errors in this automated classification can introduce noise into our subfield point clouds.
3. **Geographical & Language Bias:** Scholarly indexes historically underrepresent non-English and regional publications. This skew can distort our macro-level temporal maps of global scientific evolution.

### Academic Defense (How We Mitigate It):
*   We cite **Visser et al. (2020)** and **Scheidsteger & Lindner (2025)**, who audited OpenAlex's coverage and demonstrated that its metadata quality is statistically equivalent to commercial standards (WoS, Scopus) for large-scale science mapping.
*   **Methodological Decision:** We implement a strict data cleaning protocol: filtering out works missing abstract text, restricting our analysis to peer-reviewed publications, and acknowledging regional metadata limitations in the discussion.

---

## 5. Semantic Circularity: SPECTER2 Citation Graph Bias

### The Caveat:
SPECTER2 embeddings are pretrained using citation links. The model learns to pull citing and cited papers closer together in the embedding space.

### Mathematical & Conceptual Risks:
*   If we use SPECTER2 embeddings to measure cognitive distance and then use citation counts or citation links to validate those distances, we introduce a **methodological circularity**. The embedding space is *already* citation-informed.

### Academic Defense (How We Mitigate It):
*   We cite **Cohan et al. (2020)** and **Singh et al. (2023)** to acknowledge the citation-informed nature of SPECTER2.
*   **Methodological Decision:** We do not use citation links as our sole validation of cognitive distance. Instead, we cross-validate our morphological metrics against independent structural measures, such as the hierarchical OpenAlex concept taxonomy, co-authorship networks, and taxonomic subject classifications.
