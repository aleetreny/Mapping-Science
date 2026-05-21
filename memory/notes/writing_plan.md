# Thesis Writing Plan & Pipeline Mapping

This document outlines the active thesis direction, detailed chapter structure, and the exact mapping of specific Python scripts and data/visual outputs from the analysis pipeline to individual chapters.

---

## 1. Active Thesis Direction

The thesis is **descriptive and comparative**. It characterizes the structural morphology and longitudinal evolution of scientific disciplines using dense vector representations of title-plus-abstract text.

### Critical Methodology Constraints
- **Core Geometry**: All quantitative analyses, statistical rankings, and similarity matrices must be computed directly in the high-dimensional **original 768-dimensional SPECTER2 embedding space** to prevent distortion and projection artifacts.
- **Exploratory Typologies**: Unsupervised clustering is strictly exploratory and must be conducted *only* on the reduced 11 embedding-space metrics, not on UMAP coordinates. The clusters must be interpreted as descriptive typologies (conceptual heuristics) rather than absolute natural categories of science.
- **Integrated Visualization**: There is **no standalone visualization chapter**. Instead, UMAP coordinates and density visualizations are integrated throughout the thesis inside relevant chapters to provide interpretive and qualitative support.

---

## 2. Chapter Mapping to Pipeline Outputs

The following table maps each thesis chapter to its corresponding active pipeline scripts, raw/processed data dependencies, and final output deliverables.

| Chapter | Description / Scientific Focus | Python Scripts | Inputs & Data Dependencies | Expected Output Deliverables |
| :--- | :--- | :--- | :--- | :--- |
| **01 Intro** | Motivation, research questions (RQs), contributions. | N/A | `docs/TFM_summary.md` | Narrative overview, RQs list. |
| **02 Background** | Scientometrics literature, dense text representation, embedding geometry. | N/A | Literature databases | Reference lists, theoretical framing. |
| **03 Data & Corpus** | OpenAlex taxonomy, English filtering, 400 py sampling design, database validation. | `scripts/00_fetch_taxonomy.py`<br>`scripts/01_build_counts.py`<br>`scripts/02_build_corpus_plan.py`<br>`scripts/03_build_sample_plan.py`<br>`scripts/04_download_sampled_corpus.py`<br>`scripts/05_validate_database.py`<br>`scripts/06_build_analysis_subfields.py` | `data/raw/`<br>`data/interim/`<br>`warehouse/` | Table of subfield sampling sizes,<br>`outputs/01_corpus_construction/` validation metrics. |
| **04 Semantic Rep** | SPECTER2 model architecture, shard validation, building the row-aligned analysis matrix. | `scripts/07_validate_embeddings.py`<br>`scripts/08_prepare_analysis_matrix.py` | `data/processed/analysis_embedding_index.parquet`<br>`embeddings/.../main_embeddings.float16.npy` | Embedding dimension stats,<br>row-aligned matrix size validation,<br>`outputs/02_embedding_matrix/` diagnostics. |
| **05 Morph Metrics** | Geometric definitions and equations for the 11-metric reduced core across dispersion, density, hubness, and dimensionality. | `scripts/11_compute_embedding_space_metrics.py`<br>`scripts/12_build_reduced_metric_core.py` | `data/processed/subfield_embedding_space_metrics.parquet`<br>`docs/reduced_metric_core.md` | Formal mathematical formulations,<br>`outputs/04_reduced_metric_core/` Spearman/Pearson correlation maps. |
| **06 Static Comp** | Cross-sectional profiles of subfields, fields, and domains under the 11-metric core. | `scripts/13_analyze_static_discipline_profiles.py` | `outputs/04_reduced_metric_core/reduced_interpretable_core_metrics.parquet` | Rankings of most compact vs. dispersed subfields,<br>`outputs/05_static_comparison/` summary tables. |
| **07 Temporal Evol** | Longitudinal trajectories of the 8 non-temporal metrics across five 5-year windows (2000-2024). | `scripts/14_compute_temporal_metric_evolution.py` | `data/processed/temporal/subfield_window_embedding_metrics.parquet` | Trajectory line plots of dispersion and dimensionality,<br>`outputs/06_temporal_evolution/` shift rankings. |
| **08 Similarity** | Static and temporal morphological convergence (negative delta) and divergence (positive delta) between disciplines. | `scripts/15_compute_morphological_similarity_evolution.py` | `outputs/04_reduced_metric_core/reduced_interpretable_core_metrics.parquet` | Similarity matrices and dendrograms,<br>`outputs/07_morphological_similarity/` top converging pairs. |
| **09 Typologies** | Unsupervised clustering of subfields using the 11-metric profiles to discover descriptive typologies of disciplines. | N/A (Exploratory scripts to be developed) | `outputs/04_reduced_metric_core/reduced_interpretable_core_metrics.parquet` | Clustering dendrograms and cluster member profile tables.<br>Integrated UMAP plots for cluster visualization. |
| **10 Discussion** | Synthesis of static, temporal, similarity, and typology findings, methodological limitations, policy implications. | N/A | Empirical chapters 6, 7, 8, 9 | Conceptual synthesis, critique of transformer space biases. |
| **11 Conclusion** | Summary of contributions, final takeaways, avenues for future research. | N/A | All chapters | Key takeaways summary, future directions. |

---

## 3. Drafting Status & Readiness

- **Draftable Immediately** (Do not depend on final rerun metrics):
  - **Chapter 1: Introduction** — Ready for structure, objectives, and RQ descriptions.
  - **Chapter 2: Background and Related Work** — Ready for structural literature drafting.
  - **Chapter 3: Data and Corpus Construction** — Sampling design and counts can be written based on the `docs/data_corpus.md` and current data structures.
  - **Chapter 4: Semantic Representation** — Technical specification of SPECTER2 and matrix construction is stable.
  - **Chapter 5: Embedding-Space Morphological Metrics** — Mathematical equations and definitions for the 11 metrics are fully defined and can be written now.
- **Requires Final Pipeline Run Outputs** (Depend on final script execution):
  - **Chapter 6: Static Comparison of Scientific Disciplines** — Awaiting final rankings from `outputs/05_static_comparison/`.
  - **Chapter 7: Temporal Evolution of Scientific Morphology** — Awaiting final trajectories from `outputs/06_temporal_evolution/`.
  - **Chapter 8: Morphological Similarity, Convergence and Divergence** — Awaiting final convergence outputs from `outputs/07_morphological_similarity/`.
  - **Chapter 9: Exploratory Morphological Typologies** — Awaiting exploratory clustering run based on the 11-metric profiles.

---

## 4. Pending Literature Review Needs

To complete the introductory and background chapters, the following literature domains must be targeted:
1. **Classical Scientometrics**: Foundational science mapping based on co-citation analysis, journal classifications, and keyword networks.
2. **Dense Document Embeddings**: Semantic text representations, with focus on contrastive learning models and SPECTER/SPECTER2.
3. **High-Dimensional Geometry & Typology**: Epistemological and mathematical interpretations of dense vector space morphology, dispersion measures, intrinsic dimensionality, and taxonomic typologies of scientific disciplines.
