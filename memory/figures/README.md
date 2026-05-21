# Figures Folder Naming Convention and Guidelines

This folder contains all image assets and plots exported from the active research pipeline to be compiled into the thesis chapters.

## Naming Convention

All figures exported into this folder must adhere to a chapter-centric naming system:

```text
fig_[chapter_num]_[descriptive_snake_case_name].[ext]
```

### Examples by Chapter
- **Chapter 5: Embedding-Space Morphological Metrics**
  - `fig_05_reduced_core_spearman_heatmap.png`
  - `fig_05_reduced_core_histograms.png`
- **Chapter 6: Static Comparison of Scientific Disciplines**
  - `fig_06_domain_morphological_profiles.png`
- **Chapter 7: Temporal Evolution of Scientific Morphology**
  - `fig_07_subfield_dispersion_trajectories.png`
  - `fig_07_top_shifting_disciplines.png`
- **Chapter 8: Morphological Similarity, Convergence and Divergence**
  - `fig_08_subfield_morphological_similarity_heatmap.png`
- **Chapter 9: Visualization of the Scientific Space**
  - `fig_09_global_umap_layout.png`
  - `fig_09_smooth_density_fields.png`
  - `fig_09_temporal_centroid_paths.png`
- **Appendix C: Additional Figures**
  - `fig_C_diagnostic_knn_cv_distribution.png`

## Best Practices
- **Format**: Preferred formats are high-resolution PNG or vector PDF (if available).
- **Legibility**: Ensure axis labels, legends, and annotations are readable at thesis page scale.
- **Source**: Document which Python visualization script generated the figure.
