# Tables Folder Naming Convention and Guidelines

This folder contains all data tables, descriptive summaries, and LaTeX-formatted table files to be compiled into the thesis chapters.

## Naming Convention

All tables exported into this folder must adhere to a chapter-centric naming system:

```text
tab_[chapter_num]_[descriptive_snake_case_name].[ext]
```

### Examples by Chapter
- **Chapter 3: Data and Corpus Construction**
  - `tab_03_corpus_subfield_counts.tex`
- **Chapter 5: Embedding-Space Morphological Metrics**
  - `tab_05_reduced_core_metrics_dictionary.tex`
- **Chapter 6: Static Comparison of Scientific Disciplines**
  - `tab_06_top_compact_subfields.tex`
  - `tab_06_top_dispersed_subfields.tex`
- **Chapter 7: Temporal Evolution of Scientific Morphology**
  - `tab_07_temporal_shift_rankings.tex`
- **Chapter 8: Morphological Similarity, Convergence and Divergence**
  - `tab_08_top_converging_pairs.tex`
  - `tab_08_top_diverging_pairs.tex`

## Best Practices
- **Format**: Preferred format for direct input into LaTeX is `.tex`. Raw spreadsheets can be stored as `.csv` for reproducibility.
- **UC3M Style**: Use standard LaTeX table markers, avoiding excessive vertical lines where possible. Follow the template's table settings detailed in `memoria.tex` (utilizing `\ttabbox` macros).
