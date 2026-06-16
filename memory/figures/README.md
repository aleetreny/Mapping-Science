# Figures Folder Naming Convention

This folder contains only the figures currently compiled into the thesis, plus
the UMAP atlas images used by Appendix B. Superseded variants and diagnostics are
kept locally in `memory/legacy/`, which is ignored by Git.

## Naming Convention

Active chapter figures follow the current thesis chapter numbering:

```text
fig_[chapter_num]_[descriptive_snake_case_name].[ext]
```

## Active Figure Set

- Chapter 3: OpenAlex corpus construction pipeline
- Chapter 4: static structural morphology profiles and diagnostics
- Chapter 5: temporal evolution, centroid drift, and dynamic typology
- Chapter 6: field-pair morphology, convergence, and divergence
- Appendix B: selected UMAP atlas images in `umap_atlas/`

## Best Practices
- Prefer PDF for figures included in LaTeX, with PNG retained only as a useful
  preview of the same active figure.
- Move exploratory variants, old diagnostics, and unused images to the ignored
  legacy folder instead of keeping them in the public thesis tree.
