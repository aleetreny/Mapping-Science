# Tables Folder Naming Convention

This folder contains only the LaTeX tables currently compiled into the thesis.
Editorial drafts and unused table variants are kept locally in `memory/legacy/`,
which is ignored by Git.

## Naming Convention

Active tables follow the current thesis chapter numbering:

```text
tab_[chapter_num]_[descriptive_snake_case_name].[ext]
```

## Active Table Set

- Chapter 3: `tab_03_corpus_summary.tex`, `tab_03_corpus_filters.tex`,
  `tab_03_domain_distribution.tex`, `tab_03_metric_core.tex`
- Chapter 4: `tab_04_extreme_subfields.tex`

## Best Practices
- Use `.tex` files here only when they are directly included by `memoria.tex`
  or one of the active chapter files.
- Keep exploratory, superseded, or diagnostic tables outside the public thesis
  tree.
