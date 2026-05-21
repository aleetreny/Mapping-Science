# Output Organization

The active output tree is stage-based:

```text
outputs/
  01_corpus_construction/
  02_embedding_matrix/
  03_embedding_metrics/
  04_reduced_metric_core/
  05_static_comparison/
  06_temporal_evolution/
  07_morphological_similarity/
  08_visualization/
    per_subfield_umap_smooth_density/
    per_field_umap_smooth_density/
    per_domain_umap_smooth_density/
  archive_or_legacy/
```

- `outputs/02_embedding_matrix/` contains only lightweight summary and row-alignment validation diagnostics (`.json`), not the large matrix itself.
- `outputs/archive_or_legacy/` contains generated artifacts from the older UMAP metric, metric-family comparison, clustering, dimensionality-reduction, and semantic-distance experiments.

Ignored data archives live under:

```text
data/archive_or_legacy/
```

These local archives are safe to delete if disk space matters and the old
exploratory runs are no longer needed.
