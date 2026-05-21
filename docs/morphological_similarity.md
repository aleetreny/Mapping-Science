# Morphological Similarity

`scripts/15_compute_morphological_similarity_evolution.py` measures static and
temporal morphological similarity between disciplines using reduced
embedding-space metric profiles.

Inputs:

```text
outputs/04_reduced_metric_core/reduced_interpretable_core_metrics.parquet
data/processed/temporal/subfield_window_embedding_metrics.parquet
```

Outputs:

```text
data/processed/temporal/morphological_pair_distances_static.parquet
data/processed/temporal/morphological_pair_distances_by_window.parquet
data/processed/temporal/morphological_pair_distance_changes.parquet
outputs/07_morphological_similarity/
```

The script supports Euclidean and correlation distances at subfield, field, and
domain levels. Negative temporal deltas indicate convergence; positive deltas
indicate divergence.

Run:

```powershell
.\.venv\Scripts\python.exe scripts\15_compute_morphological_similarity_evolution.py --overwrite
```
