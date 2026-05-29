# Temporal Evolution

`scripts/14_compute_temporal_metric_evolution.py` recomputes the eight
non-temporal structural metrics in five 5-year windows:

```text
2000-2004
2005-2009
2010-2014
2015-2019
2020-2024
```

The temporal metrics cover dispersion, local density, hubness, and intrinsic
dimensionality. They do not use UMAP-derived metrics.

Outputs:

```text
data/processed/temporal/subfield_window_embedding_metrics.parquet
data/processed/temporal/subfield_metric_temporal_changes.parquet
data/processed/temporal/subfield_overall_temporal_change_ranking.parquet
outputs/06_temporal_evolution/
```

Run:

```powershell
.\.venv\Scripts\python.exe scripts\14_compute_temporal_metric_evolution.py --overwrite
```
