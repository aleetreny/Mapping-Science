# Visualization

UMAP is retained only for visualization and inspection.

Active visualization scripts:

```powershell
.\.venv\Scripts\python.exe scripts\09_build_global_umap_visualization.py --sample-per-subfield 500 --year-min 2000 --year-max 2024 --force
.\.venv\Scripts\python.exe scripts\10_build_subfield_umap_visualizations.py --year-min 2000 --year-max 2024 --overwrite
.\.venv\Scripts\python.exe scripts\10b_build_field_umap_visualizations.py --year-min 2000 --year-max 2024 --overwrite
.\.venv\Scripts\python.exe scripts\10c_build_domain_umap_visualizations.py --year-min 2000 --year-max 2024 --overwrite
.\.venv\Scripts\python.exe scripts\16_build_temporal_centroid_visualizations.py --overwrite
.\.venv\Scripts\python.exe scripts\17_build_temporal_umap_visualizations.py --overwrite
```

Outputs:

```text
outputs/08_visualization/global_umap/
outputs/08_visualization/per_subfield_umap_smooth_density/
outputs/08_visualization/per_field_umap_smooth_density/
outputs/08_visualization/per_domain_umap_smooth_density/
outputs/08_visualization/temporal_centroid_paths/
outputs/08_visualization/subfield_temporal_umap_panels/
```

The preferred domain, field, and subfield maps use the `smooth_hist` density
renderer and are stored in the `_smooth_density` folders. No UMAP-derived metric
table is part of the active thesis pipeline.
