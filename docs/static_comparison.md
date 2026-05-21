# Static Comparison

`scripts/13_analyze_static_discipline_profiles.py` compares subfields, fields,
and domains using only the reduced 11-metric embedding-space core.

It writes:

```text
outputs/05_static_comparison/static_discipline_profiles.parquet
outputs/05_static_comparison/static_profile_distances.parquet
outputs/05_static_comparison/static_metric_rankings.csv
outputs/05_static_comparison/top_static_profile_pairs.csv
outputs/05_static_comparison/summary.md
```

Use this stage for questions such as which disciplines are more compact, more
dispersed, more intrinsically dimensional, or more hub-like.

Run:

```powershell
.\.venv\Scripts\python.exe scripts\13_analyze_static_discipline_profiles.py --overwrite
```
