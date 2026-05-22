# Exploratory Morphological Typologies

Generated: 2026-05-22T16:54:46.086146+00:00

## Input

- Input table: `outputs/04_reduced_metric_core/reduced_interpretable_core_metrics.csv`
- Unit clustered: 241 subfields.
- Feature set: 11 reduced embedding-space morphology metrics.
- Missing values in selected metrics: 0.
- Scaling used for the selected solution: robust median/IQR scaling across subfields.
- Projection coordinates are not used for clustering.

## Selected solution

- Algorithm: Ward hierarchical clustering.
- Distance geometry: Euclidean distance in robust-scaled eleven-metric profile space.
- Selected k: 5.
- Silhouette: 0.133.
- Cluster sizes: 10;26;49;67;89.
- PCA visualization variance: PC1 32.0%, PC2 24.1%.

## Typologies

- T1 Broad sparse profiles: 89 subfields; high kNN median, centroid median; low kNN CV, hub Gini.
- T2 Compact low-dispersion profiles: 49 subfields; high PCA D80, hub Gini; low centroid P90, centroid median.
- T3 Low-dimensional locally uneven profiles: 67 subfields; high kNN CV, centroid drift; low PCA entropy, PCA D80.
- T4 Temporal novelty profiles: 26 subfields; high centroid drift, recent novelty; low PCA entropy, PCA D80.
- T5 Uneven dispersion outliers: 10 subfields; high centroid IQR, kNN CV; low PCA entropy, kNN median.

## Robustness

- Mean subsample ARI over 100 80% subsamples: 0.355.
- Mean subsample AMI over 100 80% subsamples: 0.444.
- Agreement with plausible alternatives:

  - z-score Ward, all 11 metrics: ARI 0.306, AMI 0.409; sizes 12;34;45;66;84.
  - robust Ward, static 8 metrics: ARI 0.372, AMI 0.405; sizes 11;40;44;64;82.
  - family-balanced robust Ward: ARI 0.548, AMI 0.561; sizes 1;40;48;71;81.
  - robust k-means, all 11 metrics: ARI 0.412, AMI 0.416; sizes 1;32;50;63;95.
  - average-link correlation: ARI 0.263, AMI 0.389; sizes 9;45;59;61;67.
  - average-link Euclidean: ARI 0.054, AMI 0.134; sizes 1;1;3;6;230.

## Interpretation

The evidence supports a weak, descriptive typology rather than a strong natural partition. The selected solution avoids singleton clusters and gives interpretable profiles, but silhouette and sensitivity diagnostics show that boundaries are fuzzy.

## Reproducibility note

Run `python scripts/18_explore_morphological_typologies.py --overwrite` from the repository root to regenerate the typology outputs, figures, and thesis tables.

The active thesis evidence is the reduced eleven-metric core. Domain and field labels are metadata for interpretation only; they are not used as clustering targets.
