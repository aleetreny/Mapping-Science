# Exploratory Morphological Typologies

Generated: 2026-06-15T11:55:32.096847+00:00

## Input

- Input table: `outputs/04_reduced_metric_core/reduced_interpretable_core_metrics.csv`
- Unit clustered: 241 subfields.
- Feature set: 8 reduced embedding-space morphology metrics.
- Missing values in selected metrics: 0.
- Scaling used for the selected solution: robust median/IQR scaling across subfields.
- Projection coordinates are not used for clustering.

## Selected solution

- Algorithm: Ward hierarchical clustering.
- Distance geometry: Euclidean distance in robust-scaled eight-metric structural profile space.
- Selected k: 5.
- Silhouette: 0.174.
- Cluster sizes: 11;40;44;64;82.
- PCA visualization variance: PC1 35.9%, PC2 31.8%.

## Typologies

- T1 Broad dispersed profiles: 11 subfields; high centroid IQR, kNN CV; low PCA entropy, kNN median.
- T2 Compact dense profiles: 40 subfields; high hub Gini, PCA D80; low centroid P90, centroid median.
- T3 Hub-concentrated uneven profiles: 82 subfields; high kNN median, PCA D80; low kNN CV, hub Gini.
- T4 Spectrally complex profiles: 44 subfields; high centroid median, centroid P90; low hub Gini, PCA entropy.
- T5 Uneven dispersion outliers: 64 subfields; high kNN CV, centroid IQR; low PCA entropy, kNN median.

## Robustness

- Mean subsample ARI over 100 80% subsamples: 0.427.
- Mean subsample AMI over 100 80% subsamples: 0.531.
- Agreement with plausible alternatives:

  - z-score Ward, structural 8 metrics: ARI 0.394, AMI 0.495; sizes 31;31;34;71;74.
  - robust Ward, structural 8 metrics: ARI 1.000, AMI 1.000; sizes 11;40;44;64;82.
  - family-balanced robust Ward: ARI 0.695, AMI 0.720; sizes 9;17;35;85;95.
  - robust k-means, structural 8 metrics: ARI 0.367, AMI 0.458; sizes 12;51;55;61;62.
  - average-link correlation: ARI 0.282, AMI 0.451; sizes 25;31;54;63;68.
  - average-link Euclidean: ARI 0.025, AMI 0.067; sizes 1;1;2;5;232.

## Interpretation

The evidence supports a weak, descriptive typology rather than a strong natural partition. The selected solution avoids singleton clusters and gives interpretable profiles, but silhouette and sensitivity diagnostics show that boundaries are fuzzy.

## Reproducibility note

Run `python scripts/18_explore_morphological_typologies.py --overwrite` from the repository root to regenerate the typology outputs, figures, and thesis tables.

The active thesis evidence is the eight-metric structural core. Domain and field labels are metadata for interpretation only; they are not used as clustering targets.
