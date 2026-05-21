# Script 16: Depuración y Diagnóstico del Núcleo Limpio de Embeddings

## Objetivo del Script
El objetivo principal de `16_build_clean_embedding_core_diagnostics.py` es consolidar una tabla depurada y optimizada de descriptores estructurales en alta dimensión, reduciendo las 32 variables calculadas inicialmente en el Script 12 a un **núcleo limpio de 14 métricas esenciales**. El script automatiza la eliminación de colinealidades extremas, variables redundantes e inestabilidades diagnósticas, generando matrices de correlación cruzada de Spearman y de Pearson para auditar matemáticamente que el conjunto final conserve la máxima representatividad sin redundancia artificial.

## Racional de Exclusión y Métricas Eliminadas
Para evitar sesgar los análisis estadísticos posteriores debido al peso inflado de variables idénticas, el pipeline eliminó **12 métricas redundantes**:

1. **Exclusiones por Redundancia Lineal o del Coseno Extremas ($\rho \ge 0.99$):**
   - `embedding_distance_to_centroid_mean`: Excluida por ser menos robusta y poseer una correlación perfecta ($\rho = 0.992$) con `embedding_distance_to_centroid_median`.
   - `embedding_knn_mean_distance`: Excluida por colinealidad matemática casi absoluta ($\rho = 0.997$) con `embedding_knn_median_distance`.
   - `embedding_graph_edge_distance_median` y `p90`: Excluidas por colinealidades extremas ($\rho = 0.999$) con sus equivalentes analíticos directos `embedding_knn_median_distance` y `embedding_knn_p90_distance`.

2. **Exclusiones de Bloques Dimensionales PCA Altamente Colineales:**
   - `embedding_pca_first_component_share` y `embedding_pca_top3_variance_share`: Eliminadas debido a colinealidades extremas ($\rho \approx 0.98$) con `embedding_pca_top5_variance_share`, la cual se conserva por ser una métrica integradora.
   - `embedding_pca_participation_ratio`: Excluida por su fuerte correlación negativa lineal con la porción explicada por las componentes principales principales.
   - `embedding_pca_dim_50`: Eliminada en favor de `embedding_pca_dim_80`, que captura un umbral de cobertura tridimensional mucho más representativo ($80\%$ de la varianza tridimensional).

3. **Exclusiones por Sensibilidad a Outliers e Inestabilidad:**
   - `embedding_tail_index_p90_median`: Excluida en favor de percentiles de vecindad directa por su propensión a inflarse en presencia de pequeños ruidos muestrales locales.
   - `embedding_annual_centroid_path_length` y `embedding_directionality_ratio`: Retiradas del núcleo de clustering principal por inestabilidad temporal en subcampos con muy baja tasa de publicación anual en las ventanas tempranas.

## El Conjunto de 14 Métricas del Núcleo Limpio (Clean Core)
Las 14 variables resultantes cubren de forma equitativa los diferentes bloques teóricos del espacio de conocimiento latente:
- **Cohesión:** `embedding_centroid_norm`, `embedding_distance_to_centroid_median`, `embedding_distance_to_centroid_iqr`, `embedding_distance_to_centroid_p90`.
- **Densidad y Hubness:** `embedding_knn_median_distance`, `embedding_knn_p90_distance`, `embedding_knn_distance_cv`, `embedding_knn_indegree_gini`.
- **Dimensionalidad:** `embedding_pca_top5_variance_share`, `embedding_pca_dim_80`, `embedding_pca_spectral_entropy`.
- **Dinámica Temporal:** `embedding_centroid_drift_early_late`, `embedding_radial_expansion_slope`, `embedding_recent_novelty_score`.

## Parámetros de Entrada y Salida
- **Configuración y Entrada:**
  - Tabla amplia de métricas del embedding: `data/processed/subfield_embedding_space_metrics.parquet`.
  - Umbral mínimo de observaciones válidas por columna: `--min-non-missing-share 0.70`.
- **Rutas de Salida de Datos (Carpeta `outputs/analysis/clean_embedding_core_metrics/`):**
  - **Tabla Saneada del Núcleo Limpio:**
    - `clean_embedding_core_metrics.parquet` y `csv` (La matriz final optimizada de 241 filas y 14 columnas de métricas).
  - **Matrices de Relaciones:**
    - `clean_core_pearson_correlation_matrix.csv` y `Spearman` equivalente.
    - `clean_core_top_abs_pearson_pairs.csv` y `Spearman` equivalente (Lista ordenada de colinealidades remanentes).
  - **Gráficos y Reportes:**
    - `clean_core_pearson_correlation_heatmap.png` y `Spearman` heatmap.
    - `summary.json` y `summary.md`.

## Resultados Reales de Correlación Remanente
Dentro del núcleo limpio, las colinealidades remanentes más altas corresponden a leyes de potencia de dispersión inevitables:
- `embedding_centroid_norm` vs `embedding_distance_to_centroid_median`: Spearman $\rho = -0.992$. (Aunque colineales, se retienen ambos en esta fase para validar la sensibilidad en análisis de robustez).
- `embedding_knn_median_distance` vs `embedding_knn_p90_distance`: Spearman $\rho = 0.951$.
- `embedding_pca_top5_variance_share` vs `embedding_pca_spectral_entropy`: Spearman $\rho = -0.911$.

## Integración en la Tesis (TFM)
La depuración realizada en este script representa el **saneamiento de datos para la reproducibilidad metodológica** del TFM. Probar algoritmos cienciométricos con variables redundantes (como poseer 4 distancias kNN colineales al $99.9\%$) inyecta un sesgo implícito que distorsiona la métrica de enlace Ward y los dendrogramas. Este script expone de forma transparente y matemáticamente justificada el descarte de las variables superfluas, garantizando que cada una de las dimensiones morfológicas del conocimiento (especificidad, dispersión, dimensionalidad y cambio) aporte un peso balanceado y único al modelo tipológico de la tesis.
