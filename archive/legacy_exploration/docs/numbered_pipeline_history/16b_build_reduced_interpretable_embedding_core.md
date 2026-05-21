# Script 16b: Definición del Núcleo Interpretable Reducido de 11 Métricas

## Objetivo del Script
El objetivo principal de `16b_build_reduced_interpretable_embedding_core.py` es definir y estructurar la selección final de descriptores estructurales que se utilizarán en los análisis de agrupamiento y dinámicas temporales del TFM. El script refina las 14 variables del núcleo limpio (Script 16) eliminando redundancias residuales específicas y agrupando los descriptores restantes en **11 métricas clave organizadas en 4 pilares teóricos**. Esto simplifica y maximiza la interpretabilidad conceptual de los resultados, aislando las variables más explicativas para caracterizar la cohesión, densidad local, dimensionalidad intrínseca y evolución temporal de cada subcampo científico.

## Estructura Teórica de los 4 Pilares del Núcleo Reducido
El núcleo interpretable final de 11 métricas se divide en cuatro dimensiones complementarias de la epistemología de la ciencia:

### Pilar 1: Dispersión Semántica Global (Global Semantic Dispersion)
Mide la cohesión general y homogeneidad conceptual del subcampo respecto a su centro de gravedad semántico en 768D:
- `embedding_distance_to_centroid_median`: La mediana de la distancia del coseno de todos los documentos al centroide disciplinario.
- `embedding_distance_to_centroid_iqr`: Rango intercuartílico de la distancia al centroide (mide la variabilidad o simetría de la dispersión).
- `embedding_distance_to_centroid_p90`: Distancia al centroide en el percentil 90 (caracteriza el límite extremo o la periferia del subcampo).

### Pilar 2: Densidad Semántica Local e Interconexión (Local Density and Hubness)
Caracteriza la microestructura de vecindad y la distribución de eminencias o nichos densos de investigación:
- `embedding_knn_median_distance`: Mediana de la distancia del coseno a los $k=15$ vecinos más cercanos de cada documento en 768D.
- `embedding_knn_distance_cv`: Coeficiente de variación de las distancias kNN (mide el grado de heterogeneidad o irregularidad de las vecindades locales).
- `embedding_knn_indegree_gini`: Índice de desigualdad de Gini para el grado de entrada (*in-degree*) en el grafo de vecinos más cercanos. Captura la presencia de "hubs" semánticos o documentos ultra-citados/atraedores en la topología local del subcampo.

### Pilar 3: Dimensionalidad Semántica Intrínseca (Intrinsic Dimensionality)
Estima la complejidad estructural o el número efectivo de direcciones ortogonales de desarrollo temático en alta dimensión:
- `embedding_pca_dim_80`: Número mínimo de componentes principales necesarios para explicar el $80\%$ de la varianza lineal en el subcampo en 768D.
- `embedding_pca_spectral_entropy`: Entropía espectral basada en los autovalores de la covarianza del subcampo (valores altos revelan que el conocimiento está disperso uniformemente en múltiples dimensiones, mientras que valores bajos indican polarización unidireccional).

### Pilar 4: Movimiento Semántico Temporal (Temporal Semantic Movement)
Cuantifica la deformación histórica y la velocidad de desplazamiento del subcampo a lo largo de los 25 años del corpus:
- `embedding_centroid_drift_early_late`: Distancia de deriva del centroide disciplinario en 768D entre la primera ventana (2000-2004) y la última (2020-2024).
- `embedding_radial_expansion_slope`: Pendiente temporal de la expansión/compresión radial del subcampo.
- `embedding_recent_novelty_score`: Puntuación de novedad reciente de las publicaciones disciplinarias.

## Exclusiones Residuales Adoptadas
Para lograr este núcleo compacto, se eliminaron **3 variables del núcleo limpio**:
- `embedding_centroid_norm`: Excluida por ser prácticamente la inversa geométrica lineal directa de `embedding_distance_to_centroid_median`.
- `embedding_knn_p90_distance`: Descartada por colinealidad extrema con `embedding_knn_median_distance`.
- `embedding_pca_top5_variance_share`: Eliminada para favorecer a `embedding_pca_spectral_entropy` y `dim_80` como descriptores dimensionales intrínsecos de base teórica más sólida.

## Parámetros de Entrada y Salida
- **Configuración y Entrada:**
  - Tabla de métricas broad de embedding: `data/processed/subfield_embedding_space_metrics.parquet`.
- **Rutas de Salida de Datos (Carpeta `outputs/analysis/reduced_interpretable_embedding_core/`):**
  - **Tabla de Métricas Reducidas Definitivas:**
    - `reduced_interpretable_core_metrics.parquet` y `csv` (La matriz estructurada final de 241 filas y 11 columnas).
  - **Matrices diagnósticas:**
    - `reduced_core_pearson_correlation_matrix.csv` y `Spearman` equivalente.
    - `reduced_core_top_absolute_pearson_pairs.csv` y `Spearman` equivalente.
  - **Gráficos e Interpretaciones:**
    - `reduced_core_histograms.png` (Histogramas detallados de los 11 descriptores seleccionados).
    - `reduced_core_pearson_correlation_heatmap.png` y `Spearman` heatmap.
    - `summary.json` y `summary.md`.

## Integración en la Tesis (TFM)
El núcleo interpretable reducido representa **el marco conceptual definitivo para el modelado matemático** del TFM. En lugar de enfrentar a un tribunal de tesis con decenas de variables de difícil interpretación física, el alumno puede argumentar que ha sintetizado toda la topología tridimensional y temporal del conocimiento en **11 variables canónicas y complementarias**, organizadas en pilares cienciométricos intuitivos. Este marco permite que los capítulos finales de la tesis sobre la dinámica temporal (Fase 7) y la evolución espacial de la ciencia posean una lectura ágil, fluida y teóricamente rigurosa, enriqueciendo la discusión epistemológica general.
