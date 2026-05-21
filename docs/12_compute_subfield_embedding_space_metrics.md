# Script 12: Extracción de Métricas Estructurales del Espacio de Embedding 768D

## Objetivo del Script
El objetivo principal de `12_compute_subfield_embedding_space_metrics.py` es calcular descriptores morfológicos y estructurales estrictos directamente sobre el espacio original de embeddings de alta dimensión (SPECTER2, 768D), evitando las distorsiones geométricas e topológicas inevitables de cualquier reducción dimensional proyectiva como UMAP. Para cada uno de los subcampos científicos que cumplen las condiciones del Manifiesto, el script extrae **26 variables estructurales del espacio latente**, estructurándolas en métricas de cohesión, dimensionalidad intrínseca, dispersión radial y topología de red de vecindades.

## Bloques Teóricos de Métricas Estructurales 768D
A diferencia de las variables proyectadas en UMAP 2D, las métricas del embedding de alta dimensión reflejan las relaciones semánticas reales del espacio vectorial original:

1. **Métricas de Cohesión Central y Dispersión (Distancia al Centroide):**
   - `embedding_centroid_norm`: La norma del vector centroide del subcampo en 768D (calculada a partir de los embeddings de documentos normalizados en L2). Cuantifica la "densidad absoluta" y especificidad temática: valores altos indican un núcleo altamente enfocado, mientras que valores bajos revelan dispersión o heterogeneidad conceptual extrema.
   - `embedding_distance_to_centroid_median`, `embedding_distance_to_centroid_mean`, `embedding_distance_to_centroid_p90` y `cv`: Descriptores continuos de la distribución de distancias de los documentos individuales respecto a su centroide disciplinario. El coeficiente de variación (`cv`) mide la homogeneidad de la distribución de dispersión.

2. **Métricas de Densidad Local de Vecindario (k-Nearest Neighbors - kNN):**
   - `embedding_knn_mean_distance`, `embedding_knn_median_distance` y `embedding_knn_p90_distance`: Calculan la distancia del coseno a los $k$ vecinos más cercanos ($k=15$) en el espacio 768D.
   - `embedding_knn_distance_cv`: Coeficiente de variación de las distancias a vecinos. Identifica subcampos con alta irregularidad en sus densidades semánticas internas (zonas ultra-densas combinadas con periferias dispersas).

3. **Métricas de Dimensionalidad Intrínseca (PCA en Alta Dimensión):**
   - `embedding_pca_first_component_share`: Fracción de la varianza explicada por el primer componente principal lineal dentro del subcampo en 768D. Mide qué tan lineal u unidireccional es la estructura conceptual del subcampo.
   - `embedding_pca_top3_variance_share`, `embedding_pca_top5_variance_share`: Fracción acumulada de varianza de los primeros 3 y 5 componentes principales.
   - `embedding_pca_participation_ratio`: Mide la dimensionalidad intrínseca efectiva del subcampo basándose en los autovalores de su matriz de covarianza. Valores altos indican que el subcampo abarca múltiples direcciones semánticas ortogonales sin una dirección obvia dominante.

4. **Métricas de Conectividad de Red en Alta Dimensión (Grafo e Índices de Cola):**
   - `embedding_graph_edge_distance_median` y `p90`: Distancias del coseno a lo largo de las aristas de un grafo de vecindades mutuas construido en 768D.
   - `embedding_tail_index_p90_median`: Mide el decaimiento de las distancias en la cola del vecindario local, capturando la presencia de "hubs" o fronteras difusas.
   - `core_periphery_ratio`: Evalúa la polarización de la disciplina entre un núcleo ultra-especializado y una periferia de trabajos dispersos o interdisciplinarios.

## Parámetros de Entrada y Salida
- **Configuración y Entrada:**
  - Matriz principal de embeddings: `embeddings/specter2_v1_2000_2024_400py/analysis/main_embeddings.float16.npy`.
  - Índice parquet alineado fila por fila: `data/processed/analysis_embedding_index.parquet`.
  - Parámetros por defecto: $k=15$ vecinos más cercanos, estado aleatorio $42$, límites temporales del corpus completo.
- **Rutas de Salida de Datos:**
  - **Tabla de Métricas Estructurales del Embedding:**
    - `data/processed/subfield_embedding_space_metrics.parquet`
    - `data/processed/subfield_embedding_space_metrics.csv`
  - **Diccionario de Métricas:**
    - `outputs/metrics/subfield_embedding_space_metrics_dictionary.csv`
  - **Resumen Diagnóstico e Histogramas:**
    - `outputs/metrics/subfield_embedding_space_metrics_summary.json`
    - Carpeta de diagnósticos visuales y matrices de correlación: `outputs/analysis/embedding_space_metric_diagnostics/`

## Resultados Reales de Diagnóstico
El diagnóstico distributivo de estas 26 variables en los 241 subcampos científicos reveló relaciones matemáticas de redundancia casi absoluta en alta dimensión:
- **Redundancia del Centroide:** `embedding_centroid_norm` exhibe una correlación de Spearman perfecta de $\rho = -1.000$ con `embedding_distance_to_centroid_mean`, demostrando que la norma del vector agregado resume de forma exacta y simétrica la cohesión promedio de sus documentos constituyentes.
- **Redundancia kNN y Grafo:** `embedding_knn_p90_distance` vs `embedding_graph_edge_distance_p90` muestran una correlación de $\rho = 0.999$, al igual que las medianas correspondientes ($\rho = 0.999$). Esto evidencia que la topología de red construida en 768D preserva de forma casi idéntica las distancias métricas euclidianas directas del espacio latente.
- **Redundancia PCA:** Las métricas de varianza acumulada (`top3_variance_share`, `top5_variance_share` y `participation_ratio`) muestran colinealidades extremas ($\rho \approx -0.97$), sugiriendo la viabilidad de utilizar un subconjunto simplificado para caracterizar la dimensionalidad interna del espacio conceptual.

## Integración en la Tesis (TFM)
En el diseño metodológico de la tesis, este script constituye el **patrón de oro de la estructura del conocimiento**. Mientras que el Script 11 captura la morfología tal y como se proyecta visualmente para el ojo humano (2D), el Script 12 captura el estado original, no distorsionado, de la ciencia en el hiperespacio semántico. Al calcular paralelamente estas dos familias de métricas, el TFM puede testear formalmente la hipótesis de la "distorsión proyectiva", cuantificando el grado en que técnicas populares como UMAP introducen artefactos geométricos y alteran las verdaderas relaciones de proximidad de los subcampos científicos.
