# Script 19: Evolución Temporal Longitudinal de Métricas de Embedding

## Objetivo del Script
El propósito principal de `19_compute_temporal_embedding_metric_evolution.py` es calcular y analizar la evolución diacrónica de los descriptores topológicos de los subcampos científicos a lo largo de un período de 25 años (2000-2024). Para evitar el ruido inherente a los análisis anuales individuales y asegurar muestras estadísticamente representativas, el script divide la línea temporal en **cinco ventanas fijas de 5 años**, recomputando las 8 métricas no temporales del núcleo de embedding en cada ventana y para cada subcampo. Esto permite trazar trayectorias históricas de expansión semántica, densificación local y transformaciones dimensionales intrínsecas a escala disciplinar.

## Metodología y Ventanas Temporales
La segmentación diacrónica se define mediante las siguientes ventanas temporales de cinco años:
1. **2000-2004** (Ventana de Referencia o Inicial)
2. **2005-2009**
3. **2010-2014**
4. **2015-2019**
5. **2020-2024** (Ventana Reciente o Final)

En cada ventana temporal, el script selecciona los documentos publicados en dicho intervalo para cada subcampo, extrae sus embeddings SPECTER2 de alta dimensión, y calcula los 8 descriptores topológicos esenciales:
- **Dispersión Global:** `distance_to_centroid_median`, `distance_to_centroid_iqr`, `distance_to_centroid_p90`.
- **Vecindad Local e Inequidad (Hubness):** `knn_median_distance`, `knn_distance_cv`, `knn_indegree_gini`.
- **Dimensionalidad Intrínseca:** `pca_dim_80`, `pca_spectral_entropy`.

### Control de Calidad y Filtrado de Muestras (QA)
Para evitar estimaciones sesgadas por bajo volumen de publicaciones, se introduce el hiperparámetro `min_papers_per_window = 100`. Si un subcampo posee menos de 100 artículos en una ventana de 5 años, dicha fila subcampo-ventana es omitida de los análisis. 
- *Alineación de Datos:* El script trabaja directamente sobre la matriz mapeada en memoria de embeddings `float16.npy` y el índice unificado, garantizando una consistencia absoluta fila a fila.
- *Cálculo de deltas estandarizados:* La variación global de un subcampo se calcula restando la métrica en la primera ventana disponible de la última ventana disponible, estandarizando la diferencia mediante z-scores en base a la desviación típica del conjunto total de observaciones subcampo-ventana.

## Parámetros de Entrada y Salida

- **Configuración y Entrada:**
  - Índice alineado de embeddings: `data/processed/analysis_embedding_index.parquet`
  - Matriz global de embeddings en memoria: `embeddings/specter2_v1_2000_2024_400py/analysis/main_embeddings.float16.npy`
  - Umbral mínimo: `min_papers_per_window = 100`, Vecinos: `k = 15`.

- **Rutas de Salida de Datos (Carpeta `data/processed/temporal/` y `outputs/analysis/temporal_embedding_metric_evolution/`):**
  - **Tablas de datos estructurados:**
    - `subfield_window_embedding_metrics.parquet` (y `.csv`): Matriz larga de 1204 observaciones con los descriptores de cada subcampo-ventana.
    - `subfield_metric_temporal_changes.parquet` (y `.csv`): Deltas absolutos y estandarizados, pendientes lineales y correlaciones de Spearman por subcampo y métrica.
    - `subfield_overall_temporal_change_ranking.parquet` (y `.csv`): Ranking de subcampos más cambiantes según la suma de deltas absolutos estandarizados.
    - `field_window_embedding_metric_trajectories.parquet` (y `domain_` / `overall_`): Trayectorias agregadas promedio a escala macro y meso.
  - **Heatmaps y Gráficos del Cambio:**
    - `temporal_metric_delta_heatmap.png` (Heatmap de deltas estandarizados).
    - `temporal_metric_slope_heatmap.png` (Heatmap de pendientes temporales continuas).
    - `top_changed_subfields_overall.png` (Perfil visual de los 10 subcampos de mayor deriva).
    - `top_changed_subfields_by_metric.png` (Mayores cambios individuales por descriptor).
    - `domain_average_metric_trajectories.png` y trayectorias específicas por pilares.

---

## Diagnósticos y Auditoría de Datos (QA Real)

- **Total de observaciones planificadas:** 1205 filas ($241\text{ subcampos} \times 5\text{ ventanas}$).
- **Total de observaciones completadas:** 1204 filas ($99.91\%$ de completitud).
- **Fila Omitida por Insuficiencia Numérica (Shortfall):**
  - **Subcampo Omitido:** `2605` *Computational Mathematics* (Matemáticas, Physical Sciences) en la ventana inicial **2000-2004**.
  - **Causa:** Solo contaba con **92 documentos** disponibles en ese quinquenio, quedando por debajo del umbral mínimo de 100.
  - **Acción Correctora:** El script de forma automática calcula sus deltas comparando la ventana *2005-2009* frente a *2020-2024*. La exclusión es registrada en los archivos diagnósticos (`subfield_window_sample_size_diagnostics.csv`) para advertir al investigador en la fase de interpretación cualitativa.

---

## Resultados y Hallazgos Epistemológicos

El script identificó los subcampos científicos que han experimentado las mayores reestructuraciones topológicas en su conocimiento. A continuación se detallan las derivas más extremas clasificadas por pilares:

### 1. El Top 5 de Subcampos más Dinámicos (Deriva Semántica Total)
1. **`1211` Philosophy** (Suma de deltas estandarizados = **8.778**): Reestructuración extrema de su espacio semántico global.
2. **`3500` General Dentistry** (Suma = **7.252**): Proceso masivo de compactación local.
3. **`1313` Molecular Medicine** (Suma = **6.364**): Expansión radial y fragmentación sub-disciplinar.
4. **`3200` General Psychology** (Suma = **5.941**): Uniformización de sus vecindades.
5. **`2725` Infectious Diseases** (Suma = **5.835**): Compactación y surgimiento de hubs inducido por crisis sanitarias globales (COVID-19/Virología).

### 2. Comportamientos Extremos por Descriptores Singulares

#### A. Expansión y Compresión de la Dispersión Global (`distance_to_centroid_iqr`)
- **Máxima Expansión Radial (Aumento del IQR):** **`1313` Molecular Medicine** ($z\text{-score delta} = +4.700$). Su nube semántica en 768D se ha vuelto sumamente heterogénea, revelando que la disciplina se ha ramificado en múltiples especialidades distantes de su centro de gravedad original.
- **Máxima Contracción Radial (Reducción del IQR):** **`1211` Philosophy** ($z\text{-score delta} = -6.537$). Su dispersión global se ha contraído a niveles críticos, evidenciando una fuerte convergencia terminológica o la concentración del canon de investigación filosófico.

#### B. Densificación Local (`knn_median_distance`)
- **Máxima Densificación (Compromiso Local de Proximidad):** **`3500` General Dentistry** ($z\text{-score delta} = -3.659$). Las distancias a los 15 vecinos más cercanos se redujeron drásticamente, revelando una cohesión y consolidación de nichos de investigación hiper-específicos. Le siguen **`2725` Infectious Diseases** ($-2.785$) y **`2730` Oncology** ($-2.738$).

#### C. Inequidad en Vecindad y Concentración de Atractores (`knn_indegree_gini`)
- **Aumento del Grado de Hubness (Gini In-degree):** **`2803` Biological Psychiatry** ($z\text{-score delta} = +1.956$) y **`2502` Biomaterials** ($+1.670$). Revela la aparición de artículos científicos ultra-atractores o metodologías transversales que centralizan la topología de vecindad del subcampo, polarizando la investigación local.

## Integración en la Tesis (TFM)
- **El Núcleo Dinámico de la Tesis:** Este script aporta **los resultados empíricos fundamentales sobre la evolución histórica de la ciencia**. Demuestra que las disciplinas no son estáticas, sino entidades topológicas que respiran, se contraen y se ramifican de manera medible en espacios vectoriales continuos.
- **Rigor Frente al Tribunal:** Proporciona un marco de auditoría impecable al identificar cuantitativamente las limitaciones de la muestra (como la exclusión controlada de *Computational Mathematics* en 2000-2004), blindando el trabajo metodológicamente contra acusaciones de generalizaciones apresuradas sobre datos insuficientes.
