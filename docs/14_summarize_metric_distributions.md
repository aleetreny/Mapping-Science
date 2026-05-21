# Script 14: Diagnóstico Distributivo y Filtro de Métricas de Baja Información

## Objetivo del Script
El objetivo principal de `14_summarize_metric_distributions.py` es realizar un filtrado estadístico y un control de calidad sobre el conjunto combinado de variables morfológicas y estructurales. El script evalúa cuantitativamente la distribución de cada una de las 33 métricas UMAP y 32 métricas de embedding de alta dimensión en los 241 subcampos científicos. Mediante el análisis de datos faltantes (*missingness*), entropía de valores únicos y dominancia de ceros, este script identifica y marca aquellas variables con bajo poder discriminatorio o colinealidad nula, sentando las bases para consolidar un núcleo de datos limpio e interpretable.

## Criterios de Selección y Banderas Diagnósticas
El script aplica umbrales paramétricos rigurosos para auditar la utilidad analítica de cada métrica:
1. **Falta de Datos (High Missingness):** Columnas con una tasa de nulos $\ge 30\%$. (Ninguna métrica del corpus final superó este umbral gracias a la robustez del muestreo).
2. **Baja Variabilidad (Low Uniqueness):** Columnas con $\le 5$ valores únicos en total o cuya tasa de valores únicos respecto al total de observaciones sea $\le 5\%$.
3. **Dominancia de Ceros (Zero Dominance):** Métricas dominadas por valores nulos en más del $80\%$ de los subcampos científicos.

## Resultados Reales de Auditoría
El script analizó 65 métricas y **marcó con banderas de alerta a 8 variables de baja información**, recomendando su exclusión del núcleo morfológico activo:

### 1. Constantes y Variables Estructurales sin Variación (Unique = 1)
Estas variables mostraron un único valor idéntico para todos los 241 subcampos, resultando inútiles para clasificaciones estadísticas:
- `embedding_graph_connected_component_count`: Siempre igual a $1.0$. Indica que el grafo de vecindades en 768D es universalmente conexo.
- `embedding_graph_largest_component_share`: Siempre igual a $1.0$, indicando que el componente gigante abarca el $100\%$ de los nodos en todas las disciplinas.
- `n_years_available`, `n_annual_centroids` y `n_pca_components_used`: Parámetros de control constantes e idénticos a lo largo del pipeline.

### 2. Dominancia Extrema de Ceros (Zero Dominated = 98%)
Estas variables resultaron nulas en casi todo el corpus, comportándose como ruido disperso:
- `outlier_share_outside_density_extent` (UMAP 2D): Tasa de ceros del **$98\%$** (solo 5 valores únicos).
- `outlier_share_r_gt_1_5` (UMAP 2D): Tasa de ceros del **$98\%$** (solo 7 valores únicos).
- *Razón física:* La eliminación de ruidos durante el filtrado y alineación de embeddings e índices limpia los bordes del mapa de forma tan eficiente que los outliers geométricos extremos desaparecen casi por completo del paisaje KDE disciplinario.

### 3. Muy Baja Unicidad
- `n_late_points`: Solo 7 valores únicos, actuando como variable discreta de control sin carácter morfológico representativo.

## Parámetros de Entrada y Salida
- **Configuración y Entrada:**
  - Métricas UMAP: `data/processed/subfield_morphology_metrics.parquet`.
  - Métricas del Embedding: `data/processed/subfield_embedding_space_metrics.parquet`.
  - Umbrales por defecto: `--high-missing-threshold 0.30`, `--low-unique-threshold 5`, `--low-unique-share-threshold 0.05`, `--zero-dominated-threshold 0.80`.
- **Rutas de Salida de Datos (Carpeta `outputs/analysis/metric_distributions/`):**
  - **Tabla de Métricas Excluidas:**
    - `low_information_metrics.csv` (Registro detallado de las 8 variables inhabilitadas y su motivo).
  - **Diagnóstico Gráfico Completo:**
    - `all_metric_boxplots_zscore.png` (Diagrama de caja y bigotes normalizado por Z-score para comparar rangos y simetrías).
    - `umap_metric_histograms.png` y `embedding_metric_histograms.png` (Distribuciones empíricas detalladas).
  - **Tablas de Resumen:**
    - `metric_distribution_summary.csv`
    - `metric_distribution_summary.md`

## Integración en la Tesis (TFM)
Este paso constituye el **filtro de robustez estadística** de la tesis. En análisis multivariantes y de *clustering*, inyectar variables colineales, constantes o vacías distorsiona el cálculo de distancias (como la euclidiana de Ward) y degrada la calidad de los clústeres. Al reportar de manera explícita y transparente esta fase de diagnóstico distributivo, el TFM justifica matemáticamente la simplificación de las métricas (Script 16), validando la solidez de las tipologías de subcampos posteriores y eliminando el ruido estadístico antes de la partición de clústeres.
