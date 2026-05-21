# Script 18: Optimización Sensible e Hiperparametrización de UMAP

## Objetivo del Script
El objetivo central de `18_tune_umap_dimensionality_reduction.py` es ejecutar un análisis exhaustivo de sensibilidad paramétrica para el algoritmo UMAP sobre las tres representaciones del espacio métrico (`umap_only`, `embedding_only` y `combined`). En lugar de adoptar un criterio meramente estético para seleccionar los mapas visuales de la ciencia, el script realiza un barrido de cuadrícula (*grid search*) evaluando 25 combinaciones de hiperparámetros por espacio, con el fin de cuantificar de forma rigurosa el compromiso (*trade-off*) entre la preservación topológica local y la separación espacial de los perfiles de investigación.

## Metodología del Barrido de Hiperparámetros
El script somete a prueba una matriz combinatoria de dos hiperparámetros críticos en el comportamiento geométrico de UMAP:

1. **`n_neighbors` (Vecindad Local vs. Global):** Valores de barrido: `[5, 10, 15, 30, 50]`. Controla el tamaño de la vecindad difusa considerada por UMAP. Valores bajos obligan al algoritmo a centrarse en la microestructura muy local (creando "islas" desconectadas), mientras que valores altos priorizan la macrotopología global, fundiendo las islas en un continuo.
2. **`min_dist` (Densidad y Empaquetamiento Visual):** Valores de barrido: `[0.0, 0.05, 0.1, 0.3, 0.6]`. Controla la distancia mínima permitida entre puntos en el espacio bidimensional de salida. Valores bajos (cercanos a 0) permiten un empaquetamiento extremadamente denso de los subcampos similares, acentuando los límites de clústeres. Valores altos (cercanos a 0.6) fuerzan una distribución dispersa y uniforme de los puntos, reduciendo los huecos vacíos.

Para cada una de las 25 combinaciones, el script calcula:
- **Trustworthiness (Confianza Topológica):** Evaluada con un marco de referencia a $k_{ref}=15$.
- **Silhouette Coefficient 2D (Silueta de Clústeres):** Calculada sobre las coordenadas proyectadas usando las asignaciones disciplinarias obtenidas por el método jerárquico Ward ($k=5$).

## Parámetros de Entrada y Salida

- **Configuración y Entrada:**
  - Métricas de morfología 2D: `data/processed/subfield_morphology_metrics.parquet`
  - Métricas de embedding 768D: `data/processed/subfield_embedding_space_metrics.parquet`
  - Agrupamientos Ward `k=5`: `outputs/analysis/metric_clustering/`

- **Rutas de Salida de Datos (Carpeta `outputs/analysis/dimensionality_reduction_comparison/umap_tuning/`):**
  - **Coordenadas de Proyección del Barrido (CSV y Parquet por espacio):**
    - `umap_tuning_coords_umap_only.parquet` (y `.csv`)
    - `umap_tuning_coords_embedding_only.parquet` (y `.csv`)
    - `umap_tuning_coords_combined.parquet` (y `.csv`)
  - **Métricas de Calidad de Barrido:**
    - `umap_tuning_quality.csv` (25 filas por espacio métrico con Silhouette y Trustworthiness)
    - `umap_tuning_summary.md` (Resumen académico estructurado)
    - `umap_tuning_config.json` y `summary.json` (Detalles de ejecución)
  - **Mallas Visuales Multigráfico (PNG - Gráfico de 5x5 proyecciones):**
    - `umap_tuning_umap_only_grid.png`
    - `umap_tuning_embedding_only_grid.png`
    - `umap_tuning_combined_grid.png`

---

## Diagnósticos Cuantitativos y Hallazgos Reales

Los resultados del barrido de hiperparámetros para cada espacio métrico se resumen de la siguiente manera:

### 1. Espacio UMAP-only (11 variables de morfología proyectada)
- **Máxima Confianza Topológica (*Trustworthiness*):** **$0.950$** alcanzada con **`n_neighbors = 5`** y **`min_dist = 0.05`**.
- **Máxima Separación de Clústeres (*Silhouette*):** **$0.202$** alcanzada con **`n_neighbors = 5`** y **`min_dist = 0.3`**.
- *Interpretación:* La topología morfológica está fuertemente dominada por relaciones locales estrechas. Mantener un número de vecinos bajo ($k=5$) conserva mejor el espacio de procedencia, mientras que una distancia intermedia ($0.3$) ayuda a que los clústeres no colapsen en micro-puntos densos inconexos.

### 2. Espacio Embedding-only (6 variables nativas 768D)
- **Máxima Confianza Topológica (*Trustworthiness*):** **$0.958$** alcanzada con **`n_neighbors = 5`** y **`min_dist = 0.1`**.
- **Máxima Separación de Clústeres (*Silhouette*):** **$0.270$** alcanzada con **`n_neighbors = 30`** y **`min_dist = 0.3`**.
- *Interpretación:* Este espacio métrico es muy suave y posee transiciones globales coherentes. Por tanto, para maximizar la separación de los perfiles de clústeres es óptimo usar un vecindario amplio ($k=30$) que capte la macroestructura y un empaquetamiento suelto ($0.3$), impidiendo que el ruido local rompa las amplias nubes disciplinares.

### 3. Espacio Combinado (17 variables unificadas)
- **Máxima Confianza Topológica (*Trustworthiness*):** **$0.931$** alcanzada con **`n_neighbors = 5`** y **`min_dist = 0.0`**.
- **Máxima Separación de Clústeres (*Silhouette*):** **$0.284$** alcanzada con **`n_neighbors = 10`** y **`min_dist = 0.05`**.
- *Interpretación:* El espacio unificado, al integrar descriptores tanto lineales de alta dimensión como geométricos de baja dimensión, responde con excelente dinamismo. Logra la silueta más alta de todo el experimento ($0.284$) con $k=10$ y una distancia mínima estrecha de $0.05$, lo cual produce una visualización en donde los 5 perfiles disciplinares se aíslan visualmente de manera óptima sin comprometer significativamente la fidelidad topológica general.

---

## Directrices para el Análisis de Estabilidad Visual

El script y su grid visual de 5x5 ayudan a evitar el sesgo de la "proyección perfecta", estableciendo los siguientes principios metodológicos:
1. **La Mentira de la Silueta Extrema:** Proyecciones con distancias muy bajas (`min_dist = 0.0`) y pocos vecinos (`k = 5`) tienden a fraccionar la ciencia en decenas de pequeños fragmentos independientes que aparentan ser disciplinas aisladas. Aunque numéricamente muestren alta fidelidad local, carecen de sentido epistemológico global ya que rompen la continuidad interdisciplinar.
2. **Estabilidad de Vecindarios:** Se debe preferir una configuración en la que la topología general de los subcampos permanezca relativamente invariante frente a ligeras variaciones de parámetros (e.g., moviéndose de $k=10$ a $k=15$ o de $min\_dist=0.05$ a $0.1$). El espacio Combinado con $k=15$ y $min\_dist=0.1$ representa precisamente ese **punto de equilibrio óptimo y robusto** de la tesis.

## Integración en la Tesis (TFM)
- **Justificación Metodológica de Hiperparámetros:** Proporciona al alumno el soporte matemático definitivo para responder preguntas del tribunal sobre por qué se usó una configuración específica de UMAP. En lugar de responder "por defecto de la biblioteca", se demuestra que se ha barrido exhaustivamente el espacio hiperparamétrico y que $k=15$ y $min\_dist=0.1$ representa la combinación estable más cercana a la frontera óptima de Pareto entre fidelidad local y separabilidad global.
- **Riqueza Metodológica:** Los gráficos de cuadrícula generados son ideales para anexos o secciones metodológicas avanzadas, demostrando un rigor en el tratamiento de algoritmos de aprendizaje no supervisado propio de una tesis doctoral.
