# Script 13: Comparación de Familias de Métricas (Proyección 2D vs Latente 768D)

## Objetivo del Script
El objetivo central de `13_compare_metric_families.py` es ejecutar el análisis comparativo sistemático y cruzado entre las dos grandes familias de métricas del TFM: las métricas morfológicas proyectadas en UMAP 2D (Script 11) y las métricas estructurales nativas calculadas directamente en el espacio de embedding de alta dimensión SPECTER2 768D (Script 12). El script une deterministicamente ambas tablas por subcampo científico y calcula matrices de correlación (Spearman y Pearson) para evaluar de forma cuantitativa la pérdida de información y distorsión topológica introducida por la reducción de dimensionalidad no lineal.

## Hallazgos Científicos Críticos del Contraste
El análisis comparativo de este script produjo **uno de los descubrimientos teóricos más importantes de la tesis**: la demostración de la **desacoplación topológica** entre la proyección visual 2D y el espacio latente original 768D.

### 1. Desacoplación de Pares Análogos Conceptuales
Al comparar métricas que miden teóricamente el mismo atributo en baja y alta dimensión, se observan correlaciones de Spearman sorprendentemente débiles:
- **Distancia al Vecino más Cercano (kNN Median):**
  - Par comparado: `knn_median_distance` (UMAP 2D) vs `embedding_knn_median_distance` (768D)
  - **Correlación de Spearman:** **$\rho = 0.098$** (cercana a cero)
  - *Implicación:* La densidad local o proximidad semántica estrecha que un investigador observa visualmente en un mapa UMAP 2D no tiene relación matemática real con la proximidad de los documentos en el espacio latente de alta dimensión. Los clusters visuales son, en gran parte, artefactos de la proyección.
- **Dispersión de la Cola y Extensión (Tail Index):**
  - Par comparado: `radial_tail_index` vs `embedding_tail_index_p90_median`
  - **Correlación de Spearman:** **$\rho = 0.122$**
  - *Implicación:* El comportamiento extremo de los trabajos fronterizos en la disciplina sufre una severa reconfiguración durante la compresión a dos dimensiones.
- **Árbol de Expansión Mínima vs Conectividad del Grafo:**
  - Par comparado: `mst_gap_index` vs `embedding_graph_edge_distance_p90`
  - **Correlación de Spearman:** **$\rho = 0.208$**

### 2. Preservación Moderada de Dinámicas Globales y Elongaciones
Ciertas dinámicas macro y formas globales sí logran cruzar la barrera de la reducción de dimensiones con una fidelidad moderada-alta:
- **Linealidad y Elongación:**
  - Par comparado: `anisotropy_ratio` vs `embedding_pca_first_component_share`
  - **Correlación de Spearman:** **$\rho = 0.650$**
  - *Implicación:* Las disciplinas que poseen una estructura conceptual lineal dominante en 768D se proyectan de forma consistente como elipses alargadas y orientadas en el mapa UMAP local 2D.
- **Relación de Direccionalidad Temporal:**
  - Par comparado: `directionality_ratio` vs `embedding_directionality_ratio`
  - **Correlación de Spearman:** **$\rho = 0.586$**
- **Deriva del Centroide (Drift):**
  - Par comparado: `centroid_drift_early_late` vs `embedding_centroid_drift_early_late`
  - **Correlación de Spearman:** **$\rho = 0.513$**

### 3. Relaciones Cruzadas Más Fuertes
Las relaciones más intensas entre familias ocurren entre la fragmentación espacial UMAP y la dimensionalidad lineal del embedding:
- `dense_component_count` vs `embedding_pca_top3_variance_share`: **$\rho = -0.666$**
- `density_peak_count` vs `embedding_pca_top3_variance_share`: **$\rho = -0.659$**
- `effective_area_90` vs `embedding_pca_top3_variance_share`: **$\rho = -0.655$**
- *Interpretación:* A menor varianza explicada por los componentes principales del embedding (alta complejidad tridimensional intrínseca del campo), el mapa proyectado UMAP tiende a fragmentarse en un mayor número de islas densas desconectadas y con un área efectiva más amplia.

## Parámetros de Entrada y Salida
- **Configuración y Entrada:**
  - Métricas UMAP: `data/processed/subfield_morphology_metrics.parquet`.
  - Métricas del Embedding: `data/processed/subfield_embedding_space_metrics.parquet`.
- **Rutas de Salida de Datos (Carpeta `outputs/analysis/metric_family_comparison/`):**
  - **Gráfico Crítico:**
    - `cross_family_spearman_heatmap.png` (Mapa de calor completo de correlaciones cruzadas).
  - **Tablas de Correlación:**
    - `cross_family_spearman_correlations.csv`
    - `cross_family_pearson_correlations.csv`
    - `top_absolute_spearman_correlations.csv`
    - `analogue_metric_pair_correlations.csv` (Correlaciones exclusivas de los pares conceptuales análogos).
  - **Reportes de Resumen:**
    - `metric_family_comparison_summary.json`
    - `metric_family_comparison_summary.md`

## Integración en la Tesis (TFM)
Este análisis empírico fundamenta la **justificación metodológica dual** de la tesis. En la literatura de minería de textos y cienciometría, es extremadamente común tratar las proyecciones visuales (t-SNE o UMAP) como descripciones fieles de la estructura conceptual de la ciencia. El TFM desafía formalmente esta convención y demuestra mediante números robustos (como la correlación $\rho=0.098$ en kNN) que **la visualización 2D y el análisis métrico estructural en alta dimensión no son sustitutos, sino complementos**. Las representaciones visuales son excelentes para mapas de navegación humana, pero las auditorías de políticas científicas o clasificaciones sistemáticas deben realizarse obligatoriamente sobre el espacio nativo de alta dimensión (768D) calculado por el Script 12.
