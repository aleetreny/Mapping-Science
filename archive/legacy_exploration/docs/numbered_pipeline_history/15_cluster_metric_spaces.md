# Script 15: Agrupamiento Multivariante de Espacios Métricos (Clustering)

## Objetivo del Script
El objetivo principal de `15_cluster_metric_spaces.py` es identificar tipologías morfológicas y perfiles de campos científicos mediante el agrupamiento multivariante (*clustering*) de los 241 subcampos en función de sus características morfológicas y estructurales. El script evalúa de forma paralela tres espacios métricos distintos: el espacio morfológico proyectado UMAP 2D, el espacio estructural de alta dimensión (Embedding 768D), y un espacio combinado. Utiliza algoritmos jerárquicos y de partición (Ward, KMeans, KMedoides y Mezclas Gaussianas) para particionar las disciplinas en perfiles morfológicos homogéneos y auditar la concordancia entre los diferentes espacios.

## Metodología y Algoritmos de Agrupamiento
El pipeline estadístico de clustering sigue un flujo riguroso para evitar sesgos:
1. **Preprocesamiento y Escalado:** Las métricas seleccionadas de cada espacio se imputan con la mediana (en caso de valores atípicos o nulos) y se estandarizan utilizando un escalador por defecto (`standard` o `robust`).
2. **Estrategia Combinada (Block-PCA):** Al construir el espacio combinado (proyección 2D + alta dimensión), inyectar directamente las métricas de ambas familias sesgaría el clustering hacia la familia con mayor número de columnas. Para evitar esto, el script aplica una estrategia de **Block-PCA**: realiza PCA por separado en cada bloque (morfológico UMAP y estructural embedding), retiene componentes que explican el $90\%$ de la varianza en cada lado y luego une los componentes resultantes ponderándolos equitativamente.
3. **Clustering Jerárquico de Ward:** Método principal aplicado sobre las coordenadas PCA. Construye un árbol jerárquico que minimiza la varianza interna de cada grupo al fusionarse.
4. **Validación Cruzada de Particiones:** Ejecuta particiones alternativas mediante KMeans y Gaussian Mixture Models (GMM) para sweeps de $k$ (desde $k=3$ hasta $k=8$, con un valor por defecto de $k=5$) y calcula métricas diagnósticas de calidad de clústeres.

## Parámetros de Entrada y Salida
- **Configuración y Entrada:**
  - Métricas UMAP: `data/processed/subfield_morphology_metrics.parquet`.
  - Métricas del Embedding: `data/processed/subfield_embedding_space_metrics.parquet`.
  - Archivo de métricas de baja información: `outputs/analysis/metric_distributions/low_information_metrics.csv`.
  - Parámetros por defecto: Ponderación Block-PCA, varianza PCA $\ge 90\%$, sweep $k \in [3, 8]$, clúster final de Ward con $k=5$.
- **Rutas de Salida de Datos (Carpeta `outputs/analysis/metric_clustering/`):**
  - **Coordenadas de Espacio UMAP de Métricas:**
    - `{space_name}_metric_umap_coordinates.csv`
    - `{space_name}_metric_umap_coordinates.parquet`
  - **Subcarpetas por Espacio (`combined/`, `embedding_only/`, `umap_only/`):**
    - `cluster_assignments.csv` y `parquet` (Asignaciones de los subcampos a clústeres para Ward, KMeans y GMM).
    - `preprocessing_report.csv` (Registro de variables estandarizadas).
    - `pca_scores.csv`, `pca_loadings.csv` y `pca_explained_variance.csv`.
    - `cluster_quality_by_k.csv` (Métricas de silueta y calidad de partición).
    - `cluster_profiles.csv` (Valores promedio de las métricas robust-scaled por clúster).
    - `cluster_representatives.csv` (Subcampos más cercanos al centroide de cada clúster).
  - **Diagnósticos Gráficos:**
    - `dendrogram.png` (Dendrograma de enlace Ward jerárquico).
    - `pca_scatter_clusters.png` (Dispersión bidimensional de las dos componentes PCA coloreada por clúster).
    - `metric_space_umap_clusters.png` (Visualización local de vecindades morfológicas en 2D UMAP).
    - `cluster_profile_heatmap.png` (Mapa de calor del perfil morfológico de métricas promedio).

## Resultados Reales de Consistencia
La comparación sistemática de las particiones (Script 16) demostró que **los mapas proyectados y la estructura del embedding agrupan la ciencia en clústeres muy distintos**:
- **Concordancia Moderada-Baja:** El índice Rand ajustado (ARI) entre la partición puramente morfológica visual (`umap_only`) y la partición tridimensional (`embedding_only`) es significativamente bajo.
- **Razón Metodológica:** La proyección UMAP 2D distorsiona las distancias locales (correlación de kNN de $0.098$, Script 13), fragmentando campos continuos en islas visuales ficticias. Por ello, el agrupamiento jerárquico en `umap_only` tiende a basarse en discontinuidades visuales artificiales, mientras que `embedding_only` se agrupa según la cohesión real y dimensionalidad de los embeddings 768D.
- **Solución Adoptada:** La tipología principal adoptada por la tesis utiliza el espacio **`combined`** estructurado mediante Block-PCA, el cual reconcilia ambas geografías sin permitir que una opaque a la otra.

## Integración en la Tesis (TFM)
Este script representa la **herramienta de clasificación cienciométrica objetiva** de la tesis. Supera la clasificación clásica manual de revistas de bases de datos como Scopus o Web of Science (a menudo subjetiva y comercial), proponiendo en su lugar una taxonomía cienciométrica de base empírica y matemática. Al agrupar los 241 subcampos según su morfología interna y no según sus tópicos, el TFM devela que disciplinas lejanas temáticamente (ej. Astronomía e Inmunología) pueden poseer perfiles morfológicos y de flujo de conocimiento idénticos en su evolución epistemológica.
