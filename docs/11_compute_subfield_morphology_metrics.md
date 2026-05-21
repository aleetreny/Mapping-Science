# Script 11: Extracción de Métricas de Morfología Proyectada UMAP 2D

## Objetivo del Script
El objetivo principal de `11_compute_subfield_morphology_metrics.py` es cuantificar de forma rigurosa y matemática la "forma" visual o relieve geográfico de cada paisaje local UMAP 2D calculado en el paso 10. Para cada uno de los subcampos completados en el manifiesto, este script extrae una batería de **25 descriptores morfológicos proyectados**, traduciendo las visualizaciones cualitativas de densidad KDE a un panel estructurado de variables cuantitativas continuas.

## Bloques Teóricos de Métricas Proyectadas 2D
El motor de análisis morfológico de este script se divide en varios bloques estructurales clave:
1. **Métricas de Concentración y Hubness Semántico:**
   - `density_peak_count`: Cuenta el número de máximos locales (picos) en la superficie de densidad KDE. Representa cuántas subespecialidades densas conviven en la disciplina.
   - `dense_component_count` y `largest_component_mass_share`: Cuantifican cuántos "bloques" discretos superan un umbral crítico de densidad y qué fracción de la masa bibliográfica total de la disciplina capturan.
   - `peak_mass_entropy`: Mide la dispersión de la masa entre los picos de densidad. Una entropía baja indica que un único subtema domina de forma absoluta.
2. **Métricas de Dispersión y Extensión Espacial:**
   - `effective_area_90`: El área geográfica mínima de la cuadrícula que encierra el 90% de la densidad acumulada.
   - `support_solidity`: Relación entre el área ocupada y el área del casco convexo (*convex hull*) que encierra los puntos UMAP. Cuantifica qué tan "compacta" o "porosa" es la geografía del campo.
3. **Métricas de Forma y Anisotropía:**
   - `anisotropy_ratio`: Proporción de la varianza a lo largo de los dos ejes principales del mapa local (mediante PCA lineal de las coordenadas $x,y$ UMAP). Valores altos indican disciplinas alargadas o dominadas por una única tensión temática lineal.
   - `boundary_complexity`: La complejidad fractal del límite externo del casco convexo del campo.
4. **Métricas de Estructura de Red Proyectada (MST):**
   - `mst_gap_index`: Relación del árbol de expansión mínima (*Minimum Spanning Tree*) calculado sobre las coordenadas UMAP locales para evaluar vacíos o fracturas internas.

## Parámetros de Entrada y Salida
- **Configuración y Entrada:**
  - Manifiesto de mapas locales de UMAP: `outputs/maps/per_subfield_umap/per_subfield_umap_manifest.parquet`.
  - Archivos de coordenadas individuales por subcampo `subfield_<subfield_id>_coords.parquet`.
- **Rutas de Salida de Datos:**
  - **Tabla de Métricas Morfológicas Proyectadas:**
    - `data/processed/subfield_morphology_metrics.parquet`
  - **Base de Datos DuckDB (`warehouse/tfm_openalex.duckdb`):**
    - Tabla `subfield_morphology_metrics_<version>`

## Integración en la Tesis (TFM)
Este script representa la **reducción cuantitativa del relieve semántico visual**. En cienciometría tradicional, la interpretación de mapas UMAP o t-SNE suele ser puramente visual y anecdótica, expuesta a sesgos subjetivos de interpretación del autor. El TFM supera esta limitación metodológica clásica al convertir cada mapa visual 2D en una fila de 25 variables continuas comparables mediante algoritmos estadísticos multivariantes. Esto permite auditorías matemáticas estrictas sobre las tipologías morfológicas de la ciencia (Script 15) basadas en relieve e interpolaciones empíricas KDE, no en impresiones visuales de diagramas de dispersión.
