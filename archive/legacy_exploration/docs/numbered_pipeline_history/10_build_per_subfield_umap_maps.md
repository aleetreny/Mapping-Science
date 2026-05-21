# Script 10: Cartografía Local e Introducción a la Densidad Espacial (KDE) por Subcampo

## Objetivo del Script
El objetivo principal de `10_build_per_subfield_umap_maps.py` es descender al nivel micro y mapear individualmente el paisaje semántico interno de **cada uno de los 241 subcampos científicos**. Ajustando un modelo UMAP *independiente y local* para cada subcampo y modelando la concentración de papers mediante estimadores de densidad de kernel (KDE), el *pipeline* logra fotografiar y aislar la morfología interna (cohesión, picos de especialización, dispersión) de cada disciplina de forma independiente.

## Metodología y Algoritmos
El script procesa recursivamente y en paralelo los subcampos aplicando:
1. **Extracción y Aislamiento de Embeddings Locales:** Filtra la matriz principal de embeddings para retener únicamente los trabajos (típicamente hasta 10.000 papers) del subcampo bajo estudio.
2. **Ajuste UMAP Local:** Ajusta un modelo UMAP dedicado sobre estos vectores 768D locales. Al aislar la disciplina, UMAP explota las variaciones semánticas más sutiles y específicas del campo (ej. separando subtemas específicos en Inteligencia Artificial como Visión Artificial vs. Procesamiento de Lenguaje Natural).
   - Parámetros locales por defecto: `n_neighbors=15`, `min_dist=0.05`, `metric='cosine'`.
3. **Cálculo de Densidad Espacial Continua (Kernel Density Estimation - KDE):**
   - Proyecta una cuadrícula de interpolación (grid contiguo de **100x100 celdas**).
   - Ajusta un estimador Gaussiano KDE bidimensional sobre las coordenadas locales $x,y$, traduciendo los puntos dispersos en una superficie topográfica de densidad de masa semántica.
   - Aplica normalización por percentiles (`vmax_percentile=0.99`) para suavizar picos espurios y uniformizar el contraste visual de los mapas.
4. **Generación de Paneles e Informes:** Produce una visualización combinada (típtica) que muestra:
   - Panel de Dispersión (*Scatter*): Los papers coloreados según citas o año de publicación.
   - Panel de Paisaje de Densidad (*Density Heatmap*): Un mapa térmico continuo que resalta los "hubs semánticos" o núcleos temáticos densamente ocupados por la literatura.
5. **Persistencia e Indexación en Manifiesto:** Guarda las coordenadas locales en un parquet individual por subcampo y añade un registro detallado en el manifiesto unificado de mapas.

## Parámetros de Entrada y Salida
- **Configuración y Entrada:**
  - Archivos unificados del paso 08 (`analysis_embedding_index.parquet` y `main_embeddings.float16.npy`).
- **Rutas de Salida de Datos (bajo `outputs/maps/per_subfield_umap/`):**
  - **Manifiesto Colectivo:**
    - `per_subfield_umap_manifest.parquet`
  - **Archivos de Coordenadas Locales (un Parquet por subcampo):**
    - `subfield_<subfield_id>_coords.parquet` (Contiene IDs de papers, coordenadas $x,y$ locales y pesos de densidad KDE).
  - **Gráficos Tripticos y de Densidad:**
    - `plots/subfield_<subfield_id>_panels.png`

## Integración en la Tesis (TFM)
Este paso constituye la **piedra angular del análisis morfológico de la ciencia**. En lugar de asumir que todos los subcampos tienen la misma estructura uniforme, la generación de mapas KDE independientes aisla la "huella digital semántica" de cada disciplina. Permite visualizar de forma directa si un subcampo es **monocéntrico** (un único pico de densidad masivo, muy centrado en un paradigma unificado, común en Física de Altas Energías), **policéntrico** (varios picos separados por valles de baja densidad, indicando múltiples subespecialidades, común en Ciencias de la Computación) o **difuso/periférico** (sin picos marcados y distribución muy dispersa, común en Humanidades). Estas características cualitativas se cuantifican de forma rigurosa en el Script 11.
