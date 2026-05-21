# Script 16 (Tablas): Perfilado e Interpretación de los 5 Clústeres Disciplinarios

## Objetivo del Script
El objetivo principal de `16_build_cluster_interpretation_tables.py` es consolidar, perfilar y exportar los resultados cuantitativos del clustering morfológico multivariante calculado por el Script 15 en el espacio combinado (Block-PCA). Este script traduce las coordenadas abstractas y las asignaciones matemáticas a tablas legibles para el texto principal de la tesis doctoral, calculando distribuciones de frecuencia de dominios OpenAlex por clúster, extrayendo los subcampos representativos reales (los más cercanos al centroide morfológico) y generando las figuras diagnósticas clave de perfilación.

## Perfil Detallado de los 5 Clústeres Morfológicos Descubiertos
La tipología resultante agrupa a las disciplinas científicas no por su temática lingüística, sino por su **morfología del conocimiento e internalización de flujos epistemológicos**:

### Clúster 1: Nubes Morfológicas Dispersas pero Cohesivas (Sociales y Humanidades)
- **Tamaño:** 55 subcampos ($22.8\%$ del corpus).
- **Composición Dominante:** Ciencias Sociales ($47\%$), Ciencias Físicas ($25\%$), Ciencias de la Vida ($18\%$).
- **Campos Principales:** Ciencias Sociales ($22\%$), Arte y Humanidades ($16\%$), Ciencias Agrícolas y Biológicas ($9\%$).
- **Subcampos Representativos:** *Visual Arts and Performing Arts*, *Language and Linguistics*, *Aquatic Science*, *Linguistics and Language*, *Cultural Studies*.
- **Métricas Extremas Estandarizadas:** `knn_median_distance` ($+1.11$), `support_solidity` ($+1.11$), `effective_area_90` ($+1.08$), `density_peak_count` ($+1.07$).
- **Racional Cienciométrico:** Disciplinas muy extendidas en área geográfica visual UMAP pero altamente sólidas y sin fracturas internas. Poseen múltiples subcampos y picos conceptuales bien definidos, con una dimensionalidad lineal intrínseca sumamente baja en alta dimensión (`top5_variance_share` de $-0.87$).

### Clúster 2: Perfiles Tecnológicos Ultra-Anisotrópicos (Tecnologías Clínicas y Aplicadas)
- **Tamaño:** 21 subcampos ($8.7\%$).
- **Composición Dominante:** Ciencias de la Salud ($43\%$), Ciencias Físicas ($33\%$), Ciencias de la Vida ($19\%$).
- **Campos Principales:** Medicina ($29\%$), Genética y Biología Molecular ($14\%$), Profesiones de la Salud ($14\%$), Ingeniería Química ($10\%$).
- **Subcampos Representativos:** *Rehabilitation*, *Physical Therapy, Sports Therapy and Rehabilitation*, *Urology*, *Radiological and Ultrasound Technology*.
- **Métricas Extremas Estandarizadas:** `embedding_pca_first_component_share` ($+2.08$), `annual_centroid_step_cv` ($+2.00$), `anisotropy_ratio` ($+1.69$).
- **Racional Cienciométrico:** Disciplinas de carácter extremadamente "alargado" o anisotropía extrema. El conocimiento fluye linealmente a lo largo de un eje unidireccional dominante (alta varianza en la primera componente lineal). Presentan baja entropía de densidad y pocos picos alternativos, comportándose como tecnologías secuenciales lineales con un alto dinamismo y pasos anuales rápidos.

### Clúster 3: Campos Semánticos Anchurosos y Difusos (Ciencias Ambientales e Interdisciplinarias)
- **Tamaño:** 75 subcampos ($31.1\%$, el más numeroso).
- **Composición Dominante:** Ciencias Físicas ($51\%$), Ciencias de la Salud ($23\%$), Ciencias de la Vida ($13\%$).
- **Campos Principales:** Medicina ($20\%$), Ingeniería ($11\%$), Ciencias Ambientales ($11\%$), Química ($7\%$).
- **Subcampos Representativos:** *Nutrition and Dietetics*, *Social Psychology*, *Biotechnology*, *Pharmacology*, *Environmental Chemistry*.
- **Métricas Extremas Estandarizadas:** `embedding_knn_median_distance` ($+0.81$), `embedding_knn_p90_distance` ($+0.81$).
- **Racional Cienciométrico:** Representa nubes de conocimiento difusas con grandes distancias locales del coseno entre documentos constituyentes. Su cohesión al centroide es sumamente débil (`centroid_norm` estandarizado de $-0.78$), y poseen una bajísima polarización núcleo-periferia, reflejando campos integradores y multidisciplinarios con fronteras muy porosas.

### Clúster 4: Hubs Clínicos Ultra-Cohesivos (Especialidades Médicas Avanzadas)
- **Tamaño:** 31 subcampos ($12.9\%$).
- **Composición Dominante:** Ciencias de la Salud ($65\%$), Ciencias de la Vida ($19\%$), Ciencias Sociales ($10\%$).
- **Campos Principales:** Medicina ($39\%$), Genética y Biología Molecular ($10\%$), Odontología ($10\%$), Enfermería ($10\%$).
- **Subcampos Representativos:** *Periodontics*, *Nephrology*, *Hepatology*, *Anesthesiology and Pain Medicine*, *Aging*.
- **Métricas Extremas Estandarizadas:** `embedding_centroid_norm` ($+1.31$), `embedding_knn_distance_cv` ($+1.07$), `embedding_knn_mean_distance` ($-1.48$).
- **Racional Cienciométrico:** Núcleos conceptuales ultra-densos y altamente enfocados. Sus distancias internas del coseno a vecinos y al centroide son extremadamente bajas (las menores de todo el TFM). Muestran una alta irregularidad de densidad interna (`knn_distance_cv` alto), revelando núcleos con una concentración de masa bibliográfica espectacular orientados a temas clínicos específicos.

### Clúster 5: Paisajes Teóricos Uniformes y Regulares (Matemáticas, Física e Ingenierías)
- **Tamaño:** 59 subcampos ($24.5\%$).
- **Composición Dominante:** Ciencias Físicas ($42\%$), Ciencias Sociales ($25\%$), Ciencias de la Vida ($17\%$).
- **Campos Principales:** Ingeniería ($10\%$), Negocios y Contabilidad ($8\%$), Ciencias de la Computación ($8\%$), Física y Astronomía ($8\%$).
- **Subcampos Representativos:** *Statistics, Probability and Uncertainty*, *Safety, Risk, Reliability and Quality*, *Pharmacy*, *Organizational Behavior and Human Resource Management*.
- **Métricas Extremas Estandarizadas:** `embedding_pca_top5_variance_share` ($+0.66$), `largest_component_mass_share` ($+0.52$), `dense_component_count` ($-0.72$), `density_peak_count` ($-0.71$).
- **Racional Cienciométrico:** Disciplinas conceptualmente compactas pero uniformes. Poseen un área efectiva muy comprimida y pocos componentes discretos o picos morfológicos, lo que indica paisajes conceptuales planos y consolidados con flujos metodológicos homogéneos y estables.

## Parámetros de Entrada y Salida
- **Configuración y Entrada:**
  - Carpeta de agrupamiento multivariante: `outputs/analysis/metric_clustering/`.
  - Parámetros por defecto: Espacio principal `--main-space combined`.
- **Rutas de Salida de Datos (Carpeta `outputs/analysis/cluster_interpretation/`):**
  - **Tablas de la Tesis:**
    - `subfield_cluster_master_table.parquet` y `csv` (La tabla maestra que asocia cada subcampo con sus coordenadas morfológicas UMAP y PCA, sus métricas crudas y sus asignaciones a clústeres).
    - `combined_cluster_domain_composition.csv` y `combined_cluster_field_composition.csv`.
    - `combined_cluster_size_summary.csv`.
    - `combined_cluster_representatives_readable.csv` (El listado definitivo de subcampos representativos de cada clúster).
    - `combined_cluster_metric_profile_long.csv` (Métricas estandarizadas para graficar perfiles).
  - **Diagnósticos y Reportes Académicos:**
    - `combined_cluster_interpretation_summary.md` (La descripción cualitativa y cuantitativa en español de los clústeres).
    - `cluster_family_agreement_summary.md` y `csv`.
  - **Gráficos de Alta Resolución:**
    - `combined_cluster_domain_stacked_bar.png` (Composición porcentual apilada de los Dominios OpenAlex por clúster).
    - `combined_metric_space_umap_clusters.png` (Mapa de los subcampos proyectado en el espacio métrico UMAP coloreado por clúster).
    - `combined_cluster_metric_profile_heatmap.png` (Mapa de calor de las métricas morfológicas clave para cada perfil).
    - `combined_metric_space_pca_diagnostic.png` (Gráfico de dispersión de las dos componentes PCA principales coloreado por clúster).

## Integración en la Tesis (TFM)
Esta sección constituye el **corazón del análisis cienciométrico descriptivo** del TFM. Al perfilar detalladamente estos 5 clústeres morfológicos con métricas del mundo real, el alumno puede escribir los capítulos de resultados demostrando que la geografía de la ciencia posee "tipos de paisajes" bien delimitados epistemológicamente. Esto permite contrastar tesis clásicas de la sociología de la ciencia (como la división de Whitley o Biglan de ciencias "duras vs blandas", "puras vs aplicadas") con una taxonomía empírica rigurosa de 241 subcampos, superando de forma definitiva las categorizaciones cualitativas tradicionales.
