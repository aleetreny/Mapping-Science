# Índice general preliminar del TFM

## Título provisional

**Measuring the Shape of Science: Morphological Indicators and Evolution of Research Fields**

---

## 1. Introduction

Presentación del problema y motivación.

La idea central es estudiar los campos científicos no solo por volumen de publicaciones o citas, sino por la **forma de su espacio semántico**.

Preguntas principales:

- ¿Pueden compararse disciplinas científicas mediante métricas morfológicas del espacio de embeddings?
- ¿Qué diferencias estructurales existen entre subfields, fields y domains?
- ¿Cómo evolucionan esas estructuras semánticas a lo largo del tiempo?
- ¿Qué disciplinas convergen o divergen morfológicamente?

---

## 2. Background and Related Work

Revisión breve de literatura relevante.

Áreas principales:

- science mapping y bibliometría;
- representación semántica de papers científicos;
- embeddings científicos, especialmente SPECTER/SPECTER2;
- análisis de espacios semánticos;
- reducción dimensional como herramienta visual;
- estudios de evolución y convergencia científica.

---

## 3. Data and Corpus Construction

Descripción del corpus usado.

Elementos principales:

- fuente: OpenAlex;
- unidad principal: subfields;
- agregaciones: fields y domains;
- periodo: 2000–2024;
- texto usado: título + abstract;
- filtros de calidad;
- diseño de muestreo;
- almacenamiento en Parquet / DuckDB;
- validación del corpus.

---

## 4. Semantic Representation

Construcción del espacio semántico.

Elementos principales:

- generación de embeddings SPECTER2;
- matriz de embeddings;
- alineación entre papers, metadata y filas de la matriz;
- justificación de usar el espacio original del embedding como geometría principal;
- UMAP solo como herramienta auxiliar de visualización.

---

## 5. Embedding-Space Morphological Metrics

Definición del núcleo metodológico del TFM.

El análisis principal usa el **reduced interpretable embedding core** de 11 métricas.

### 5.1 Global Semantic Dispersion

- `embedding_distance_to_centroid_median`
- `embedding_distance_to_centroid_iqr`
- `embedding_distance_to_centroid_p90`

### 5.2 Local Semantic Density and Hubness

- `embedding_knn_median_distance`
- `embedding_knn_distance_cv`
- `embedding_knn_indegree_gini`

### 5.3 Intrinsic Dimensionality

- `embedding_pca_dim_80`
- `embedding_pca_spectral_entropy`

### 5.4 Temporal Semantic Movement

- `embedding_centroid_drift_early_late`
- `embedding_radial_expansion_slope`
- `embedding_recent_novelty_score`

---

## 6. Static Comparison of Scientific Disciplines

Comparación de disciplinas usando las 11 métricas del embedding-space.

Niveles de análisis:

- subfield;
- field;
- domain.

Preguntas principales:

- qué disciplinas son más compactas;
- cuáles son más dispersas;
- cuáles tienen mayor dimensionalidad;
- cuáles presentan más hubness;
- qué disciplinas tienen perfiles morfológicos similares;
- qué diferencias estructurales aparecen entre domains.

---

## 7. Temporal Evolution of Scientific Morphology

Análisis de evolución temporal mediante ventanas quinquenales.

Ventanas principales:

- 2000–2004;
- 2005–2009;
- 2010–2014;
- 2015–2019;
- 2020–2024.

Objetivo:

- estudiar qué disciplinas se expanden;
- cuáles se compactan;
- cuáles se diversifican;
- cuáles se desplazan semánticamente;
- cuáles muestran mayor novedad reciente;
- cómo cambian los perfiles morfológicos por field y domain.

---

## 8. Morphological Similarity, Convergence and Divergence

Análisis de distancia morfológica entre disciplinas.

Preguntas principales:

- qué disciplinas son morfológicamente similares;
- cuáles son más diferentes;
- qué pares de disciplinas convergen con el tiempo;
- qué pares divergen;
- cómo cambian las distancias a nivel de subfield, field y domain.

Métricas posibles:

- distancia euclídea sobre métricas estandarizadas;
- distancia de correlación;
- cambios entre ventanas iniciales y finales.

---

## 9. Exploratory Typologies

Bloque opcional.

Clustering exploratorio usando solo las 11 métricas del embedding-space.

Objetivo:

- resumir perfiles morfológicos recurrentes;
- identificar tipos de disciplinas;
- encontrar ejemplos representativos.

No debe presentarse como descubrimiento de clases naturales, sino como herramienta descriptiva.

---

## 10. Discussion

Interpretación de los resultados.

Ideas centrales:

- la ciencia puede compararse por su estructura semántica interna;
- algunas disciplinas son compactas, otras dispersas o multidimensionales;
- la evolución temporal revela expansión, especialización, desplazamiento o novedad;
- la convergencia/divergencia permite estudiar relaciones dinámicas entre áreas científicas;
- UMAP ayuda a visualizar, pero la base cuantitativa está en el embedding original.

---

## 11. Limitations

Limitaciones principales:

- dependencia de OpenAlex y su taxonomía;
- posibles errores de clasificación;
- sesgos de cobertura en títulos y abstracts;
- limitaciones del modelo SPECTER2;
- pérdida de información por muestreo;
- métricas descriptivas, no causales;
- visualizaciones UMAP no equivalen a geometría real;
- ventanas temporales afectadas por cobertura desigual en años antiguos.

---

## 12. Conclusion

Resumen de contribuciones.

Contribuciones principales:

- construcción de un pipeline reproducible para estudiar la morfología semántica de disciplinas;
- propuesta de un núcleo interpretable de métricas del embedding-space;
- comparación estática entre regiones científicas;
- análisis temporal de evolución morfológica;
- estudio de convergencia y divergencia entre disciplinas.

---

## Estructura mínima si hay que recortar

1. Introduction  
2. Data and Semantic Representation  
3. Embedding-Space Morphological Metrics  
4. Static Comparison of Disciplines  
5. Temporal Evolution  
6. Convergence and Divergence  
7. Discussion and Conclusion  
