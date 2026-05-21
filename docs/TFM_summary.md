# Resumen del Proyecto: Morfología de Subcampos Científicos en OpenAlex

Este repositorio contiene un _pipeline_ de datos reproducible diseñado para medir, comparar y entender la **morfología y topología 2D y multidimensional de los subcampos científicos** usando OpenAlex. El enfoque es puramente estadístico, comparativo y descriptivo (no busca predecir, ni entrenar modelos de regresión, ni construir clasificadores).

La unidad de análisis elegida es el **subcampo científico de OpenAlex** (lo suficientemente amplio como para poseer vecindarios semánticos ricos, pero manejable computacionalmente). 

## 1. Diseño de Arquitectura y Datos
- **Fuente de Textos:** Corpus conformado por los Títulos y _Abstracts_ de artículos científicos en inglés (desde OpenAlex).
- **Extracción Temporal:** Se usa el set `2000_2024_400py`, una ventana que va del 2000 al 2024 extrayendo unas **400 publicaciones estratificadas al año por subcampo** válido. En total, unos 10.000 artículos por campo para modelado longitudinal y estático. 
- **Modelo de Embedding:** **SPECTER2** (un modelo derivado de BERT pre-entrenado en citas científicas) genera representaciones semánticas flotantes (Float16) de 768 dimensiones.
- **Almacenamiento:** Los metadatos de las obras se alojan en una base de datos **DuckDB** (`warehouse/tfm_openalex.duckdb`), y los datasets analíticos, extracciones y matrices se guardan en `.parquet` y `.npy`.

---

## 2. Pipeline de Flujo de Datos
El abanico de ejecución (los _scripts_) se estructura en fases muy claras:

1. **Extracción y Validación de Corpus (Scripts 01-06):** Interroga de forma segura y paginada a la API de OpenAlex, descarga la muestra estratificada de los textos mediante iteraciones, rellena faltantes (_backfills_), genera recuentos y tablas maestras para definir qué dominios poseen los +2500 textos necesarios para ser elegibles para el análisis principal.
2. **Construcción de Embeddings y Matrices (Scripts 07 y 08):** Se encarga de validar los archivos fragmentados (_shards_) de embedding construidos remotamente de SPECTER2 y crea una **matriz analítica alineada** deterministicamente.
3. **Proyecciones UMAP 2D (Scripts 09 y 10):** Construyen mapas exploratorios topológicos. Se ajusta un modelo UMAP 2D *independiente* por cada subcampo para entender su propio "paisaje" (aislando la morfología).
4. **Cálculo de Topología y Morfología Semántica (Scripts 11 y 12):** Calculan paralelamente las propiedades de la forma visual de esos mapas locales frente a las propiedades espaciales del embedding de 768 dimensiones.
5. **Diagnósticos Estadísticos y Sub-Selección Núcleo (Scripts 13-16b):** Saneamiento y limpieza de redundancias entre las métricas estudiadas.

---

## 3. Análisis Morfológicos y Estructurales
Para extraer características, el pipeline extrae dos familias complementarias de características; en lugar de mezclarlo ciegamente, explota ambas perspectivas:

* **Familia UMAP (Morfología Proyectada):** Conformada por 25 métricas. Computan propiedades geográficas del espacio proyectado en 2D de las incrustaciones de los papers (densidad de picos, masa de los componentes densos, entropía espacial y límites morfológicos).
* **Familia Embedding (Morfología Estructural Multidimensional):** Conformada por 26 métricas medidas directamente sobre el espacio normalizado (L2) originado por SPECTER2.

Para evitar altas covarianzas (redundancias) descubiertas a través de matrices de correlación Pearson y Spearman previas, **la familia del Embedding se reduce a métricas Core Interpretables de 11 variables** condensadas en 4 grandes pilares teóricos:
1. **Dispersión Semántica Global:** Mediana y varianza de las distancias directas al centroide (índices globales).
2. **Densidad Semántica Local y Hubness:** Mediana de redes de distancia del vecino más cercano (kNN).
3. **Dimensionalidad Intrínseca:** Entropía espectral vía Componentes Principales (PCA) para entender cuánto varía intrínsecamente.
4. **Evolución Temporal-Semántica:** Rapidez en el derrape anual (drift del centroide vs. años pioneros), novedad reciente y la tasa de velocidad en la que crecen geométricamente a través de la expansión radial espaciotemporal.

---

## 4. Resultados Clave y Conclusiones (Hasta la fecha)

El _script_ 15 (Clustering Ward) ejecutó agrupaciones exploratorias buscando paralelismos en los estilos morfológicos de las disciplinas estudiadas en OpenAlex. 

De los diagnósticos destacan las siguientes observaciones concluyentes:

### A. La Disimilitud de Familias Topológicas
El estudio paramétrico reportó que algunas de las variables que se diseñaron como analogías idénticas en baja dimensión (ej. kNN 2D en UMAP vs. kNN espectral, o longitud de trayectoria histórica de los agrupamientos a los años en el embedding) arrojan **blandas correlaciones mutuas (rho entre 0.1 a 0.5)**. 
**Conclusión:** La estructura latente en UMAP reduce dinámicas topológicas genuinas del campo, confirmando que la métrica visual proyectada evalúa características cualitativamente distintas y útiles a las del hiper-espacio SPECTER.

### B. Tipologías de Subcampos (Los 5 Clústeres Nucleares)
El análisis combinado (_Block PCA Combined_ sobre 241 subcampos validados) expuso exitosamente a **5 clústeres representativos** (mismo K arrojado como óptimo por Ward/KMedias). Los campos científicos resultaron diferir estadísticamente según su estructura de embedding en la ciencia moderna:

* **Clúster 1 (Ciencias Sociales y Artes - 22.8%):** Perfil centrado en subcampos como *Artes Visuales* o *Lingüística*. Presentan muy variables distancias a vecinos más cercanos (*k-nn altas*) y baja varianza PCA. Denotan estructuras heterogéneas menos cohesionadas alrededor de axiomas estrechos, o gran disparidad lingüística al clasificar sub-temas.
* **Clúster 2 (Tecnologías Aplicadas y Clínicas - 8.7%):** Ej. *Ultrasonido, Rehabilitación, Terapias Físicas*. Extrema asimetría anisotrópica con fuerte cuota en primeros componentes PCA. Son ciencias compactas, muy dominadas por uno o dos vectores temáticos altamente repetitivos (pocos componentes dictan el total de su morfología latente).
* **Clúster 3 (Ciencias Físicas Ambientales, Ingenierías de Grafo y Química Larga - 31.1%):** Ej. *Farmacología, Nutrición, Biotecnología*. Tienen alta dispersión de grafo global con baja norma de centroide. Su topología indica campos polifacéticos distribuidos ampliamente en distancias amplias, no formando macro-islas pegadas.
* **Clúster 4 (Medicina Avanzada y Profunda - 12.9%):** Ej. *Hepatología, Anestesiología, Odontología Clínica*. Destacan de toda la ciencia por exhibir un ecosistema **enormemente hiper-denso**. Tienen cortas y aplastantes distancias medias entre subtemas (_knn mean distance_ abismalmente bajas). Sus conceptos crecen fuertemente amarrados sobre paradigmas compactos en bloques aislados. Sus campos vecinos literarios siempre son semánticamente idénticos entre sí.
* **Clúster 5 (Matemáticas, Riesgo y RRHH - 24.5%):** Ej. *Estadística, Probabilidad, Farmacia y Recursos Humanos*. Se caracterizan por tener un conteo nulo de "componentes o picos densos", compensándolo con extrema alta variabilidad general (_high variance PCA shares_ equilibradas). Es una estructura en "nube" con baja densidad focalización y topologías extremadamente uniformes pero extensas.