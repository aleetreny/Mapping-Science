# Script 02: Planificación del Corpus Analítico y Ajuste de Capacidad

## Objetivo del Script
El objetivo principal de `02_build_corpus_plan.py` es calcular la capacidad analítica y planificar de forma determinista el tamaño de muestra requerido por cada celda espacio-temporal (definida por la intersección de un subcampo y un año de publicación). Este plan asume el objetivo balanceado de descargar hasta **400 publicaciones válidas por año por subcampo** (equivalente a 10.000 papers por subcampo sobre los 25 años del horizonte temporal `2000_2024_400py`).

## Metodología
1. **Definición de Filtros de Selección de Población:** Lee el recuento poblacional calculado en el script anterior (`counts_subfield.parquet`), utilizando específicamente la columna de publicaciones válidas en inglés con abstract indexado (`n_works_article_preprint_en_with_abstract`).
2. **Evaluación de Disponibilidad por Celda:** Para cada celda, compara la población real de papers válidos disponibles frente a la meta objetivo (400 papers).
3. **Cálculo de Topes de Muestreo (Capping):**
   - **Caso de Suficiencia (Población >= 400):** Establece el tamaño de muestra planeado estrictamente en **400**.
   - **Caso de Escasez (Población < 400):** Ajusta el plan al 100% de la población disponible en esa celda. Registra la escasez analítica (*shortfall*) resultante, la cual será informada en las fases de control de calidad.
4. **Agregaciones de Ventana Longitudinal:** Calcula la suma total acumulada de trabajos planeados por subcampo a lo largo de todo el rango temporal, identificando preliminarmente qué disciplinas alcanzarán el umbral representativo mínimo establecido por el proyecto (ej. 2500 trabajos totales para habilitación de análisis).

## Parámetros de Entrada y Salida
- **Configuración de Entrada:**
  - `config.yaml`: Define la meta anual de muestreo (400 trabajos/año) y la versión de extracción.
  - Parquet intermedio `counts_subfield.parquet` del paso 01.
- **Rutas de Salida de Datos:**
  - **Archivo Parquet:**
    - `data/interim/corpus_plan.parquet`
  - **Base de Datos DuckDB (`warehouse/tfm_openalex.duckdb`):**
    - Tabla `corpus_plan_<version>`

## Integración en la Tesis (TFM)
En el TFM, este script representa el **diseño muestral estratificado**. El diseño equilibrado de 400 papers/año por subcampo protege la investigación de verse desbordada por disciplinas masivas hiper-prolíficas (como Oncología o Física de Partículas) en detrimento de disciplinas con menor volumen anual (como Historia de la Ciencia o Antropología). Al fijar un tamaño máximo de celda, el *pipeline* garantiza que los paisajes locales UMAP y las métricas espaciales del embedding sean morfológicamente comparables entre sí, neutralizando distorsiones derivadas del puro volumen de publicación.
