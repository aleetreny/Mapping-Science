# Script 08: Ensamblado y Consolidación de la Matriz Analítica Principal

## Objetivo del Script
El objetivo principal de `08_prepare_analysis_matrix.py` es compilar y consolidar los shards dispersos de embeddings validados en una **única matriz binaria de análisis principal** y un **índice de metadatos perfectamente indexado fila por fila**. Este paso unifica todo el corpus elegible (excluyendo subcampos no válidos) en un formato matricial unificado de alta velocidad para soportar las fases computacionales intensivas de distancias semánticas y proyecciones UMAP.

## Metodología y Flujo de Procesamiento
1. **Identificación de Subcampos Elegibles:** Carga la tabla de elegibilidad (`analysis_subfields.parquet`) generada en el paso 06 para saber exactamente qué subcampos científicos califican para el estudio morfológico.
2. **Carga y Filtrado Selectivo:** Lee de forma secuencial y eficiente los archivos de metadatos Parquet de los 37 shards de embeddings, identificando las filas correspondientes a trabajos de subcampos elegibles.
3. **Indexación y Consolidación en Memoria:**
   - Construye un DataFrame indexado unificado (`analysis_embedding_index.parquet`) que asocia cada fila de la matriz final con su identificador de trabajo (`work_id`), subcampo, campo, dominio, año de publicación y recuento de citas.
   - Extrae únicamente las filas vectoriales deseadas de los archivos binarios `.npy` de cada shard y las concatena en una gran matriz bidimensional contigua en formato binario contiguo.
4. **Persistencia de Alto Rendimiento:**
   - Guarda la matriz semántica principal de Float16 en formato plano de alta densidad de NumPy (`main_embeddings.float16.npy`).
   - Guarda el índice alineado en formato Parquet indexado.

## Parámetros de Entrada y Salida
- **Configuración y Entrada:**
  - Directorio de shards de embeddings validado en el paso anterior.
  - Parquet de elegibilidad `data/processed/analysis_subfields.parquet`.
- **Rutas de Salida de Datos:**
  - **Índice Alineado de Análisis:**
    - `data/processed/analysis_embedding_index.parquet`
  - **Matriz de Incrustaciones Unificada (768D, Float16):**
    - `embeddings/specter2_v1_2000_2024_400py/analysis/main_embeddings.float16.npy`
  - **Registro en DuckDB:**
    - Tabla `analysis_embedding_index_<version>`

## Integración en la Tesis (TFM)
Este script representa la **congelación y consolidación de la matriz experimental**. Al unificar todas las incrustaciones semánticas bajo un único binario NumPy y un índice Parquet mapeado a nivel de fila, se habilita una infraestructura de cómputo ultra-rápida. Algoritmos de cálculo de grafos kNN de vecinos cercanos o reducciones PCA dinámicas (Scripts 12 y 15) requieren accesos vectoriales continuos en memoria. Este empaquetado deterministicamente alineado garantiza que no existan descalces de filas y reduce drásticamente los tiempos de lectura de disco de horas a escasos segundos, sentando las bases operacionales de todo el análisis longitudinal subsiguiente.
