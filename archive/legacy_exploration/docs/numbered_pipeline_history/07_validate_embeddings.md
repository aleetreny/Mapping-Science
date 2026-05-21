# Script 07: Validación Fina de Incrustaciones Semánticas (Embeddings SPECTER2)

## Objetivo del Script
El objetivo principal de `07_validate_embeddings.py` es auditar la integridad estructural de los archivos fragmentados (*shards*) de embeddings semánticos generados remotamente en la GPU de Kaggle (ver manual `embed_specter2_kaggle.py`). El script asegura que los vectores latentes de alta dimensión ($768$ dimensiones) y sus metadatos asociados se hayan transferido sin pérdidas ni corrupción, garantizando una alineación deterministicamente exacta para las etapas analíticas.

## Metodología y Criterios de Auditoría
El script escanea el directorio local de embeddings (configurado mediante la variable de entorno `LOCAL_EMBEDDINGS_DIR`, ej. `embeddings/specter2_v1_2000_2024_400py`) aplicando cuatro niveles de pruebas estrictas:
1. **Verificación de Estructura de Shards:** Comprueba la presencia del número esperado de shards (por defecto **37 shards**). Para cada shard, exige la existencia de una tríada perfecta de archivos:
   - Archivo de vectores: `shard_<N>.npy` (formato binario NumPy).
   - Archivo de metadatos: `shard_<N>.metadata.parquet` (Pandas Parquet).
   - Archivo de resumen: `shard_<N>.summary.json` (JSON descriptivo).
2. **Auditoría de Dimensiones y Formato Binario:** Abre cada archivo NumPy `.npy` y valida que posea una dimensionalidad matemática exacta de **768** por fila y un tipo de dato binario compacto strictly de precisión media de punto flotante (**Float16** / `numpy.float16`).
3. **Consistencia Interna de Filas (Alineación 1 a 1):** Comprueba que para cada shard, el número exacto de filas en la matriz NumPy `.npy` coincida fila por fila con el número de identificadores únicos indexados en su archivo de metadatos Parquet.
4. **Validación de Integridad Referencial con DuckDB:** Muestra una muestra aleatoria de identificadores de obras (*work_ids*) del corpus de embeddings y comprueba mediante cruces rápidos con DuckDB que correspondan a registros existentes en la tabla `works`, garantizando que no existan vectores huérfanos sin metadatos bibliométricos.

## Parámetros de Entrada y Salida
- **Configuración de Entrada:**
  - Variables de entorno en `.env` (`LOCAL_EMBEDDINGS_DIR`, que apunta a la carpeta de shards).
  - Parámetros CLI: `--embedding-dir`, `--expected-shards` (por defecto 37).
  - Base de datos DuckDB: `warehouse/tfm_openalex.duckdb`
- **Rutas de Salida de Datos:**
  - **JSON de Resumen Técnico y Logs de Consistencia:**
    - Generados temporalmente en consola y validados en el flujo interactivo.
    - Habilitación del flag interno de validación para proceder al paso de ensamblado de matriz.

## Integración en la Tesis (TFM)
En el TFM, este script representa el **cortafuegos de consistencia binaria**. Los embeddings semánticos de alta dimensión generados en infraestructuras GPU distribuidas (Kaggle) son vulnerables a fallas sutiles durante la descarga recursiva desde Google Drive. Una fila desalineada por un fallo de red desfasaría por completo las distancias de todos los trabajos subsiguientes en la matriz analítica. Al auditar la concordancia 1 a 1 entre los vectores de 768D y sus índices bibliométricos, se garantiza que cada posición geométrica estudiada en los Scripts 12 y 22 responda con precisión matemática absoluta al contenido semántico del paper correspondiente.
