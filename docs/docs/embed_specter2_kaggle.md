# Script de Producción: Generación de Embeddings SPECTER2 en Kaggle GPU

## Objetivo del Script
El script `embed_specter2_kaggle.py` constituye el motor de extracción de representaciones vectoriales densas de alta dimensión para el TFM. Dado que el corpus procesado consta de cientos de miles de artículos científicos de OpenAlex (2000-2024), la computación local en CPU es inviable. Este script está diseñado para ejecutarse en entornos de computación en la nube basados en GPU de Kaggle (usualmente NVIDIA T4 x2 o P100), implementando un pipeline altamente optimizado y tolerante a fallos para procesar, fragmentar y almacenar los embeddings directamente en una carpeta sincronizada de Google Drive mediante `rclone`.

---

## Arquitectura del Modelo y Pipeline de Procesamiento

### 1. Modelo Base y Adaptador SPECTER2
Para capturar de forma óptima el significado conceptual de la literatura académica, se utiliza la suite de modelos **SPECTER2** de Allen Institute for AI:
*   **Modelo de Entrada (`model_base`):** `allenai/specter2_base` (una arquitectura basada en SciBERT pre-entrenada con millones de textos científicos).
*   **Adaptador de Tareas (`adapter`):** `allenai/specter2` (utiliza la biblioteca `adapters` de Hugging Face). La incorporación del adaptador modulariza el embedding y especializa la representación en tareas de similitud científica general, ad-hoc y clasificación de citas.
*   **Pooling de Salida:** Se extrae el vector de estado oculto correspondiente al token de inicio de secuencia `[CLS]` (primera posición de salida `[:, 0, :]`), obteniendo una representación vectorial unificada de **768 dimensiones**.

### 2. Preparación y Concatenación del Texto
El texto que se inyecta en el tokenizador de SPECTER2 se construye estructurando los campos de metadatos cienciométricos:
$$\text{Texto de Entrada} = \text{Título} + \text{[SEP]} + \text{Resumen}$$
Donde `[SEP]` representa el token especial de separación del modelo (`tokenizer.sep_token` o su defecto `" [SEP] "`).
El tokenizador restringe la longitud de la secuencia a un máximo de **512 tokens** (`max_length=512`), truncando aquellas publicaciones cuyo resumen exceda dicho límite, asegurando un uso óptimo y homogéneo del espacio de memoria de vídeo de la GPU.

### 3. Precisión en Punto Flotante y Almacenamiento
*   **Tipo de Datos (`dtype`):** Se utiliza `float16` de manera por defecto mediante automezcla de precisión (`torch.autocast` en CUDA). Esto reduce los requerimientos de almacenamiento y transferencia en un **50%** en comparación con el estándar `float32` sin degradar la fidelidad métrica de los embeddings resultantes.
*   **Batch Size:** Se establece en `32` como el tamaño óptimo de lote para evitar el desbordamiento de memoria (*Out of Memory* - OOM) en las GPU convencionales de 16 GB de VRAM.

---

## Fragmentación (Sharding) y Sincronización en la Nube

Debido a las estrictas limitaciones físicas de los entornos virtuales de Kaggle (límite de 9 a 12 horas de ejecución por sesión y almacenamiento en disco temporal restringido), el script implementa una estrategia de **procesamiento por fragmentos independientes (shards)**:

### 1. Definición del Sharding
*   La tabla maestra de trabajos (`works_text.parquet`) se ordena de manera estable y determinista por su jerarquía:
    $$\text{Ordenación} = \text{subfield\_id} \to \text{publication\_year} \to \text{work\_id}$$
*   El corpus se subdivide en bloques homogéneos de **20,000 registros** (`shard_size=20000`).
*   Para cada shard $i$, el script genera tres archivos de salida bien delimitados:
    1.  `shard_xxxx_embeddings.npy` (Matriz NumPy binaria de dimensiones $[N_{\text{shard}}, 768]$ en tipo `float16`).
    2.  `shard_xxxx_metadata.parquet` (Tabla Parquet con los 18 metadatos cienciométricos correspondientes, alineados registro a registro con la matriz anterior).
    3.  `shard_xxxx_summary.json` (Archivo de metadatos del shard: tiempo de cálculo en segundos, recuento de registros, límites temporales de publicación, subcampos involucrados).

### 2. Protocolo de Resiliencia y Recuperación Automática (`--resume`)
*   **Autenticación en la Nube:** El script lee un secreto cifrado de Kaggle (`RCLONE_CONF`) y escribe automáticamente la configuración de `rclone` en `/root/.config/rclone/rclone.conf`.
*   **Políticas de Tolerancia a Fallos:**
    *   Al iniciar, el script escanea tanto el almacenamiento local en disco como la carpeta remota de Google Drive (`rclone lsf`).
    *   Si los tres archivos correspondientes a un shard $i$ ya se encuentran completos en Google Drive o localmente, **el script lo omite automáticamente**, saltando al siguiente shard pendiente.
    *   Una vez calculado y guardado localmente el shard, se sube inmediatamente al almacenamiento en la nube.
    *   Si se habilita `--delete-local-after-upload`, los archivos locales se eliminan inmediatamente tras una subida exitosa, permitiendo procesar millones de registros con apenas espacio en el disco temporal de Kaggle.

---

## Estructura del Dataset de Salida en Producción

El proceso en producción generó exitosamente el dataset ubicado en la ruta corporativa de la nube `gdrive:TFM/openalex_subfields/embeddings/specter2_v1_2000_2024_400py`, constando de:
*   **37 Fragmentos Completos:** Numerados secuencialmente desde `shard_0000_*` hasta `shard_0036_*`.
*   **111 Archivos de Datos:** 37 matrices NumPy, 37 tablas Parquet de metadatos y 37 ficheros JSON de resumen.
*   **Archivo `embedding_config.json`:** Registro histórico con todos los hiperparámetros y metadatos de configuración aplicados en la nube.
*   **Archivo `embedding_run_manifest.csv`:** Historial de ejecución agregando el desempeño de la GPU por cada fragmento.

---

## Descarga Local Automatizada y Validación

Para traer el set completo de embeddings y metadatos desde la nube a la infraestructura local del TFM, se incorporan scripts de descarga inteligente concurrentes:
*   **Windows (PowerShell):** `scripts/download_embeddings_from_drive.ps1`
*   **Linux / macOS (Bash):** `scripts/download_embeddings_from_drive.sh`

### Características de la Descarga Inteligente:
*   Importación automatizada de variables de entorno mediante un archivo de configuración `.env`.
*   Sincronización multi-hilo optimizada mediante `rclone copy` configurado con 4 transferencias concurrentes (`--transfers 4`), 8 comprobadores simultáneos (`--checkers 8`) y tamaño de bloque de lectura de drive optimizado a 64 MB (`--drive-chunk-size 64M`).
*   **QA de Integridad Local:** Tras concluir la transferencia, los scripts verifican rigurosamente que el directorio local contenga exactamente los 111 componentes (37 embeddings, 37 metadatos y 37 resúmenes). En caso de existir alguna discrepancia o archivo corrupto, se lanza una excepción crítica para abortar los pipelines subsecuentes.

---

## Comandos Operativos y Pruebas

### 1. Verificación Inicial de Conectividad con la Nube:
```bash
python scripts/embed_specter2_kaggle.py \
  --setup-rclone-from-secret RCLONE_CONF \
  --check-remote-only
```

### 2. Prueba Acotada de Depuración en GPU (Sin Subida):
```bash
python scripts/embed_specter2_kaggle.py \
  --setup-rclone-from-secret RCLONE_CONF \
  --limit-rows 1000 \
  --end-shard 1 \
  --no-upload
```

### 3. Lanzamiento de la Producción Completa (Modo Resiliente):
```bash
python scripts/embed_specter2_kaggle.py \
  --setup-rclone-from-secret RCLONE_CONF \
  --resume
```

---

## Integración en la Tesis (TFM)
*   **Reproducibilidad Metodológica:** Este script y su runbook demuestran la viabilidad técnica y el diseño de ingeniería detrás del TFM. Describe cómo se sortearon las limitaciones computacionales y de almacenamiento mediante una arquitectura desacoplada en la nube y sincronización diacrónica robusta.
*   **Trazabilidad del Corpus:** La ordenación estable inicial y la fragmentación en shards garantizan una trazabilidad determinista del 100% de los datos. Permite asegurar que cualquier registro en las matrices NumPy de embeddings locales corresponde de forma exacta y matemática al mismo registro en las tablas de metadatos cienciométricos, sentando las bases analíticas para todos los scripts y modelos posteriores de la tesis.
