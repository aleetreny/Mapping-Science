# Script 04: Motor de Extracción Resiliente y Descarga del Corpus

## Objetivo del Script
El objetivo principal de `04_download_sampled_corpus.py` es ejecutar la descarga masiva y remota del corpus de textos científicos (Títulos y Abstracts) a partir de la API de OpenAlex, guiándose estrictamente por el plan de muestreo. Este script está diseñado como una máquina de estados tolerante a fallos, capaz de gestionar caídas de red, límites de velocidad severos (errores HTTP 429) y reinicios de proceso sin duplicar peticiones.

## Metodología y Robustez Técnica
1. **Petición con Semillas Deterministas a la API:** Para las celdas marcadas como `sampled`, construye consultas a la API inyectando el parámetro `sample` con la semilla estable de 32 bits generada en el paso anterior. Esto obliga a la API de OpenAlex a seleccionar de forma probabilística pero determinista los trabajos correspondientes.
2. **Normalización en Caliente de Publicaciones:** Para cada trabajo devuelto, valida inmediatamente los metadatos esenciales en memoria:
   - Traduce el abstract indexado en formato de índice invertido (*Inverted Abstract Index*) a texto lineal legible.
   - Limpia etiquetas HTML e impurezas del texto.
   - Verifica la presencia real de texto en inglés con abstract no nulo.
   - Acorta y normaliza la clave de ID único (ej. `W309183421`).
3. **Máquina de Estados de Descarga (Resumable Manifest):** Para evitar la pérdida de progreso tras fallos de conexión en descargas largas (que pueden durar horas), el script mantiene un **manifiesto de descarga** (`download_manifest.parquet`). Cada vez que una celda espacio-temporal finaliza con éxito, escribe el resultado en el manifiesto. Al reiniciar, lee este archivo y descarta automáticamente las celdas ya descargadas.
4. **Manejo de Errores y Estrategia Backoff:** Captura de forma robusta las excepciones de cuota y red. Implementa reintentos automáticos con retraso exponencial (*exponential backoff*) y pausas controladas de desconexión ante límites de tasa de la API pública (`OpenAlexRateLimitError`).

## Parámetros de Entrada y Salida
- **Configuración y Entrada:**
  - `config.yaml`: Token de cortesía (*Polite Email*) de OpenAlex, parámetros de tasa y búferes.
  - Parquet intermedio `sample_plan.parquet` del paso 03.
- **Rutas de Salida de Datos:**
  - **Manifiesto de Progreso:**
    - `data/interim/download_manifest.parquet`
  - **Base de Datos DuckDB (`warehouse/tfm_openalex.duckdb`):**
    - Tabla `works_<version>`: Contiene metadatos de las obras (años, citas, identificadores, idiomas).
    - Tabla `works_text_<version>`: Almacena de forma indexada los textos concatenados del título y abstract para posterior tokenización/embeddings.

## Integración en la Tesis (TFM)
Este motor de descarga representa el **núcleo de extracción del corpus empírico**. Su capacidad de reanudación y normalización inmediata en caliente asegura la estabilidad e integridad de la base de datos DuckDB. Al reconstruir los abstracts indexados invertidos en texto plano y limpiar las impurezas en caliente, consolida un repositorio de texto libre de ruido lingüístico, lo que resulta indispensable para que el codificador espectral de SPECTER2 (Script 07 y Kaggle) funcione de forma óptima sin sesgos derivados de mala puntuación o texto dañado.
