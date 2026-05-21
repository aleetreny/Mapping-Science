# Script 05: Validación Científica y Control de Calidad de la Base de Datos

## Objetivo del Script
El objetivo principal de `05_validate_database.py` es someter al almacén DuckDB (`tfm_openalex.duckdb`) a una batería rigurosa de pruebas de control de calidad. El script busca certificar la integridad referencial, consistencia de esquemas y completitud de datos descargados frente a las metas planeadas originalmente, documentando cualquier brecha de descarga (*shortfall*) que pudiese restringir el alcance empírico del TFM.

## Metodología y Pruebas Aplicadas
El script realiza un diagnóstico multi-nivel que incluye:
1. **Validación de Presencia de Tablas y Esquemas:** Comprueba que todas las tablas planificadas en el *pipeline* relacional (ej. `works`, `works_text`, `corpus_plan`, `sample_plan`, `subfield`) existan en DuckDB y posean los tipos de datos requeridos.
2. **Evaluación Cuantitativa de Mermas (Shortfall Analysis):**
   - Compara, celda por celda (subcampo y año), el número real de registros indexados en la tabla `works` frente al tamaño de muestra planificado en `sample_plan`.
   - Identifica y clasifica desviaciones. Las mermas suelen ocurrir por discrepancias menores de la API de OpenAlex al filtrar dinámicamente campos durante la descarga masiva (ej. abstracts indexados como válidos que resultaron estar en blanco o codificados incorrectamente).
3. **Auditoría de Integridad y Calidad de Texto:**
   - **Prueba de Nulos:** Calcula el porcentaje de abstracts vacíos o nulos en `works_text` (el cual debe ser estrictamente 0%, dado que un abstract ausente invalida la codificación en SPECTER2).
   - **Prueba de Idioma y Longitud de Token:** Diagnóstica la longitud en caracteres del texto concatenado (título + abstract) para detectar truncamientos anómalos o registros corruptos.
4. **Resumen de Cobertura Longitudinal:** Genera estadísticas globales del corpus extraído, facilitando la comprensión sintética del tamaño muestral final.

## Parámetros de Entrada y Salida
- **Configuración de Entrada:**
  - `config.yaml`: Versión de extracción.
  - Base de datos DuckDB: `warehouse/tfm_openalex.duckdb`
- **Rutas de Salida de Datos:**
  - **JSON de Diagnóstico Completo:**
    - `data/processed/database_validation_summary.json` (Parámetros y recuentos)
  - **Base de Datos DuckDB (`warehouse/tfm_openalex.duckdb`):**
    - Tabla `validation_shortfalls_<version>`: Registro exhaustivo de las desviaciones de descarga por celda.

## Integración en la Tesis (TFM)
Este paso representa la **declaración de transparencia de datos** de la tesis. Todo TFM riguroso basado en análisis masivos de datos debe reportar explícitamente sus desviaciones muestrales. La tabla de *shortfalls* y el archivo JSON resultantes alimentan los capítulos metodológicos, permitiendo cuantificar que las posibles mermas de descarga de OpenAlex son residuales (generalmente inferiores al 1%) y no comprometen la significancia de los modelos locales UMAP (Script 10) ni de los descriptores morfológicos agregados.
