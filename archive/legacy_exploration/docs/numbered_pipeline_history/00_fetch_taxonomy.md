# Script 00: Extracción de la Taxonomía de OpenAlex

## Objetivo del Script
El objetivo principal de `00_fetch_taxonomy.py` es interrogar a la API pública de OpenAlex para descargar e inicializar la taxonomía completa de la ciencia estructurada en tres niveles de granularidad jerárquica: **Dominios (Domains)**, **Campos (Fields)** y **Subcampos (Subfields)**. Este script asienta las bases jerárquicas e institucionales sobre las cuales se organizarán y clasificarán todos los trabajos de investigación (*works*) en las fases posteriores del *pipeline*.

## Metodología
El script ejecuta las siguientes etapas metodológicas:
1. **Conexión a OpenAlex:** Inicializa un cliente de conexión a la API (`OpenAlexClient`) respetando las políticas de cortesía y límites de velocidad.
2. **Paginación Dinámica:** Solicita de forma iterativa y paginada todos los registros correspondientes a los tres endpoints jerárquicos:
   - `/domains` (nivel macro, ej. Ciencias Físicas, Ciencias de la Salud)
   - `/fields` (nivel intermedio, ej. Química, Medicina, Ciencias de la Computación)
   - `/subfields` (nivel micro, ej. Inteligencia Artificial, Ecología, Enfermería)
3. **Normalización de Esquemas:** Extrae y limpia los campos clave del JSON retornado por la API, transformándolos en registros tubulares estructurados mediante Pandas:
   - Limpieza de IDs de OpenAlex a formatos cortos (ej. de `https://openalex.org/subfields/1702` a `1702`).
   - Mapeo de relaciones jerárquicas directas (`subfield -> field -> domain`).
   - Retención de metadatos cuantitativos para auditoría (`works_count`, `cited_by_count`).
4. **Persistencia:** Almacena los resultados resultantes en dos formatos complementarios para garantizar flexibilidad:
   - Archivos Parquet intermedios para lecturas rápidas e independientes de Pandas.
   - Tablas relacionales indexadas en la base de datos DuckDB para consultas SQL eficientes.

## Parámetros de Entrada y Salida
- **Configuración de Entrada:**
  - `config.yaml`: Archivo de configuración global que determina la ubicación de la base de datos DuckDB.
- **Rutas de Salida de Datos:**
  - **Archivos Parquet:**
    - `data/interim/domain.parquet` (Dominios normalizados)
    - `data/interim/field.parquet` (Campos normalizados)
    - `data/interim/subfield.parquet` (Subcampos normalizados)
  - **Base de Datos DuckDB (`warehouse/tfm_openalex.duckdb`):**
    - Tabla `domain`
    - Tabla `field`
    - Tabla `subfield`

## Integración en la Tesis (TFM)
En el contexto del Trabajo de Fin de Máster, este script asienta el **modelo de datos jerárquico**. A diferencia de otros enfoques que analizan disciplinas académicas de forma aislada, el presente TFM explota la clasificación jerárquica nativa de OpenAlex para posibilitar análisis morfológicos y de similitud agregados a nivel macro (Dominios y Campos) a partir del comportamiento detallado de sus Subcampos micro. Esto es crucial en los Scripts 22 y 23 para agrupar y ponderar las trayectorias dinámicas de cambio semántico.
