# Script 01: Construcción de Recuentos Históricos Stratificados

## Objetivo del Script
El objetivo principal de `01_build_counts.py` es calcular la población de publicaciones científicas disponibles en OpenAlex para cada año y nivel taxonómico, evaluando progresivamente la aplicación de los filtros científicos y computacionales del TFM. Este script determina cuántos trabajos existen potencialmente en el universo científico y asienta la factibilidad cuantitativa de extracción para cada subcampo.

## Metodología
1. **Especificación de Filtros Acumulativos:** Define cuatro escenarios de conteo progresivo:
   - `n_works_total`: Todo tipo de publicación sin restricciones.
   - `n_works_article_preprint`: Restringido exclusivamente a artículos de revistas científicas o preprints (eliminando libros, actas de congresos breves, patentes, etc.).
   - `n_works_article_preprint_en`: Trabajos anteriores cuyo idioma catalogado sea estrictamente inglés.
   - `n_works_article_preprint_en_with_abstract`: Trabajos anteriores que tengan indexado un abstract (requerimiento imprescindible para generar embeddings de SPECTER2).
2. **Petición del Endpoint Group-By de la API:** En lugar de paginar trabajos individuales (lo que consumiría millones de llamadas API), el script aprovecha la funcionalidad avanzada de agregación y agrupamiento (`group_by`) de la API de OpenAlex. Agrupa las obras por nivel taxonómico (`primary_topic.subfield.id`, `primary_topic.field.id`, `primary_topic.domain.id`) y año de publicación.
3. **Pilas de Peticiones Estratificadas:** Ejecuta consultas concurrentes para cada año definido en el rango analítico de la versión de extracción activa (ej. del 2000 al 2024 para la versión activa `2000_2024_400py`).
4. **Persistencia e Indexación en DuckDB:** Junta y formatea las métricas resultantes en paneles longitudinales, escribiendo tablas de base de datos relacionales para análisis SQL posteriores.

## Parámetros de Entrada y Salida
- **Configuración e Inicialización:**
  - `config.yaml`: Rango de años y versión de extracción.
  - Tabla `subfield` de DuckDB para validar e identificar los subcampos existentes.
- **Rutas de Salida de Datos:**
  - **Archivos Parquet:**
    - `data/interim/counts_subfield.parquet`
    - `data/interim/counts_field.parquet`
    - `data/interim/counts_domain.parquet`
  - **Base de Datos DuckDB (`warehouse/tfm_openalex.duckdb`):**
    - Tabla `counts_subfield_<version>`
    - Tabla `counts_field_<version>`
    - Tabla `counts_domain_<version>`

## Integración en la Tesis (TFM)
Este script representa la **auditoría de disponibilidad poblacional**. Permite cuantificar objetivamente la "población de estudio" frente a la "muestra extraída". Al calcular la diferencia entre `n_works_total` y `n_works_article_preprint_en_with_abstract`, la tesis puede justificar formalmente el sesgo idiomático y metodológico de la muestra analizada, sentando una base rigurosa de control de calidad sobre la representatividad estadística de los 241 subcampos.
