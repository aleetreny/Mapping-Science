# Script 06: Determinación de Subcampos Científicos Elegibles para el Análisis Principal

## Objetivo del Script
El objetivo principal de `06_build_analysis_subfields.py` es definir y etiquetar la lista oficial de subcampos científicos elegibles para el análisis estadístico y espacial del TFM. El script filtra aquellos subcampos con volumen de publicación insuficiente o mermas de descarga severas, asegurando que solo las disciplinas con un tamaño de muestra estadísticamente representativo (ej. **mínimo de 2500 trabajos válidos en total**) entren al pipeline de cálculo morfológico longitudinal.

## Metodología
1. **Unión de Plan vs Muestra Real:** Integra el recuento de trabajos reales almacenados en DuckDB (proporcionado por el script de validación 05) con la planificación teórica original.
2. **Aplicación del Criterio de Cobertura Mínima:**
   - **Umbral de Volumen Absoluto:** Evalúa que el número total de trabajos válidos y validados descargados para el subcampo a lo largo de todo el horizonte temporal (2000-2024) sea estrictamente mayor o igual a **2500**.
   - Este umbral garantiza que en las sub-ventanas temporales (ej. de 5 años cada una) existan al menos 400 a 500 papers en promedio para sostener cálculos estables de densidad KDE y coordenadas UMAP locales.
3. **Generación de Flags de Elegibilidad:** Crea una variable booleana binaria (`eligible`) para cada subcampo. Aquellos subcampos que no alcancen el criterio son excluidos del flujo de análisis espacial subsiguiente.
4. **Construcción de la Tabla de Análisis Principal:** Exporta la lista limpia clasificada jerárquicamente por Dominios y Campos de OpenAlex. En la versión activa `2000_2024_400py`, **241 subcampos** cumplieron con este riguroso criterio de representatividad científica.

## Parámetros de Entrada y Salida
- **Configuración de Entrada:**
  - `config.yaml`: Parámetros de umbral mínimo de trabajos (`min_works_for_analysis`, por defecto 2500).
  - Parquet intermedio `sample_plan.parquet` (Paso 03).
  - DuckDB: Tabla `works` para los conteos reales descargados.
- **Rutas de Salida de Datos:**
  - **Archivo Parquet:**
    - `data/processed/analysis_subfields.parquet`
  - **Base de Datos DuckDB (`warehouse/tfm_openalex.duckdb`):**
    - Tabla `analysis_subfields_<version>`

## Integración en la Tesis (TFM)
Este paso constituye el **filtro de significancia estadística**. En sociología y cienciometría, los subcampos pequeños o marginales (ej. lenguas clásicas muertas o tecnologías nicho obsoletas con menos de 100 publicaciones en 25 años) no poseen vecindarios semánticos estables. Al imponer un filtro objetivo de $\ge 2500$ publicaciones, la metodología del TFM asegura que todas las morfologías analizadas (como picos de densidad o diámetros de expansión radial) se calculen sobre muestras densas y comparables, blindando el análisis de agrupamiento (Script 15) contra artefactos e inestabilidad estadística debida a la escasez de datos.
