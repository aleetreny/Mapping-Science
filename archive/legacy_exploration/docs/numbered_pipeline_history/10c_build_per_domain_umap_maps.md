# Script 10c: Paisajes de Densidad y Cartografía Espacial a Nivel de Dominios Científicos

## Objetivo del Script
El objetivo principal de `10c_build_per_domain_umap_maps.py` es cartografiar y proyectar visualmente la estructura a nivel de **Dominios (Domains)** (ej. Ciencias de la Salud, Ciencias Físicas, Ciencias Sociales, Ciencias de la Vida). Este análisis a macro-escala permite evaluar la uniformidad morfológica global de los grandes pilares del conocimiento humano y contrastar si existen valles o fronteras marcadas entre la ciencia aplicada y la ciencia teórica básica.

## Metodología
El script utiliza el runner común de categoría de la siguiente forma:
1. **Llamada de Enrutamiento:** Importa la función de ejecución centralizada `main` de `src.per_category_umap_runner`.
2. **Parámetro del Nivel Fijo:** Establece la variable de nivel de agregación macro `fixed_level = "domain"`.
3. **Agrupación Masiva de Trabajos:** Agrupa todos los trabajos correspondientes a los 4 grandes dominios de OpenAlex (Physical Sciences, Health Sciences, Life Sciences, Social Sciences).
4. **Proyección UMAP Macro:** Ajusta modelos UMAP locales a gran escala para mapear la geografía interna de cada Dominio en 2D.
5. **Cálculo de KDE a Macro-escala:** Genera mapas térmicos continuos que identifican las grandes concentraciones de masa semántica y áreas periféricas del conocimiento.

## Parámetros de Entrada y Salida
- **Configuración y Entrada:**
  - Base de datos DuckDB `warehouse/tfm_openalex.duckdb`.
  - Matriz de embeddings consolidada `main_embeddings.float16.npy`.
- **Rutas de Salida de Datos (bajo `outputs/maps/per_domain_umap/` o similar):**
  - **Manifiesto Colectivo:**
    - `per_domain_umap_manifest.parquet`
  - **Archivos de Coordenadas de Dominio (un Parquet por dominio):**
    - `domain_<domain_id>_coords.parquet`
  - **Gráficos de Paneles de Densidad:**
    - `plots/domain_<domain_id>_panels.png`

## Integración en la Tesis (TFM)
En el TFM, este script representa el **análisis de macro-escala topológica**. Ofrece una visión a gran altitud que revela cómo se articulan las grandes divisiones del conocimiento. Al proyectar de forma aislada dominios como "Social Sciences" frente a "Physical Sciences", se puede contrastar cuantitativamente si las Ciencias Sociales presentan una geografía semántica más fluida, dispersa y heterogénea (menores picos de densidad locales) que las Ciencias Físicas, las cuales suelen agruparse en polos muy compactos y definidos experimentalmente. Esto provee un marco conceptual interpretativo fundamental para los capítulos finales de discusión teórica sobre la sociología de las ciencias.
