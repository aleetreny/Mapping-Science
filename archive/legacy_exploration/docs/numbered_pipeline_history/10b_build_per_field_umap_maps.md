# Script 10b: Proyección Topológica y Cartografía de Campos Disciplinares

## Objetivo del Script
El objetivo principal de `10b_build_per_field_umap_maps.py` es generar mapas de proyección topológica agregada a nivel intermedio de **Campos (Fields)** (ej. Ciencias de la Computación, Química, Medicina). El script permite contrastar cómo se auto-organizan las publicaciones en el mapa visual 2D a una escala intermedia, agregando múltiples subcampos afines y facilitando un entendimiento de la cohesión interna de las grandes ramas del conocimiento.

## Metodología
El script utiliza el runner parametrizado de la siguiente forma:
1. **Llamada de Enrutamiento:** Importa la función de ejecución centralizada `main` de `src.per_category_umap_runner`.
2. **Parámetro del Nivel Fijo:** Define la variable de nivel de agregación fija `fixed_level = "field"`.
3. **Agregación de Datos:** Agrupa todos los trabajos correspondientes a cada campo de OpenAlex que alcancen la masa crítica requerida de papers.
4. **Reducción Dimensional Local UMAP:** Ejecuta modelos UMAP independientes por cada Campo sobre los embeddings de sus trabajos.
5. **Cálculo de Densidad Continua:** Computa mapas de calor de densidad KDE de $100 \times 100$ puntos para identificar concentraciones de actividad literaria a nivel macro-disciplinar.

## Parámetros de Entrada y Salida
- **Configuración y Entrada:**
  - Base de datos DuckDB `warehouse/tfm_openalex.duckdb`.
  - Matriz de embeddings `main_embeddings.float16.npy`.
- **Rutas de Salida de Datos (bajo `outputs/maps/per_field_umap/` o similar):**
  - **Manifiesto Colectivo:**
    - `per_field_umap_manifest.parquet`
  - **Archivos de Coordenadas de Campo (un Parquet por campo):**
    - `field_<field_id>_coords.parquet`
  - **Gráficos de Paneles de Densidad:**
    - `plots/field_<field_id>_panels.png`

## Integración en la Tesis (TFM)
En el TFM, este script representa el **análisis de meso-escala espacial**. Mientras que el Script 10 mapea subcampos extremadamente específicos (ej. Inteligencia Artificial) de forma aislada, este script revela el paisaje global del que forman parte (ej. el Campo completo de Ciencias de la Computación, que agrupa Inteligencia Artificial, Redes de Datos, Software, etc.). Permite responder científicamente a preguntas sobre la cohesión y uniformidad de las disciplinas tradicionales: ¿es la Química un espacio semántico continuo y denso, o está fracturada en islas hiper-especializadas e inconexas? Esto sirve de puente conceptual e interpretativo para validar el agrupamiento morfológico.
