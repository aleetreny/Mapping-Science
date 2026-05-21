# Script 10b (Deprecado): Visualización por Categorías de OpenAlex

## Objetivo del Script
El script `10b_build_per_category_umap_maps.py` es un punto de entrada heredado diseñado para ejecutar la proyecciones semánticas UMAP agregadas utilizando la antigua clasificación taxonómica por "Categorías" de OpenAlex. Actualmente se encuentra marcado como **deprecado (deprecated)** debido a que la taxonomía oficial de OpenAlex migró hacia una estructura más robusta y estándar basada en Campos y Dominios académicos (los cuales son cubiertos de manera activa por los scripts hermanos `10b_build_per_field_umap_maps.py` y `10c_build_per_domain_umap_maps.py`).

## Metodología y Funcionamiento Interno
El script actúa puramente como un envoltorio (*wrapper*) de compatibilidad:
1. **Llamada de Enrutamiento:** Importa la función de ejecución centralizada `main` de `src.per_category_umap_runner`.
2. **Flag de Deprecación:** Ejecuta `main(deprecated=True)`. Esto instruye al cargador interno para mapear registros utilizando los esquemas taxonómicos antiguos si aún estuvieran presentes en base de datos intermedios o para levantar alertas de omisión.

## Parámetros de Entrada y Salida
- **Entrada:**
  - Base de datos DuckDB.
- **Salida:**
  - No genera archivos en la carpeta de producción analítica, permaneciendo retenido por motivos de trazabilidad histórica del código.

## Integración en la Tesis (TFM)
Este script no aporta directamente resultados al cuerpo principal del TFM actual. Su retención en el repositorio se justifica bajo principios de **auditoría de software y reproducibilidad**. Documenta la transición y evolución del diseño metodológico del proyecto, evidenciando el paso de un esquema taxonómico preliminar e inestable hacia el modelo jerárquico unificado actual (Dominios, Campos, Subcampos) sancionado por OpenAlex en 2024.
