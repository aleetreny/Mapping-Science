# Script 09: Construcción del Mapa Topológico Global Semántico

## Objetivo del Script
El objetivo principal de `09_build_first_umap_maps.py` es generar el primer **paisaje visual global** de la ciencia indexada en el corpus del TFM. El script ajusta un modelo UMAP 2D sobre una muestra balanceada y representativa de todos los subcampos elegibles, facilitando una cartografía interactiva preliminar de las cercanías semánticas cruzadas entre disciplinas científicas e identificando los límites macro de los dominios del conocimiento.

## Metodología y Parámetros
El script realiza los siguientes pasos computacionales:
1. **Muestreo Equilibrado entre Subcampos:** Extrae del índice analítico (`analysis_embedding_index.parquet`) un subconjunto estrictamente balanceado de publicaciones (por defecto **500 papers por subcampo**) correspondientes a un rango temporal (ej. 2010 a 2025). Esto asegura representatividad democrática de todos los subcampos en el espacio visual final, independientemente del volumen poblacional absoluto.
2. **Ajuste del Modelo UMAP Global:** Configura y ejecuta la técnica de reducción de dimensionalidad no lineal UMAP sobre los vectores SPECTER2 de 768 dimensiones.
   - Parámetros por defecto: `n_neighbors=15`, `min_dist=0.1`, `metric='cosine'` (métricas basadas en coseno son altamente eficientes para capturar relaciones angulares en representaciones textuales de alta dimensión).
3. **Cartografía Visual y Codificación Jerárquica:** Genera gráficos de dispersión de alta resolución coloreando cada punto (paper) según su Campo o Dominio de OpenAlex y etiquetando los centroides geométricos de las disciplinas para hacer el mapa legible.
4. **Persistencia de Coordenadas:** Guarda las coordenadas UMAP 2D ($x, y$) resultantes junto con metadatos de citas y campos.

## Parámetros de Entrada y Salida
- **Configuración y Entrada:**
  - Matriz de embeddings consolidada (`main_embeddings.float16.npy`).
  - Índice Parquet `analysis_embedding_index.parquet`.
  - Parámetros CLI de muestreo (ej. `--sample-per-subfield=500`).
- **Rutas de Salida de Datos:**
  - **Coordenadas Globales:**
    - `outputs/maps/first_umap/first_umap_coordinates.parquet`
  - **Figuras Cartográficas de Alta Resolución:**
    - `outputs/maps/first_umap/first_umap_scatter_domain.png` (Coloreado por macros-dominios)
    - `outputs/maps/first_umap/first_umap_scatter_field.png` (Coloreado por campos disciplinares)

## Integración en la Tesis (TFM)
Este mapa representa la **cartografía macro de la ciencia**. Es la figura de portada del TFM, proporcionando un retrato intuitivo e integrador de la estructura global del conocimiento. Permite verificar visualmente si la topología y cercanías semánticas coinciden con las clasificaciones taxonómicas institucionales de OpenAlex (ej. si los subcampos de Inteligencia Artificial se ubican adyacentes a Matemáticas y Robótica en el mapa 2D, o si surgen áreas de intersección interdisciplinarias transfronterizas como Bioinformática entre Medicina y Ciencias de la Computación). Esto sirve de validación empírica preliminar de que los embeddings de SPECTER2 preservan coherencia temática de alto nivel al proyectarse a 2D.
