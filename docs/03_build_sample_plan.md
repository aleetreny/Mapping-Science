# Script 03: Diseño de Muestreo Reproducible y Generación de Semillas Estables

## Objetivo del Script
El objetivo principal de `03_build_sample_plan.py` es definir de manera matemática y reproducible el método de muestreo y las semillas de aletoriedad estable para cada celda espacio-temporal del corpus. Este script garantiza que, al interrogar a la API interactiva de OpenAlex en el paso de descarga, la muestra de papers seleccionada sea determinista, auditable y robusta a la repetición del *pipeline*.

## Metodología
El script procesa cada subcampo y año mediante las siguientes reglas:
1. **Asignación del Método de Muestreo:**
   - **Método `all` (Censo Total):** Si la población disponible es menor o igual al objetivo planeado (población <= 400), se asigna el método `all`. En este caso, no hay selección probabilística, pues se extraerá el 100% de la población de la celda.
   - **Método `sampled` (Muestreo Probabilístico):** Si la población es mayor que 400, se asigna el método `sampled`. La API de OpenAlex seleccionará aleatoriamente un subconjunto utilizando el algoritmo interno de dispersión por semilla.
2. **Generación de Semillas Criptográficas Estables:** Para evitar el sesgo de usar una semilla única global, calcula una semilla aleatoria estructurada de tipo entero de 32 bits (`stable_sample_seed`) basada en el hash criptográfico SHA-256 de la combinación de la ID del subcampo y el año de publicación:
   $$\text{Seed} = \text{Hash}_{\text{SHA-256}}(\text{Subfield ID} + \text{"\_"} + \text{Year}) \pmod{2^{31} - 1}$$
3. **Cálculo del Búfer de API (Oversampling):** Al realizar descargas remotas, algunos trabajos pueden ser catalogados incorrectamente con abstract por la API pero carecer de él en la práctica. Para neutralizar pérdidas en el procesamiento posterior, calcula un tamaño de muestra inicial sobre-dimensionado para la API (`api_initial_sample_size`) con un factor de búfer configurable.

## Parámetros de Entrada y Salida
- **Configuración de Entrada:**
  - `config.yaml`: Parámetros de factor de búfer y versión.
  - Parquet intermedio `corpus_plan.parquet` del paso 02.
- **Rutas de Salida de Datos:**
  - **Archivo Parquet:**
    - `data/interim/sample_plan.parquet`
  - **Base de Datos DuckDB (`warehouse/tfm_openalex.duckdb`):**
    - Tabla `sample_plan_<version>`

## Integración en la Tesis (TFM)
Este paso constituye la **garantía de reproducibilidad científica**. En la ciencia de datos empírica, el muestreo dinámico sobre APIs web vivas suele introducir variabilidad incontrolada en ejecuciones sucesivas, lo que alteraría los paisajes de densidad local de UMAP (Script 10). La implantación de semillas criptográficas estables y deterministas por cada combinación de subcampo-año asegura que cualquier investigador que vuelva a ejecutar el TFM obtendrá exactamente el mismo corpus de textos, garantizando la validez externa de los clústeres morfológicos calculados posteriormente.
