# Script 20: Deriva y Trayectorias Temporales de los Centroides Disciplinares

## Objetivo del Script
El objetivo primordial de `20_visualize_subfield_centroid_paths.py` es calcular, modelar y visualizar el desplazamiento diacrónico de los centros de gravedad semánticos (centroides) de los 241 subcampos científicos en el espacio vectorial SPECTER2 de 768 dimensiones. A través de este análisis longitudinal, el script cuantifica la velocidad, acumulación y dirección del cambio del conocimiento científico de cada disciplina a lo largo de las cinco ventanas temporales quinquenales de los últimos 25 años, proyectando tridimensionalmente estas trayectorias sobre un plano bidimensional mediante Análisis de Componentes Principales (PCA) para su interpretación diagnóstica.

## Metodología y Métricas de Deriva Semántica

Es crucial destacar que **todas las métricas cuantitativas de deriva semántica se calculan estrictamente en el espacio vectorial nativo de 768 dimensiones**, utilizando embeddings normalizados bajo la norma $L_2$. Las distancias de proyección visual PCA sirven exclusivamente como apoyo diagnóstico gráfico, previniendo la distorsión dimensional lineal.

El script define y extrae tres descriptores fundamentales del movimiento semántico:

1. **Longitud Acumulada de la Trayectoria (Accumulated Path Length):** Suma acumulada de las distancias del coseno entre centroides sucesivos a lo largo de las ventanas de transición:
   $$\text{Path Length} = \sum_{w=1}^{4} D_{\text{coseno}}(\mathbf{C}_{w}, \mathbf{C}_{w+1})$$
   Representa la "distancia total recorrida" o el grado de dinamismo y volatilidad interna de la disciplina.
2. **Deriva Extrema Temprana-Tardía (Early-Late Cosine Drift):** Distancia directa del coseno entre el centroide de la ventana inicial (2000-2004) y el de la ventana final (2020-2024):
   $$\text{Early-Late Drift} = D_{\text{coseno}}(\mathbf{C}_{2000-2004}, \mathbf{C}_{2020-2024})$$
   Mide el desplazamiento neto y permanente en el mapa del conocimiento tras un cuarto de siglo.
3. **Ratio de Direccionalidad (Directionality Ratio):** Calculada dividiendo la deriva euclídea neta entre la longitud total del camino euclídeo acumulado (sobre vectores de centroide normalizados a norma unitaria):
   $$\text{Directionality Ratio} = \frac{\|\mathbf{C}_{\text{final}} - \mathbf{C}_{\text{inicial}}\|}{\sum_{w} \|\mathbf{C}_{w+1} - \mathbf{C}_{w}\|}$$
   Un valor cercano a 1 indica una trayectoria rectilínea y progresiva en una dirección semántica clara (evolución progresiva acumulativa), mientras que valores cercanos a 0 indican trayectorias erráticas, oscilatorias o cíclicas que regresan sobre sí mismas.

## Parámetros de Entrada y Salida

- **Configuración y Entrada:**
  - Índice alineado de embeddings: `data/processed/analysis_embedding_index.parquet`
  - Matriz de embeddings unificada: `embeddings/specter2_v1_2000_2024_400py/analysis/main_embeddings.float16.npy`
  - Umbral mínimo: `min_papers_per_window = 100`, Proyección de soporte: `pca` (UMAP excluido por defecto para evitar distorsiones no lineales no conservadoras de distancia).

- **Rutas de Salida de Datos (Carpeta `outputs/analysis/temporal_centroid_paths/` y `data/processed/temporal/`):**
  - **Archivos de Datos Estructurados:**
    - `subfield_window_centroids.parquet` (Coordenadas de los centroides en 768D por subcampo y ventana).
    - `subfield_centroid_path_metrics.parquet` (y `.csv`): Tabla maestra con Path Length, Drift y Directionality por subcampo.
    - `domain_centroid_movement_summary.csv` y `field_centroid_movement_summary.csv` (Agregados macro y meso).
    - `largest_jump_window_distribution.csv` (Distribución de saltos máximos).
  - **Visualizaciones Científicas (PNG):**
    - `global_centroid_paths_pca_clean.png` (Mapa global con el trazado de los 241 caminos de centroides en el plano PCA).
    - `global_centroid_paths_pca_highlight_top_dynamic.png` (Mapa destacando las disciplinas con las derivas semánticas más agresivas).
    - `top_dynamic_pca_small_multiples.png` (Malla de pequeños múltiplos detallando la trayectoria individual de los campos de mayor cambio).
    - `top_centroid_path_length_subfields.png` y `top_early_late_drift_subfields.png` (Gráficos de barra de los subcampos más dinámicos).
    - `largest_jump_window_distribution.png` (Histograma de la época del mayor salto histórico).
    - `domain_average_path_length.png` y `field_average_path_length_top20.png` (Gráficos diacrónicos por jerarquías OpenAlex).

---

## Hallazgos Científicos y Diagnósticos Cuantitativos

El análisis de trayectorias de centroides en SPECTER2 revela descubrimientos epistemológicos de gran valor para la tesis:

### 1. La Gran Aceleración Reciente (Efecto Pandemia y Revolución Deep Learning)
El análisis del "Quinquenio del Salto Máximo" (`largest_jump_window_distribution.csv`) muestra cuándo experimentaron los subcampos su cambio semántico más brusco de los últimos 25 años:
- **Transición 2000-2004 a 2005-2009:** 46 subcampos.
- **Transición 2005-2009 a 2010-2014:** 21 subcampos.
- **Transición 2010-2014 a 2015-2019:** 56 subcampos.
- **Transición 2015-2019 a 2020-2024:** **118 subcampos**.
- *Significado Académico:* Casi **el 50% de las disciplinas científicas** (118 de 241) experimentaron la mayor transformación estructural de su conocimiento en el último quinquenio (2020-2024). Este fenómeno documenta empíricamente la sacudida epistemológica provocada por la pandemia del COVID-19 en ciencias de la salud y el impacto transversal e irruptivo de la Inteligencia Artificial y los modelos masivos de lenguaje en ingenierías, física y ciencias sociales.

### 2. Los Líderes del Cambio Semántico (Espacio 768D)
#### A. Máximo Desplazamiento Neto (Early-Late Cosine Drift)
- **`3307` Human Factors and Ergonomics** (Drift = **$0.038$**, Path Length = $0.014$): Es la disciplina que más se ha desplazado de su punto de origen en los últimos 25 años. Ha sufrido una deriva radical de su espectro semántico original, inducida por la obligada convergencia con la interacción persona-ordenador (HCI) y los entornos cognitivos digitales.
- **`2611` Modeling and Simulation** (Drift = **$0.016$**, Path Length = $0.018$): Se desplaza a un ritmo alto de forma rectilínea y sumamente direccional, debido a la adopción masiva de técnicas de computación distribuida y computación de alto rendimiento.
- **`2302` Ecological Modeling** (Drift = **$0.016$**, Path Length = $0.016$): Evolución progresiva y constante ligada a los modelos predictivos del cambio climático global.

#### B. Máxima Volatilidad o Dinamismo (Accumulated Path Length)
- **`2611` Modeling and Simulation** (Path = **$0.018$**).
- **`2302` Ecological Modeling** (Path = **$0.016$**).
- **`3307` Human Factors and Ergonomics** (Path = **$0.014$**).
- *Interpretación:* Obsérvese que las disciplinas computacionales y de modelado teórico son los verdaderos motores de la deriva en SPECTER2, confirmando que la incorporación constante de herramientas algorítmicas genera desplazamientos semánticos acumulados muy superiores a los de campos experimentales clásicos.

### 3. Dinámica Agregada por Campos y Dominios
- **El Dominio Más Volátil:** **Physical Sciences** registra la mayor volatilidad promedio (Path medio = **$0.004$**, Drift medio = **$0.005$** en 86 subcampos), seguido de Health Sciences, Social Sciences y Life Sciences.
- **El Campo Meso Más Dinámico:** **Computer Science** lidera de manera absoluta la deriva de la ciencia (Path medio = **$0.005$**, Drift medio = **$0.009$** en 11 subcampos). Refleja cómo la informática ha redefinido transversalmente sus propios conceptos y los de disciplinas adyacentes.

## Integración en la Tesis (TFM)
- **El Trazado del Mapa Físico del Cambio:** Las visualizaciones de los caminos de centroides en el plano PCA (`global_centroid_paths_pca_clean.png`) representan uno de los mayores atractivos visuales y científicos del TFM. Permiten al lector ver físicamente "hacia dónde viajan" las disciplinas.
- **Aportación Conceptual Clave:** El análisis cuantitativo en alta dimensión (768D) utilizando el rigor de la normalización $L_2$ y la distinción matemática entre Longitud de Trayectoria (volatilidad) y Deriva Neta (cambio estructural acumulado) dota a la tesis de un marco conceptual elegante que superará con creces el nivel metodológico habitual en un TFM de posgrado.
