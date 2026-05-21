# Script 22: Dinámica Semántica Longitudinal y Convergencia-Divergencia Disciplinar

## Objetivo del Script
El objetivo principal de `22_compute_temporal_semantic_distances.py` es medir y analizar cómo evoluciona la distancia semántica diacrónica entre pares de disciplinas científicas a lo largo del período 2000-2024. El script calcula matrices de distancia por parejas (*pairwise distance matrices*) para cada una de las cinco ventanas temporales quinquenales a tres niveles jerárquicos de OpenAlex: **Subcampo** (28,920 pares), **Campo** (325 pares) y **Dominio** (6 pares). Esto permite detectar de forma empírica y a escalas macro, meso y micro cuáles son las áreas del conocimiento que se están integrando conceptualmente (convergencia semántica) y cuáles se están fragmentando o aislando (divergencia semántica) en el siglo XXI.

## Metodología y Concepto de Convergencia-Divergencia
El análisis utiliza como base los centroides vectoriales en el espacio SPECTER2 de 768 dimensiones (calculados en el Script 20).
1. **Mapeo de Distancias por Parejas:** Para cada ventana temporal $w$ y para cada par de disciplinas $(i, j)$, se calcula la distancia del coseno entre sus respectivos centroides vectoriales normalizados $\mathbf{C}_i$ y $\mathbf{C}_j$:
   $$D_{w}(i, j) = 1 - \frac{\mathbf{C}_i^{(w)} \cdot \mathbf{C}_j^{(w)}}{\|\mathbf{C}_i^{(w)}\| \|\mathbf{C}_j^{(w)}\|}$$
2. **Cálculo del Cambio de Distancia Semántica ($\Delta\text{Dist}$):** Representa la diferencia neta de proximidad conceptual entre el quinquenio inicial y el final:
   $$\Delta D(i, j) = D_{2020-2024}(i, j) - D_{2000-2004}(i, j)$$
   Los signos de la métrica resultante se interpretan rigurosamente así:
   - **Delta Negativo ($\Delta D < 0$) = Convergencia Semántica:** Las dos disciplinas se han aproximado conceptualmente en el espacio de representación. Sugiere la aparición de frentes interdisciplinares comunes, adopción de marcos metodológicos compartidos o fusión temática.
   - **Delta Positivo ($\Delta D > 0$) = Divergencia Semántica:** Las disciplinas se han alejado semánticamente, sugiriendo una especialización excluyente, bifurcación epistemológica o desinterés interdisciplinar mutuo.

### Método de Agregación Jerárquica
Para los niveles superiores (Campo y Dominio), el script utiliza por defecto la estrategia `weighted_subfield_centroids`. Los centroides de un Campo se derivan como la media ponderada de los centroides de sus subcampos componentes, ponderados por el volumen de publicaciones de cada subcampo en esa ventana temporal.

## Parámetros de Entrada y Salida

- **Configuración y Entrada:**
  - Tabla de centroides por ventana: `data/processed/temporal/subfield_window_centroids.parquet`
  - Jerarquía: `subfield`, `field`, `domain`.
  - Agregación: `weighted_subfield_centroids`.

- **Rutas de Salida de Datos (Carpeta `outputs/analysis/temporal_semantic_distances/` y `data/processed/temporal/`):**
  - **Archivos de Datos Estructurados:**
    - `semantic_pair_distances_by_window.parquet` (Matriz larga de 146,015 observaciones con distancias de pares por ventana).
    - `semantic_pair_distance_changes.parquet` (y `.csv`): Deltas de cambio de distancia por pareja de disciplinas.
    - `top_semantic_converging_pairs.csv` y `top_semantic_diverging_pairs.csv` (Tablas maestras firmadas sin mezclar).
    - Carpetas de matrices: `matrices/subfield_[window]_distance_matrix.csv`, `matrices/field_distance_delta_matrix.csv`, etc.
  - **Visualizaciones de Dinámica Relacional (Heatmaps y Redes):**
    - Heatmaps de distancia basal, final y delta para los tres niveles jerárquicos (ej: `field_distance_delta_heatmap.png`).
    - Trípticos diacrónicos inicial-final-delta (ej: `domain_distance_initial_final_delta_triptych.png`).
    - `subfield_semantic_convergence_divergence_network.png` (Grafo de red mostrando los lazos más fuertes de atracción y repulsión interdisciplinar).

---

## Resultados y Revelaciones de Integración Científica

Los datos del TFM revelan procesos fascinantes de evolución cienciométrica diacrónica:

### 1. Nivel Macro: Dominios en Convergencia y Divergencia
- **Máxima Convergencia Semántica:** **Health Sciences <-> Social Sciences** ($\Delta D = -0.001829$). Documenta la creciente hibridación entre el modelado social y el cuidado de la salud (salud pública, epidemiología social, políticas de salud pública).
- **Máxima Divergencia Semántica:** **Life Sciences <-> Social Sciences** ($\Delta D = +0.000610$). Retrata un distanciamiento entre la investigación biológica y médica básica frente a las corrientes interpretativas de las humanidades y ciencias sociales.

### 2. Nivel Meso: Campos de Convergencia Excepcional
A escala de Campos, surge un fenómeno cienciométrico imprevisto en el TFM: **la irrupción del campo *Economics, Econometrics and Finance* como el mayor polo de convergencia interdisciplinar de toda la ciencia contemporánea**:
- **Economics <-> Immunology and Microbiology** ($\Delta D = -0.011190$): Es **la mayor convergencia registrada** en todo el estudio meso. Se explica por el desarrollo masivo de la economía de la salud global, modelos matemáticos epidemiológicos integrados a decisiones macroeconómicas y estudios cienciométricos conjuntos catalizados por la pandemia de la COVID-19.
- **Economics <-> Dentistry** ($\Delta D = -0.010349$).
- **Economics <-> Medicine** ($\Delta D = -0.008849$).
- **Computer Science <-> Economics** ($\Delta D = -0.008732$): Refleja la explosión del *Fintech*, la computación algorítmica de mercados y la teoría de juegos computacional.

### 3. Nivel Meso: Campos de Mayor Especialización (Divergencia)
En el otro extremo, **el campo de la Enfermería (*Nursing*) encabeza los procesos de divergencia semántica**, mostrando un distanciamiento activo respecto a las ciencias pesadas y físicas:
- **Chemical Engineering <-> Nursing** ($\Delta D = +0.015960$): **La mayor divergencia neta del experimento**. La enfermería consolida su identidad semántica orientándose de manera exclusiva hacia el cuidado clínico humanizado y la medicina preventiva, alejándose de los vocabularios de la física de procesos industriales y el diseño de reactores.
- **Energy <-> Nursing** ($\Delta D = +0.014824$).
- **Biochemistry <-> Nursing** ($\Delta D = +0.014566$).

---

## Integración en la Tesis (TFM)
- **El Capítulo de Discusión Interdisciplinar:** Este script aporta **el núcleo empírico del análisis de hibridación de la ciencia**. Permite demostrar con datos precisos que la interdisciplinariedad no es un concepto etéreo, sino un vector de acercamiento observable en espacios semánticos multivariantes.
- **Evidencia contra la Fragmentación del Saber:** Los resultados de la convergencia de la economía con la inmunología y la medicina son perfectos para estructurar una discusión de alto nivel sobre cómo las crisis sistémicas globales (como las pandemias) moldean instantáneamente la estructura lógica del conocimiento humano, forzando la hibridación terminológica de disciplinas tradicionalmente inconexas.
