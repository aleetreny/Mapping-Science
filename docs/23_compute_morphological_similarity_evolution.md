# Script 23: Evolución de la Similitud Morfológica y Topológica Interdisciplinar

## Objetivo del Script
El objetivo principal de `23_compute_morphological_similarity_evolution.py` es medir y analizar diacrónicamente (2000-2024) la similitud morfológica entre disciplinas científicas. A diferencia del Script 22, que analiza la distancia semántica (el "dónde" se ubican las disciplinas en el espacio de conocimiento a través de sus centroides de contenido), este script estudia la **morfología espacial latente** (el "cómo" se estructura internamente cada disciplina, analizando su grado de dispersión, dimensionalidad intrínseca, densidad, presencia de hubs y compacidad). El script calcula distancias morfológicas diacrónicas basándose en los perfiles de métricas morfológicas escaladas de forma robusta, abarcando tres niveles de agregación de OpenAlex: **Subcampo**, **Campo** y **Dominio**.

## Metodología y Conceptos Morfológicos

### 1. Perfiles Morfológicos y Escalado Robusto
El perfil de cada entidad científica se compone de un vector multivariante que sintetiza su topología interna en el espacio de embeddings:
*   **Análisis Estático:** Emplea las 11 métricas reducidas e interpretables definidas en el Script 16b (que incluyen métricas morfológicas locales/globales y métricas directas en 768D).
*   **Análisis Temporal:** Utiliza un conjunto de 8 métricas morfológicas netamente estructurales y no dependientes del tiempo, recalculadas de manera independiente para cada una de las 5 ventanas quinquenales.
*   **Normalización Robusta:** Para asegurar la comparabilidad de métricas con escalas físicas y rangos muy heterogéneos, los perfiles se transforman utilizando un escalado robusto (`RobustScaler`):
    $$\tilde{x} = \frac{x - \text{mediana}(x)}{\text{IQR}(x)}$$
    Esto minimiza la influencia de subcampos atípicos (*outliers*) o comportamientos extremos de nicho científico. Los perfiles a nivel de Campo y Dominio se construyen a partir del promedio ponderado de los perfiles robustos de sus subcampos componentes.

### 2. Distancias Morfológicas por Parejas (Pairwise Distances)
Para evaluar la similitud morfológica entre cada par de disciplinas $(i, j)$ en una ventana $w$, se calculan dos tipos de métricas de distancia complementarias:
*   **Distancia Euclídea Morfológica ($D_{\text{Euc}}$):** Compara la magnitud absoluta de los perfiles morfológicos escalados. Mide la diferencia cuantitativa en el volumen de dispersión, densidad espacial latente o dimensionalidad efectiva.
    $$D_{\text{Euc}}^{(w)}(i, j) = \sqrt{\sum_{k} (\tilde{x}_{i, k}^{(w)} - \tilde{x}_{j, k}^{(w)})^2}$$
*   **Distancia de Correlación Morfológica ($D_{\text{Corr}}$):** Compara el patrón proporcional o la silueta relativa de las métricas morfológicas, ignorando su magnitud absoluta. Mide si la firma morfológica de dos campos es cualitativamente análoga.
    $$D_{\text{Corr}}^{(w)}(i, j) = 1 - r(\tilde{\mathbf{x}}_i^{(w)}, \tilde{\mathbf{x}}_j^{(w)})$$
    Donde $r$ es el coeficiente de correlación de Pearson entre los dos perfiles.

### 3. Dinámica Temporal: Convergencia y Divergencia Estructural
El cambio morfológico diacrónico ($\Delta\text{Dist}$) se calcula restando la distancia del primer quinquenio de la del último:
$$\Delta D(i, j) = D_{2020-2024}(i, j) - D_{2000-2004}(i, j)$$
*   **Delta Negativo ($\Delta D < 0$) = Convergencia Morfológica:** Las estructuras latentes de ambas disciplinas se asemejan en forma y magnitud. Sugiere que están adoptando modos organizativos y de cohesión similares (por ejemplo, consolidando grados análogos de especialización o dispersión temática).
*   **Delta Positivo ($\Delta D > 0$) = Divergencia Morfológica:** Sus estructuras internas se vuelven morfológicamente heterogéneas. Sugiere dinámicas evolutivas dispares (por ejemplo, una disciplina se compacta en torno a consensos metodológicos fuertes mientras la otra se fragmenta y dispersa).

---

## Control de Calidad y Retención de Entidades (QA)
El script aplica un estricto protocolo de auditoría e integridad de datos al comparar las ventanas temporal inicial (2000-2004) y final (2020-2024):
*   **Entidades Evaluadas:**
    *   **Nivel Subcampo:** 241 subcampos iniciales. Se retuvieron exitosamente **240** y se excluyó únicamente **1** (`2605` Computational Mathematics).
    *   **Nivel Campo:** **26** de 26 retenidos (100% de éxito).
    *   **Nivel Dominio:** **4** de 4 retenidos (100% de éxito).
*   **Diagnóstico de la Exclusión:** El subcampo **Computational Mathematics** (`2605`) fue descartado debido a que en el primer quinquenio (2000-2004) contenía únicamente 92 trabajos en la base de datos muestreada, cifra por debajo del umbral cienciométrico de representatividad mínima ($N=100$). Al carecer de perfil morfológico en la ventana inicial, no era metodológicamente viable estimar su delta de cambio de distancia.

---

## Parámetros de Entrada y Salida

*   **Entradas:**
    *   Métricas estáticas reducidas: `outputs/analysis/reduced_interpretable_embedding_core/reduced_interpretable_core_metrics.parquet`
    *   Métricas temporales por ventana: `data/processed/temporal/subfield_window_embedding_metrics.parquet`
*   **Salidas de Datos Clave (Carpeta `outputs/analysis/morphological_similarity_evolution/`):**
    *   `top_morphological_converging_pairs.csv` y `top_morphological_diverging_pairs.csv` (Tablas maestras de dinámicas diacrónicas).
    *   `dropped_entities.csv` (Registro detallado del subcampo 2605 descartado).
    *   `entity_retention_diagnostics.csv` (Metadatos globales de calidad y retención).
    *   `top_pair_entity_frequency.csv` y `top_pair_driver_diagnostics.csv` (Identificación de entidades sesgadas o dominantes en los extremos).
    *   Carpeta `matrices/` (Contiene 48 archivos CSV con matrices estáticas, diacrónicas, deltas y pendientes de regresión lineal para todos los niveles).
*   **Visualizaciones Generadas:**
    *   Heatmaps de distancias basales, finales y deltas (ej: `subfield_morphological_distance_delta_heatmap.png`).
    *   Trípticos comparativos inicial-final-delta para Euclidean y Correlation (ej: `field_correlation_initial_final_delta_triptych.png`).
    *   Gráficos de dispersión estática de parejas más cercanas e inconexas (ej: `field_euclidean_static_closest_farthest_pairs.png`).

---

## Resultados y Revelaciones Científicas del TFM

### 1. Resultados de Similitud Morfológica Estática (Nivel Campo)
El análisis estático describe qué campos comparten una organización homóloga de conocimiento a nivel macro y cuáles representan islas formales incomparables:

#### A. Cercanía Morfológica Euclídea (Magnitud de Estructura Homóloga):
1.  **Biochemistry, Genetics and Molecular Biology <-> Medicine** ($0.788$): Comparten un volumen y un patrón idéntico de dispersión, con una alta concentración en núcleos clínicos combinada con extensiones metodológicas robustas.
2.  **Neuroscience <-> Psychology** ($0.812$): Refleja una densidad latente y compactación estructural altamente equiparable, propia de las ciencias conductuales y neurobiológicas.
3.  **Chemical Engineering <-> Medicine** ($0.829$).

#### B. Lejanía Morfológica Euclídea (Fuerte Disparidad Formal):
1.  **Computer Science <-> Pharmacology, Toxicology and Pharmaceutics** ($5.908$): Muestra que la ciencia de la computación opera formalmente de un modo opuesto a la farmacología (ej. CS posee múltiples focos dinámicos y gran dispersión, mientras que Farmacología se agrupa de forma hipercompacta).
2.  **Computer Science <-> Dentistry** ($5.508$).
3.  **Chemistry <-> Computer Science** ($5.167$).

#### C. Similitud de Patrón Morfológico (Correlación):
1.  **Chemistry <-> Materials Science** ($0.0827$): A pesar de sus posibles diferencias de tamaño o dispersión absoluta, la silueta proporcional de sus métricas morfológicas es prácticamente idéntica.
2.  **Dentistry <-> Immunology and Microbiology** ($0.124$).
3.  **Business, Management and Accounting <-> Neuroscience** ($0.127$).

---

### 2. Resultados de Evolución Temporal (Convergencia y Divergencia)
El estudio diacrónico de deltas revela transformaciones estructurales críticas:

#### A. Convergencia Euclídea (Aproximación Formal):
*   **Materials Science <-> Mathematics** ($\Delta D = -0.895$): Documenta la progresiva matematización y formalización de la ciencia de materiales, adoptando estructuras métricas muy similares en su espacio de representación.
*   **Decision Sciences <-> Environmental Science** ($\Delta D = -0.846$): Refleja cómo la ciencia ambiental ha adoptado estructuras morfológicas típicas de las ciencias de decisión y gestión del riesgo.
*   **Economics, Econometrics and Finance <-> Psychology** ($\Delta D = -0.829$): Evidencia la asimilación estructural impulsada por la economía conductual.

#### B. Divergencia Euclídea (Bifurcación Estructural):
*   **Computer Science <-> Immunology and Microbiology** ($\Delta D = +2.146$): Es **la mayor divergencia morfológica registrada**. Retrata la mutación de la informática hacia configuraciones estructurales sumamente dispersas y singulares, mientras la inmunología permanece fuertemente compactada en torno a sus núcleos teóricos estables.

#### C. Convergencia de Correlación (Homologación en la Firma del Perfil):
*   **Health Professions <-> Medicine** ($\Delta D = -1.659$): Homologación casi perfecta en la forma del perfil distributivo del conocimiento.
*   **Health Professions <-> Veterinary** ($\Delta D = -1.502$).
*   **Health Professions <-> Immunology and Microbiology** ($\Delta D = -1.473$).

#### D. Divergencia de Correlación (Bifurcación de la Firma del Perfil):
*   **Engineering <-> Materials Science** ($\Delta D = +1.314$): A pesar de su cercanía intuitiva, la firma morfológica interna de la ingeniería y la ciencia de materiales se ha diferenciado dramáticamente.
*   **Economics <-> Materials Science** ($\Delta D = +1.127$).
*   **Chemistry <-> Engineering** ($\Delta D = +1.067$).

---

### 3. Entidades Dominantes en los Extremos Morfológicos
El TFM devela que ciertos subcampos y campos actúan como polos dominantes que sesgan las parejas extremas, lo cual debe interpretarse como propiedades intrínsecas del comportamiento de tales áreas científicas:
*   **Philosophy (Subcampo):** Aparece en **10 de los pares morfológicos de mayor convergencia euclídea**. Refleja una progresiva "estandarización" formal de la filosofía respecto a otras disciplinas del conocimiento a nivel de estructura de embeddings (ej. reducción de vacíos terminológicos extremos o atenuación de una dispersión anárquica).
*   **Molecular Medicine (Subcampo):** Aparece en **10 de los pares morfológicos de mayor divergencia euclídea**. Demuestra una deriva estructural única de la medicina molecular, diferenciándose formalmente de manera masiva respecto al resto de la ciencia.
*   **Computer Science (Campo):** Lidera con presencia en **9 de las parejas meso de mayor divergencia euclídea**, reafirmando su morfología singular y anómala frente a los campos científicos analógicos.
*   **Health Professions (Campo):** Domina la convergencia de correlación diacrónica (participa en 8 pares extremos de convergencia), consolidando una firma formal que imita activamente los campos clínicos y de ciencias biológicas adyacentes.

---

## Integración en la Tesis (TFM)
*   **Morfología frente a Semántica:** Este capítulo es crucial para demostrar el aporte teórico-metodológico original de la tesis. Permite argumentar que el análisis cienciométrico tradicional (basado en cercanía de palabras clave o citas) es ciego a la **morfología del conocimiento**. Dos campos pueden mantener una gran distancia semántica (ej. no citarse ni usar las mismas palabras, como Neuroscience y Social Sciences) pero exhibir una similitud morfológica extrema ($0.865$), lo que significa que el conocimiento en ambos campos se organiza, ramifica y consolida siguiendo leyes estructurales idénticas en el cerebro científico.
*   **El Fenómeno Informático (Computer Science):** Permite sostener empíricamente que la informática no es meramente una "ingeniería aplicada", sino una disciplina metodológica con un comportamiento estructural disociativo y divergente, cuya forma latente evoluciona hacia una enorme complejidad interna inapreciable en las ingenierías convencionales.
