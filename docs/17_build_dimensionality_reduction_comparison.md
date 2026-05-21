# Script 17: Comparación Sensible de Métodos de Reducción Dimensional

## Objetivo del Script
El propósito principal de `17_build_dimensionality_reduction_comparison.py` es evaluar y contrastar sistemáticamente cómo diferentes algoritmos de reducción de dimensionalidad lineales y no lineales proyectan los espacios métricos disciplinarios a 2D. Esta comparación cuantitativa y visual permite analizar si la agrupación y vecindad observadas localmente son artefactos de una técnica particular (como UMAP) o si representan estructuras topológicas robustas y persistentes que emergen consistentemente a través de diversas familias algorítmicas (PCA, MDS, t-SNE, UMAP, Isomap y PHATE).

## Metodología y Algoritmos Evaluados
El script aplica seis técnicas distintas sobre las tres representaciones del espacio métrico (`umap_only`, `embedding_only` y `combined`). En cada caso, los descriptores seleccionados son imputados por mediana, escalados uniformemente (usando `StandardScaler` o `RobustScaler`), y reducidos mediante un PCA preliminar (para conservar un umbral mínimo del $90\%$ de la varianza lineal) antes de someterse a las siguientes técnicas:

1. **PCA (Principal Component Analysis):** Método lineal de proyección ortogonal que conserva las direcciones de máxima varianza lineal global, sirviendo como línea de base conservadora.
2. **MDS (Multidimensional Scaling):** Algoritmo clásico no lineal enfocado en preservar de manera óptima las distancias euclídeas por parejas entre todos los subcampos en el espacio multidimensional original.
3. **t-SNE (t-Distributed Stochastic Neighbor Embedding):** Algoritmo probabilístico no lineal que mapea las distancias de alta dimensión a probabilidades conjuntas bajo una distribución normal, y las de baja dimensión bajo una t de Student, penalizando severamente la distorsión de distancias cortas (vecindades ultra-locales).
4. **UMAP (Uniform Manifold Approximation and Projection):** Técnica basada en geometría riemanniana y topología algebraica que asume que los datos yacen sobre una variedad localmente conexa. Equilibra la estructura local y global mediante el ajuste del hiperparámetro de vecinos ($k=15$).
5. **Isomap (Isometric Feature Mapping):** Extensión clásica de MDS que estima distancias geodésicas sobre un grafo de vecindades locales en lugar de distancias euclídeas directas, revelando si los subcampos yacen sobre un colector o variedad suave continua.
6. **PHATE (Potential of Heat-diffusion for Affinity-based Transition Embedding):** Método diseñado específicamente para visualizar estructuras continuas de transición y ramificaciones biológicas/semánticas mediante la difusión de calor sobre una matriz de afinidad local, preservando distancias de paso globales.

## Parámetros de Entrada y Salida

- **Configuración y Entrada:**
  - Métricas de morfología 2D: `data/processed/subfield_morphology_metrics.parquet`
  - Métricas de embedding 768D: `data/processed/subfield_embedding_space_metrics.parquet`
  - Directorio con agrupamientos base (Ward `k=5`): `outputs/analysis/metric_clustering/`
  - Parámetros por defecto: `random_state=42`, `n_neighbors=15` (para UMAP/Isomap/PHATE), `perplexity=30.0` (para t-SNE), `mds_n_init=4`, `decay=40` (PHATE).

- **Rutas de Salida de Datos (Carpeta `outputs/analysis/dimensionality_reduction_comparison/`):**
  - **Coordenadas de Proyección (CSV y Parquet por cada espacio):**
    - `dr_coords_umap_only.parquet` (y `.csv`)
    - `dr_coords_embedding_only.parquet` (y `.csv`)
    - `dr_coords_combined.parquet` (y `.csv`)
  - **Métricas Diagnósticas Cuantitativas:**
    - `dr_projection_quality.csv` (Tabla completa de calidad de proyección)
    - `dr_comparison_summary.md` (Resumen académico rápido)
    - `dr_comparison_config.json` y `summary.json` (Parámetros y metadatos del run)
  - **Paneles Visuales Multigráfico (PNG):**
    - `dr_comparison_umap_only_grid.png` (Malla de 6 gráficos UMAP-only)
    - `dr_comparison_embedding_only_grid.png` (Malla de 6 gráficos Embedding-only)
    - `dr_comparison_combined_grid.png` (Malla de 6 gráficos del espacio Combinado)

---

## Diagnósticos Cuantitativos Reales (Resultados del Corpus)

Los resultados de calidad de proyección se evalúan bajo dos métricas fundamentales:
1. **Trustworthiness (Confianza Topológica a $k=15$):** Mide el grado en que los vecinos más cercanos en la proyección 2D eran realmente vecinos en el espacio original de alta dimensión (evitando la introducción de falsas vecindades). Oscila entre 0 y 1 (donde 1 indica preservación perfeita de vecindad).
2. **Silhouette Coefficient (Coeficiente de Silueta 2D):** Evalúa el grado de separación de los 5 perfiles disciplinares (Clústeres Ward) proyectados en el plano bidimensional.

A continuación se presenta la tabla detallada de diagnósticos cuantitativos extraídos de `dr_projection_quality.csv`:

| Espacio Métrico (`feature_set`) | Algoritmo | Trustworthiness (Confianza) | Silhouette 2D (Separación Clústeres) | Notas de Comportamiento / Parámetros Clave |
| :--- | :--- | :---: | :---: | :--- |
| **`umap_only`** *(11 métricas)* | **PCA** | 0.8794 | 0.0653 | Lineal conservador. Explica $33.4\%$ (PC1) y $18.6\%$ (PC2) de varianza. |
| | **MDS** | 0.9085 | 0.2414 | Preserva distancias euclídeas globales de morfología 2D. |
| | **t-SNE** | **0.9284** | 0.2516 | **Máxima confianza topológica**. Genera vecindades muy cohesivas. |
| | **UMAP** | 0.9139 | 0.1355 | Mantiene un balance, pero la separación visual de clústeres es menor. |
| | **Isomap** | 0.8807 | **0.2794** | **Máxima separación de clústeres** en 2D en este espacio métrico. |
| | **PHATE** | 0.8664 | 0.1278 | Conserva transiciones continuas. Menor confianza topológica. |
| **`embedding_only`** *(6 métricas)*| **PCA** | 0.9181 | 0.2691 | Alto rendimiento lineal. PC1 ($42.6\%$) y PC2 ($27.7\%$) de varianza. |
| | **MDS** | **0.9320** | 0.2521 | **Máxima confianza global** de todo el experimento. |
| | **t-SNE** | 0.9306 | 0.2079 | Confianza excelente, similar a MDS. |
| | **UMAP** | 0.9195 | 0.2506 | Balance sólido entre confianza y cohesión local. |
| | **Isomap** | 0.9231 | **0.2825** | **Máxima separación en 2D** para variables nativas de embedding 768D. |
| | **PHATE** | 0.8850 | 0.1590 | Tendencia a la difusión de trayectorias. |
| **`combined`** *(17 métricas)* | **PCA** | 0.8510 | 0.2208 | Estrategia *Block-PCA*. PC1 ($26.8\%$) y PC2 ($21.2\%$) de varianza. |
| | **MDS** | 0.8810 | 0.1930 | Preservación de distancias en espacio híbrido ponderado. |
| | **t-SNE** | **0.8921** | 0.2375 | **Máxima confianza topológica** en el espacio unificado. |
| | **UMAP** | 0.8861 | **0.2580** | **Máxima separación de clústeres** en el espacio unificado. |
| | **Isomap** | 0.8482 | 0.2308 | Estructura geodésica. |
| | **PHATE** | 0.8463 | 0.2184 | Difusión de afinidad. |

---

## Hallazgos Científicos y Visuales

1. **La Paradoja de la Confianza Topológica:**
   El análisis cuantitativo revela que **t-SNE y MDS superan a UMAP en Trustworthiness** en la mayoría de los espacios evaluados (por ejemplo, en `umap_only`, t-SNE alcanza $0.928$ frente al $0.913$ de UMAP). Esto demuestra rigurosamente que t-SNE, a pesar de su reputación de "romper" la estructura global, es extremadamente preciso al preservar el orden de las vecindades locales en comparación con las aproximaciones de colectores locales de UMAP.
2. **Isomap como Revelador de Estructuras Discretas:**
   Sorprendentemente, **Isomap exhibe los coeficientes de Silueta 2D más altos** en los espacios individuales (0.279 en `umap_only` y 0.282 en `embedding_only`). Al calcular distancias geodésicas (caminos más cortos a lo largo de vecindades), Isomap exagera la brecha o "vacío" entre subcampos densamente poblados y regiones desérticas del mapa semántico, actuando como un excelente demarcador visual de discontinuidades epistemológicas.
3. **El Efecto Saneador de PCA en Alta Dimensión:**
   En el espacio de embeddings nativos (`embedding_only`), PCA y MDS registran puntuaciones de confianza excepcionalmente altas ($\ge 0.918$). Esto sugiere que las 6 variables esenciales extraídas del núcleo semántico 768D poseen una estructura matemática inherentemente más lineal y menos distorsionada que las 11 variables morfológicas de proyección 2D, las cuales muestran mayor no-linealidad intrínseca.

## Integración en la Tesis (TFM)
Este análisis comparativo es un elemento fundamental para el **capítulo de validación metodológica y análisis de sensibilidad** del TFM:
- **Desmitificación de Visualizaciones Únicas:** Permite justificar científicamente que la estructura de la ciencia identificada no es un sesgo impuesto por las decisiones por defecto de UMAP, sino una realidad matemática observable bajo múltiples lentes.
- **Rigor Matemático:** La inclusión de métricas formalizadas como *Trustworthiness* y *Silhouette* aleja la sección de visualizaciones de ser una mera "galería estética de mapas" y la posiciona como un estudio robusto de topología cuantitativa y fidelidad métrica.
- **Robustez del Espacio Combinado:** Muestra cómo la proyección del espacio híbrido (`combined`) mediante UMAP ($Silueta = 0.258$, $Trustworthiness = 0.886$) y t-SNE ($Silueta = 0.237$, $Trustworthiness = 0.892$) proporciona un plano de lectura bidimensional altamente fiel donde las propiedades morfológicas y las semánticas se complementan sin distorsionarse destructivamente.
