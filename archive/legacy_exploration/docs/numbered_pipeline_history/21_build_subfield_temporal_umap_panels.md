# Script 21: Mapeo de Paneles Temporales UMAP y Redistribución de Densidad Semántica

## Objetivo del Script
El objetivo primordial de `21_build_subfield_temporal_umap_panels.py` es generar diagnósticos visuales avanzados de la dinámica semántica interna para los 25 subcampos científicos más dinámicos de la ciencia (identificados en el análisis de centroides del Script 20). Mediante el trazado de publicaciones sobre **coordenadas visuales fijas y alineadas en el tiempo**, el script calcula mapas de densidad de probabilidad mediante estimación de densidad por kernel (KDE) y matrices de diferencia diacrónica. Esto permite observar geográficamente cómo el "peso" o masa conceptual de una disciplina se desplaza, expande o coloniza nuevos territorios dentro de su propio mapa topológico a lo largo de los últimos 25 años.

## Metodología de Alineación y Coordenadas Fijas
Un problema metodológico clásico al realizar mapas temporales con algoritmos estocásticos como UMAP es la "inestabilidad de la proyección": si se ejecuta UMAP por separado para cada año o quinquenio, los mapas resultantes tendrán rotaciones, reflexiones o contracciones arbitrarias que impiden cualquier comparación visual.

Para solucionar rigurosamente este obstáculo, el script implementa la siguiente estrategia:
1. **Coordenadas de Referencia Estáticas:** Se genera o reutiliza un único espacio bidimensional UMAP ajustado sobre la totalidad de los artículos del subcampo a lo largo de todo el período (2000-2024). Estas coordenadas actúan como el "suelo geográfico" o mapa base inalterable (representado visualmente como una nube de fondo gris).
2. **Alineación de Ventanas Temporales:** Para cada uno de los cinco quinquenios evaluados (desde 2000-2004 hasta 2020-2024), el script proyecta sobre el mapa base exclusivamente las publicaciones pertenecientes a ese intervalo de tiempo (dibujadas en color), revelando el posicionamiento histórico cambiante del conocimiento sobre las mismas coordenadas geográficas.

---

## Modelado y Cálculo de Densidad (KDE) y Diferencia Semántica

Para trascender la mera inspección visual de puntos dispersos, el script introduce análisis de densidad cuantitativa bidimensional:

### A. Paneles de Densidad Normalizada (Density Panels)
Sobre la cuadrícula fija de 220x220 puntos, el script aplica un histograma bidimensional suavizado mediante un kernel Gaussiano con una desviación estándar $\sigma = 2.0$:
$$\text{Density}_{w} = \text{Gaussian\_Filter}(\text{Histogram2D}(X_{w}, Y_{w}), \sigma=2.0)$$
La densidad calculada se normaliza de manera independiente dentro de cada ventana de cinco años. Este paso metodológico es crucial: al normalizarse internamente, **los paneles reflejan la redistribución geográfica relativa de la masa semántica** (dónde se concentra la atención del subcampo) y no las variaciones absolutas en el volumen anual de publicaciones científicas.

### B. Paneles de Diferencia de Densidad (Difference-Density Panels)
Para caracterizar con precisión microscópica la mutación conceptual, el script implementa la técnica de sustracción topográfica. Resta la matriz de densidad de la ventana inicial o basal (2000-2004) de la densidad de las ventanas subsiguientes $t$:
$$\Delta\text{Density}_{t} = \text{Density}_{t} - \text{Density}_{2000-2004}$$
Para su visualización, se utiliza una escala cromática de color simétrica y centrada en cero (escala divergente). Los valores resultantes se leen así:
- **Regiones Rojas (Valores Positivos / Ganancia Semántica):** Indican zonas geográficas del mapa disciplinar que han ganado masa semántica y concentración de publicaciones recientes, representando frentes de colonización científica o paradigmas emergentes.
- **Regiones Azules (Valores Negativos / Pérdida Semántica):** Zonas del mapa que han sufrido despoblación científica relativa respecto al quinquenio inicial, representando metodologías obsoletas, paradigmas consolidados pero inactivos o líneas de investigación secundarias.

---

## Parámetros de Entrada y Salida

- **Configuración y Entrada:**
  - Índice alineado de embeddings: `data/processed/analysis_embedding_index.parquet`
  - Matriz de embeddings unificada: `embeddings/specter2_v1_2000_2024_400py/analysis/main_embeddings.float16.npy`
  - Coordenadas de UMAP locales precalculadas: `outputs/maps/per_subfield_umap/coordinates/`
  - Métricas de camino de centroide (para selección automática): `data/processed/temporal/subfield_centroid_path_metrics.parquet`
  - Selección: `top_dynamic_n = 25` (barrido automático de los 25 subcampos de mayor deriva).
  - Parámetros KDE: `density_grid_size = 220`, `density_sigma = 2.0`, `min_points_for_density = 30`.

- **Rutas de Salida de Datos (Carpeta `outputs/analysis/subfield_temporal_umap_panels/`):**
  - **Manifiestos Diagnósticos:**
    - `manifest.parquet` (y `.csv`): Registro detallado indicando qué subcampos y ventanas procesaron datos, tamaño muestral por fotograma, y ubicaciones de archivos.
    - `summary.json` y `summary.md`.
  - **Subcarpetas de Visualizaciones (PNG):**
    - `static_panels/`: Paneles secuenciales horizontales de 5 imágenes con los puntos coloreados por quinquenio.
    - `density_panels/`: Paneles horizontales con el mapa de contornos de calor KDE por quinquenio.
    - `density_difference_panels/`: Paneles que exhiben visualmente las matrices de ganancia y pérdida semántica ($\Delta\text{Density}_{t}$), aislando con precisión quirúrgica los frentes de colonización interdisciplinar.

---

## Casos Diagnósticos de Alta Relevancia Epistemológica

1. **General Dentistry (Dentistry / `3500`):**
   La sustracción diacrónica muestra un vaciamiento masivo de la periferia del mapa y una concentración brutal de masa en un único núcleo ultra-cohesivo. Esto representa la estandarización clínica e industrial de las metodologías odontológicas avanzadas y la convergencia de su literatura científica.
2. **Infectious Diseases (Medicine / `2725`):**
   La transición al quinquenio 2020-2024 exhibe una gigantesca región de ganancia semántica (rojo intenso) concentrada en una coordenada aislada del mapa, rodeada de regiones azules de pérdida. Este comportamiento retrata visualmente el "Monocultivo Pandémico", en el cual la literatura médica mundial detuvo sus líneas tradicionales (VIH, tuberculosis, malaria) para volcarse unánimemente hacia la investigación de la COVID-19.

## Integración en la Tesis (TFM)
- **El Atractivo Visual Definitivo:** Los paneles de diferencia de densidad (`density_difference_panels/`) constituyen **las figuras metodológicas más impactantes de la tesis**. Hacen visible el cambio científico, transformando vectores matemáticos de 768D de difícil comprensión en mapas intuitivos de colonización intelectual.
- **Rigor en la Demostración:** Protege la tesis al sustentar empíricamente que la deriva del centroide del Script 20 no se debe a ruido aleatorio de la muestra, sino a un desplazamiento coordinado e inequívoco de la producción intelectual hacia coordenadas semánticas específicas del colector de conocimiento.
