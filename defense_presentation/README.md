# Presentacion de defensa

Archivos principales:

- `presentation.tex`: fuente Beamer final (22 diapositivas principales + 18 de respaldo, 16:9).
- `presentation.pdf`: version compilada y revisada visualmente pagina a pagina.
- `make_defense_figures.py`: genera las figuras a medida de las diapositivas (`assets/def_*.pdf`) a partir de los datos ya calculados del pipeline (`outputs/` y `data/processed/temporal/`). Ejecutar desde la raiz del repo con el venv.
- `assets/`: logo UC3M, figuras a medida y figuras vectoriales de la memoria. La carpeta es autocontenida para compilar.

Diseno:

- Tema Beamer Madrid con el azul corporativo UC3M (RGB 0,0,102, el mismo `azulUC3M` de la portada de la memoria) como color de estructura.
- La seccion de medicion dedica una diapositiva a cada familia de metricas (`def_family_spread`, `def_family_packing`, `def_family_hubs`, `def_family_dims`): ilustracion conceptual arriba y la tira real de los 241 subcampos con sus extremos nombrados debajo.
- La evidencia principal usa figuras a medida generadas desde los datos: ranking de subcampos extremos (`def_opt_leaderboard`), parejas de campos mas parecidas/diferentes como capsulas de color de dominio (`def_alike_pairs`), curvas de empaquetamiento (`def_packing_curves`), top de drift semantico (`def_drift_movers`) y barras divergentes de convergencia/divergencia (`def_convergence`).
- El respaldo (paginas 24-41) cubre el detalle metodologico y los resultados completos: eleccion de SPECTER2, medicion en 768 dimensiones, definiciones y formulas, distribuciones crudas, correlaciones, distribuciones por dominio, espacio de perfiles, perfiles espejo de dos campos opuestos, las 325 parejas, naturaleza de las afirmaciones, patrones de trayectoria con silhouette/ARI, trayectorias por dominio, evolucion de Computer Science, vecinos mas cercanos e isolates, justificacion del cap de muestreo, validacion del corpus y agenda de robustez.
- Titulos de diapositiva en estilo asercion-evidencia: cada titulo enuncia el mensaje y la diapositiva lo prueba.
- Las diapositivas de respaldo (tras `\appendix`) no inflan el contador del pie: el total mostrado es 22.
- Variantes de diseno no usadas (glifos, morfoespacio, huellas de barras, ranking por amplitud) siguen disponibles como funciones en `make_defense_figures.py`.

Compilacion recomendada (dos pasadas para indice y totales):

```powershell
pdflatex -interaction=nonstopmode -halt-on-error presentation.tex
pdflatex -interaction=nonstopmode -halt-on-error presentation.tex
```

La presentacion esta redactada en ingles para mantener coherencia con la memoria y el master. El hilo narrativo sigue las cinco secciones del indice: problema y medicion, evidencia estatica, evolucion temporal, relaciones entre campos y sintesis final.
