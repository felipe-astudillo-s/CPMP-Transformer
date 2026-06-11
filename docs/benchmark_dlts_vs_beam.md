# Benchmark: DLTS-DFS vs Beam Search vs Greedy

## ¿Qué es DLTS-DFS?

DLTS-DFS (*Deep Learning assisted Heuristic Tree Search — Depth First Search*) es un algoritmo de búsqueda en árbol propuesto en el paper de Hottung, Tanaka y Tierney (2020) para el problema de pre-marshalling de contenedores (CPMP).

A diferencia de un greedy simple que siempre sigue el movimiento más probable, DLTS-DFS realiza una **búsqueda en profundidad con backtracking**, guiada por dos modelos de aprendizaje profundo:

- **Branching model**: decide qué movimientos explorar en cada nodo del árbol. Genera una distribución de probabilidad sobre todos los movimientos válidos y selecciona aquellos cuya probabilidad supera un umbral `p` relativo al movimiento más probable.
- **Bounding model**: estima el costo restante desde el estado actual. Permite **podar ramas** que no pueden mejorar la mejor solución conocida hasta el momento.

### Mecanismos clave

**Branch pruning (parámetro `p`):** dado el movimiento más probable con probabilidad `r`, solo se exploran los movimientos con probabilidad ≥ `r * (1 - p)`. Con `p=0` se comporta como greedy; con `p=1` se exploran todos los movimientos válidos.

**Bounding (parámetros `k` y `d`):** cada `k` niveles de profundidad se recalcula un *lower bound* heurístico usando el bounding model. Si `pasos_actuales + d * estimación_costo ≥ upper_bound`, la rama se poda. El factor `d < 1` hace el bound más conservador.

**Upper bound inicial:** antes de la búsqueda, se ejecuta un greedy warm-start para obtener un upper bound inicial razonable que guíe la poda desde el principio.

---

## Configuración del experimento

| Parámetro | Valor |
|---|---|
| Branching model | `v9_dataBSG_250k` |
| Bounding model | `v2_cost` (solo DLTS-DFS) |
| Tiempo límite | 15 s por instancia |
| Beam width | W = 32 |
| DLTS-DFS `p`, `k`, `d` | 0.3, 3, 0.8 |
| Instancias | 380 (19 categorías CVS × 20 instancias) |

Beam Search y Greedy usan **únicamente** el branching model, para aislar el efecto de la estrategia de búsqueda.

---

## Resultados

### Tasa de resolución y pasos promedio

| Solver | Resueltas | Avg pasos | vs Greedy | Avg tiempo |
|---|---|---|---|---|
| Greedy | 380/380 | 30.4 | — | 0.040 s |
| DLTS-DFS | 380/380 | 27.1 | **-11.0%** | 8.28 s |
| Beam Search | 380/380 | 26.8 | **-12.0%** | 1.84 s |

### Resultados por categoría

| Cat | H | S | Greedy (pasos) | DLTS-DFS (pasos / tiempo) | Beam (pasos / tiempo) |
|---|---|---|---|---|---|
| 3-3 | 3 | 3 | 11.8 | 10.7 / 0.06 s | 10.1 / 0.35 s |
| 3-4 | 3 | 4 | 10.2 | 9.3 / 0.07 s | 8.9 / 0.33 s |
| 3-5 | 3 | 5 | 13.4 | 11.0 / 0.31 s | 11.1 / 0.51 s |
| 3-6 | 3 | 6 | 14.3 | 12.6 / 2.20 s | 12.9 / 0.69 s |
| 3-7 | 3 | 7 | 16.1 | 13.8 / 5.36 s | 14.5 / 0.88 s |
| 3-8 | 3 | 8 | 16.8 | 14.6 / 10.42 s | 14.8 / 0.95 s |
| 4-4 | 4 | 4 | 20.8 | 17.4 / 0.46 s | 16.5 / 0.68 s |
| 4-5 | 4 | 5 | 20.6 | 18.8 / 1.95 s | 18.6 / 0.90 s |
| 4-6 | 4 | 6 | 22.5 | 19.6 / 7.82 s | 19.6 / 1.10 s |
| 4-7 | 4 | 7 | 25.9 | 23.6 / 11.66 s | 23.6 / 1.54 s |
| 5-4 | 5 | 4 | 33.0 | 24.8 / 3.32 s | 24.2 / 1.10 s |
| 5-5 | 5 | 5 | 32.4 | 26.9 / 9.35 s | 28.1 / 1.49 s |
| 5-6 | 5 | 6 | 36.1 | 32.1 / 14.26 s | 31.6 / 1.95 s |
| 5-7 | 5 | 7 | 40.4 | 35.3 / 14.75 s | 36.1 / 2.57 s |
| 5-8 | 5 | 8 | 45.5 | 42.0 / 15.08 s | 40.6 / 3.20 s |
| 5-9 | 5 | 9 | 47.0 | 44.3 / 15.09 s | 44.8 / 3.80 s |
| 5-10 | 5 | 10 | 52.2 | 48.9 / 15.10 s | 47.5 / 4.17 s |
| 6-6 | 6 | 6 | 50.8 | 44.0 / 14.88 s | 41.9 / 2.67 s |
| 6-10 | 6 | 10 | 68.7 | 65.0 / 15.14 s | 63.4 / 6.11 s |

### DLTS-DFS vs Beam Search (instancia a instancia)

| | Cantidad | Porcentaje |
|---|---|---|
| DLTS-DFS gana | 122 | 32.1% |
| Empate | 135 | 35.5% |
| Beam Search gana | 123 | 32.4% |

---

## Análisis

**Calidad similar, velocidad muy diferente.** Tanto DLTS-DFS como Beam Search reducen los pasos del greedy en ~11-12%, y al compararse instancia a instancia están prácticamente empatados (32.1% vs 32.4%). Sin embargo, Beam Search es **4.5× más rápido** en promedio (1.84 s vs 8.28 s).

**DLTS-DFS satura el tiempo límite en instancias medianas.** A partir de categorías como 3-7, 4-6 o 5-5, el tiempo de DLTS-DFS ya supera los 5 segundos y en las categorías 5-6 en adelante llega consistentemente al límite de 15 s. En esas instancias la búsqueda queda truncada y la poda no alcanza a explorar el árbol de forma efectiva.

**El bounding model no compensa el costo de la búsqueda.** DLTS-DFS incorpora un modelo adicional para podar ramas, lo que debería permitirle encontrar soluciones más cortas. En la práctica, la mejora obtenida no supera la que logra Beam Search con una estrategia más simple y rápida. Esto sugiere que el bounding model, o los hiperparámetros `p`, `k`, `d`, no están bien calibrados para las instancias CVS evaluadas.

**Beam Search es el mejor balance calidad/tiempo** para este conjunto de instancias con el modelo `v9_dataBSG_250k`. Con un tiempo promedio de 1.84 s, deja margen para aumentar el beam width o explorar instancias más grandes sin sacrificar escalabilidad.
