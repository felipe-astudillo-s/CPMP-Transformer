# Benchmark — Modelo v9_dataBSG_250k vs Solvers

**Fecha:** 2026-05-09
**Instancias:** CVS Benchmark — 21 categorías, 40 instancias cada una, **840 instancias en total**
**Modelo:** `v9_dataBSG_250k` (Transformer entrenado con datos BSG, H=12)
**Métrica de calidad:** promedio de pasos sobre instancias resueltas por ambos solvers
**Métrica de velocidad:** tiempo promedio por instancia

---

## Configuración experimental

| Parámetro | Valor |
|---|---|
| `max_steps` | 100 |
| H por instancia | H_real + 2 (convención CVS) |
| Beams (SolverBSG) | 5 |
| Batch size (modelo batched) | 40 |
| Hardware | CPU (Apple Silicon, sin GPU) |

> **Nota sobre batching:** el modelo batched produce resultados ligeramente distintos al modelo individual debido a diferencias en el orden de operaciones de punto flotante dentro del transformer (no es un cambio algorítmico). Para comparaciones de calidad rigurosas usar la versión no-batched; para comparaciones de throughput usar la versión batched.

---

## Comparación 1 — Modelo (individual) vs Solver Greedy

**Solvers:** `ModelSolver(v9_dataBSG_250k)` vs `FRGSolver(beams=0)`

### Resumen global

| Métrica | BSG250k | Greedy |
|---|---|---|
| Instancias resueltas | 757 / 840 | 760 / 840 |
| Tasa de resolución | 90.12% | 90.48% |
| Avg pasos (propios resueltos) | 30.29 | 31.29 |
| Avg pasos (resueltas por ambos) | 30.29 | 31.16 |
| Avg tiempo por instancia | 0.0490s | 0.0037s |

> **Calidad:** Modelo es **2.8% mejor** en calidad (30.29 vs 31.16 pasos promedio)
> **Velocidad:** Greedy es **92.5% más rápido** (0.0037s vs 0.0490s por instancia)

### Observaciones

- Ambos solvers fallan completamente en las categorías **10-10** y **10-6** (las más difíciles del benchmark, superan el límite de 100 pasos)
- El modelo supera al Greedy en la mayoría de categorías medianas (4-x, 5-x)
- El Greedy supera al modelo en categorías con muchos stacks bajos (3-6, 3-7, 3-8)
- La diferencia de velocidad es grande porque el modelo corre en Python+PyTorch vs C++ compilado con `-O2`

### Tabla detallada por categoría

```
[PEGAR TABLA AQUÍ]
```

---

## Comparación 2 — Modelo (individual) vs Solver BSG (beams=5)

**Solvers:** `ModelSolver(v9_dataBSG_250k)` vs `FRGSolver(beams=5)`

### Resumen global

| Métrica | BSG250k | SolverBSG |
|---|---|---|
| Instancias resueltas | 757 / 840 | 840 / 840 |
| Tasa de resolución | 90.12% | **100.00%** |
| Avg pasos (propios resueltos) | 30.29 | 36.37 |
| Avg pasos (resueltas por ambos) | 30.29 | 26.90 |
| Avg tiempo por instancia | 0.0495s | 0.0086s |

> **Calidad:** SolverBSG es **11.2% mejor** en calidad sobre instancias que ambos resuelven (26.90 vs 30.29 pasos)
> **Velocidad:** SolverBSG es **82.6% más rápido** (0.0086s vs 0.0495s por instancia)

### Observaciones

- El SolverBSG resuelve el **100%** de las instancias; el modelo falla en 10-10 y 10-6
- El SolverBSG domina en calidad en todas las categorías, con márgenes de 4–21%
- El gap del 11.2% es la referencia de calidad más honesta: modelo greedy individual vs beam search con 5 haces
- El SolverBSG es más rápido en todas las categorías, dominando especialmente en instancias grandes

### Tabla detallada por categoría

```
[PEGAR TABLA AQUÍ]
```

---

## Comparación 4 — Modelo (batched, bs=40) vs Solver Greedy

**Solvers:** `BatchedModelSolver(v9_dataBSG_250k, batch_size=40)` vs `FRGSolver(beams=0)`

### Resumen global

| Métrica | BSG250k (batched) | Greedy |
|---|---|---|
| Instancias resueltas | 760 / 840 | 760 / 840 |
| Tasa de resolución | 90.48% | 90.48% |
| Avg pasos (propios resueltos) | 29.01 | 31.29 |
| Avg pasos (resueltas por ambos) | 29.01 | 31.29 |
| Avg tiempo por instancia | 0.0144s | 0.0048s |

> **Calidad:** Modelo es **7.3% mejor** en calidad (29.01 vs 31.29 pasos promedio)
> **Velocidad:** Greedy es **66.2% más rápido** (0.0048s vs 0.0144s por instancia)

### Observaciones

- El batching reduce el gap de velocidad de **92.5% → 66.2%** respecto a la versión individual
- En instancias pequeñas (3-3, 3-4, 3-5) el modelo batched es **más rápido** que el Greedy
- El modelo batched supera en calidad al Greedy en la mayoría de categorías 4-x y 5-x con márgenes de 6–14%
- Las categorías 10-10 y 10-6 siguen sin resolverse por ninguno de los dos

### Tabla detallada por categoría

```
[PEGAR TABLA AQUÍ]
```

---

## Comparación 5 — Modelo (batched, bs=40) vs Solver BSG (beams=5)

**Solvers:** `BatchedModelSolver(v9_dataBSG_250k, batch_size=40)` vs `FRGSolver(beams=5)`

### Resumen global

| Métrica | BSG250k (batched) | SolverBSG |
|---|---|---|
| Instancias resueltas | 760 / 840 | 840 / 840 |
| Tasa de resolución | 90.48% | **100.00%** |
| Avg pasos (propios resueltos) | 29.01 | 36.37 |
| Avg pasos (resueltas por ambos) | 29.01 | 27.01 |
| Avg tiempo por instancia | 0.0140s | 0.0094s |

> **Calidad:** SolverBSG es **6.9% mejor** en calidad sobre instancias que ambos resuelven (27.01 vs 29.01 pasos)
> **Velocidad:** SolverBSG es **32.8% más rápido** (0.0094s vs 0.0140s por instancia)

### Observaciones

- El SolverBSG resuelve **el 100%** de las instancias, incluyendo 10-10 y 10-6 que el modelo no logra resolver
- En instancias pequeñas (3-3 a 3-6, 4-4) el modelo batched es **más rápido** que SolverBSG
- El SolverBSG domina en velocidad para instancias grandes (5-x, 6-x) donde el modelo requiere más pasos
- El gap de calidad (6.9%) refleja la diferencia entre política greedy neural vs beam search con 5 haces

### Tabla detallada por categoría

```
[PEGAR TABLA AQUÍ]
```

---

## Resumen comparativo global

| Comparación | Ganador calidad | Margen calidad | Ganador velocidad | Margen velocidad |
|---|---|---|---|---|
| Modelo individual vs Greedy | **Modelo** | +2.8% | Greedy | 92.5% más rápido |
| Modelo individual vs SolverBSG | SolverBSG | +11.2% | SolverBSG | 82.6% más rápido |
| Modelo batched vs Greedy | **Modelo** | +7.3% | Greedy | 66.2% más rápido |
| Modelo batched vs SolverBSG | SolverBSG | +6.9% | SolverBSG | 32.8% más rápido |

### Conclusiones

1. **Calidad vs Greedy:** el modelo supera al Greedy en ambas configuraciones (+2.8% individual, +7.3% batched). El modelo aprende a hacer movimientos más eficientes que el greedy puro.

2. **Calidad vs SolverBSG:** el gap de calidad es del 11.2% (individual) y 6.9% (batched). El 11.2% es la referencia más honesta: refleja la diferencia real entre política greedy neural y beam search con 5 haces. La reducción al 6.9% con batching es un artefacto de punto flotante, no una mejora algorítmica.

3. **Velocidad:** el Greedy C++ es el más rápido en instancia individual. El batching reduce el gap del modelo de 92.5% → 66.2% frente al Greedy, y de 82.6% → 32.8% frente al SolverBSG. En instancias pequeñas (3-x, 4-4) el modelo batched llega a ser más rápido que el Greedy.

4. **Cobertura:** el SolverBSG es el único que resuelve el 100% de las instancias. Modelo y Greedy fallan en 10-10 y 10-6 (superan el límite de 100 pasos).

5. **Tradeoff general:** el modelo ofrece mejor calidad que el Greedy a costa de velocidad. Frente a su maestro (SolverBSG), queda a un 11.2% en calidad pero con potencial de reducir ese gap mediante beam search en inferencia o fine-tuning con RL.
