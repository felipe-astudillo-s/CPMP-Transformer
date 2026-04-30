# Generación de datos — Pipeline V9

Este documento describe el pipeline de generación de datasets supervisados
para entrenar los CPMP Transformer (V8/V9/V10 y posteriores). Implementa la
lógica descrita en `Gen_datos.pdf` con los refinamientos discutidos para
balanceo por dificultad, diversidad de altura y eficiencia en buckets duros.

El notebook que ejecuta este pipeline es
[`notebooks/GenDataCPMP_V9.ipynb`](../notebooks/GenDataCPMP_V9.ipynb).

---

## 1. Resumen ejecutivo

| Aspecto | Decisión |
|---|---|
| Etiqueta `Y` | `DefaultMovesAdapter` multilabel `S·(S−1)` (idéntica al pipeline anterior) |
| Resolutor | C++ greedy (`beams=0`) + 1 paso de lookahead en Python (`get_best_moves`) — **idéntico al actual** |
| `S` (pilas) | Fijo por dataset (requerido por `DefaultMovesAdapter`) |
| `T` (altura física) | **Variable por instancia** ∈ [3, `H_pad`], padding a `H_pad` con `-1` |
| `F` (fill rate) | Sorteado por instancia ∈ [50%, 85%] |
| `k` (grupos de prioridad) | Sorteado por instancia ∈ [⌈N/4⌉, N] |
| Balanceo | Cuotas globales por bucket de dificultad (15/25/30/20/10%) |
| Eficiencia en buckets duros | Caminata sesgada hacia movimientos que generan bloqueos |
| Modularidad | Cualquier `LayoutDataAdapter` enchufable; misma `seed` reproduce el dataset |
| Salida | `.data` HDF5 con `S`, `X`, `Y`, `C`, más metadata por instancia |

---

## 2. Diferencias vs. el pipeline anterior

| Punto | Pipeline anterior (`generation.ipynb`) | Pipeline V9 (`GenDataCPMP_V9.ipynb`) |
|---|---|---|
| Tamaño de bahía | `(H, S, N)` fijos por dataset | `S` fijo, `T`/`F`/`k` sorteados por instancia |
| Cantidad de movimientos aleatorios | `r` fijo (e.g., 50) para todas las instancias | Adaptativo según el bucket objetivo |
| Estado base | Random insertion (`generate_stacks` con `random.randint`) | Estado **perfectamente ordenado** (round-robin + shuffle de columnas) |
| Anti-ciclo en caminata | Sí (un paso) | Sí (un paso) |
| Sesgo en caminata | No | **Sí** para buckets `medio`/`difícil`/`extremo` |
| Balanceo por dificultad | No (todo `r=50`) | Sí (cuotas por bucket sobre %bloqueados real) |
| Verificación 0%-bloqueados del estado base | No (no aplicaba) | Sí (`assert` interno) |
| Verificación de resolución | Solo `cost != inf` | Solo `cost != inf` (igual) |
| Lookahead / cómputo de `Y` | `get_best_moves` (1-paso) | `get_best_moves` (1-paso) — **misma función reutilizada** |
| Metadata por instancia | Solo `cost` | `cost`, `T`, `k`, `F`, `N`, `bucket`, `%bloqueados`, opcional `gap`/`difficulty` |

**No cambia el cómputo de `(X, Y)` dado un `Layout`** — solo cambia cómo se
generan los `Layout` y qué `T` (altura física) llega al adapter.

---

## 3. Pipeline en 3 fases

### Fase 1 — Generación de estados (serial, rápida)

Para cada instancia, hasta cubrir la cuota total:

1. **Elegir bucket objetivo** mediante muestreo ponderado por la cuota
   restante (los buckets con más cupo libre son más probables).
2. **Sortear parámetros** `(T, F, k, N)` con las restricciones del PDF:
   `T ∈ [3, H_pad]`, `F ∈ [50%, 85%]`, `N = ⌊S·T·F⌋` (forzado a
   `N ≤ S·T − 1` para garantizar al menos un slot vacío),
   `k ∈ [⌈N/4⌉, N]`.
3. **Construir el estado objetivo (ordenado)**:
   - Asignar prioridades `p ∈ {1..k}` a los `N` contenedores. Si `N` no es
     múltiplo de `k`, los primeros `N mod k` grupos reciben `⌈N/k⌉`
     contenedores y el resto `⌊N/k⌋`.
   - Distribuir los contenedores por **round-robin** sobre las pilas en
     orden descendente de prioridad. Con esto cada pila queda ordenada
     (no-creciente bottom→top) y los grupos quedan distribuidos de manera
     uniforme — no hay pilas "monogrupo".
   - **Permutar columnas** aleatoriamente (elimina sesgo espacial).
   - **Checkpoint interno**: `count_blocked() == 0` (assert).
4. **Caminata inversa** sobre el estado ordenado:
   - Sortear el número de pasos en `[lo·N, hi·N]` según el bucket
     objetivo (tabla abajo).
   - En cada paso, elegir un movimiento válido (origen no vacío, destino
     con espacio, distinto del origen) que **no sea el inverso inmediato**
     del último movimiento.
   - **Sesgo opcional** (`p_bias = 0.7`, solo en buckets `medio`/`difícil`/`extremo`):
     entre los movimientos válidos, preferir los que apilan un contenedor
     de mayor valor sobre uno menor (genera bloqueos directamente). Si no
     hay tales movimientos, fallback a aleatorio uniforme.
5. **Medir y asignar bucket**:
   - `%bloqueados = (#contenedores con max_prioridad_arriba > su prioridad) / N · 100`
   - Si el bucket real tiene cuota libre → aceptar.
   - Si no, intentar reasignar al bucket vecino (`bucket±1`) si tiene
     cuota libre **y** `%bloqueados` está a ≤12 puntos del centro de ese
     vecino. Esto evita descartar instancias casi-en-rango.
   - Si ningún bucket compatible tiene cuota → descartar y reintentar.

### Fase 2 — Resolución con C++ (paralela, lenta)

`ProcessPoolExecutor` distribuye las instancias entre procesos. Por cada una:

1. Reconstruir el `Layout(stacks, T_phys)`.
2. `best_moves, cost = get_best_moves(layout, T_phys, max_steps)`:
   - Para cada uno de los `S·(S−1)` primeros movimientos legales, copiar el
     layout, aplicar el movimiento y correr el binario C++ `frg` con
     `beams=0` (greedy puro).
   - Quedarse con el conjunto de primeros movimientos que minimizan el
     costo (multilabel — empates incluidos).
3. **Opcional (`compute_difficulty=True`)**: correr greedy puro **sin**
   primer movimiento para obtener `cost_pure`, y derivar
   `gap = (cost_pure − cost_full) / cost_full`.

Es **exactamente la misma resolución** que usa el pipeline anterior — la
función `get_best_moves` se importa sin modificar de
`src/generation/data.py`.

### Fase 3 — Empaquetado (serial, rápida)

Para cada instancia resuelta:

1. `layout_vec = layout_adapter.layout_2_vec(layout, T_phys)` — el adapter
   recibe `T_phys` (no `H_pad`), así que `X[i][4]` (altura máxima en el
   adapter 5D) y los free-spaces tácticos (en V10) corresponden a la
   altura física real de la instancia.
2. La matriz `S` resultante (shape `(S_stacks, T_phys, ...)`) se padea a
   `(S_stacks, H_pad, ...)` con `-1` a lo largo de la dimensión de
   altura. Las máscaras `(S == -1)` del modelo absorben este padding sin
   modificación.
3. `moves_vec = moves_adapter.moves_2_vec(best_moves, S_stacks)` — `Y`
   multilabel idéntico al pipeline anterior.
4. Acumular en los adapters y en arrays de metadata.

Todo se vuelca a un único `.data` HDF5.

---

## 4. Tabla de buckets

| Bucket | %bloqueados objetivo | Pasos inversos (×N) | Cuota global |
|---|---|---|---|
| 0 — trivial | 0 — 15 % | 1·N — 3·N | 15 % |
| 1 — fácil   | 15 — 35 % | 3·N — 6·N | 25 % |
| 2 — medio   | 35 — 55 % | 6·N — 10·N | 30 % |
| 3 — difícil | 55 — 75 % | 10·N — 15·N | 20 % |
| 4 — extremo | 75 — 90 % | 15·N — 25·N | 10 % |

Con `total_instances = 250 000`: `[37500, 62500, 75000, 50000, 25000]`. El
remanente por redondeo se absorbe en el bucket `medio`.

---

## 5. Parámetros del notebook

Todos viven en la celda principal de generación
(`Generación principal — 250k instancias`):

| Parámetro | Default | Descripción |
|---|---|---|
| `OUTPUT_NAME` | `'V9_S5_H10_250k_5D'` | Nombre del archivo `.data` (sin extensión). Se guarda en `DATA_FOLDER`. |
| `S_STACKS` | `5` | Número de pilas. **Fijo en todo el dataset** (requerido por `DefaultMovesAdapter`). |
| `H_PAD` | `10` | Altura de padding. Las instancias tendrán `T ∈ [3, H_PAD]`. |
| `TOTAL_INSTANCES` | `250_000` | Tamaño objetivo del dataset. |
| `MAX_STEPS` | `100` | Tope de movimientos para `greedy_solve` en C++. Si el solver no resuelve en `MAX_STEPS`, retorna `inf` y la instancia se descarta. Para `H_PAD=10` y `S=5`, `100` deja margen suficiente. |
| `SEED` | `42` | Semilla maestra. Misma `SEED` con misma config reproduce el dataset bit a bit. |
| `COMPUTE_DIFF` | `False` | Si `True`, corre greedy puro adicional para medir gap → roughly 2x más lento. La etiqueta `Y` no cambia; solo añade `meta_gap` y `meta_difficulty`. |
| `layout_adapter` | `EnrichedStackMatrix5DAdapter()` | Adaptador para construir la matriz `S` y el vector `X`. Cualquier subclase de `LayoutDataAdapter` sirve. |
| `moves_adapter` | `DefaultMovesAdapter()` | Adaptador para `Y`. **No cambiar** si quieres mantener compatibilidad con el entrenamiento actual. |

Parámetros internos del pipeline (modificables editando la celda de
definiciones, no documentados como API pública):

- `T_min = 3` (altura mínima física por instancia).
- `F_min, F_max = 0.50, 0.85` (rango del fill rate).
- `p_bias = 0.7` (prob. de elegir un movimiento que genera bloqueo en
  buckets duros). Subir a `0.85` si el bucket `extremo` tarda mucho en
  llenarse.
- `max_walk_retries = 3` (reintentos por instancia en Fase 1 antes de
  abandonarla).
- `acceptance_band = 12` (puntos de %bloqueados de tolerancia para
  reasignar al bucket vecino).
- `chunksize = 64` para el `ProcessPoolExecutor` de Fase 2.

---

## 6. Estructura del archivo `.data`

HDF5 con las siguientes claves (el shape exacto depende del adapter; el
ejemplo asume `EnrichedStackMatrix5DAdapter` con `S=5`, `H_pad=10`):

| Clave | Shape | Tipo | Descripción |
|---|---|---|---|
| `S` | `(N, 5, 10, 2)` | `float32` | Matriz por pila × altura × (valor_normalizado, flag_bloqueo). Padding `-1`. |
| `X` | `(N, 5, 5)` | `float32` | Vector por pila × 5 features (depende del adapter). |
| `Y` | `(N, 20)` | `int32` | Multilabel sobre los `S·(S−1)=20` movimientos posibles. `1` en cada primer movimiento óptimo. |
| `C` | `(N,)` | `int32` | Costo (movimientos del greedy desde el estado X). `-1` si falló. |
| `meta_T` | `(N,)` | `int32` | Altura física `T` de cada instancia. |
| `meta_k` | `(N,)` | `int32` | Número de grupos de prioridad. |
| `meta_F` | `(N,)` | `float32` | Fill rate sorteado. |
| `meta_N` | `(N,)` | `int32` | Cantidad de contenedores. |
| `meta_bucket` | `(N,)` | `int32` | Índice de bucket [0..4]. |
| `meta_blocked_pct` | `(N,)` | `float32` | %bloqueados real. |
| `meta_gap` | `(N,)` | `float32` | (Solo si `COMPUTE_DIFF=True`) Gap relativo del greedy puro vs full. `-1` si no se pudo calcular. |
| `meta_difficulty` | `(N,)` | `int32` | (Solo si `COMPUTE_DIFF=True`) `0=fácil, 1=medio, 2=difícil`. |

Atributos del archivo (`f.attrs`):
- `key_order`: orden de claves del adapter (compatibilidad con el
  pipeline de entrenamiento existente).
- `S_stacks`, `H_pad`: parámetros del dataset.
- `pipeline`: `'V9'`.
- `compute_difficulty`: bool.

El esquema **no rompe** la compatibilidad con el entrenamiento actual: las
claves `S`, `X`, `Y`, `C` están presentes con la misma forma y dtype que
en los `.data` previos. Las claves `meta_*` son aditivas — el código
existente las ignora.

---

## 7. Cómo correr

### En Google Colab

1. Subir `GenDataCPMP_V9.ipynb` a Colab.
2. Ejecutar las celdas de **Setup** (clonar repo, compilar solver, paths).
   Si ya tienes el repo clonado en Drive, descomenta solo la celda de
   compilación.
3. Ejecutar la celda de **Pipeline V9** (define todas las funciones).
4. Ajustar parámetros en la celda **Generación principal** y ejecutar.
5. (Opcional) Inspeccionar el dataset con la última celda.

**Sanity check antes de tirar 250k**: ejecutar primero con
`TOTAL_INSTANCES = 5_000` y verificar que:
- La distribución por bucket queda cerca de las cuotas
  (15/25/30/20/10 %).
- El `.data` carga sin errores con tu pipeline de entrenamiento.
- Un forward de V9/V10 con un batch del dataset corre sin shape mismatch.

Pasado el sanity check, escalar a 250k.

### Local

```bash
cd CPMP-Framework
g++ Codigo_C_solver/Greedy.cpp Codigo_C_solver/Layout.cpp \
    Codigo_C_solver/Bsg.cpp Codigo_C_solver/main_cpmp.cpp \
    -o Codigo_C_solver/frg -O3 -std=c++11
chmod +x Codigo_C_solver/frg
jupyter notebook notebooks/GenDataCPMP_V9.ipynb
```

(Idéntico al pipeline anterior; el binario `frg` ya viene compilado en el
repo si trabajas en Linux/Mac.)

---

## 8. Modularidad — regenerar con otro adapter

Para producir el mismo conjunto de instancias con otra representación de
`X` (por ejemplo, para entrenar V10 que consume `TacticalStackMatrixAdapter`):

```python
generate_dataset_v9(
    output_name='V9_S5_H10_250k_TACTICAL',
    S_stacks=5,
    H_pad=10,
    total_instances=250_000,
    layout_adapter=TacticalStackMatrixAdapter(),
    moves_adapter=DefaultMovesAdapter(),
    max_steps=100,
    seed=42,  # MISMO seed → MISMAS instancias, distinto X
)
```

La misma `seed` con la misma `(S_STACKS, H_PAD, TOTAL_INSTANCES)` reproduce
exactamente la misma secuencia de estados `(stacks, T, k, F)`. Solo cambia
cómo se proyectan a tensor.

---

## 9. Multi-config — varios `S` en paralelo

Para un dataset con varios tamaños de bahía, lanzar el pipeline una vez
por cada `S`:

```python
configs = [
    # (S, H_pad, total)
    (4, 10,  60_000),
    (5, 10, 100_000),
    (6, 10,  60_000),
    (7, 10,  30_000),
]
for S_cfg, Hp_cfg, total_cfg in configs:
    generate_dataset_v9(
        output_name=f'V9_S{S_cfg}_H{Hp_cfg}_{total_cfg//1000}k_5D',
        S_stacks=S_cfg, H_pad=Hp_cfg, total_instances=total_cfg,
        layout_adapter=EnrichedStackMatrix5DAdapter(),
        moves_adapter=DefaultMovesAdapter(),
        max_steps=100, seed=42 + S_cfg,
    )
```

Cada `.data` por separado. La combinación se hace en el pipeline de
entrenamiento (con `preprocessing.dataset.generate_dataset` u otro
mecanismo equivalente).

**Nota**: cada `.data` es internamente coherente (`S` fijo). El modelo
recibe `H_pad` constante en el batch, pero `T_phys` varía por instancia y
queda codificado en `X[i][4]` (V9) o `X[i][1]` (V10), que es lo que
permite al modelo razonar sobre alturas heterogéneas.

---

## 10. Compatibilidad con los modelos

| Modelo | Adapter requerido | Estado |
|---|---|---|
| V8 | `EnrichedStackMatrix4DAdapter` | ✅ compatible (cambiar `layout_adapter`) |
| V9 | `EnrichedStackMatrix5DAdapter` | ✅ compatible (default del notebook) |
| V10 | `TacticalStackMatrixAdapter` | ✅ compatible (cambiar `layout_adapter`) |
| Modelos posteriores | Cualquier `LayoutDataAdapter` | ✅ mientras respeten el contrato `(S == -1)` para padding y deriven `free_spaces` desde `X` |

Puntos críticos verificados:
- Padding de la matriz `S` con `-1` en la dimensión de altura — coincide
  con `(S == -1).all(dim=-1)` que usan V9 y V10 para detectar slots
  vacíos/padeados.
- Para V9: `X[i][4] = T_phys` (altura física por instancia), no `H_pad`.
  Esto es lo que hace que `free_spaces = X[:,:,4] - X[:,:,3]` retorne el
  espacio libre real en la bahía y no el "espacio libre incluyendo el
  padding fantasma".
- Para V10: `X[i][1] = T_phys - len(stack)` (calculado por el adapter
  con `H = T_phys`).
- Pilas vacías (puede pasar tras la caminata): `(S == -1).all(dim=-1).all(dim=-1) = True`
  → `is_origin_empty` correctamente las marca como inválidas.

---

## 11. Tiempos esperados

Estimación para Colab gratuito (2 cores, CPU):

| Etapa | 5k | 50k | 250k |
|---|---|---|---|
| Fase 1 (generación) | ~10 s | ~2 min | ~10 min |
| Fase 2 (resolución, `compute_difficulty=False`) | ~2 min | ~20 min | ~2-3 h |
| Fase 2 (`compute_difficulty=True`) | ~4 min | ~40 min | ~4-6 h |
| Fase 3 (empaquetado) | <5 s | <30 s | ~3 min |

Si el bucket `extremo` se atasca (la traza de Fase 1 muestra que
`remaining[extremo]` baja muy lento), opciones:
- Subir `p_bias` de `0.70` a `0.85` (más agresivo en sesgo).
- Ampliar la `acceptance_band` de `12` a `15`.
- Reducir la cuota del bucket `extremo` (cambiar el `0.10` en
  `BUCKETS[4]`).

---

## 12. Troubleshooting

**`'No se logró sortear params válidos'`**: ocurre si `S_STACKS` y
`H_PAD` son tan pequeños que ningún `(T, F)` cumple las restricciones.
Subir `H_PAD` o `S_STACKS`, o ampliar el rango de `F`.

**`assert blocked == 0` en `build_ordered_layout`**: bug en la
construcción del estado base. No debería ocurrir; si pasa, abrir issue
con el seed y los parámetros de la instancia.

**Tasa de descarte alta en Fase 1 (>50 %)**: significa que la caminata
está saliéndose de los buckets objetivo. Suele pasar si subiste
`p_bias` demasiado y los buckets fáciles se quedan sin candidatos
porque siempre se generan bloqueos. Bajar `p_bias` o ampliar
`acceptance_band`.

**Costo `inf` frecuente en Fase 2**: `MAX_STEPS` demasiado bajo para los
estados generados. Subirlo (e.g., `200`).

**Distribución final por bucket sesgada**: revisar la traza de Fase 1
— si `remaining[extremo]` no llegó a 0, las cuotas no se cumplieron.
Esto pasa cuando el solver C++ falla preferentemente en estados duros.
Mitigación: subir `MAX_STEPS` o relajar la cuota del bucket extremo.

---

## 13. Validación incremental sugerida

Antes de entrenar con un dataset nuevo:

1. **Carga**: abrir el `.data` con `h5py`, verificar shapes y dtypes
   (celda 6 del notebook).
2. **Distribución**: `np.unique(meta_bucket, return_counts=True)` cerca
   de las cuotas.
3. **Sanidad de `Y`**: `Y.sum(axis=1) >= 1` para todas las instancias
   (cada estado tiene al menos un primer movimiento óptimo).
4. **Forward**: cargar V9/V10, hacer un batch del nuevo dataset, ver
   que `model(S, X)` no rompa.
5. **Equivalencia con pipeline antiguo** (opcional, paranoia): para
   instancias con `T = H_PAD` (sin padding extra), las representaciones
   `S` y `X` deben coincidir bit a bit con las del pipeline anterior.
