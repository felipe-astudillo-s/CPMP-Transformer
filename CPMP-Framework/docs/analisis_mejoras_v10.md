# Análisis de arquitectura y features — CPMP Transformer V10

Este documento recoge un análisis detallado del modelo V10 y del adaptador
`TacticalStackMatrixAdapter` recién incorporado. El objetivo es identificar
oportunidades concretas de mejora en precisión, tanto a nivel de
**features** (información que recibe el modelo) como de **arquitectura**
(cómo la consume). Se prioriza por relación impacto/esfuerzo.

No se propone reemplazar V10 — sólo listar cambios acumulativos que se
pueden validar de manera incremental.

---

## 1. Resumen ejecutivo

| # | Propuesta | Categoría | Impacto esperado | Esfuerzo |
|---|-----------|-----------|------------------|----------|
| 1 | Arreglar máscara de stacks fantasma vs. vacíos reales en `inter_stack_attention` | Arquitectura (bug) | **Alto** | Bajo |
| 2 | Normalizar por `num_groups` en vez de `max_val` local del layout | Feature | **Alto** | Bajo |
| 3 | Exponer `reachable_height(i)` como feature (ya está computado, no se usa) | Feature | **Alto** | Bajo |
| 4 | Inyectar `Top Move Cost` y compatibilidad de tops como **bias de atención** en la matriz de logits | Arquitectura | Alto | Medio |
| 5 | Añadir features de profundidad: posición del min y del top-misplaced | Feature | Medio-Alto | Bajo |
| 6 | FiLM conditioning: modular embeddings de contenedor con X antes de intra-attention | Arquitectura | Medio-Alto | Medio |
| 7 | Cabezal de scoring multi-cabeza (bilinear multi-head en lugar de 1 head) | Arquitectura | Medio | Bajo |
| 8 | Token global CLS sobre toda la pizarra, inyectado a cada pila | Arquitectura | Medio | Medio |
| 9 | Pérdida auxiliar: predecir misplaced count / reachable height | Entrenamiento | Medio | Medio |
| 10 | Augmentation por permutación de pilas | Entrenamiento | Medio | Bajo |
| 11 | Sustituir `-1e4` por `torch.finfo(dtype).min` en masking | Numérico | Bajo | Bajo |
| 12 | Eliminar redundancia entre Sorted Status y Top Move Cost=0 | Feature | Bajo | Bajo |

Las propuestas 1-3 son las que más probabilidad tienen de dar ganancia
directa sin coste arquitectónico. Las 4-6 son los cambios con mayor techo.

---

## 2. Análisis de features (`TacticalStackMatrixAdapter`)

### 2.1 Revisión dimensión por dimensión

| Dim | Nombre | Rango | Observaciones |
|-----|--------|-------|---------------|
| 0 | Sorted Status | {0.0, 1.0} | Redundante con (Misplaced Count == 0) AND (len > 0). Útil como atajo, pero introduce colinealidad. |
| 1 | Free Space | [0, H] | Valor absoluto. Usado directamente como máscara de destino. Crucial. |
| 2 | Misplaced Count | [0, H-1] | Valor absoluto. Señala "cuánto trabajo" tiene la pila. |
| 3 | Top Element | [0, 1] o 1.0 | Normalizado por `max_val` del layout (ver problema abajo). |
| 4 | Top Move Cost | {0.0, 1.0, 2.0} | Señal táctica fuerte pero sólo alimenta la fusión — no influye el scoring. |

### 2.2 Problemas detectados

#### P1. Normalización inestable por `max_val` local

`get_X` y `StackMatrix4DAdapter` normalizan valores con
`max_val = max(valores_del_layout)`. Esto significa que **dos estados con
estructura idéntica pero distinto conjunto de grupos se verán distintos
para el modelo**:

```
Estado A: [[3,2,1],[5,4]]  -> max_val = 5  -> top_de_1 = 0.2
Estado B: [[3,2,1],[8,7]]  -> max_val = 8  -> top_de_1 = 0.125
```

La prioridad relativa no cambió, pero la señal sí. Con datasets que mezclan
instancias de distinto número de grupos esto genera ruido. Peor aún: tras
cada movimiento, si el grupo de máxima prioridad sale del tablero, todo se
re-normaliza.

**Fijar**: normalizar por `num_groups` de la instancia (constante en toda
la trayectoria) o usar rangos ordinales. `num_groups` se puede pasar al
adaptador a través del `Layout` (añadir atributo) o derivar como
`max(todos los valores vistos durante la generación)`.

#### P2. `reachable_height()` existe pero no se expone

`src/cpmp/layout.py:215` define `reachable_height(layout, i)` — calcula
hasta qué altura una pila ordenada puede crecer consumiendo estorbos de
otras pilas. Es una cota superior de "calidad de destino" mucho más rica
que `Free Space`: una pila con `free=4` pero `reachable=1` es mucho peor
destino que una con `free=4` y `reachable=4`.

**Coste**: cero conceptual, ya está implementado. Sólo hay que añadirlo al
vector X.

#### P3. Redundancias parcialmente útiles

- `Sorted Status == 1.0` es casi equivalente a `Top Move Cost == 0.0`.
- `Sorted Status == 0.0 AND len == 0` es equivalente a `Top Element == 1.0`.

No son bugs, pero la colinealidad penaliza velocidad de entrenamiento y
deja parámetros "huérfanos" en `x_projection`. Se podría consolidar o
mantener pero con claridad de roles.

#### P4. Falta señal de *profundidad del problema*

El Misplaced Count dice "cuántos estorbos hay", pero no dice **dónde están**.
No es lo mismo:

```
Pila A: [10, 5, 3, 100]  -> 1 misplaced en la cima (fácil, 1 movimiento)
Pila B: [100, 5, 3, 2]   -> 3 misplaced enterrados (difícil, hay que desarmar)
```

Ambas tienen misplaced_count = diferente, pero incluso cuando coinciden,
la ubicación del estorbo es clave. Una sola métrica complementaria lo
resuelve:

- `depth_of_deepest_misplaced` — posición (0 = cima) del estorbo más
  profundo. Es la cota inferior del *número de movimientos* necesarios para
  limpiar la pila sin ayuda externa.

#### P5. Falta información sobre el contenedor más urgente

El contenedor con valor mínimo (mayor prioridad en la convención actual) de
la pila es el "destino final" del trabajo en esa pila. Saber:

- `min_value` de cada pila (normalizado consistentemente)
- `min_depth` — cuántos contenedores lo tapan

Le permite al modelo razonar sobre prioridad global: "el min de la pila 3
es el más urgente del tablero y está enterrado bajo 4 contenedores".

#### P6. Top Element normalizado pierde cardinalidad cuando se usa como bias

El modelo aprende comparaciones top_origen vs top_destino a partir de
embeddings. Si normalizamos por `max_val` local, la comparación **numérica**
sigue siendo coherente dentro del mismo estado, pero inestable *entre*
estados. Mejor normalizar por `num_groups` fijo de la instancia.

### 2.3 Mejoras propuestas en features (resumen)

Ampliar el vector X de 5 a 8 dimensiones, preservando las 5 existentes:

```
[0] Sorted Status                 (igual)
[1] Free Space                    (igual)
[2] Misplaced Count               (igual)
[3] Top Element                   (NORMALIZADO por num_groups, no max_val)
[4] Top Move Cost                 (igual)
[5] Reachable Height              (NUEVA — usar layout.reachable_height)
[6] Depth of Deepest Misplaced    (NUEVA — 0 si no hay misplaced)
[7] Min Value / Min Depth         (NUEVA — combinable, p.ej. prioridad * profundidad)
```

Además, añadir un flag explícito de `is_real_stack` (6ª dimensión siempre
1.0 para pilas reales, 0.0 para fantasmas de padding). Esto desambigua el
problema P1 de la sección 3.2 (máscara de fantasma en inter_stack).

Implementación: crear un `TacticalStackMatrixAdapterV2` hermano, sin tocar
V1. Patrón idéntico al que se usó entre V7 y V10.

---

## 3. Análisis de arquitectura (`cpmp_transformer_v10.py`)

### 3.1 Flujo resumido

```
S (B, S_len, H, 2)  →  input_projection  →  where(padding, empty_embed, x)
                    →  pos_encoder       →  prepend CLS
                    →  intra_stack_attention (self-attn sobre altura)
                    →  tomar CLS         →  stack_vertical_info (B, S_len, d)

X (B, S_len, 5)     →  x_projection      →  (B, S_len, d)

concat(CLS, xX)     →  fusion_layer      →  LayerNorm
                    →  inter_stack_attention
                    →  origin_proj / dest_proj
                    →  matmul escalado   →  masking  →  flatten  →  logits
```

### 3.2 Problemas detectados

#### P1. Máscara de fantasmas borra a las pilas vacías reales

`src/models/cpmp_transformer_v10.py:103` hace:

```python
stack_padding_mask = (S == -1).all(dim=-1).all(dim=-1)
x_global = self.inter_stack_attention(stack_embeddings,
                                       src_key_padding_mask=stack_padding_mask)
```

El propósito es enmascarar stacks fantasma introducidos por
`pad_batch_collate` (`src/training/training.py:91`). Sin embargo, **una
pila vacía real también tiene todas sus celdas en -1**, así que la máscara
también la enmascara.

Efecto: las pilas vacías reales **no son atendidas** por las demás durante
`inter_stack_attention`. Sus embeddings sí se actualizan (sus queries
atacan), pero su información nunca llega a las otras. En CPMP, una pila
vacía es el **destino más deseable** porque acepta cualquier top.

Solución directa: el collate rellena X con 0.0 para fantasmas. Para pilas
reales, `X[1] > 0` (free space) si hay espacio, o `X[3] > 0` (top
normalizado, o 1.0 si vacía). Basta con:

```python
# Fantasma = X es todo cero (collate rellena con 0.0)
stack_padding_mask = (X.abs().sum(dim=-1) == 0)
```

O, mucho más limpio, añadir `is_real_stack` como 6ª dimensión de X (1.0
para reales, 0.0 para padding):

```python
stack_padding_mask = (X[:, :, IS_REAL_IDX] == 0)
```

Este es probablemente el bug con mayor impacto directo en precisión actual.

#### P2. `Top Move Cost` sólo entra por fusión, no modula el scoring

El scoring es `logits[i,j] = (q_i · k_j) / sqrt(d)`, donde `q_i` viene de
`origin_proj(x_global)` y `k_j` de `dest_proj(x_global)`. La información
táctica de X ya pasó por `fusion_layer`, pero **la geometría específica
origen→destino nunca recibe bias explícito**.

Mejora: añadir un bias de atención a la matriz de logits con señales
tácticas par-a-par computables directamente desde X:

```python
top_i = X[:, :, TOP_IDX].unsqueeze(2)    # (B, S_len, 1)
top_j = X[:, :, TOP_IDX].unsqueeze(1)    # (B, 1, S_len)
# Bonus si top_j >= top_i (movimiento limpio disponible)
clean_move_bias = (top_j >= top_i).float() * self.clean_bonus  # parámetro aprendible

# Penalización si la pila origen ya está ordenada (cost_i == 0)
cost_i = X[:, :, COST_IDX].unsqueeze(2)
already_sorted_penalty = (cost_i == 0).float() * self.sorted_penalty

logits_matrix = logits_matrix + clean_move_bias - already_sorted_penalty
```

Estos bias son *aprendibles* (no forzados) — el modelo decide cuánto
apoyarse en ellos. Equivale a darle al decoder un *prior* táctico fuerte
sin quitarle capacidad de sobrescribirlo.

#### P3. La fusión es un cuello de botella lineal

`fusion_layer = Linear(d_model * 2, d_model)` concatena el CLS con
`x_projection(X)` y lo comprime con una única capa afín. Cuando X contiene
señales fuertes (como Top Move Cost), una sola proyección afín puede no
separar bien clases.

Alternativas (impacto medio, esfuerzo bajo):

1. **MLP 2 capas con residual:**
   ```python
   self.fusion = nn.Sequential(
       nn.Linear(d_model * 2, d_model * 4),
       nn.GELU(),
       nn.Dropout(p),
       nn.Linear(d_model * 4, d_model),
   )
   # x = stack_vertical_info + self.fusion(concat(...))
   ```
2. **Gating (GLU):**
   ```python
   gate = torch.sigmoid(self.gate_proj(x_external_info))
   stack_embeddings = stack_vertical_info * gate + x_external_info * (1 - gate)
   ```

#### P4. El vector X no influye en la atención intra-pila

Hoy X sólo entra tras la agregación por CLS. Pero X describe propiedades
globales de la pila (ordenada, misplaced, etc.) que podrían modular cómo
el CLS agrega las posiciones. Una **capa FiLM** resuelve esto sin aumentar
parámetros notoriamente:

```python
# FiLM: condicionar intra-stack embeddings con X
gamma = self.film_gamma(X)   # (B, S_len, d_model)
beta  = self.film_beta(X)    # (B, S_len, d_model)
# Expandir a (B*S_len, 1, d_model) y aplicar a x antes del encoder
x = x * (1 + gamma_flat) + beta_flat
```

Así cada pila entra al intra-attention con una "coloración" táctica
específica.

#### P5. Scoring de cabezal único

`q · k` es una forma bilinear simple de rango 1 por batch. Para capturar
simultáneamente distintos criterios (limpieza, altura alcanzable,
compatibilidad de grupos) conviene **scoring multi-cabeza**:

```python
# En lugar de una proyección:
self.origin_proj = nn.Linear(d_model, d_model)
self.dest_proj   = nn.Linear(d_model, d_model)
# Usar varias cabezas y promediar:
self.n_score_heads = 4
self.origin_heads = nn.Linear(d_model, d_model)  # d_model = n_score_heads * d_head
self.dest_heads   = nn.Linear(d_model, d_model)
# logits_matrix_multi = sum_h (q_h · k_h) / sqrt(d_head)
```

Equivalente a añadir una capa de cross-attention explícita entre origins y
destinations. Ya existe `CrossAttentionBlock` en `src/models/base/
attention.py` — se puede reutilizar.

#### P6. El CLS token no ve el estado global

Cada pila se codifica independientemente (intra-stack). El estado global
sólo emerge de `inter_stack_attention`. Para acciones que dependen del
balance global del tablero (ej. "¿hay alguna pila vacía disponible?") el
modelo tiene que "descubrirlo" por atención. Un **token global GLOBAL**,
prepended en la etapa inter-stack, da una representación reciclable:

```python
global_token = self.global_token.expand(B, 1, d_model)
stack_embeddings_ext = torch.cat([global_token, stack_embeddings], dim=1)
x_global_ext = self.inter_stack_attention(stack_embeddings_ext, ...)
# Descartar el global en la salida
x_global = x_global_ext[:, 1:, :]
```

Y opcionalmente pasar `x_global_ext[:, 0, :]` (el global refinado) como
bias aditivo a cada pila antes del scoring.

#### P7. Valor `-1e4` en `masked_fill`

Con mixed-precision en FP16 (el entrenamiento usa autocast) `-1e4` es
seguro, pero marginal: `softmax([logit_valido ~ +5, -1e4])` todavía puede
filtrar ~0.0001 de probabilidad a acciones inválidas. Con batches grandes,
esto se nota en la pérdida. Usar:

```python
MASK_VALUE = torch.finfo(logits_matrix.dtype).min
logits_matrix = logits_matrix.masked_fill(invalid_action_mask, MASK_VALUE)
```

`finfo(float16).min ≈ -6.5e4`, `finfo(bfloat16).min ≈ -3.4e38`. Más seguro
en ambos regímenes.

#### P8. Sin augmentation por permutación

El modelo es **permutation-equivariant por construcción** en el eje de
pilas (inter_stack_attention no tiene positional encoding sobre el eje
de stacks). Eso es un regalo: la permutación de pilas es una augmentation
gratis que multiplica efectivamente el dataset por `S_len!` sin coste de
etiqueta.

En `pad_batch_collate` basta con añadir (opcional, modo entrenamiento):

```python
perm = torch.randperm(S_len)
S = S[perm]
X = X[perm]
Y_mat = Y_mat[perm][:, perm]  # permutar filas y columnas antes del flatten
```

Suele dar 1-3 puntos de accuracy en problemas de este tipo.

#### P9. Sin pérdida auxiliar

Las dimensiones X[2] (misplaced_count), X[4] (top_move_cost) y la futura
X[5] (reachable_height) son **labels estructuradas conocidas** que el
modelo podría predecir como tarea auxiliar:

```python
# Además del output principal (logits sobre movimientos):
aux_misplaced = self.aux_head_misplaced(stack_embeddings)    # regresión
aux_reachable = self.aux_head_reachable(stack_embeddings)    # regresión
# loss total = CE_principal + lambda_aux * (MSE_misplaced + MSE_reachable)
```

Efecto: ancla las representaciones de pila a cantidades interpretables y
acelera la convergencia. Lambda típica: 0.1-0.3. No tiene coste de
inferencia (sólo en training, las heads auxiliares no se usan).

### 3.3 Otros detalles menores

- `cls_token` y `empty_embed` se inicializan con `torch.randn` (var=1).
  Conviene inicializar con `nn.init.trunc_normal_(std=0.02)` — estándar en
  transformers modernos.
- `dropout=0.1` nunca se ha tuneado: probar {0.05, 0.15, 0.2}.
- `ff_dim_multiplier=4` es razonable. Si se sube `d_model`, 2 suele bastar.

---

## 4. Plan incremental sugerido (4 etapas)

El orden respeta impacto esperado y aislamiento de cada cambio (cada fase
es testeable por separado contra V10 de baseline).

### Fase 1 — Bug fixes + normalización estable (1 PR)

- **F1.1** Añadir `is_real_stack` como 6ª dimensión en un nuevo
  `TacticalStackMatrixAdapterV2`. Usar esa columna (no la matriz S) para
  construir `stack_padding_mask` en el modelo.
- **F1.2** Cambiar `-1e4` por `torch.finfo(dtype).min`.
- **F1.3** Normalizar `Top Element` por `num_groups` (constante por
  instancia).

Output: modelo V11 = V10 + fixes. Entrenar y medir diferencia.

### Fase 2 — Features tácticas adicionales (1 PR)

- **F2.1** Añadir `reachable_height` (dim 7) al adaptador.
- **F2.2** Añadir `depth_of_deepest_misplaced` (dim 8).
- **F2.3** Añadir `min_depth` o `min_priority_score` (dim 9).

Output: V12 = V11 + features extra.

### Fase 3 — Scoring y bias táctico (1 PR)

- **F3.1** Añadir bias aprendible de compatibilidad de tops a la matriz de
  logits.
- **F3.2** Opcional: reemplazar el scoring bilinear actual por un
  `CrossAttentionBlock` (ya existe).

Output: V13.

### Fase 4 — Capacidad arquitectónica y training (1 PR)

- **F4.1** FiLM conditioning sobre intra-stack.
- **F4.2** MLP fusion con residual.
- **F4.3** Augmentation por permutación en `pad_batch_collate`.
- **F4.4** Pérdida auxiliar (misplaced / reachable) con `lambda=0.2`.

Output: V14.

---

## 5. Experimentos para validar

Cada fase debe validarse contra **la misma instancia V10 ya entrenada**
(no sobre checkpoint aleatorio). Métricas a reportar:

1. Accuracy top-1 del movimiento óptimo (elección estricta).
2. Accuracy top-k (k=3) — útil dada la existencia de movimientos óptimos
   empatados.
3. Fracción de estados donde el modelo eligió un movimiento **sucio
   innecesariamente** (cuando existía un limpio). Métrica táctica directa.
4. Fracción de estados donde el modelo eligió el movimiento correcto
   cuando `Top Move Cost == 2` (movimiento sucio forzado). Prueba que el
   bias táctico funciona.
5. Longitud promedio de solución ejecutada vs. óptimo conocido — si hay
   simulador/roll-out disponible.

Idealmente, además, se reporta **accuracy desagregada por tamaño de
instancia (S, H) y por dificultad (misplaced_count global inicial)** para
detectar si las mejoras benefician sobre todo a las instancias fáciles o
también a las duras.

---

## 6. Referencias al código actual

- Adaptador actual: `src/generation/adapters.py:256` (clase
  `TacticalStackMatrixAdapter`).
- Modelo V10: `src/models/cpmp_transformer_v10.py:25` (clase
  `CPMPTransformer`).
- Máscara de fantasma (bug P1): `src/models/cpmp_transformer_v10.py:103`.
- `pad_batch_collate`: `src/training/training.py:91`.
- `reachable_height` (sin usar): `src/cpmp/layout.py:215`.
- Loss + targets normalizados: `src/training/metrics.py:69`.
- Bloques de atención reutilizables: `src/models/base/attention.py`.

---

## 7. Notas finales

- Ninguna mejora listada requiere romper V10. Todas se materializan como
  nuevas clases/archivos siguiendo el patrón establecido
  (`V10` → `V11` → …, `TacticalStackMatrixAdapter` → `…V2` → …).
- Las ganancias más probables están en la Fase 1 (bug de máscara y
  normalización estable) antes que en cambios arquitectónicos.
- La Fase 4, aunque más ambiciosa, es la que fija el techo del modelo si
  se pretende escalar a instancias más grandes.
