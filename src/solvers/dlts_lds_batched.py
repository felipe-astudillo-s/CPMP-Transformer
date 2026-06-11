import copy
import heapq
import time
import torch
import torch.nn.functional as F

from solvers.solver import Solver
from solvers.model import ModelSolver
from cpmp.layout import read_file


class DLTSLDSBatchedSolver(Solver):
    """
    DLTS-LDS con inferencia batched por nivel de discrepancia.

    En lugar de procesar nodos uno a uno (un forward pass por nodo), extrae
    todos los nodos con la misma discrepancia mínima del heap y ejecuta un
    único forward pass del branching model sobre el batch completo. El bounding
    model también se invoca en batch, pero solo sobre los nodos cuyo depth
    cumple depth % k == 0.

    Parámetros
    ----------
    branching_model : CPMPTransformer
        Genera probabilidades sobre movimientos.
    bounding_model : CostPredictorTransformer, opcional
        Predice log(costo restante) para poda heurística.
    bounding_adapter : EnrichedLayoutAdapter, opcional
        Adapter pre-instanciado para el bounding_model.
    p : float
        Parámetro de branch pruning [0, 1]. MP-Constant = r*(1-p).
    k : int
        Cada cuántos niveles de profundidad se recalcula el lower bound.
    d : float
        Factor de escala sobre la estimación del bounding model.
    z : int
        Profundidad hasta la cual la discrepancia no se acumula.
    time_limit : float
        Límite de tiempo en segundos.
    max_disc : int | None
        Discrepancia máxima a explorar. None = sin límite (comportamiento
        equivalente al original). Valores típicos: 3-5.
    batch_size : int
        Máximo de layouts por forward pass dentro de un nivel de discrepancia.
    """

    def __init__(self, branching_model, bounding_model=None, bounding_adapter=None,
                 p=0.3, k=3, d=0.8, z=0, time_limit=30.0,
                 max_disc=None, batch_size=256):
        super().__init__("DLTSLDSBatchedSolver")
        self.branching_model = branching_model
        self.bounding_model = bounding_model
        self.bounding_adapter = bounding_adapter
        self.p = p
        self.k = k
        self.d = d
        self.z = z
        self.time_limit = time_limit
        self.max_disc = max_disc
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def solve_from_layout(self, layout, H, max_steps):
        # Warm-start greedy para obtener upper bound inicial
        greedy_solver = ModelSolver(self.branching_model)
        layout_greedy = copy.deepcopy(layout)
        greedy_solved, greedy_steps = greedy_solver.solve_from_layout(
            layout_greedy, H, max_steps
        )
        ub = greedy_steps if greedy_solved else max_steps
        best_steps = ub if greedy_solved else None

        start_time = time.monotonic()
        counter = 0

        root = copy.deepcopy(layout)
        Q = []
        # Tupla del heap: (disc, counter, depth, state, hlb)
        # depth se almacena explícitamente para no acceder a state.steps durante el drain
        heapq.heappush(Q, (0, counter, root.steps, root, float('-inf')))

        while Q and (time.monotonic() - start_time) < self.time_limit:
            current_disc = Q[0][0]

            if self.max_disc is not None and current_disc >= self.max_disc:
                break

            # Extraer todos los nodos con discrepancia == current_disc
            level_nodes = []
            while Q and Q[0][0] == current_disc:
                disc, _, depth, state, hlb = heapq.heappop(Q)
                level_nodes.append((depth, state, hlb))

            # Pre-filtrar: nodos terminados y poda obvia por depth/hlb
            to_process = []
            for depth, state, hlb in level_nodes:
                if state.is_sorted():
                    cost = state.steps
                    if cost < ub:
                        ub = cost
                        best_steps = cost
                    continue
                if depth >= ub or hlb >= ub:
                    continue
                # Lista mutable para poder actualizar hlb tras bounding batch
                to_process.append([depth, state, hlb])

            if not to_process:
                continue

            # Procesar en chunks de tamaño <= batch_size
            for chunk_start in range(0, len(to_process), self.batch_size):
                if time.monotonic() - start_time >= self.time_limit:
                    break

                chunk = to_process[chunk_start: chunk_start + self.batch_size]

                # --- 1. Bounding batch para nodos que lo necesitan ---
                if self.bounding_model is not None:
                    needs_bound_idx = [
                        i for i, (depth, _, _hlb) in enumerate(chunk)
                        if depth % self.k == 0
                    ]
                    if needs_bound_idx:
                        bound_layouts = [chunk[i][1] for i in needs_bound_idx]
                        estimates = self._get_bounding_estimate_batch(bound_layouts, H)
                        for list_pos, chunk_idx in enumerate(needs_bound_idx):
                            depth = chunk[chunk_idx][0]
                            chunk[chunk_idx][2] = depth + estimates[list_pos] * self.d

                # --- 2. Re-filtrar con hlb posiblemente actualizado ---
                chunk = [node for node in chunk if node[2] < ub and node[0] < ub]
                if not chunk:
                    continue

                # --- 3. Batch branching inference ---
                layouts_for_branch = [node[1] for node in chunk]
                ranked_list = self._get_probabilities_batch(layouts_for_branch, H)

                # --- 4. Generar hijos y pushear al heap ---
                for (depth, state, hlb), ranked in zip(chunk, ranked_list):
                    if not ranked:
                        continue

                    r = ranked[0][1]
                    threshold = self._mp_constant(r, self.p)

                    for rank, ((src, dst), prob) in enumerate(ranked):
                        if prob < threshold:
                            break

                        child = copy.deepcopy(state)
                        child.move(src, dst)

                        child_steps = child.steps
                        if child_steps >= ub:
                            continue

                        child_disc = current_disc + (rank if depth >= self.z else 0)

                        if self.max_disc is not None and child_disc >= self.max_disc:
                            continue

                        counter += 1
                        heapq.heappush(Q, (child_disc, counter, child_steps, child, hlb))

        if best_steps is not None:
            return True, best_steps
        if greedy_solved:
            return True, greedy_steps
        return False, max_steps

    def solve_from_path(self, instance_path, H, max_steps):
        layout = read_file(instance_path, H)
        return self.solve_from_layout(layout, H, max_steps)

    # ------------------------------------------------------------------
    # Inferencia batch
    # ------------------------------------------------------------------

    def _get_probabilities_batch(self, layouts, H):
        """Forward pass batched del branching model sobre una lista de layouts.

        Retorna una lista de listas de ((src, dst), prob) ordenadas por prob desc,
        una por layout, con solo movimientos válidos.
        """
        device = next(self.branching_model.parameters()).device

        batch_data_lists = []
        for layout in layouts:
            ld = list(self.branching_model.layout_adapter.layout_2_vec(layout, H))
            for j in range(len(ld)):
                val = ld[j]
                if isinstance(val, (int, float)):
                    ld[j] = torch.tensor([val]).to(device)
                else:
                    ld[j] = torch.from_numpy(val).unsqueeze(0).to(device)
            batch_data_lists.append(ld)

        n_args = len(batch_data_lists[0])
        batched = [
            torch.cat([bd[j] for bd in batch_data_lists], dim=0)
            for j in range(n_args)
        ]

        with torch.no_grad():
            logits = self.branching_model(*batched)      # (N, S*(S-1))
            probs_batch = F.softmax(logits, dim=1)

        results = []
        for b_idx, layout in enumerate(layouts):
            S = len(layout.stacks)
            probs = probs_batch[b_idx]
            ranked = []
            for idx in range(S * (S - 1)):
                src = idx // (S - 1)
                r   = idx % (S - 1)
                dst = r if r < src else r + 1
                if not layout.stacks[src]:
                    continue
                if len(layout.stacks[dst]) >= H:
                    continue
                ranked.append(((src, dst), probs[idx].item()))
            ranked.sort(key=lambda x: x[1], reverse=True)
            results.append(ranked)

        return results

    def _get_bounding_estimate_batch(self, layouts, H):
        """Forward pass batched del bounding model sobre una lista de layouts.

        Retorna una lista de floats con el costo estimado restante para cada layout.
        """
        device = next(self.bounding_model.parameters()).device

        batch_data_lists = []
        for layout in layouts:
            data = list(self.bounding_adapter.input_2_vec(layout, H))
            for j in range(len(data)):
                val = data[j]
                if isinstance(val, (int, float)):
                    data[j] = torch.tensor([val]).to(device)
                else:
                    data[j] = torch.from_numpy(val).unsqueeze(0).to(device)
            batch_data_lists.append(data)

        n_args = len(batch_data_lists[0])
        batched = [
            torch.cat([bd[j] for bd in batch_data_lists], dim=0)
            for j in range(n_args)
        ]

        with torch.no_grad():
            log_costs = self.bounding_model(*batched)    # (N,) o (N,1)
            costs = torch.exp(log_costs).view(-1)        # siempre (N,)

        return costs.tolist()

    def _mp_constant(self, r, p):
        return r * (1.0 - p)
