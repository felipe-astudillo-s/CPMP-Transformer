import copy
import heapq
import time
import torch
import torch.nn.functional as F

from solvers.solver import Solver
from solvers.model import ModelSolver
from cpmp.layout import read_file


class DLTSLDSSolver(Solver):
    """
    Deep Learning Heuristic Tree Search — Limited Discrepancy Search (DLTS-LDS).

    Implementa el Algorithm 2 del paper:
      "Deep learning assisted heuristic tree search for the container
       pre-marshalling problem" (Hottung, Tanaka, Tierney, 2020).

    Explora el árbol por número de discrepancias: primero el camino greedy puro
    (0 desviaciones del modelo), luego caminos con 1 desviación, luego 2, etc.
    Usa una cola de prioridad ordenada por discrepancia (generalized LDS,
    Furcy & Koenig, 2005).

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
        Factor de escala sobre la estimación del bounding model (< 1 = más conservador).
    z : int
        Profundidad hasta la cual la discrepancia no se acumula (permite explorar
        más ramas cerca de la raíz antes de aplicar LDS).
    time_limit : float
        Límite de tiempo en segundos.
    """

    def __init__(self, branching_model, bounding_model=None, bounding_adapter=None,
                 p=0.3, k=3, d=0.8, z=0, time_limit=30.0):
        super().__init__("DLTSLDSSolver")
        self.branching_model = branching_model
        self.bounding_model = bounding_model
        self.bounding_adapter = bounding_adapter
        self.p = p
        self.k = k
        self.d = d
        self.z = z
        self.time_limit = time_limit

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def solve_from_layout(self, layout, H, max_steps):
        """Resuelve a partir de un Layout ya cargado."""
        greedy_solver = ModelSolver(self.branching_model)
        layout_greedy = copy.deepcopy(layout)
        greedy_solved, greedy_steps = greedy_solver.solve_from_layout(
            layout_greedy, H, max_steps
        )
        ub = greedy_steps if greedy_solved else max_steps
        best_steps = ub if greedy_solved else None

        start_time = time.monotonic()
        counter = 0

        # Cola de prioridad: (discrepancy, counter, layout, hlb)
        # hlb = heuristic lower bound heredado del último recálculo
        root = copy.deepcopy(layout)
        Q = []
        heapq.heappush(Q, (0, counter, root, float('-inf')))

        while Q and (time.monotonic() - start_time) < self.time_limit:
            disc, _, state, hlb = heapq.heappop(Q)

            depth = state.steps

            # Nodo completo: actualizar mejor solución
            if state.is_sorted():
                cost = state.steps
                if cost < ub:
                    ub = cost
                    best_steps = cost
                continue

            # Recalcular lower bound heurístico cada k niveles
            if self.bounding_model is not None and depth % self.k == 0:
                estimate = self._get_bounding_estimate(state, H)
                hlb = depth + estimate * self.d

            # Poda: pasos actuales o lower bound ya superan la mejor solución
            if depth >= ub:
                continue
            if hlb >= ub:
                continue

            # Branching: ordenar movimientos por probabilidad
            ranked = self._get_probabilities(state, H)
            if not ranked:
                continue

            r = ranked[0][1]
            threshold = self._mp_constant(r, self.p)

            # Añadir hijos a la cola con su discrepancia acumulada
            for rank, ((src, dst), prob) in enumerate(ranked):
                if prob < threshold:
                    break

                child = copy.deepcopy(state)
                child.move(src, dst)

                # Pasos ya exceden upper bound
                if child.steps >= ub:
                    continue

                # Discrepancia: si depth < z no se acumula (exploración libre en raíz)
                child_disc = disc + (rank if depth >= self.z else 0)

                counter += 1
                heapq.heappush(Q, (child_disc, counter, child, hlb))

        if best_steps is not None:
            return True, best_steps
        if greedy_solved:
            return True, greedy_steps
        return False, max_steps

    def solve_from_path(self, instance_path, H, max_steps):
        layout = read_file(instance_path, H)
        return self.solve_from_layout(layout, H, max_steps)

    # ------------------------------------------------------------------
    # Helpers internos (misma lógica que DLTSDFSSolver)
    # ------------------------------------------------------------------

    def _get_probabilities(self, layout, H):
        S = len(layout.stacks)
        device = next(self.branching_model.parameters()).device

        with torch.no_grad():
            layout_data = list(
                self.branching_model.layout_adapter.layout_2_vec(layout, H)
            )
            for i, val in enumerate(layout_data):
                if isinstance(val, (int, float)):
                    layout_data[i] = torch.tensor([val]).to(device)
                else:
                    layout_data[i] = torch.from_numpy(val).unsqueeze(0).to(device)

            logits = self.branching_model(*layout_data)
            probs = F.softmax(logits, dim=1).squeeze(0)

        result = []
        for idx in range(S * (S - 1)):
            src = idx // (S - 1)
            r   = idx % (S - 1)
            dst = r if r < src else r + 1

            if not layout.stacks[src]:
                continue
            if len(layout.stacks[dst]) >= H:
                continue

            result.append(((src, dst), probs[idx].item()))

        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def _get_bounding_estimate(self, layout, H):
        device = next(self.bounding_model.parameters()).device

        with torch.no_grad():
            data = list(self.bounding_adapter.input_2_vec(layout, H))
            for i, val in enumerate(data):
                if isinstance(val, (int, float)):
                    data[i] = torch.tensor([val]).to(device)
                else:
                    data[i] = torch.from_numpy(val).unsqueeze(0).to(device)

            log_cost = self.bounding_model(*data)
            cost = torch.exp(log_cost).item()

        return cost

    def _mp_constant(self, r, p):
        return r * (1.0 - p)
