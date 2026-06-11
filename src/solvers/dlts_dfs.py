import copy
import time
import torch
import torch.nn.functional as F

from solvers.solver import Solver
from solvers.model import ModelSolver
from cpmp.layout import read_file


class DLTSDFSSolver(Solver):
    """
    Deep Learning Heuristic Tree Search — Depth First Search (DLTS-DFS).

    Implementa el Algorithm 1 del paper:
      "Deep learning assisted heuristic tree search for the container
       pre-marshalling problem" (Hottung, Tanaka, Tierney, 2020).

    Parámetros
    ----------
    branching_model : CPMPTransformer
        Modelo entrenado que genera logits sobre movimientos (branching DNN).
    bounding_model : CostPredictorTransformer
        Modelo entrenado que predice log(costo restante) (bounding DNN).
    bounding_adapter : EnrichedLayoutAdapter
        Adapter pre-instanciado con S_max y H_max compatibles con bounding_model.
    p : float
        Parámetro de branch pruning [0, 1].
        p=0 → greedy puro (solo el movimiento más probable).
        p=1 → explorar todos los movimientos válidos.
    k : int
        Cada cuántos niveles de profundidad se recalcula el lower bound heurístico.
    d : float
        Factor de escala sobre la estimación del bounding model.
        Valores < 1 hacen el bound más conservador (menos poda, más seguro).
    time_limit : float
        Límite de tiempo de búsqueda en segundos.
    """

    def __init__(self, branching_model, bounding_model, bounding_adapter,
                 p=0.3, k=3, d=1.0, time_limit=30.0):
        super().__init__("DLTSDFSSolver")
        self.branching_model = branching_model
        self.bounding_model = bounding_model
        self.bounding_adapter = bounding_adapter
        self.p = p
        self.k = k
        self.d = d
        self.time_limit = time_limit

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def solve_from_layout(self, layout, H, max_steps):
        """Resuelve a partir de un Layout ya cargado."""
        # Warm-start greedy para obtener ub_ini
        greedy_solver = ModelSolver(self.branching_model)
        layout_greedy = copy.deepcopy(layout)
        greedy_solved, greedy_steps = greedy_solver.solve_from_layout(
            layout_greedy, H, max_steps
        )

        ub_ini = greedy_steps if greedy_solved else max_steps
        ub = [ub_ini]
        start_time = time.monotonic()

        best_steps = self._dlts_dfs(
            layout=copy.deepcopy(layout),
            H=H,
            hlb=0.0,
            ub=ub,
            k=self.k,
            d=self.d,
            p=self.p,
            start_time=start_time,
            time_limit=self.time_limit,
            last_move=None,
        )

        if best_steps is not None:
            return True, best_steps
        if greedy_solved:
            return True, greedy_steps
        return False, max_steps

    def solve_from_path(self, instance_path, H, max_steps):
        layout = read_file(instance_path, H)
        return self.solve_from_layout(layout, H, max_steps)

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _get_probabilities(self, layout, H):
        """
        Devuelve lista de ((src, dst), prob) ordenada descendente por probabilidad.
        Solo incluye movimientos válidos (src no vacío, dst no lleno).
        """
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

            logits = self.branching_model(*layout_data)       # [1, S*(S-1)]
            probs = F.softmax(logits, dim=1).squeeze(0)       # [S*(S-1)]

        result = []
        for idx in range(S * (S - 1)):
            src = idx // (S - 1)
            r   = idx % (S - 1)
            dst = r if r < src else r + 1

            if not layout.stacks[src]:           # src vacío
                continue
            if len(layout.stacks[dst]) >= H:     # dst lleno
                continue

            result.append(((src, dst), probs[idx].item()))

        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def _get_bounding_estimate(self, layout, H):
        """
        Llama al bounding model y devuelve el costo restante estimado.
        El modelo predice log(costo), por lo que se aplica exp() al output.
        """
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
        """MP-Constant: umbral de poda = r * (1 - p)."""
        return r * (1.0 - p)

    # ------------------------------------------------------------------
    # Algoritmo 1 del paper: DLTS-DFS
    # ------------------------------------------------------------------

    def _dlts_dfs(self, layout, H, hlb, ub, k, d, p,
                  start_time, time_limit, last_move=None):
        """
        Búsqueda DFS recursiva guiada por DNNs.

        Parámetros
        ----------
        layout    : Layout actual (copia independiente por llamada)
        H         : Altura máxima del bay
        hlb       : Heuristic lower bound heredado del padre
        ub        : Lista mutable [upper_bound] compartida entre llamadas
        k, d, p   : Parámetros de búsqueda
        start_time: Tiempo de inicio (time.monotonic())
        time_limit: Límite en segundos
        last_move : (src, dst) del movimiento anterior (para evitar reversas)

        Retorna
        -------
        int con el mejor costo encontrado bajo este nodo, o None si se podó.
        """
        # 1. Terminal: solución encontrada
        if layout.is_sorted():
            cost = layout.steps
            if cost < ub[0]:
                ub[0] = cost
            return cost

        # 2. Recalcular lower bound heurístico cada k niveles
        if layout.steps % k == 0:
            bounding_estimate = self._get_bounding_estimate(layout, H)
            hlb = layout.steps + bounding_estimate * d

        # 3. Condiciones de poda
        if layout.steps >= ub[0]:
            return None
        if hlb >= ub[0]:
            return None
        if time.monotonic() - start_time >= time_limit:
            return None

        # 4. Branching: construir conjunto B de sucesores a explorar
        ranked = self._get_probabilities(layout, H)
        if not ranked:
            return None

        r = ranked[0][1]
        threshold = self._mp_constant(r, p)
        B = [(move, prob) for move, prob in ranked if prob >= threshold]

        # 5. DFS recursivo sobre B (orden descendente de probabilidad)
        best = None
        for (src, dst), _ in B:
            # Evitar reversa inmediata del movimiento anterior
            if last_move is not None and src == last_move[1] and dst == last_move[0]:
                continue

            child = copy.deepcopy(layout)
            child.move(src, dst)

            result = self._dlts_dfs(
                layout=child,
                H=H,
                hlb=hlb,
                ub=ub,
                k=k,
                d=d,
                p=p,
                start_time=start_time,
                time_limit=time_limit,
                last_move=(src, dst),
            )

            if result is not None and (best is None or result < best):
                best = result

        return best
