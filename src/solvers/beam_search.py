import copy
import time
import torch
import torch.nn.functional as F

from solvers.solver import Solver
from cpmp.layout import read_file


class BeamSearchSolver(Solver):
    """
    Beam Search guiado por el branching DNN (solo usa el branching model).

    Mantiene un beam de W layouts parciales. En cada paso expande todos los
    estados del beam con sus top-E movimientos más probables según el modelo,
    y conserva los W candidatos con mayor probabilidad acumulada (log-sum).

    Solo usa el branching model — sin bounding model — para aislar el efecto
    de la estrategia de búsqueda respecto a DLTS-DFS.

    Parámetros
    ----------
    branching_model : CPMPTransformer
        Genera probabilidades sobre movimientos.
    beam_width : int
        Número de estados a mantener en el beam (W).
    expansions_per_state : int
        Top-E movimientos a explorar por estado en cada expansión.
        Si es None, se usan todos los movimientos válidos.
    time_limit : float
        Límite de tiempo en segundos.
    """

    def __init__(self, branching_model, beam_width=5,
                 expansions_per_state=None, time_limit=30.0):
        super().__init__("BeamSearchSolver")
        self.branching_model = branching_model
        self.beam_width = beam_width
        self.expansions_per_state = expansions_per_state
        self.time_limit = time_limit

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def solve_from_layout(self, layout, H, max_steps):
        start_time = time.monotonic()

        # Beam: lista de (log_prob_acumulada, layout)
        beam = [(0.0, copy.deepcopy(layout))]
        visited = {tuple(tuple(s) for s in layout.stacks)}

        while beam:
            if time.monotonic() - start_time >= self.time_limit:
                break

            for _, state in beam:
                if state.is_sorted():
                    return True, state.steps

            if beam[0][1].steps >= max_steps:
                break

            candidates = []
            for log_prob, state in beam:
                ranked = self._get_probabilities(state, H)
                moves = ranked[:self.expansions_per_state]  # None → todos

                for (src, dst), prob in moves:
                    child = copy.deepcopy(state)
                    child.move(src, dst)
                    key = tuple(tuple(s) for s in child.stacks)
                    if key not in visited:
                        visited.add(key)
                        # Acumular log-probabilidad: mayor = mejor camino
                        candidates.append((log_prob + (prob + 1e-12), child))

            if not candidates:
                break

            # Conservar los W de mayor log-prob acumulada
            candidates.sort(key=lambda x: x[0], reverse=True)
            beam = candidates[:self.beam_width]

        return False, max_steps

    def solve_from_path(self, instance_path, H, max_steps):
        layout = read_file(instance_path, H)
        return self.solve_from_layout(layout, H, max_steps)

    # ------------------------------------------------------------------
    # Helper
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
