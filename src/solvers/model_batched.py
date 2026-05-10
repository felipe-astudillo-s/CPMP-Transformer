import os
import torch
from solvers.solver import Solver
from cpmp.layout import read_file
from generation.adapters import *
from settings import INSTANCE_FOLDER


class BatchedModelSolver(Solver):
    """
    Versión batched de ModelSolver.

    En lugar de resolver instancias una a una (un forward pass por paso por instancia),
    agrupa varias instancias activas en un solo forward pass por paso, amortizando
    el costo fijo del transformer (~1.2ms) entre todas las instancias del batch.

    Uso:
        solver = BatchedModelSolver(model, batch_size=32)

        # Opción 1 — lista de archivos (recomendado para aprovechar batching):
        solved, steps, times = solver.solve_batch(dat_files, H, max_steps)

        # Opción 2 — instancia individual (compatible con interfaz Solver):
        solved, steps = solver.solve_from_path(filepath, H, max_steps)

    Nota: solve_from_path delega a solve_batch con una sola instancia, así que
    no hay diferencia de velocidad en ese caso. El beneficio aparece al llamar
    solve_batch con muchas instancias a la vez.
    """

    def __init__(self, model, batch_size=32):
        super().__init__("BatchedModelSolver")
        self.model = model
        self.batch_size = batch_size

    # ── Interfaz pública ────────────────────────────────────────────────────

    def solve_from_path(self, instance_path, H, max_steps):
        solved_list, steps_list, _ = self.solve_batch([str(instance_path)], H, max_steps)
        return solved_list[0], steps_list[0]

    def solve_batch(self, instance_paths, H, max_steps):
        """
        Resuelve una lista de instancias en paralelo usando forward passes batched.

        Retorna:
            solved_list : List[bool]
            steps_list  : List[int]
            time_list   : List[float]  — tiempo por instancia en segundos
        """
        import time
        all_solved, all_steps, all_times = [], [], []
        for i in range(0, len(instance_paths), self.batch_size):
            chunk = instance_paths[i:i + self.batch_size]
            t0 = time.perf_counter()
            s, t = self._solve_chunk(chunk, H, max_steps)
            elapsed = time.perf_counter() - t0
            per_instance = elapsed / len(chunk)
            all_solved.extend(s)
            all_steps.extend(t)
            all_times.extend([per_instance] * len(chunk))
        return all_solved, all_steps, all_times

    def solve_from_folder(self, folder, H, max_steps):
        folder_path = INSTANCE_FOLDER / folder
        filepaths = sorted([
            str(folder_path / f)
            for f in os.listdir(folder_path)
            if (folder_path / f).is_file()
        ])
        solved, steps, _ = self.solve_batch(filepaths, H, max_steps)
        return solved, steps

    # ── Núcleo del batching ─────────────────────────────────────────────────

    def _solve_chunk(self, instance_paths, H, max_steps):
        try:
            layouts = [read_file(p, H) for p in instance_paths]
        except Exception:
            return [False] * len(instance_paths), [max_steps] * len(instance_paths)

        n = len(layouts)
        S = len(layouts[0].stacks)

        visited = [set() for _ in range(n)]
        done = [layout.is_sorted() for layout in layouts]

        with torch.no_grad():
            while not all(done):
                active = [i for i in range(n) if not done[i]]

                # Registrar estado actual de cada instancia activa
                for idx in active:
                    state = tuple(tuple(s) for s in layouts[idx].stacks)
                    visited[idx].add(state)

                # Construir tensores para cada instancia activa
                batch_data = []
                for idx in active:
                    ld = list(self.model.layout_adapter.layout_2_vec(layouts[idx], H))
                    for j in range(len(ld)):
                        val = ld[j]
                        if isinstance(val, (int, float)):
                            ld[j] = torch.tensor([val])
                        else:
                            ld[j] = torch.from_numpy(val).unsqueeze(0)
                    batch_data.append(ld)

                # Un solo forward pass para todas las instancias activas
                n_args = len(batch_data[0])
                batched = [
                    torch.cat([bd[j] for bd in batch_data], dim=0)
                    for j in range(n_args)
                ]
                logits = self.model(*batched)  # (len(active), S*(S-1))

                _, top_indices = torch.sort(logits, dim=1, descending=True)

                # Aplicar el mejor movimiento válido a cada instancia
                for batch_pos, idx in enumerate(active):
                    layout = layouts[idx]
                    stacks = layout.stacks

                    moved = False
                    for i in range(top_indices.shape[1]):
                        best = top_indices[batch_pos, i].item()
                        src = int(best / (S - 1))
                        r   = best % (S - 1)
                        dst = r if r < src else r + 1

                        if not stacks[src]:
                            continue

                        top = stacks[src][-1]
                        next_state = tuple(
                            tuple(stacks[k][:-1])      if k == src else
                            tuple(stacks[k]) + (top,)  if k == dst else
                            tuple(stacks[k])
                            for k in range(S)
                        )

                        if next_state not in visited[idx]:
                            layout.move(src, dst)
                            moved = True
                            break

                    if layout.is_sorted() or layout.steps >= max_steps or not moved:
                        done[idx] = True

        return (
            [layout.unsorted_stacks == 0 for layout in layouts],
            [layout.steps for layout in layouts],
        )
