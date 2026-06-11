from settings import INSTANCE_FOLDER, DATA_FOLDER, FRG_PATH
import subprocess
from generation.instances import read_instance
import copy
import os
import h5py
import numpy as np
from generation.adapters import *
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import torch

def greedy(layout, H, max_steps, beams=0):
    pid = os.getpid()
    os.makedirs(INSTANCE_FOLDER, exist_ok=True)
    filepath = INSTANCE_FOLDER / f"tmp_{pid}.txt"

    flag = "--no-assignment" if beams == 0 else "--compound"

    try:
        lay2file(layout, filename=filepath)

        result = subprocess.run(
            [FRG_PATH, str(H), filepath, "1.2", str(max_steps), str(beams), flag, "2"],
            check=True,
            text=True,
            capture_output=True
        )
        output_str = result.stdout.split('\t')[0].strip()
        if not output_str.isdigit():
            return float('inf')

        return int(output_str)
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

def lay2file(layout, filename):
    S = layout.stacks

    with open(filename, "w") as f:
        num_sublists = len(S)
        sum_lengths = sum(len(sublist) for sublist in S)
        f.write(f"{num_sublists} {sum_lengths}\n")
        for sublist in S:
            f.write(str(len(sublist)) +" " + " ".join(str(x) for x in sublist) + "\n")

def get_feasible_moves(layout):
    moves = []
    num_stacks = len(layout.stacks)

    for i in range(num_stacks):
        if len(layout.stacks[i]) > 0:
            for j in range(num_stacks):
                if i != j and len(layout.stacks[j]) < layout.H:
                    moves.append((i, j))

    return moves
    
def get_best_moves(layout, H, max_steps, beams=0):
    moves = get_feasible_moves(layout)
    best_moves = []
    min_cost = float('inf')

    for (i, j) in moves:
        lay_copy = copy.deepcopy(layout)
        lay_copy.move(i, j)
        cost = greedy(lay_copy, H, max_steps, beams=beams)

        if cost < min_cost:
            min_cost = cost
            best_moves = [(i, j)]
        elif cost == min_cost:
            best_moves.append((i, j))

    return best_moves, cost

def generate_data_from_file(filepath, H, max_steps, layout_cls, moves_cls, beams=0):
    layout = read_instance(filepath, H)
    if layout.unsorted_stacks == 0:
        return None

    layout_vec = layout_cls.layout_2_vec(layout, H)
    S = len(layout.stacks)

    best_moves, cost = get_best_moves(layout, H, max_steps, beams=beams)
    if cost == float('inf'):
        return None

    moves_vec = moves_cls.moves_2_vec(best_moves, S)

    return layout_vec, moves_vec, cost

def generate_data(folder, H, max_steps, layout_adapter: LayoutDataAdapter, moves_adapter: MovesDataAdapter, output_name=None, beams=0):
    filepaths = [os.path.join(INSTANCE_FOLDER / folder, f) for f in os.listdir(INSTANCE_FOLDER / folder)]

    l_cls = layout_adapter.__class__
    m_cls = moves_adapter.__class__

    with ProcessPoolExecutor() as executor:
        task = partial(generate_data_from_file, H=H, max_steps=max_steps, layout_cls=l_cls, moves_cls=m_cls, beams=beams)
        results = list(executor.map(task, filepaths))

    costs = []
    for result in results:
        if result is None:
            continue

        layout_vec, moves_vec, cost = result
        layout_adapter.add(layout_vec)
        moves_adapter.add(moves_vec)
        costs.append(cost)

    layout_data = layout_adapter.get()
    moves_data = moves_adapter.get()
    data = {**layout_data, **moves_data}

    if output_name is None:
        output_path = DATA_FOLDER / f"{folder}.data"
    else:
        output_path = DATA_FOLDER / f"{output_name}.data"

    with h5py.File(output_path, "w") as f:
        keys_order = [k for k in data.keys() if k != 'C']
        f.attrs['key_order'] = [k for k in keys_order]

        for key in data:
            f.create_dataset(key, data=data[key])
        f.create_dataset("C", data=np.stack(costs, dtype=np.int32))

    print(f"Datos guardados en: {output_path} (Tamaño {layout_adapter.count()})")


# ── Expert Iteration (RL): etiquetas generadas por el modelo (GPU batched) ────

def _rollout_batched(next_layouts, H, max_steps, model, device):
    """Rollout de k layouts simultáneamente con inferencia batched en GPU.
    Todos los layouts deben tener el mismo S y H (vienen del mismo padre).
    Muta los layouts in-place — el caller debe pasar copias.
    """
    n = len(next_layouts)
    active = list(range(n))
    solved = [False] * n
    steps = [0] * n
    visited = [set() for _ in range(n)]
    adapter = model.layout_adapter  # clase estática, e.g. EnrichedStackMatrix5DAdapter

    with torch.no_grad():
        while active:
            vecs = [adapter.layout_2_vec(next_layouts[i], H) for i in active]
            S_batch = torch.from_numpy(np.stack([v[0] for v in vecs])).to(device)
            X_batch = torch.from_numpy(np.stack([v[1] for v in vecs])).to(device)

            logits = model(S_batch, X_batch)
            _, top_indices = torch.sort(logits, dim=1, descending=True)

            next_active = []
            for batch_pos, orig_idx in enumerate(active):
                layout = next_layouts[orig_idx]
                S = len(layout.stacks)
                moved = False

                for rank in range(top_indices.shape[1]):
                    best_idx = top_indices[batch_pos, rank].item()
                    src = best_idx // (S - 1)
                    r = best_idx % (S - 1)
                    dst = r if r < src else r + 1

                    if src >= S or dst >= S or not layout.stacks[src]:
                        continue
                    if len(layout.stacks[dst]) >= H:
                        continue

                    top = layout.stacks[src][-1]
                    next_state = tuple(
                        tuple(layout.stacks[k][:-1])     if k == src else
                        tuple(layout.stacks[k]) + (top,) if k == dst else
                        tuple(layout.stacks[k])
                        for k in range(S)
                    )

                    if next_state not in visited[orig_idx]:
                        visited[orig_idx].add(next_state)
                        layout.move(src, dst)
                        steps[orig_idx] += 1
                        moved = True
                        break

                if layout.is_sorted():
                    solved[orig_idx] = True
                elif not moved or steps[orig_idx] >= max_steps:
                    solved[orig_idx] = False
                else:
                    next_active.append(orig_idx)

            active = next_active

    return [(solved[i], steps[i]) for i in range(n)]


def generate_data_rl(folder, H, max_steps, layout_adapter: LayoutDataAdapter, moves_adapter: MovesDataAdapter, model, output_name=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    filepaths = [os.path.join(INSTANCE_FOLDER / folder, f) for f in os.listdir(INSTANCE_FOLDER / folder)]
    print(f"  [{folder}] {len(filepaths)} instancias — dispositivo: {device}")

    costs = []
    skipped = 0
    total_rollouts = 0
    solved_rollouts = 0

    for filepath in filepaths:
        layout = read_instance(filepath, H)
        if layout.unsorted_stacks == 0:
            skipped += 1
            continue

        S = len(layout.stacks)
        moves = get_feasible_moves(layout)
        if not moves:
            skipped += 1
            continue

        layout_vec = layout_adapter.layout_2_vec(layout, H)

        # Una copia por movimiento candidato
        next_layouts = []
        for (i, j) in moves:
            lay_copy = copy.deepcopy(layout)
            lay_copy.move(i, j)
            next_layouts.append(lay_copy)

        # Rollout batched: todos los k candidatos avanzan juntos en cada paso
        results = _rollout_batched(next_layouts, H, max_steps, model, device)

        total_rollouts += len(results)
        solved_rollouts += sum(1 for (s, _) in results if s)

        best_moves = []
        min_cost = float('inf')
        for move, (move_solved, move_steps) in zip(moves, results):
            if not move_solved:
                continue
            if move_steps < min_cost:
                min_cost = move_steps
                best_moves = [move]
            elif move_steps == min_cost:
                best_moves.append(move)

        if not best_moves:
            skipped += 1
            continue

        moves_vec = moves_adapter.moves_2_vec(best_moves, S)
        layout_adapter.add(layout_vec)
        moves_adapter.add(moves_vec)
        costs.append(min_cost + 1)

    solved_pct = 100 * solved_rollouts / total_rollouts if total_rollouts > 0 else 0
    print(f"  [{folder}] Generados: {len(costs)} | Saltados: {skipped} | Rollouts resueltos: {solved_rollouts}/{total_rollouts} ({solved_pct:.1f}%)")

    layout_data = layout_adapter.get()
    moves_data = moves_adapter.get()
    data = {**layout_data, **moves_data}

    if output_name is None:
        output_path = DATA_FOLDER / f"{folder}_rl.data"
    else:
        output_path = DATA_FOLDER / f"{output_name}.data"

    with h5py.File(output_path, "w") as f:
        keys_order = [k for k in data.keys() if k != 'C']
        f.attrs['key_order'] = [k for k in keys_order]
        for key in data:
            f.create_dataset(key, data=data[key])
        f.create_dataset("C", data=np.stack(costs, dtype=np.int32))

    print(f"  Datos guardados en: {output_path} (Tamaño {layout_adapter.count()})")