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

def greedy(layout, H, max_steps):
    pid = os.getpid()
    filepath = INSTANCE_FOLDER / f"tmp_{pid}.txt"

    try:
        lay2file(layout, filename=filepath)

        result = subprocess.run(
            [FRG_PATH, str(H), filepath, "1.2", str(max_steps), "0", "--no-assignment", "2"],
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
    
def get_best_moves(layout, H, max_steps):
    moves = get_feasible_moves(layout)
    best_moves = []
    min_cost = float('inf')

    for (i, j) in moves:
        lay_copy = copy.deepcopy(layout)
        lay_copy.move(i, j)
        cost = greedy(lay_copy, H, max_steps)

        if cost < min_cost:
            min_cost = cost
            best_moves = [(i, j)]
        elif cost == min_cost:
            # Si hay empates en la jugada óptima, guarda ambas
            best_moves.append((i, j))

    return best_moves, cost

def generate_data_from_file(filepath, H, max_steps, layout_cls, moves_cls):
    layout = read_instance(filepath, H)
    if layout.unsorted_stacks == 0: 
        return None

    layout_vec = layout_cls.layout_2_vec(layout, H)
    S = len(layout.stacks)

    best_moves, cost = get_best_moves(layout, H, max_steps)
    if cost == float('inf'):
        return None

    moves_vec = moves_cls.moves_2_vec(best_moves, S)

    return layout_vec, moves_vec, cost

def generate_data(folder, H, max_steps, layout_adapter: LayoutDataAdapter, moves_adapter: MovesDataAdapter, output_name=None):
    filepaths = [os.path.join(INSTANCE_FOLDER / folder, f) for f in os.listdir(INSTANCE_FOLDER / folder)]
    
    # Extraemos las clases de las instancias recibidas
    l_cls = layout_adapter.__class__
    m_cls = moves_adapter.__class__

    with ProcessPoolExecutor() as executor:
        task = partial(generate_data_from_file, H=H, max_steps=max_steps, layout_cls=l_cls, moves_cls=m_cls)
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