import os
from settings import INSTANCE_FOLDER
from cpmp.layout import Layout
import random

def generate_stacks(H, S, N):
    stacks = []
    for _ in range(S):
        stacks.append([])

    for j in range(N):
        s = random.randint(0,S-1)
        while len(stacks[s])==H:
            s = random.randint(0,S-1)
        g = N-j
        stacks[s].append(g)

    return stacks

def random_moves(stacks, H, r):
    last_move = (None, None)
    moves_made = 0
    
    while moves_made < r:
        # 1. Elegir un origen que no esté vacío
        valid_origins = [i for i, s in enumerate(stacks) if len(s) > 0]
        if not valid_origins:
            break  # No hay movimientos posibles
            
        origin_idx = random.choice(valid_origins)
        
        # 2. Elegir un destino que no esté lleno y no sea el origen
        valid_destinations = [
            i for i, s in enumerate(stacks) 
            if i != origin_idx and len(s) < H
        ]
        
        if not valid_destinations:
            continue # Reintentar con otro origen
            
        dest_idx = random.choice(valid_destinations)
        
        # 3. Validar que no anule el movimiento anterior
        # El inverso de (a, b) es (b, a)
        if (dest_idx, origin_idx) == last_move:
            continue
            
        # Ejecutar el movimiento
        container = stacks[origin_idx].pop()
        stacks[dest_idx].append(container)
        
        # Registrar rastro
        last_move = (origin_idx, dest_idx)
        moves_made += 1

    return stacks

def generate_instance(filepath, H, S, N, r):
    stacks = generate_stacks(H, S, N)
    stacks = random_moves(stacks, H, r)

    with open(filepath, 'w') as f:
        f.write(f"{S} {N}")
        for s in stacks:
            f.write("\n")
            f.write(f"{len(s)} ")
            for g in s:
                f.write(f"{g} ")

def generate_instances(basename, H, S, N, amount, r=5, seed=42):
    os.makedirs(INSTANCE_FOLDER / basename, exist_ok=True)
    random.seed(seed)

    for i in range(amount):
        filepath = INSTANCE_FOLDER / basename / f'{basename}-{i}.txt'
        generate_instance(filepath, H, S, N, r)

def read_instance(file, H):
    with open(file) as f:
        S, C = [int(x) for x in next(f).split()] # read first line
        stacks = []
        for line in f: # read rest of lines
            stack = [int(x) for x in line.split()[1::]]
            #if stack[0] == 0: stack.pop()
            stacks.append(stack)
            
        layout = Layout(stacks,H)
    return layout