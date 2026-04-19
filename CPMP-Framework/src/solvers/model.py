import torch
from solvers.solver import Solver
import numpy as np
from cpmp.layout import read_file
import copy
from generation.adapters import *


class ModelSolver(Solver): 
    def __init__(self, model):
        super().__init__("ModelSolver")
        self.model = model
            
    def solve_from_path(self, instance_path, H, max_steps):
        layout = read_file(instance_path, H)
        S = len(layout.stacks)
        
        # Conjunto para almacenar los estados visitados (como tuplas inmutables)
        visited_states = set()
        
        with torch.no_grad():
            while not layout.is_sorted():
                # Guardamos el estado actual antes de mover
                # Convertimos cada stack a tupla para que sea "hasheable"
                current_state = tuple(tuple(stack) for stack in layout.stacks)
                visited_states.add(current_state)

                layout_data = list(self.model.layout_adapter.layout_2_vec(layout, H))
                for i in range(len(layout_data)):
                    val = layout_data[i]
                    if isinstance(val, (int, float)):
                        layout_data[i] = torch.tensor([val])
                    else:
                        layout_data[i] = torch.from_numpy(val).unsqueeze(0)
                    
                logits = self.model(*layout_data)
                
                # Ordenamos todos los índices de mejor a peor
                _, top_indices = torch.sort(logits, dim=1, descending=True)
                top_indices = top_indices.squeeze(0)

                for i in range(len(top_indices)):
                    best_index = top_indices[i].item()
                    src = int(best_index / (S-1))
                    r = best_index % (S-1)
                    dst = r if r < src else r + 1

                    # 1. Previsualizamos el movimiento con deepcopy
                    temp_layout = copy.deepcopy(layout)
                    temp_layout.move(src, dst)
                    next_state = tuple(tuple(stack) for stack in temp_layout.stacks)
                    
                    # 2. Verificamos si el estado resultante ya fue visitado
                    if next_state not in visited_states:
                        layout.move(src, dst)
                        break

                if layout.steps >= max_steps:
                    break

        solved = layout.unsorted_stacks == 0
        return solved, layout.steps