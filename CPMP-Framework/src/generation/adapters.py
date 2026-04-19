import numpy as np
from abc import ABC, abstractmethod
from cpmp.layout import Layout

class DataAdapter(ABC):
    def __init__(self, data_keys):
        super().__init__()
        self.data = {
            k: [] for k in data_keys
        }
        self.data_keys = data_keys

    @abstractmethod
    def add(self, layout_data):
        pass

    def get(self) -> dict:
        return {
            k: np.stack(v, dtype=self.data_keys[k]) for k, v in self.data.items()
        }

    def count(self):
        return len(self.data[list(self.data.keys())[0]])

class LayoutDataAdapter(DataAdapter):
    def __init__(self, data_keys):
        super().__init__(data_keys)

    @staticmethod
    @abstractmethod
    def layout_2_vec(layout: Layout, H: int):
        pass

class MovesDataAdapter(DataAdapter):
    def __init__(self, data_keys):
        super().__init__(data_keys)

    @staticmethod
    @abstractmethod
    def moves_2_vec(moves, S):
        pass

class GPIAdapter(LayoutDataAdapter):
    def __init__(self):
        super().__init__({
            "G": np.int32,
            "P": np.int32,
            "I": np.int32,
            "S": np.int32,
            "H": np.int32, 
        })

    @staticmethod
    def layout_2_vec(layout, H):
        G = [] # Valores de grupo
        P = [] # Dónde se ubica el contenedor en su respectiva pila
        I = [] # En qué pila se encuentra el contenedor
        S = len(layout.stacks) # Número de pilas

        for i in range(S):
            for j in range(len(layout.stacks[i])):
                G.append(layout.stacks[i][j])
                P.append(j)
                I.append(i)

        return np.array(G), np.array(P), np.array(I), S, H
    
    def add(self, layout_data):
        G, P, I, S, H = layout_data

        self.data['G'].append(G)
        self.data['P'].append(P)
        self.data['I'].append(I)
        self.data['S'].append(S)
        self.data['H'].append(H)

class StackMatrixAdapter(LayoutDataAdapter):
    def __init__(self):
        super().__init__({
            "S": np.float32
        })

    def add(self, layout_data):
        S_matrix = layout_data[0]
        self.data['S'].append(S_matrix)

class StackMatrix3DAdapter(StackMatrixAdapter):
    def __init__(self):
        super().__init__()

    @staticmethod
    def layout_2_vec(layout, H):
        stacks_matrix = []
        
        all_vals = [c for s in layout.stacks for c in s]
        max_val = max(all_vals) if all_vals else 1

        for stack in layout.stacks:
            normalized_stack = [val / max_val for val in stack]
            padding_size = H - len(normalized_stack)
            padded_stack = normalized_stack + [-1] * padding_size
            stacks_matrix.append(padded_stack)
            
        return (np.array(stacks_matrix, dtype=np.float32), )
    
class StackMatrix4DAdapter(StackMatrixAdapter):
    def __init__(self):
        super().__init__()

    @staticmethod
    def layout_2_vec(layout, H):
        stacks_matrix = []
        
        all_vals = [c for s in layout.stacks for c in s]
        max_val = max(all_vals) if all_vals else 1

        for i in range(len(layout.stacks)):
            stack = []
            
            # Variables para rastrear el estado de bloqueo
            is_blocked = False
            prev_val = None

            for j in range(len(layout.stacks[i])):
                current_val = layout.stacks[i][j]
                normalized_c = current_val / max_val
                
                # Lógica de Bloqueo (Definición 1)
                # 1. El primero (j == 0) nunca está bloqueado inicialmente.
                # 2. Si ya está bloqueado un nivel superior, el resto hacia abajo también.
                # 3. Si el valor actual es mayor que el anterior, se bloquea.
                if j > 0:
                    if is_blocked or current_val > prev_val:
                        is_blocked = True
                
                # Asignar 1 si está bloqueado, 0 de lo contrario
                blocked_val = 1.0 if is_blocked else 0.0
                
                stack.append([normalized_c, blocked_val])
                prev_val = current_val # Actualizamos para la siguiente iteración
            
            # Padding: Ahora cada elemento es un par [val, blocked]
            # Usamos [-1, -1] para mantener la consistencia con tu código
            padding_size = H - len(stack)
            padded_stack = stack + [[-1.0, -1.0]] * padding_size
            stacks_matrix.append(padded_stack)

        return (np.array(stacks_matrix, dtype=np.float32), )
    
class EnrichedStackMatrixAdapter(LayoutDataAdapter):
    def __init__(self):
        super().__init__({
            "S": np.float32,
            "X": np.float32
        })

    @staticmethod
    def get_X(layout: Layout, H: int):
        X = np.zeros((len(layout.stacks), 3), dtype=np.float32)

        for i in range(len(layout.stacks)):
            X[i][0] = 1.0 if layout.is_sorted_stack(i) else 0.0
            X[i][1] = len(layout.stacks[i]) / H
            X[i][2] = (layout.sorted_elements[i] / len(layout.stacks[i])) if len(layout.stacks[i]) != 0.0 else 1

        return np.array(X, dtype=np.float32)

    def add(self, layout_data):
        S_matrix, X = layout_data

        self.data['S'].append(S_matrix)
        self.data['X'].append(X)
    
class EnrichedStackMatrix3DAdapter(EnrichedStackMatrixAdapter):
    def __init__(self):
        super().__init__()

    @staticmethod
    def layout_2_vec(layout: Layout, H: int):
        S = StackMatrix3DAdapter.layout_2_vec(layout, H)[0]
        X = EnrichedStackMatrixAdapter.get_X(layout, H)
        return S, X
    
class EnrichedStackMatrix4DAdapter(EnrichedStackMatrixAdapter):
    def __init__(self):
        super().__init__()

    @staticmethod
    def layout_2_vec(layout: Layout, H: int):
        S = StackMatrix4DAdapter.layout_2_vec(layout, H)[0]
        X = EnrichedStackMatrixAdapter.get_X(layout, H)
        return S, X
    
class DefaultMovesAdapter(MovesDataAdapter):
    def __init__(self):
        super().__init__({
            "Y": np.int32
        })
    
    @staticmethod
    def moves_2_vec(moves, S):
        Y = np.zeros(S*(S-1), dtype=np.int32)

        for move in moves:
            src, dst = move[0], move[1]
            # Implementación de la fórmula: A = src * (S - 1) + (dst - [dst > src])
            idx = src * (S - 1) + (dst - int(dst > src))
            Y[idx] = 1.0

        return Y
    
    def add(self, moves_data):
        self.data['Y'].append(moves_data)

class EnrichedStackMatrix5DAdapter(EnrichedStackMatrixAdapter):
    """
    Adaptador modular para el Transformer V7.
    Reutiliza la matriz S (g, b, t) del 4DAdapter, pero expande el vector X a 5 dimensiones:
    [0] Estado de orden
    [1] Tamaño relativo
    [2] Ratio de orden
    [3] Tamaño absoluto (NUEVO)
    [4] Altura máxima física H (NUEVO)
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_X(layout: Layout, H: int):
        # Inicializamos con 5 dimensiones
        X = np.zeros((len(layout.stacks), 5), dtype=np.float32)

        for i in range(len(layout.stacks)):
            stack_len = len(layout.stacks[i])
            
            # Variables originales (para mantener el entendimiento estadístico)
            X[i][0] = 1.0 if layout.is_sorted_stack(i) else 0.0
            X[i][1] = stack_len / H
            X[i][2] = (layout.sorted_elements[i] / stack_len) if stack_len != 0.0 else 1.0
            
            # Nuevas variables (para el escudo físico y consciencia del CLS)
            X[i][3] = float(stack_len)       # Tamaño absoluto de la pila
            X[i][4] = float(H)               # Límite físico de la instancia

        return np.array(X, dtype=np.float32)

    @staticmethod
    def layout_2_vec(layout: Layout, H: int):
        # 1. Obtenemos S del adaptador 4D (que ya tiene la lógica g, b, t)
        S = StackMatrix4DAdapter.layout_2_vec(layout, H)[0]
        # 2. Obtenemos nuestro nuevo X de 5 dimensiones
        X = EnrichedStackMatrix5DAdapter.get_X(layout, H)
        return S, X
