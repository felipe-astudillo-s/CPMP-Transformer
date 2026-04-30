import os
import torch
import pandas as pd
import numpy as np
import sys

# 1. Calculamos la ruta hacia la carpeta 'src'
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

# --- Importaciones de tu framework ---
# Asegúrate de que el nombre del modelo coincida con el que estás usando (v8 o v9)
from models.cpmp_transformer_v9 import CPMPTransformer
from generation.adapters import EnrichedStackMatrix5DAdapter
from cpmp.layout import Layout

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# --- CONFIGURACIÓN ---
BENCHMARK_DIR = os.path.join(base_dir, "instances", "benchmarks", "CVS")
# Cambia a "v8.pth" si ese es el archivo que guardaste en el paso anterior
MODEL_PATH = os.path.join(base_dir, "models", "v9_new_gen_data.pth") 
MAX_STEPS = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Iniciando Benchmark CVS en {device}...")

# --- 1. CARGAR EL MODELO ---
# H_MODELO debe ser la altura máxima estructural que usaste en el entrenamiento (ej. 10)
H_MODELO = 12
C_dim = 2
X_dim = 5
d_model = 64
nhead = 4
num_layers = 4
ff_dim_multiplier = 2
dropout = 0.3

model = CPMPTransformer(
    H=H_MODELO,
    C_dim=C_dim,
    X_dim=X_dim,
    d_model=d_model,
    nhead=nhead,
    num_layers=num_layers,
    ff_dim_multiplier=ff_dim_multiplier,
    dropout=dropout
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.to(device)
model.eval()

adapter = EnrichedStackMatrix5DAdapter()

# --- 2. FUNCIÓN PARA LEER ARCHIVO .DAT ---
def read_cvs_instance(filepath, H_real): 
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    S = int(lines[0].split()[0])
    
    stacks = []
    for i in range(1, S + 1):
        line_data = list(map(int, lines[i].split()))
        if len(line_data) > 0:
            stack = line_data[1:] 
            stacks.append(stack)
        else:
            stacks.append([])
    
    return Layout(stacks, H_real) 

# --- 3. FUNCIÓN DE RESOLUCIÓN (INFERENCIA) ---
def solve_layout(layout, H_modelo, H_real, model, max_steps):
    steps = 0
    current_layout = layout
    last_move = None # Memoria para evitar bucles infinitos
    
    with torch.no_grad():
        while not current_layout.is_sorted() and steps < max_steps:
            # 1. Transformar a tensores usando H_modelo para la estructura
            S_tensor, X_tensor = adapter.layout_2_vec(current_layout, H_modelo)
            
            # Inyectamos el H físico real en la dimensión 5 del vector X
            X_tensor[:, 4] = H_real 
            
            # 2. Agregar dimensión de batch
            S_batch = torch.tensor(S_tensor, device=device).unsqueeze(0)
            X_batch = torch.tensor(X_tensor, device=device).unsqueeze(0)
            
            # 3. Predicción
            logits = model(S_batch, X_batch).squeeze(0)
            
            # Penalizar el movimiento inverso al anterior para evitar atascos (Ej: 1->2 y luego 2->1)
            if last_move is not None:
                S_len = len(current_layout.stacks)
                idx_inverso = last_move[1] * (S_len - 1) + (last_move[0] - int(last_move[0] > last_move[1]))
                logits[idx_inverso] -= 1000 # Penalización fuerte para descartar este movimiento
            
            # 4. Seleccionar el mejor movimiento (que sea válido)
            sorted_indices = torch.argsort(logits, descending=True)
            
            move_made = False
            for idx in sorted_indices:
                idx_val = idx.item()
                S_len = len(current_layout.stacks)
                src = idx_val // (S_len - 1)
                dst_offset = idx_val % (S_len - 1)
                dst = dst_offset if dst_offset < src else dst_offset + 1
                
                # Validación física real
                if len(current_layout.stacks[src]) > 0 and len(current_layout.stacks[dst]) < H_real:
                    if hasattr(current_layout, 'apply_move'):
                        nuevo_layout = current_layout.apply_move(src, dst)
                        if nuevo_layout is not None:
                            current_layout = nuevo_layout
                    else:
                        contenedor = current_layout.stacks[src].pop()
                        current_layout.stacks[dst].append(contenedor)
                    
                    steps += 1
                    last_move = (src, dst)
                    move_made = True
                    break
            
            if not move_made:
                return False, steps
                
    return current_layout.is_sorted(), steps

# --- 4. BUCLE PRINCIPAL DE EVALUACIÓN ---
resultados_globales = []

for size_folder in sorted(os.listdir(BENCHMARK_DIR)):
    folder_path = os.path.join(BENCHMARK_DIR, size_folder)
    
    if not os.path.isdir(folder_path): continue
    
    print(f"\n📁 Evaluando categoría: {size_folder}")
    
    # Extraemos S y H real del nombre de la carpeta (ej: "10-6" -> S=10, H=6)
    try:
        partes = size_folder.split('-')
        S_folder = int(partes[0])
        H_real_instancia = int(partes[1])
    except:
        print(f"   ⚠️ No se pudo deducir S y H de la carpeta {size_folder}, saltando...")
        continue

    instancias_resueltas = 0
    total_instancias = 0
    pasos_totales_resueltas = []
    
    for filename in os.listdir(folder_path):
        if not filename.endswith('.dat'): continue
        
        filepath = os.path.join(folder_path, filename)
        layout = read_cvs_instance(filepath, H_real_instancia)
        total_instancias += 1
        
        resuelto, pasos = solve_layout(layout, H_MODELO, H_real_instancia, model, MAX_STEPS)
        
        if resuelto:
            instancias_resueltas += 1
            pasos_totales_resueltas.append(pasos)
            
    tasa_exito = (instancias_resueltas / total_instancias) * 100 if total_instancias > 0 else 0
    pasos_promedio = np.mean(pasos_totales_resueltas) if instancias_resueltas > 0 else 0
    
    print(f"   ➤ Resueltas: {instancias_resueltas}/{total_instancias} ({tasa_exito:.1f}%) | Pasos promedio: {pasos_promedio:.2f}")
    
    resultados_globales.append({
        "Categoría (S-H)": size_folder,
        "Total Instancias": total_instancias,
        "Resueltas": instancias_resueltas,
        "Tasa de Éxito (%)": round(tasa_exito, 2),
        "Pasos Promedio": round(pasos_promedio, 2)
    })

# --- 5. MOSTRAR TABLA FINAL ---
print("\n" + "="*50)
print("📊 RESUMEN FINAL DEL BENCHMARK CVS")
print("="*50)

df = pd.DataFrame(resultados_globales)
print(df.to_string(index=False))

df.to_csv("benchmark_cvs_resultados.csv", index=False)
print("\n✅ Resultados guardados en 'benchmark_cvs_resultados.csv'")