import sys, os, json, copy, time
import torch
import numpy as np

MAIN_SRC = os.path.abspath('src')
REPO_SRC = os.path.abspath('Repo_Oscar/CPMP-Framework/src')
MODEL_DIR = os.path.abspath('models')

sys.path.insert(0, MAIN_SRC)

from training.training import load_model
from models.cpmp_transformer_v9 import CPMPTransformer
from solvers.model import ModelSolver
from solvers.dlts_dfs import DLTSDFSSolver
from solvers.beam_search import BeamSearchSolver
from cpmp.layout import read_file
from settings import INSTANCE_FOLDER

for k in [k for k in sys.modules if k == 'generation' or k.startswith('generation.')]:
    del sys.modules[k]
sys.path.insert(0, REPO_SRC)

from models.cost.cost_predictor_V2 import CostPredictorTransformer
from generation.adapters.input.enriched_layout_adapter import EnrichedLayoutAdapter
from generation.adapters.input.layout.layout_4D_adapter_V2 import Layout4DAdapterV2
from generation.adapters.input.stack_features.stack_features_adapter_V1 import StackFeaturesAdapterV1

branching_model = load_model(CPMPTransformer, 'v9_dataBSG_250k')
branching_model.eval()

def load_cost_model(name):
    hp = json.load(open(os.path.join(MODEL_DIR, 'hyperparameters', f'{name}.json')))
    m = CostPredictorTransformer(**hp)
    m.load_state_dict(torch.load(os.path.join(MODEL_DIR, f'{name}.pth'),
                                  weights_only=True, map_location='cpu'))
    m.eval()
    return m

bounding_model = load_cost_model('v2_cost')
bounding_adapter = EnrichedLayoutAdapter(
    layout_adapter=Layout4DAdapterV2,
    stack_features_adapter=StackFeaturesAdapterV1,
    S_max=10, H_max=12
)

T_LIM = 15.0
W = 5

greedy_solver = ModelSolver(branching_model)
dlts_solver = DLTSDFSSolver(
    branching_model=branching_model,
    bounding_model=bounding_model,
    bounding_adapter=bounding_adapter,
    p=0.3, k=3, d=0.8, time_limit=T_LIM
)
beam_solver = BeamSearchSolver(
    branching_model=branching_model,
    beam_width=W,
    expansions_per_state=W,
    time_limit=T_LIM
)

def avg(vals, mask=None):
    data = [v for v, ok in zip(vals, mask or [True]*len(vals)) if ok]
    return sum(data) / len(data) if data else float('nan')

def run_solver(solver, files, H_inf, max_steps):
    solved_list, steps_list, time_list = [], [], []
    for path in files:
        layout = read_file(path, H_inf)
        t0 = time.perf_counter()
        try:
            solved, steps = solver.solve_from_layout(layout, H_inf, max_steps)
        except Exception as e:
            solved, steps = False, max_steps
        elapsed = time.perf_counter() - t0
        solved_list.append(solved)
        steps_list.append(steps)
        time_list.append(elapsed)
    return solved_list, steps_list, time_list

CVS_PATH = INSTANCE_FOLDER / 'benchmarks' / 'CVS'
MAX_STEPS = 100
N_PER_CAT = 20

ALL_CATS = sorted([d for d in os.listdir(CVS_PATH)
                   if (CVS_PATH / d).is_dir() and not d.startswith('10-')])

SEP = '-' * 120
print(f"{'Cat':>6} {'H':>3} {'S':>3}  "
      f"{'Greedy':>7} {'pasos':>6} {'t(s)':>6}  "
      f"{'DLTS-DFS':>8} {'pasos':>6} {'t(s)':>6}  "
      f"{'BeamW=5':>7} {'pasos':>6} {'t(s)':>6}  "
      f"{'DLTS>G':>6} {'Beam>G':>6} {'DLTS>B':>7}", flush=True)
print(SEP, flush=True)

all_results = {}

for folder_name in ALL_CATS:
    H_r, S_r = [int(x) for x in folder_name.split('-')]
    H_inf = H_r + 2
    folder_path = CVS_PATH / folder_name
    files = sorted([str(folder_path / f)
                    for f in os.listdir(folder_path) if f.endswith('.dat')])[:N_PER_CAT]
    if not files:
        continue

    print(f"  Procesando {folder_name}...", flush=True)
    s_g, st_g, t_g = run_solver(greedy_solver, files, H_inf, MAX_STEPS)
    s_d, st_d, t_d = run_solver(dlts_solver,   files, H_inf, MAX_STEPS)
    s_b, st_b, t_b = run_solver(beam_solver,   files, H_inf, MAX_STEPS)

    all_results[folder_name] = dict(
        s_g=s_g, st_g=st_g, t_g=t_g,
        s_d=s_d, st_d=st_d, t_d=t_d,
        s_b=s_b, st_b=st_b, t_b=t_b,
    )

    n = len(files)
    ng, nd, nb_ = sum(s_g), sum(s_d), sum(s_b)

    dlts_beats_g = sum(1 for ok_d, ok_g, pd, pg in zip(s_d, s_g, st_d, st_g)
                       if ok_d and ok_g and pd < pg)
    beam_beats_g = sum(1 for ok_b, ok_g, pb, pg in zip(s_b, s_g, st_b, st_g)
                       if ok_b and ok_g and pb < pg)
    dlts_beats_b = sum(1 for ok_d, ok_b, pd, pb in zip(s_d, s_b, st_d, st_b)
                       if ok_d and ok_b and pd < pb)

    print(f"{folder_name:>6} {H_r:>3} {S_r:>3}  "
          f"{ng:>3}/{n:<2} {avg(st_g,s_g):>6.1f} {avg(t_g,s_g):>6.3f}  "
          f"{nd:>3}/{n:<2} {avg(st_d,s_d):>6.1f} {avg(t_d,s_d):>6.2f}  "
          f"{nb_:>3}/{n:<2} {avg(st_b,s_b):>6.1f} {avg(t_b,s_b):>6.2f}  "
          f"{dlts_beats_g:>6} {beam_beats_g:>6} {dlts_beats_b:>7}", flush=True)

print(SEP, flush=True)

all_sg  = [s for r in all_results.values() for s in r['s_g']]
all_stg = [s for r in all_results.values() for s in r['st_g']]
all_sd  = [s for r in all_results.values() for s in r['s_d']]
all_std = [s for r in all_results.values() for s in r['st_d']]
all_sb  = [s for r in all_results.values() for s in r['s_b']]
all_stb = [s for r in all_results.values() for s in r['st_b']]
all_tg  = [s for r in all_results.values() for s in r['t_g']]
all_td  = [s for r in all_results.values() for s in r['t_d']]
all_tb  = [s for r in all_results.values() for s in r['t_b']]
n_tot = len(all_sg)

print(f"{'TOTAL':>6} {'':>3} {'':>3}  "
      f"{sum(all_sg):>3}/{n_tot:<2} {avg(all_stg,all_sg):>6.1f} {avg(all_tg,all_sg):>6.3f}  "
      f"{sum(all_sd):>3}/{n_tot:<2} {avg(all_std,all_sd):>6.1f} {avg(all_td,all_sd):>6.2f}  "
      f"{sum(all_sb):>3}/{n_tot:<2} {avg(all_stb,all_sb):>6.1f} {avg(all_tb,all_sb):>6.2f}", flush=True)

# Resumen numérico
mask3 = [a and b and c for a, b, c in zip(all_sg, all_sd, all_sb)]
n3 = sum(mask3)
st_g3 = [s for s, ok in zip(all_stg, mask3) if ok]
st_d3 = [s for s, ok in zip(all_std, mask3) if ok]
st_b3 = [s for s, ok in zip(all_stb, mask3) if ok]

print(f'\n=== RESUMEN (instancias donde los 3 resolvieron: {n3}) ===', flush=True)
print(f'  Greedy     : avg pasos = {np.mean(st_g3):.2f}', flush=True)
print(f'  DLTS-DFS   : avg pasos = {np.mean(st_d3):.2f}  ({(np.mean(st_d3)/np.mean(st_g3)-1)*100:+.1f}% vs greedy)', flush=True)
print(f'  Beam Search: avg pasos = {np.mean(st_b3):.2f}  ({(np.mean(st_b3)/np.mean(st_g3)-1)*100:+.1f}% vs greedy)', flush=True)

d_beats_b = sum(1 for d, b in zip(st_d3, st_b3) if d < b)
b_beats_d = sum(1 for d, b in zip(st_d3, st_b3) if b < d)
equal     = sum(1 for d, b in zip(st_d3, st_b3) if d == b)
print(f'\nDLTS-DFS vs Beam Search (en las {n3} instancias comunes):', flush=True)
print(f'  DLTS mejor : {d_beats_b:>4} ({d_beats_b/n3:>5.1%})', flush=True)
print(f'  Igual       : {equal:>4} ({equal/n3:>5.1%})', flush=True)
print(f'  Beam mejor  : {b_beats_d:>4} ({b_beats_d/n3:>5.1%})', flush=True)

# Guardar resultados para análisis posterior
import pickle
with open('benchmark_dlts_vs_beam_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)
print('\nResultados guardados en benchmark_dlts_vs_beam_results.pkl', flush=True)
