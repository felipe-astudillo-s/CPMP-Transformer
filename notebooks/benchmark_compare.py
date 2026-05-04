import sys
import os

# Ruta absoluta basada en la ubicación del script, independiente del cwd
SRC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, os.path.normpath(SRC_PATH))

from models.cpmp_transformer_v9 import CPMPTransformer
from training.training import load_model
from solvers.model import ModelSolver
from solvers.utils import summary
from settings import INSTANCE_FOLDER

# ── Cargar modelos ────────────────────────────────────────────────────────────
model_ale     = load_model(CPMPTransformer, 'v9_Original')
model_robusto = load_model(CPMPTransformer, 'v9_ROBUSTO')

H_MODEL = model_ale.hyperparams['H']
print(f"H_MODEL: {H_MODEL}")
print(f"v9_Original     cargado - H={model_ale.hyperparams['H']}")
print(f"v9_ROBUSTO cargado - H={model_robusto.hyperparams['H']}")

# ── Solvers ───────────────────────────────────────────────────────────────────
class SafeModelSolver(ModelSolver):
    """Envuelve ModelSolver capturando errores de movimientos inválidos."""
    def solve_from_path(self, instance_path, H, max_steps):
        try:
            return super().solve_from_path(instance_path, H, max_steps)
        except (ValueError, RuntimeError, IndexError):
            return False, max_steps

solver_ale     = SafeModelSolver(model_ale)
solver_robusto = SafeModelSolver(model_robusto)

# ── Benchmark CVS ─────────────────────────────────────────────────────────────
CVS_PATH = INSTANCE_FOLDER / 'benchmarks' / 'CVS'

cvs_folders = sorted(
    [d for d in os.listdir(CVS_PATH) if (CVS_PATH / d).is_dir()]
)

results = {}
total_ale = total_rob = total_inst = 0

def avg_steps(solved, steps):
    s = [t for ok, t in zip(solved, steps) if ok]
    return sum(s) / len(s) if s else float('nan')

SEP = '-' * 90
print(f"\n{'Categoria':>10}  {'H':>3} {'S':>3}  |  {'ALE resuelto':>14} {'avg pasos':>10}  |  {'ROBUSTO resuelto':>16} {'avg pasos':>10}")
print(SEP)

for folder_name in cvs_folders:
    H_real, S_real = [int(x) for x in folder_name.split('-')]
    max_steps = 100
    folder_path = CVS_PATH / folder_name

    dat_files = sorted([
        str(folder_path / f)
        for f in os.listdir(folder_path)
        if f.endswith('.dat')
    ])

    H_inf = H_real + 2  # convención CVS: capacidad de pila = H_real + 2

    solved_ale, steps_ale = [], []
    solved_rob, steps_rob = [], []
    for filepath in dat_files:
        s, t = solver_ale.solve_from_path(filepath, H_inf, max_steps)
        solved_ale.append(s); steps_ale.append(t)
        s, t = solver_robusto.solve_from_path(filepath, H_inf, max_steps)
        solved_rob.append(s); steps_rob.append(t)

    n     = len(solved_ale)
    n_ale = sum(solved_ale)
    n_rob = sum(solved_rob)
    avg_ale = avg_steps(solved_ale, steps_ale)
    avg_rob = avg_steps(solved_rob, steps_rob)

    results[folder_name] = {
        'H': H_real, 'S': S_real, 'n': n,
        'solved_ale': solved_ale, 'steps_ale': steps_ale,
        'solved_rob': solved_rob, 'steps_rob': steps_rob,
    }

    total_ale  += n_ale
    total_rob  += n_rob
    total_inst += n

    avg_ale_str = f'{avg_ale:>8.1f}' if n_ale > 0 else '       -'
    avg_rob_str = f'{avg_rob:>8.1f}' if n_rob > 0 else '       -'

    print(
        f'{folder_name:>10}  {H_real:>3} {S_real:>3}  |  '
        f'{n_ale:>3}/{n:<3} ({n_ale/n:>5.1%}) {avg_ale_str}  |  '
        f'{n_rob:>3}/{n:<3} ({n_rob/n:>5.1%}) {avg_rob_str}'
    )

print(SEP)
all_avg_ale = avg_steps(
    [s for r in results.values() for s in r['solved_ale']],
    [s for r in results.values() for s in r['steps_ale']]
)
all_avg_rob = avg_steps(
    [s for r in results.values() for s in r['solved_rob']],
    [s for r in results.values() for s in r['steps_rob']]
)
print(
    f"{'TOTAL':>10}  {'':>3} {'':>3}  |  "
    f'{total_ale:>3}/{total_inst:<3} ({total_ale/total_inst:>5.1%}) {all_avg_ale:>8.1f}  |  '
    f'{total_rob:>3}/{total_inst:<3} ({total_rob/total_inst:>5.1%}) {all_avg_rob:>8.1f}'
)

# ── Resumen final (mismo formato que validation.ipynb) ────────────────────────
all_solved_ale = [s for r in results.values() for s in r['solved_ale']]
all_steps_ale  = [s for r in results.values() for s in r['steps_ale']]
all_solved_rob = [s for r in results.values() for s in r['solved_rob']]
all_steps_rob  = [s for r in results.values() for s in r['steps_rob']]

print('\n=== v9_ALE ===')
summary(all_solved_ale, all_steps_ale)

print('\n=== v9_ROBUSTO ===')
summary(all_solved_rob, all_steps_rob)
