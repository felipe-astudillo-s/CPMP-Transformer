"""
Benchmark CVS comparativo: v9 vs v9_new_gen_data.
Usa ModelSolver y benchmark_cvs (utilidad real del framework) en lugar
de duplicar lógica. Genera dos CSVs por modelo:
  - benchmark_<tag>_per_instance.csv (folder, S, H, file, solved, steps)
  - benchmark_<tag>_summary.csv (Categoría, Total, Resueltas, Tasa de Éxito, Pasos Promedio)
"""
import os
import sys
import argparse
import torch

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

from models.cpmp_transformer_v9 import CPMPTransformer
from training.training import load_model
from solvers.model import ModelSolver
from solvers.benchmark import benchmark_cvs, folder_stats, save_results_csv, print_benchmark_results


class SafeModelSolver(ModelSolver):
    """ModelSolver que captura errores de forma de tensor (e.g. stacks que
    desbordan H_solver porque el modelo eligió un movimiento inválido tras
    agotar estados no visitados) y los reporta como no-resuelto."""
    def solve_from_path(self, instance_path, H, max_steps):
        try:
            return super().solve_from_path(instance_path, H, max_steps)
        except (ValueError, RuntimeError, IndexError) as e:
            return False, max_steps

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(base_dir, "results")
CVS_DIR = os.path.join(base_dir, "instances", "CVS")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run(tag):
    # ModelSolver no traslada tensores a CUDA, mantenemos el modelo en CPU
    # para evitar errores de device mismatch.
    print(f"\n{'='*72}\n>> Benchmark {tag} (cpu)\n{'='*72}")
    model = load_model(CPMPTransformer, tag)
    model.eval()
    solver = SafeModelSolver(model)
    max_steps_fn = lambda s, h: max(50, s * h * 4)
    results = benchmark_cvs(solver, cvs_path=CVS_DIR, max_steps_fn=max_steps_fn)
    print_benchmark_results(results)
    # Per-instance CSV
    per_inst_csv = os.path.join(RESULTS_DIR, f"benchmark_{tag}_per_instance.csv")
    save_results_csv(results, per_inst_csv)
    # Summary CSV
    import csv
    summary_csv = os.path.join(RESULTS_DIR, f"benchmark_{tag}_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Categoria", "S", "H", "Total", "Resueltas", "Tasa_Exito_pct", "Pasos_Promedio", "Pasos_Min", "Pasos_Max"])
        for folder_name, fr in sorted(results.items()):
            st = folder_stats(fr)
            w.writerow([folder_name, fr["S"], fr["H"], st["total"], st["solved"],
                        round(st["solve_rate"], 2), round(st["avg_steps"], 2),
                        st["min_steps"], st["max_steps"]])
    print(f"Summary saved to: {summary_csv}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, required=True)
    args = parser.parse_args()
    run(args.tag)
