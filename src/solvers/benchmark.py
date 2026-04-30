import os
import csv
from pathlib import Path
from settings import INSTANCE_FOLDER


# ---------------------------------------------------------------------------
# Core benchmark functions
# ---------------------------------------------------------------------------

def benchmark_folder(solver, folder_path, H, max_steps):
    """
    Run solver on all .dat files in folder_path.

    Returns a list of dicts: {file, solved, steps}
    """
    results = []
    folder_path = Path(folder_path)

    for filename in sorted(os.listdir(folder_path)):
        if not filename.endswith(".dat"):
            continue
        filepath = folder_path / filename
        solved, steps = solver.solve_from_path(str(filepath), H, max_steps)
        results.append({"file": filename, "solved": solved, "steps": steps})

    return results


def benchmark_cvs(solver, cvs_path=None, max_steps_fn=None):
    """
    Run benchmark on every sub-folder inside the CVS directory.

    Each sub-folder is expected to follow the naming convention  'S-H'
    (e.g. '5-7', '10-10'), where S = number of stacks, H = stack height.

    Parameters
    ----------
    solver       : any Solver instance (e.g. ModelSolver)
    cvs_path     : path to the CVS folder; defaults to instances/benchmarks/CVS
    max_steps_fn : callable(S, H) -> int.  Defaults to max(50, S * H * 4).

    Returns
    -------
    dict  keyed by folder name, each value is:
        {S, H, max_steps, instances: [{file, solved, steps}, ...]}
    """
    if cvs_path is None:
        cvs_path = INSTANCE_FOLDER / "benchmarks" / "CVS"
    cvs_path = Path(cvs_path)

    if max_steps_fn is None:
        max_steps_fn = lambda s, h: max(50, s * h * 4)

    # CVS instances are fully packed (all stacks at H_folder capacity).
    # We add a fixed buffer of +2 so the solver has room to move containers,
    # matching the existing validation convention (data5-5 solved with H=7 = 5+2).
    H_BUFFER = 2

    all_results = {}

    for folder_name in sorted(os.listdir(cvs_path)):
        folder_path = cvs_path / folder_name
        if not folder_path.is_dir():
            continue

        try:
            parts = folder_name.split("-")
            H_folder, S = int(parts[0]), int(parts[1])  # convention: H-S (e.g. "10-6" = H=10, S=6)
        except (ValueError, IndexError):
            continue

        H_solver = H_folder + H_BUFFER
        max_steps = max_steps_fn(S, H_solver)
        print(f"  [{folder_name}]  S={S}  H={H_folder}  H_solver={H_solver}  max_steps={max_steps} ...", end=" ", flush=True)

        instances = benchmark_folder(solver, folder_path, H_solver, max_steps)
        all_results[folder_name] = {
            "S": S,
            "H": H_folder,
            "H_solver": H_solver,
            "max_steps": max_steps,
            "instances": instances,
        }

        solved_count = sum(r["solved"] for r in instances)
        print(f"solved {solved_count}/{len(instances)}")

    return all_results


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def folder_stats(folder_result):
    """
    Compute summary statistics for one folder result entry.

    Returns dict: {total, solved, solve_rate, avg_steps, min_steps, max_steps}
    """
    instances = folder_result["instances"]
    total = len(instances)
    solved_instances = [r for r in instances if r["solved"]]
    solved = len(solved_instances)
    solve_rate = (solved / total * 100) if total > 0 else 0.0
    solved_steps = [r["steps"] for r in solved_instances]

    return {
        "total": total,
        "solved": solved,
        "solve_rate": solve_rate,
        "avg_steps": (sum(solved_steps) / len(solved_steps)) if solved_steps else 0.0,
        "min_steps": min(solved_steps) if solved_steps else 0,
        "max_steps": max(solved_steps) if solved_steps else 0,
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_folder_result(folder_name, folder_result):
    """Print a one-line summary for a single folder."""
    stats = folder_stats(folder_result)
    s = folder_result["S"]
    h = folder_result["H"]
    print(
        f"  {folder_name:<8}  S={s:<3} H={h:<3}  "
        f"solved={stats['solved']}/{stats['total']}  "
        f"({stats['solve_rate']:5.1f}%)  "
        f"avg_steps={stats['avg_steps']:6.1f}  "
        f"[{stats['min_steps']}-{stats['max_steps']}]"
    )


def print_benchmark_results(all_results):
    """Print a formatted summary table for all CVS folders."""
    print("\n" + "=" * 80)
    print(f"{'Folder':<8}  {'S':<4} {'H':<4} {'H_s':<4}  {'Solved':>10}  {'Rate':>7}  "
          f"{'Avg':>8}  {'Min-Max':>12}")
    print("-" * 80)

    total_inst = total_solved = total_steps_sum = total_steps_n = 0

    for folder_name, folder_result in sorted(all_results.items()):
        stats = folder_stats(folder_result)
        s = folder_result["S"]
        h = folder_result["H"]
        hs = folder_result.get("H_solver", h)
        steps_info = (
            f"{stats['avg_steps']:6.1f}  [{stats['min_steps']}-{stats['max_steps']}]"
            if stats["solved"] > 0
            else "     -  [-]"
        )
        print(
            f"  {folder_name:<8}  {s:<4} {h:<4} {hs:<4}  "
            f"{stats['solved']:>4}/{stats['total']:<4}  "
            f"{stats['solve_rate']:6.1f}%  "
            f"{steps_info}"
        )
        total_inst += stats["total"]
        total_solved += stats["solved"]
        if stats["solved"] > 0:
            total_steps_sum += stats["avg_steps"] * stats["solved"]
            total_steps_n += stats["solved"]

    overall_rate = (total_solved / total_inst * 100) if total_inst > 0 else 0.0
    overall_avg = (total_steps_sum / total_steps_n) if total_steps_n > 0 else 0.0

    print("-" * 80)
    print(
        f"  {'TOTAL':<8}  {'':4} {'':4} {'':4}  "
        f"{total_solved:>4}/{total_inst:<4}  "
        f"{overall_rate:6.1f}%  "
        f"{overall_avg:6.1f}"
    )
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_results_csv(all_results, output_path):
    """
    Save per-instance benchmark results to a CSV file.

    Columns: folder, S, H, file, solved, steps
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["folder", "S", "H", "file", "solved", "steps"]
        )
        writer.writeheader()
        for folder_name, folder_result in sorted(all_results.items()):
            for inst in folder_result["instances"]:
                writer.writerow(
                    {
                        "folder": folder_name,
                        "S": folder_result["S"],
                        "H": folder_result["H"],
                        "file": inst["file"],
                        "solved": inst["solved"],
                        "steps": inst["steps"],
                    }
                )

    print(f"Results saved to: {output_path}")
