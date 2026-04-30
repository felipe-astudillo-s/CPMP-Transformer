from settings import INSTANCE_FOLDER, FRG_PATH
from solvers.solver import Solver
from cpmp.layout import read_file
from generation.adapters import *
import subprocess
import os


class FRGSolver(Solver): 
    def __init__(self):
        super().__init__("FRG")
     
    def solve_from_path(self, instance_path, H, max_steps):
        layout = read_file(instance_path, H)
        pid = os.getpid()
        filepath = INSTANCE_FOLDER / f"tmp_{pid}.txt"

        try:
            self.lay2file(layout, filepath)

            result = subprocess.run(
                [FRG_PATH, str(H), filepath, "1.2", str(max_steps), "0", "--no-assignement", "2"],
                check=True,
                text=True,
                capture_output=True
            )
            
            output_str = result.stdout.split('\t')[0].strip()
            if not output_str.isdigit():
                return False, float('inf')

            return True, int(output_str)
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def lay2file(self, layout, filename):
        S = layout.stacks

        with open(filename, "w") as f:
            num_sublists = len(S)
            sum_lengths = sum(len(sublist) for sublist in S)
            f.write(f"{num_sublists} {sum_lengths}\n")
            for sublist in S:
                f.write(str(len(sublist)) +" " + " ".join(str(x) for x in sublist) + "\n")