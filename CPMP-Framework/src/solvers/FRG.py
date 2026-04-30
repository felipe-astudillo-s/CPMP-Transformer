from settings import INSTANCE_FOLDER, FRG_PATH, FRG_LINUX_PATH
from solvers.solver import Solver
from cpmp.layout import read_file
from generation.adapters import *
import subprocess
import sys
import os


def _windows_to_wsl_path(path):
    s = str(path)
    if len(s) >= 2 and s[1] == ':':
        drive = s[0].lower()
        rest = s[2:].replace('\\', '/')
        return f'/mnt/{drive}{rest}'
    return s.replace('\\', '/')


class FRGSolver(Solver):
    def __init__(self):
        super().__init__("FRG")

    def solve_from_path(self, instance_path, H, max_steps):
        layout = read_file(instance_path, H)
        pid = os.getpid()
        filepath = INSTANCE_FOLDER / f"tmp_{pid}.txt"

        try:
            self.lay2file(layout, filepath)

            if sys.platform == 'win32':
                wsl_bin = _windows_to_wsl_path(FRG_LINUX_PATH)
                wsl_file = _windows_to_wsl_path(filepath)
                cmd = ["wsl", "-d", "Ubuntu", "--", wsl_bin,
                       str(H), wsl_file, "1.2", str(max_steps), "0", "--no-assignement", "2"]
            else:
                cmd = [FRG_PATH, str(H), filepath, "1.2", str(max_steps), "0", "--no-assignement", "2"]

            result = subprocess.run(cmd, check=True, text=True, capture_output=True)

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
