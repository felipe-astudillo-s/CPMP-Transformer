import sys, os
sys.path.insert(0, os.path.abspath('src'))
from pathlib import Path

bf_path = Path('instances/benchmarks/BF')
print('Contenido de BF:')
for item in sorted(bf_path.iterdir()):
    print('  ' + item.name)

print()
for size_folder in sorted(bf_path.iterdir()):
    if not size_folder.is_dir():
        continue
    files = sorted(size_folder.glob('*.dat'))
    n_files = len(files)
    print('=== BF/' + size_folder.name + ' (' + str(n_files) + ' instancias) ===')
    for f in files[:4]:
        with open(f) as fh:
            first = next(fh).split()
            S, C = int(first[0]), int(first[1])
            stacks = []
            for _ in range(S):
                line = [int(x) for x in next(fh).split()]
                T = line[0]
                stacks.append(line[1:T+1])
        all_vals = [c for s in stacks for c in s]
        n = len(all_vals)
        k_unique = len(set(all_vals))
        heights = [len(s) for s in stacks]
        max_h = max(heights)
        fill = n / (S * max_h)
        print('  ' + f.name + ': S=' + str(S) + ', C=' + str(C) + ', N=' + str(n) + ', k_grupos=' + str(k_unique) + ', H_max=' + str(max_h) + ', alturas=' + str(heights) + ', fill=' + str(round(fill*100, 1)) + '%')
