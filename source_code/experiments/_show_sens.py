"""Quick sensitivity progress viewer."""
import os, pickle, time, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL  = os.path.join(BASE, "results", "sensitivity_results_checkpoint.pkl")

sys.path.insert(0, BASE)
import config

if not os.path.exists(PKL):
    print("No checkpoint yet.")
    sys.exit(0)

d = pickle.load(open(PKL, "rb"))
mtime = os.path.getmtime(PKL)
print(f"Checkpoint mtime: {time.strftime('%H:%M:%S', time.localtime(mtime))} "
      f"({time.time() - mtime:.0f}s ago)")
print()

# total cells
total = sum(len(v) for v in config.SENSITIVITY_PARAMS.values()) \
        * len(config.SENSITIVITY_DATASETS)

done = 0
print(f"{'Param':15s} {'Dataset':15s} {'Value':>8s} {'Fitness':>10s} {'+-std':>10s} {'Runs':>6s}")
print("-" * 70)
for pn in config.SENSITIVITY_PARAMS:
    if pn not in d:
        continue
    for ds in d[pn]:
        for v, r in d[pn][ds].items():
            n = len(r.get("fitness_all", []))
            if n >= config.SENSITIVITY_RUNS:
                done += 1
            print(f"{pn:15s} {ds:15s} {str(v):>8s} "
                  f"{r['fitness_mean']:>10.4f} "
                  f"{r.get('fitness_std', 0):>10.4f} "
                  f"{n:>6d}")

print("-" * 70)
print(f"Progress: {done}/{total} cells complete "
      f"({100.0 * done / max(total, 1):.1f}%)")
