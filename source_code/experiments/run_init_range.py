"""
Clip-range comparison experiment (revision T2.7, reviewer R3-5).

Runs AMPA-FS with the paper's continuous-space clip bound [-6, 6] against a
narrower [-2, 2] bound on a set of representative datasets, using exactly the
same protocol as the main experiment (same seeds, same 10-fold-CV final
evaluation). Output: ``results/init_range_results.pkl`` and a summary CSV
``results/init_range_summary.csv``.

Resumable: already-computed (dataset, bound) cells are skipped.
"""

import os
import sys
import time
import pickle
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from datasets.data_loader import load_dataset
from utils.fitness import evaluate_final
from algorithms.ampa_fs import AMPA_FS

DEFAULT_DATASETS = ["Wine", "BreastCancer", "Ionosphere", "Sonar", "Vehicle"]
BOUNDS = [6.0, 2.0]
PKL = os.path.join(config.RESULTS_DIR, "init_range_results.pkl")
CSV = os.path.join(config.RESULTS_DIR, "init_range_summary.csv")


def make_ampa(clip_bound, seed):
    return AMPA_FS(
        pop_size=config.POP_SIZE, max_iter=config.MAX_ITER,
        alpha=config.ALPHA, beta=config.BETA, seed=seed,
        fads_prob=config.FADS_PROB, stag_threshold=config.STAG_THRESHOLD,
        mutation_f=config.MUTATION_F, lambda_max=config.LAMBDA_MAX,
        gamma_min=config.GAMMA_MIN, gamma_max=config.GAMMA_MAX,
        tent_z0=config.TENT_Z0,
        use_chaotic_init=True, use_adaptive_step=True,
        use_elite_mutation=True, clip_bound=clip_bound,
    )


def main(datasets=None, num_runs=30):
    datasets = datasets or DEFAULT_DATASETS
    results = {}
    if os.path.exists(PKL):
        with open(PKL, "rb") as f:
            results = pickle.load(f)
        print(f"[RESUME] loaded {list(results.keys())}")

    for ds_name in datasets:
        X, y = load_dataset(ds_name)
        results.setdefault(ds_name, {})
        for bound in BOUNDS:
            key = f"clip{bound:g}"
            if len(results[ds_name].get(key, {}).get("final_rf_acc", [])) >= num_runs:
                print(f"[SKIP] {ds_name} {key}")
                continue
            rr = {"fitness": [], "n_selected": [], "time": [],
                  "final_rf_acc": [], "best_solutions": []}
            t0 = time.time()
            for run in range(num_runs):
                seed = run * 100 + 42
                alg = make_ampa(bound, seed)
                res = alg.optimize(X, y)
                rr["fitness"].append(res["best_fitness"])
                rr["n_selected"].append(res["best_n_selected"])
                rr["time"].append(res["elapsed_time"])
                rr["best_solutions"].append(res["best_solution"])
                final_res, _ = evaluate_final(res["best_solution"], X, y,
                                              random_state=seed)
                rr["final_rf_acc"].append(final_res["RF"]["accuracy_mean"])
            results[ds_name][key] = rr
            with open(PKL, "wb") as f:
                pickle.dump(results, f)
            print(f"[OK] {ds_name:13s} {key}: RF={np.mean(rr['final_rf_acc']):.4f}"
                  f"+/-{np.std(rr['final_rf_acc']):.4f} fit={np.mean(rr['fitness']):.4f} "
                  f"sel={np.mean(rr['n_selected']):.1f} ({time.time()-t0:.0f}s)")

    rows = []
    for ds_name, bounds in results.items():
        for key, rr in bounds.items():
            rows.append({
                "Dataset": ds_name, "Bound": key,
                "RF_Acc_Mean": np.mean(rr["final_rf_acc"]),
                "RF_Acc_Std": np.std(rr["final_rf_acc"]),
                "Fitness_Mean": np.mean(rr["fitness"]),
                "Selected_Mean": np.mean(rr["n_selected"]),
                "Time_Mean": np.mean(rr["time"]),
            })
    pd.DataFrame(rows).to_csv(CSV, index=False)
    print(f"[DONE] summary -> {CSV}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--runs", type=int, default=30)
    args = p.parse_args()
    main(args.datasets, args.runs)
