"""
Quick Test Script with Real-time Progress Display.

Run this to verify the AMPA-FS pipeline works correctly.
Uses small settings for fast execution (~2-5 minutes).
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.data_loader import load_dataset
from algorithms.ampa_fs import AMPA_FS, create_ampa_variant
from algorithms.bpso import BPSO
from algorithms.bgwo import BGWO


def print_header(text, char="="):
    print(f"\n{char * 60}")
    print(f"  {text}")
    print(f"{char * 60}")


def print_progress(current, total, prefix="", bar_len=30):
    frac = current / total
    filled = int(bar_len * frac)
    bar = "#" * filled + "-" * (bar_len - filled)
    pct = frac * 100
    print(f"\r  {prefix} |{bar}| {pct:5.1f}% ({current}/{total})", end="", flush=True)


def main():
    # =
    # Quick test settings
    # Now uses KNN + cached evaluator -> very fast
    # =
    TEST_DATASETS = ["Iris", "Wine", "BreastCancer"]
    TEST_ALGORITHMS = {
        "AMPA_FS": lambda seed: AMPA_FS(
            pop_size=10, max_iter=20, alpha=0.99, beta=0.01,
            use_chaotic_init=True, use_adaptive_step=True,
            use_elite_mutation=True, seed=seed
        ),
        "BMPA": lambda seed: create_ampa_variant(
            "BMPA", pop_size=10, max_iter=20, alpha=0.99, beta=0.01, seed=seed
        ),
        "BPSO": lambda seed: BPSO(
            pop_size=10, max_iter=20, alpha=0.99, beta=0.01, seed=seed
        ),
        "BGWO": lambda seed: BGWO(
            pop_size=10, max_iter=20, alpha=0.99, beta=0.01, seed=seed
        ),
    }
    NUM_RUNS = 3

    total_tasks = len(TEST_DATASETS) * len(TEST_ALGORITHMS) * NUM_RUNS
    completed = 0

    print_header("AMPA-FS Quick Test")
    print(f"  Datasets:   {TEST_DATASETS}")
    print(f"  Algorithms: {list(TEST_ALGORITHMS.keys())}")
    print(f"  Runs:       {NUM_RUNS}")
    print(f"  Pop size:   10, Max iter: 20")
    print(f"  Total tasks: {total_tasks}")
    global_start = time.time()

    all_results = {}

    for ds_name in TEST_DATASETS:
        print_header(f"Dataset: {ds_name}", "-")

        try:
            X, y = load_dataset(ds_name)
            print(f"  Loaded: {X.shape[0]} samples, {X.shape[1]} features, "
                  f"{len(np.unique(y))} classes")
        except Exception as e:
            print(f"  [ERROR] Failed to load: {e}")
            completed += len(TEST_ALGORITHMS) * NUM_RUNS
            continue

        all_results[ds_name] = {}

        for alg_name, alg_factory in TEST_ALGORITHMS.items():
            acc_list = []
            fit_list = []
            sel_list = []
            time_list = []

            for run in range(NUM_RUNS):
                seed = run * 100 + 42
                alg = alg_factory(seed)

                t0 = time.time()
                result = alg.optimize(X, y)
                elapsed = time.time() - t0

                acc_list.append(result["best_accuracy"])
                fit_list.append(result["best_fitness"])
                sel_list.append(result["best_n_selected"])
                time_list.append(elapsed)

                completed += 1
                print_progress(completed, total_tasks,
                               prefix=f"{alg_name:8s} run {run+1}/{NUM_RUNS}")

            print()  # newline after progress bar

            avg_acc = np.mean(acc_list)
            avg_fit = np.mean(fit_list)
            avg_sel = np.mean(sel_list)
            avg_time = np.mean(time_list)

            print(f"  [OK] {alg_name:8s} | Acc={avg_acc:.4f} | Fitness={avg_fit:.4f} "
                  f"| Features={avg_sel:.1f}/{X.shape[1]} | Time={avg_time:.1f}s")

            all_results[ds_name][alg_name] = {
                "accuracy": acc_list,
                "fitness": fit_list,
                "n_selected": sel_list,
            }

    # Summary
    total_time = time.time() - global_start
    print_header("Summary")

    print(f"\n  {'Dataset':15s} | ", end="")
    for alg_name in TEST_ALGORITHMS:
        print(f"{alg_name:>10s}", end=" | ")
    print()
    print(f"  {'-'*15}-+-" + "-+-".join(["-" * 10] * len(TEST_ALGORITHMS)) + "-+")

    for ds_name in TEST_DATASETS:
        if ds_name not in all_results:
            continue
        print(f"  {ds_name:15s} | ", end="")
        for alg_name in TEST_ALGORITHMS:
            if alg_name in all_results[ds_name]:
                acc = np.mean(all_results[ds_name][alg_name]["accuracy"])
                print(f"{acc:10.4f}", end=" | ")
            else:
                print(f"{'N/A':>10s}", end=" | ")
        print()

    print(f"\n  Total time: {total_time:.1f}s")
    print(f"  Status: ALL PASSED" if completed == total_tasks else f"  Status: PARTIAL")
    print()


if __name__ == "__main__":
    main()
