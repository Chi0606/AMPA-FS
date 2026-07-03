"""
Benchmark: Directly run single optimization runs with full settings
to get accurate time estimates for the complete experiment.
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.data_loader import load_dataset
from algorithms.ampa_fs import AMPA_FS
from algorithms.bpso import BPSO
from algorithms.bgwo import BGWO
from utils.fitness import evaluate_final


def main():
    print("=" * 70)
    print("  AMPA-FS Time Benchmark (actual run measurement)")
    print("  Using KNN evaluator with solution caching")
    print("=" * 70)

    datasets = ["Iris", "Wine", "BreastCancer"]
    alg_factories = {
        "AMPA_FS": lambda: AMPA_FS(pop_size=30, max_iter=100, seed=42),
        "BPSO":    lambda: BPSO(pop_size=30, max_iter=100, seed=42),
        "BGWO":    lambda: BGWO(pop_size=30, max_iter=100, seed=42),
    }

    print(f"\n  Running 1 full run (pop=30, iter=100) per (dataset, algorithm)...\n")

    results = []

    for ds_name in datasets:
        try:
            X, y = load_dataset(ds_name)
        except Exception:
            print(f"  {ds_name}: SKIP")
            continue

        n_samples, n_features = X.shape
        print(f"  --- {ds_name} ({n_samples}x{n_features}) ---")

        for alg_name, factory in alg_factories.items():
            alg = factory()
            t0 = time.time()
            result = alg.optimize(X, y)
            elapsed = time.time() - t0

            # Also measure RF final report time
            t1 = time.time()
            final_res, n_sel = evaluate_final(result["best_solution"], X, y)
            rf_time = time.time() - t1

            print(f"    {alg_name:10s} | Search: {elapsed:6.2f}s | "
                  f"RF report: {rf_time:.2f}s | "
                  f"Acc(KNN)={result['best_accuracy']:.4f} | "
                  f"Acc(RF)={final_res['RF']['accuracy_mean']:.4f} | "
                  f"Feats={result['best_n_selected']}/{n_features}")

            results.append({
                "dataset": ds_name,
                "algorithm": alg_name,
                "features": n_features,
                "search_time": elapsed,
                "rf_report_time": rf_time,
            })

    # Summary and extrapolation
    print(f"\n{'=' * 70}")
    print(f"  Full Experiment Time Estimates")
    print(f"{'=' * 70}")

    n_algorithms = 10
    n_runs = 30

    # Average times per dataset
    ds_times = {}
    for r in results:
        ds = r["dataset"]
        if ds not in ds_times:
            ds_times[ds] = {"search": [], "report": []}
        ds_times[ds]["search"].append(r["search_time"])
        ds_times[ds]["report"].append(r["rf_report_time"])

    total_est = 0
    print(f"\n  Per-dataset estimate (x{n_algorithms} algorithms x {n_runs} runs):\n")
    for ds, times in ds_times.items():
        avg_search = np.mean(times["search"])
        avg_report = np.mean(times["report"])
        # Scale: n_algorithms/3 tested x n_runs
        ds_total = avg_search * n_algorithms * n_runs + avg_report * n_algorithms * n_runs
        total_est += ds_total
        print(f"    {ds:15s} | avg search/run: {avg_search:.2f}s | "
              f"total: {ds_total/60:.1f} min ({ds_total/3600:.2f} h)")

    avg_per_ds = total_est / len(ds_times)

    print(f"\n  Extrapolated total experiment time:")
    for n_ds in [3, 10, 15, 20]:
        est = avg_per_ds * n_ds
        if est < 3600:
            print(f"    {n_ds:2d} datasets: {est/60:.0f} min")
        else:
            print(f"    {n_ds:2d} datasets: {est/3600:.1f} hours")

    print()


if __name__ == "__main__":
    main()
