"""
Ablation Study Runner for AMPA-FS.

Tests all 8 ablation variants (V0-V7) on all datasets.
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from datasets.data_loader import load_dataset
from algorithms.ampa_fs import create_ampa_variant


def run_ablation(dataset_names=None, num_runs=None):
    """
    Run ablation study for all variants with per-variant resume support.

    Parameters
    ----------
    dataset_names : list of str or None
    num_runs : int or None
    """
    if dataset_names is None:
        dataset_names = config.DATASETS
    if num_runs is None:
        num_runs = config.NUM_RUNS

    variants = config.ABLATION_VARIANTS
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # Resume from checkpoint
    ckpt_path = os.path.join(config.RESULTS_DIR, "ablation_results_checkpoint.pkl")
    all_results = {}
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, "rb") as f:
                all_results = pickle.load(f)
            done_summary = {k: list(v.keys()) for k, v in all_results.items()}
            print(f"[RESUME] Loaded checkpoint with existing data:")
            for ds, vars_done in done_summary.items():
                print(f"  {ds}: {len(vars_done)}/{len(variants)} variants done")
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint: {e}. Starting fresh.")
            all_results = {}

    # Count tasks: total planned vs. actually to-run (skipping completed)
    total = len(dataset_names) * len(variants) * num_runs
    skipped = 0
    for ds in dataset_names:
        if ds in all_results:
            for v in variants:
                if v in all_results[ds] and len(all_results[ds][v].get("fitness", [])) >= num_runs:
                    skipped += num_runs
    to_run = total - skipped
    completed = 0  # newly completed this session
    t_start = time.time()

    print("=" * 60)
    print(f"  AMPA-FS Ablation Study")
    print(f"  Datasets: {len(dataset_names)}  Variants: {len(variants)}  Runs: {num_runs}")
    print(f"  Total tasks: {total}  (skip: {skipped}, to-run: {to_run})")
    print("=" * 60)

    for ds_name in dataset_names:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        try:
            X, y = load_dataset(ds_name)
        except Exception as e:
            print(f"  [ERROR] Failed to load {ds_name}: {e}")
            continue

        if ds_name not in all_results:
            all_results[ds_name] = {}

        for var_idx, var_name in enumerate(variants):
            # Skip if already fully done
            if (var_name in all_results[ds_name]
                    and len(all_results[ds_name][var_name].get("fitness", [])) >= num_runs):
                r = all_results[ds_name][var_name]
                print(f"  [{var_idx+1}/{len(variants)}] {var_name:10s} [SKIP - done] "
                      f"Fit={np.mean(r['fitness']):.4f} Acc={np.mean(r['accuracy']):.4f}")
                continue

            run_results = {
                "fitness": [], "accuracy": [], "n_selected": [],
                "time": [], "convergence": [],
            }
            var_t0 = time.time()

            for run in range(num_runs):
                seed = run * 100 + 42
                common = dict(
                    pop_size=config.POP_SIZE,
                    max_iter=config.MAX_ITER,
                    alpha=config.ALPHA,
                    beta=config.BETA,
                    fads_prob=config.FADS_PROB,
                    stag_threshold=config.STAG_THRESHOLD,
                    mutation_f=config.MUTATION_F,
                    lambda_max=config.LAMBDA_MAX,
                    gamma_min=config.GAMMA_MIN,
                    gamma_max=config.GAMMA_MAX,
                    tent_z0=config.TENT_Z0,
                    seed=seed,
                )
                alg = create_ampa_variant(var_name, **common)

                try:
                    result = alg.optimize(X, y)
                    run_results["fitness"].append(result["best_fitness"])
                    run_results["accuracy"].append(result["best_accuracy"])
                    run_results["n_selected"].append(result["best_n_selected"])
                    run_results["time"].append(result["elapsed_time"])
                    run_results["convergence"].append(result["convergence_curve"])
                except Exception as e:
                    print(f"\n  [ERROR] {var_name} run {run}: {e}")
                    run_results["fitness"].append(np.inf)
                    run_results["accuracy"].append(0.0)
                    run_results["n_selected"].append(X.shape[1])
                    run_results["time"].append(0.0)
                    run_results["convergence"].append([np.inf] * config.MAX_ITER)

                completed += 1
                elapsed = time.time() - t_start
                # ETA based on to-run tasks only
                eta = elapsed / completed * (to_run - completed) if completed > 0 else 0
                progress_abs = skipped + completed
                print(f"\r  {var_name:10s} run {run+1:2d}/{num_runs} "
                      f"[{progress_abs}/{total}] {progress_abs*100/total:.1f}% "
                      f"ETA: {eta/60:.1f}m",
                      end="", flush=True)

            avg_acc = np.mean(run_results["accuracy"])
            avg_fit = np.mean(run_results["fitness"])
            var_elapsed = time.time() - var_t0
            print(f"\n  [OK] {var_name:10s} | Fitness={avg_fit:.4f} "
                  f"| Acc={avg_acc:.4f} | {var_elapsed:.0f}s")
            all_results[ds_name][var_name] = run_results

            # Per-variant checkpoint (save after EACH variant)
            try:
                with open(ckpt_path, "wb") as f:
                    pickle.dump(all_results, f)
            except Exception as e:
                print(f"\n  [WARN] Checkpoint save failed: {e}")

    # Save final results
    pkl_path = os.path.join(config.RESULTS_DIR, "ablation_results.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(all_results, f)

    # Summary CSV
    rows = []
    for ds_name in dataset_names:
        if ds_name not in all_results:
            continue
        for var_name in variants:
            if var_name not in all_results[ds_name]:
                continue
            r = all_results[ds_name][var_name]
            rows.append({
                "Dataset": ds_name,
                "Variant": var_name,
                "Fitness_Mean": np.mean(r["fitness"]),
                "Fitness_Std": np.std(r["fitness"]),
                "Accuracy_Mean": np.mean(r["accuracy"]),
                "Accuracy_Std": np.std(r["accuracy"]),
                "Selected_Mean": np.mean(r["n_selected"]),
                "Time_Mean": np.mean(r["time"]),
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(config.RESULTS_DIR, "ablation_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nAblation results saved to {csv_path}")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AMPA-FS Ablation Study")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--runs", type=int, default=None)
    args = parser.parse_args()

    run_ablation(args.datasets, args.runs)
