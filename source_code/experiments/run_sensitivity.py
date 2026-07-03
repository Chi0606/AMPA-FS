"""
Parameter Sensitivity Analysis for AMPA-FS (with checkpoint resume).

For each (param_name, dataset, param_value) triple we run SENSITIVITY_RUNS
independent trials and record the best-fitness distribution.

Checkpointing: the dictionary is persisted to
    results/sensitivity_results_checkpoint.pkl
after every completed (param, dataset, value) cell. On re-launch the script
reloads the checkpoint and skips any cells already containing >= num_runs
finished trials.
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
from algorithms.ampa_fs import AMPA_FS


CHECKPOINT_FILE = os.path.join(config.RESULTS_DIR,
                               "sensitivity_results_checkpoint.pkl")


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"  [WARN] could not read checkpoint: {e}")
    return {}


def save_checkpoint(results):
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    tmp = CHECKPOINT_FILE + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(results, f)
    os.replace(tmp, CHECKPOINT_FILE)  # atomic on Windows & Linux


def run_sensitivity(dataset_names=None, num_runs=None):
    """Run parameter sensitivity analysis with checkpoint resume."""
    if dataset_names is None:
        dataset_names = config.SENSITIVITY_DATASETS
    if num_runs is None:
        num_runs = config.SENSITIVITY_RUNS

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    all_results = load_checkpoint()
    if all_results:
        print(f"[RESUME] loaded checkpoint with "
              f"{sum(len(all_results[p].get(d, {})) for p in all_results for d in all_results[p])}"
              f" pre-computed cells")

    # ---- Total task count for progress estimation ----
    total_cells = sum(len(vals) for vals in config.SENSITIVITY_PARAMS.values()) \
                  * len(dataset_names)
    total_runs  = total_cells * num_runs
    done_runs   = sum(len(all_results[p][d][v].get("fitness_all", []))
                      for p in all_results
                      for d in all_results.get(p, {})
                      for v in all_results[p].get(d, {}))
    print(f"\nTotal scope: {total_cells} (param,dataset,value) cells x "
          f"{num_runs} runs = {total_runs} trials")
    print(f"Already done: {done_runs} trials "
          f"({100.0 * done_runs / max(total_runs, 1):.1f}%)\n")

    t_start = time.time()
    cells_processed = 0

    for param_name, param_values in config.SENSITIVITY_PARAMS.items():
        print(f"\n{'='*60}\nParameter: {param_name}\n{'='*60}")
        all_results.setdefault(param_name, {})

        for ds_name in dataset_names:
            try:
                X, y = load_dataset(ds_name)
            except Exception as e:
                print(f"  [ERROR] {ds_name}: {e}")
                continue

            all_results[param_name].setdefault(ds_name, {})

            for val in param_values:
                cells_processed += 1

                # Resume: skip if already complete
                existing = all_results[param_name][ds_name].get(val, {})
                if len(existing.get("fitness_all", [])) >= num_runs:
                    print(f"  [{cells_processed:3d}/{total_cells}] "
                          f"{ds_name:15s} | {param_name}={val} "
                          f"| [SKIP] Fitness={existing['fitness_mean']:.4f}")
                    continue

                fitness_runs = []
                t_cell_start = time.time()

                for run in range(num_runs):
                    seed = run * 100 + 42
                    kwargs = dict(
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

                    if param_name == "pop_size":
                        kwargs["pop_size"] = val
                    elif param_name == "stag_threshold":
                        kwargs["stag_threshold"] = val
                    elif param_name == "alpha":
                        kwargs["alpha"] = val
                        kwargs["beta"] = 1.0 - val
                    elif param_name == "gamma_max":
                        kwargs["gamma_max"] = val

                    alg = AMPA_FS(**kwargs)
                    try:
                        result = alg.optimize(X, y)
                        fitness_runs.append(result["best_fitness"])
                    except Exception as e:
                        print(f"    [RUN-ERROR] {e}")
                        fitness_runs.append(np.inf)

                avg_fit  = float(np.mean(fitness_runs))
                std_fit  = float(np.std(fitness_runs))
                all_results[param_name][ds_name][val] = {
                    "fitness_mean": avg_fit,
                    "fitness_std":  std_fit,
                    "fitness_all":  fitness_runs,
                }
                save_checkpoint(all_results)

                # ---- progress print ----
                dt_cell = time.time() - t_cell_start
                elapsed = time.time() - t_start
                cells_remaining = total_cells - cells_processed
                eta_s = (elapsed / max(1, cells_processed)) * cells_remaining
                pct = 100.0 * cells_processed / total_cells
                print(f"  [{cells_processed:3d}/{total_cells}] "
                      f"{ds_name:15s} | {param_name}={val} "
                      f"| Fitness={avg_fit:.4f}+-{std_fit:.4f} "
                      f"| {dt_cell:.0f}s "
                      f"| {pct:.1f}% ETA: {eta_s/60:.1f}m")

    # ---- Final save (also to "results.pkl", the non-checkpoint name) ----
    pkl_path = os.path.join(config.RESULTS_DIR, "sensitivity_results.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(all_results, f)

    # ---- CSV summary ----
    rows = []
    for param_name in all_results:
        for ds_name in all_results[param_name]:
            for val, res in all_results[param_name][ds_name].items():
                rows.append({
                    "Parameter": param_name,
                    "Value": val,
                    "Dataset": ds_name,
                    "Fitness_Mean": res["fitness_mean"],
                    "Fitness_Std": res["fitness_std"],
                })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(config.RESULTS_DIR, "sensitivity_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[DONE] Sensitivity results saved to:\n  {pkl_path}\n  {csv_path}")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AMPA-FS Parameter Sensitivity")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--runs", type=int, default=None)
    args = parser.parse_args()

    run_sensitivity(args.datasets, args.runs)
