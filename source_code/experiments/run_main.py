"""
Main Experiment Runner for AMPA-FS.

Runs all algorithms on all datasets for multiple independent runs.
Uses KNN evaluator for fast search + RF/KNN final reporting.
Real-time terminal progress display.
"""

import os
import sys
import time
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from datasets.data_loader import load_dataset
from utils.fitness import evaluate_final
from algorithms.ampa_fs import AMPA_FS
from algorithms.bpso import BPSO
from algorithms.bga import BGA
from algorithms.bgwo import BGWO
from algorithms.bwoa import BWOA
from algorithms.bssa import BSSA
from algorithms.bhho import BHHO
from algorithms.bsca import BSCA
from algorithms.bgoa import BGOA
from algorithms.bwso import BWSO
from algorithms.beho import BEHO
from algorithms.bcmaes import BCMAES


# -- helpers ------------------------------------------------------------------

def _bar(current, total, width=25):
    frac = current / max(total, 1)
    filled = int(width * frac)
    return "#" * filled + "-" * (width - filled)


def _fmt_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def get_algorithm(name, seed=None):
    common = dict(
        pop_size=config.POP_SIZE,
        max_iter=config.MAX_ITER,
        alpha=config.ALPHA,
        beta=config.BETA,
        seed=seed,
    )
    if name == "AMPA_FS":
        return AMPA_FS(
            fads_prob=config.FADS_PROB,
            stag_threshold=config.STAG_THRESHOLD,
            mutation_f=config.MUTATION_F,
            lambda_max=config.LAMBDA_MAX,
            gamma_min=config.GAMMA_MIN,
            gamma_max=config.GAMMA_MAX,
            tent_z0=config.TENT_Z0,
            use_chaotic_init=True, use_adaptive_step=True,
            use_elite_mutation=True, **common,
        )
    elif name == "BMPA":
        return AMPA_FS(
            fads_prob=config.FADS_PROB,
            use_chaotic_init=False, use_adaptive_step=False,
            use_elite_mutation=False, **common,
        )
    factories = {
        "BPSO": BPSO, "BGA": BGA, "BGWO": BGWO, "BWOA": BWOA,
        "BSSA": BSSA, "BHHO": BHHO, "BSCA": BSCA, "BGOA": BGOA,
        "BWSO": BWSO, "BEHO": BEHO, "BCMAES": BCMAES,
    }
    if name in factories:
        return factories[name](**common)
    raise ValueError(f"Unknown algorithm: {name}")


# -- main experiment ----------------------------------------------------------

def run_experiment(dataset_names=None, algorithm_names=None, num_runs=None,
                   resume=True):
    if dataset_names is None:
        dataset_names = config.DATASETS
    if algorithm_names is None:
        algorithm_names = config.ALGORITHMS
    if num_runs is None:
        num_runs = config.NUM_RUNS

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    n_ds = len(dataset_names)
    n_alg = len(algorithm_names)
    total_tasks = n_ds * n_alg * num_runs
    t_start = time.time()

    # ── Resume from checkpoint ──────────────────────────────────────
    all_results = {}
    ckpt_path = os.path.join(config.RESULTS_DIR, "main_results_checkpoint.pkl")
    partial_path = os.path.join(config.RESULTS_DIR, "main_results_partial.pkl")
    partial_results = {}  # Partial results for current/interrupted dataset
    if resume and os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, "rb") as f:
                all_results = pickle.load(f)
            skipped = list(all_results.keys())
            print(f"  [RESUME] Loaded checkpoint: {skipped}")
        except Exception as e:
            print(f"  [WARN] Could not load checkpoint: {e}")
            all_results = {}
    if resume and os.path.exists(partial_path):
        try:
            with open(partial_path, "rb") as f:
                partial_results = pickle.load(f)
            for ds, alg_data in partial_results.items():
                if ds not in all_results:
                    print(f"  [RESUME] Partial checkpoint: {ds} "
                          f"({list(alg_data.keys())})")
        except Exception as e:
            print(f"  [WARN] Could not load partial checkpoint: {e}")
            partial_results = {}
    completed = 0       # Total tasks processed (including skips, for display)
    real_completed = 0  # Only actual computed runs (for ETA rate calculation)

    print("=" * 65)
    print("  AMPA-FS Main Experiment")
    print("=" * 65)
    print(f"  Datasets:    {n_ds}  |  Algorithms: {n_alg}  |  Runs: {num_runs}")
    print(f"  Total tasks: {total_tasks}  |  Already done: {completed}")
    print(f"  Search:      KNN(k=5) + cached evaluator")
    print(f"  Report:      RF/KNN with 10-fold CV")
    print("=" * 65)

    for ds_idx, ds_name in enumerate(dataset_names):
        # Skip if all algorithms already done for this dataset
        if ds_name in all_results and all(
            alg in all_results[ds_name] and
            len(all_results[ds_name][alg].get("final_rf_acc", [])) >= num_runs
            for alg in algorithm_names
        ):
            completed += n_alg * num_runs
            print(f"\n  [{ds_idx+1}/{n_ds}] Dataset: {ds_name}  [SKIP - checkpoint]")
            continue

        print(f"\n{'=' * 65}")
        print(f"  [{ds_idx+1}/{n_ds}] Dataset: {ds_name}")
        print(f"{'=' * 65}")

        try:
            # High-dim revision datasets get filter pre-screening so the
            # wrapper KNN search stays tractable (Reviewer 3.4 / 4.11 / 5.4).
            prescreen = 200 if ds_name in getattr(config, "REVISION_DATASETS", []) else None
            X, y = load_dataset(ds_name, prescreen_k=prescreen)
            n_samples, n_features = X.shape
            n_classes = len(np.unique(y))
            print(f"  Loaded: {n_samples} samples, {n_features} features, "
                  f"{n_classes} classes")
        except Exception as e:
            print(f"  [ERROR] {e}")
            completed += n_alg * num_runs
            continue

        # Restore partial results for this dataset (if interrupted mid-way),
        # merging with any algorithms already present from the checkpoint so
        # completed cells are never recomputed.
        all_results.setdefault(ds_name, {})
        if ds_name in partial_results:
            for alg, data in partial_results[ds_name].items():
                all_results[ds_name].setdefault(alg, data)
            done_algs = list(all_results[ds_name].keys())
            print(f"  [RESUME] Restoring partial: {done_algs}")

        for alg_idx, alg_name in enumerate(algorithm_names):
            # Skip if this algorithm already done for this dataset (partial resume)
            if (alg_name in all_results.get(ds_name, {}) and
                    len(all_results[ds_name][alg_name].get("final_rf_acc", [])) >= num_runs):
                completed += num_runs
                avg_acc = np.mean(all_results[ds_name][alg_name]["final_rf_acc"])
                print(f"  [{alg_idx+1}/{n_alg}] {alg_name:10s}  [SKIP - partial] "
                      f"RF_Acc={avg_acc:.4f}")
                continue


            run_results = {
                "fitness": [], "accuracy": [], "n_selected": [],
                "time": [], "convergence": [], "best_solutions": [],
                "final_rf_acc": [], "final_knn_acc": [],
            }

            alg_t0 = time.time()

            for run in range(num_runs):
                completed += 1
                real_completed += 1
                print(f"\r  {alg_name:10s} run {run+1:2d}/{num_runs} "
                      f"|{_bar(completed, total_tasks)}| "
                      f"{completed*100/total_tasks:5.1f}% "
                      f"({completed}/{total_tasks})",
                      end="", flush=True)

                seed = run * 100 + 42
                alg = get_algorithm(alg_name, seed=seed)

                try:
                    result = alg.optimize(X, y)
                    run_results["fitness"].append(result["best_fitness"])
                    run_results["accuracy"].append(result["best_accuracy"])
                    run_results["n_selected"].append(result["best_n_selected"])
                    run_results["time"].append(result["elapsed_time"])
                    run_results["convergence"].append(result["convergence_curve"])
                    run_results["best_solutions"].append(result["best_solution"])

                    # Final evaluation with RF/KNN (10-fold CV)
                    final_res, _ = evaluate_final(result["best_solution"], X, y,
                                                   random_state=seed)
                    run_results["final_rf_acc"].append(final_res["RF"]["accuracy_mean"])
                    run_results["final_knn_acc"].append(final_res["KNN"]["accuracy_mean"])

                except Exception as e:
                    print(f"\n  [ERROR] {alg_name} run {run}: {e}")
                    run_results["fitness"].append(np.inf)
                    run_results["accuracy"].append(0.0)
                    run_results["n_selected"].append(n_features)
                    run_results["time"].append(0.0)
                    run_results["convergence"].append([np.inf] * config.MAX_ITER)
                    run_results["best_solutions"].append(np.ones(n_features))
                    run_results["final_rf_acc"].append(0.0)
                    run_results["final_knn_acc"].append(0.0)

            alg_elapsed = time.time() - alg_t0
            total_elapsed = time.time() - t_start
            # Use real_completed for ETA rate (excludes skipped tasks)
            real_remaining = total_tasks - completed
            if real_completed > 0:
                eta = total_elapsed / real_completed * real_remaining
            else:
                eta = 0

            all_results[ds_name][alg_name] = run_results

            avg_acc = np.mean(run_results["final_rf_acc"])
            avg_sel = np.mean(run_results["n_selected"])
            avg_fit = np.mean(run_results["fitness"])

            print(f"\n  [OK] {alg_name:10s} | RF_Acc={avg_acc:.4f} | "
                  f"Fit={avg_fit:.4f} | Sel={avg_sel:.1f}/{n_features} | "
                  f"{_fmt_time(alg_elapsed)} | ETA: {_fmt_time(eta)}")

            # Per-algorithm partial checkpoint
            try:
                if ds_name not in partial_results:
                    partial_results[ds_name] = {}
                partial_results[ds_name][alg_name] = run_results
                with open(partial_path, "wb") as f:
                    pickle.dump(partial_results, f)
            except Exception as e:
                print(f"  [WARN] Partial checkpoint save failed: {e}")

        # Per-dataset save (checkpoint)
        try:
            _save_results(all_results, dataset_names, algorithm_names,
                          suffix="_checkpoint")
            # Remove this dataset from partial checkpoint now that it's fully done
            if ds_name in partial_results:
                del partial_results[ds_name]
                with open(partial_path, "wb") as f:
                    pickle.dump(partial_results, f)
        except Exception as e:
            print(f"\n  [WARN] Checkpoint save failed: {e}")

    total_time = time.time() - t_start
    print(f"\n{'=' * 65}")
    print(f"  EXPERIMENT COMPLETE  |  Total time: {_fmt_time(total_time)}")
    print(f"{'=' * 65}")

    _save_results(all_results, dataset_names, algorithm_names)
    _print_summary_table(all_results, dataset_names, algorithm_names)

    return all_results


def _save_results(all_results, dataset_names, algorithm_names, suffix=""):
    pkl_path = os.path.join(config.RESULTS_DIR, f"main_results{suffix}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(all_results, f)

    rows = []
    for ds_name in dataset_names:
        if ds_name not in all_results:
            continue
        for alg_name in algorithm_names:
            if alg_name not in all_results[ds_name]:
                continue
            r = all_results[ds_name][alg_name]
            rows.append({
                "Dataset": ds_name,
                "Algorithm": alg_name,
                "Fitness_Mean": np.mean(r["fitness"]),
                "Fitness_Std": np.std(r["fitness"]),
                "KNN_Acc_Mean": np.mean(r["accuracy"]),
                "RF_Acc_Mean": np.mean(r.get("final_rf_acc", [0])),
                "RF_Acc_Std": np.std(r.get("final_rf_acc", [0])),
                "Selected_Mean": np.mean(r["n_selected"]),
                "Selected_Std": np.std(r["n_selected"]),
                "Time_Mean": np.mean(r["time"]),
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(config.RESULTS_DIR, f"main_summary{suffix}.csv")
    df.to_csv(csv_path, index=False)
    if not suffix:
        print(f"  Results saved to {pkl_path}")
        print(f"  Summary saved to {csv_path}")


def _print_summary_table(all_results, dataset_names, algorithm_names):
    print(f"\n{'=' * 65}")
    print("  RF Accuracy Summary (mean +/- std)")
    print(f"{'=' * 65}")

    header = f"  {'Dataset':14s}"
    for alg in algorithm_names:
        header += f" | {alg:>10s}"
    print(header)
    print("  " + "-" * (15 + 13 * len(algorithm_names)))

    for ds_name in dataset_names:
        if ds_name not in all_results:
            continue
        row = f"  {ds_name:14s}"
        for alg_name in algorithm_names:
            if alg_name not in all_results[ds_name]:
                row += f" | {'N/A':>10s}"
                continue
            r = all_results[ds_name][alg_name]
            acc = np.mean(r.get("final_rf_acc", [0]))
            row += f" |     {acc:.4f}"
        print(row)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AMPA-FS Main Experiment")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Dataset names")
    parser.add_argument("--algorithms", nargs="+", default=None,
                        help="Algorithm names")
    parser.add_argument("--runs", type=int, default=None,
                        help="Number of independent runs (default: 30)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore checkpoint and start fresh")
    args = parser.parse_args()

    run_experiment(args.datasets, args.algorithms, args.runs,
                   resume=not args.no_resume)
