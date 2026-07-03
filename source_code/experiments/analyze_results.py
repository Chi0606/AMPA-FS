"""
Analyze main experiment results: rankings, statistical tests, win/loss.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def main():
    for fname in ("main_results_checkpoint.pkl", "main_results.pkl"):
        pkl_path = os.path.join(config.RESULTS_DIR, fname)
        if os.path.exists(pkl_path):
            print(f"[Loading: {fname}]")
            with open(pkl_path, "rb") as f:
                all_results = pickle.load(f)
            break
    else:
        print("ERROR: No results file found.")
        return

    datasets = list(all_results.keys())
    algorithms = config.ALGORITHMS

    print("=" * 75)
    print("  AMPA-FS Experiment Analysis (30 runs)")
    print("=" * 75)

    # - 1. RF Accuracy Table -
    print(f"\n{'=' * 75}")
    print("  1. RF Accuracy (mean +/- std)")
    print(f"{'=' * 75}")

    acc_matrix = {}  # dataset -> {alg -> mean_acc}
    header = f"  {'Dataset':14s}"
    for a in algorithms:
        header += f" | {a:>8s}"
    print(header)
    print("  " + "-" * (15 + 11 * len(algorithms)))

    for ds in datasets:
        row = f"  {ds:14s}"
        acc_matrix[ds] = {}
        for alg in algorithms:
            r = all_results[ds][alg]
            mean_acc = np.mean(r["final_rf_acc"])
            std_acc = np.std(r["final_rf_acc"])
            acc_matrix[ds][alg] = mean_acc
            row += f" |   {mean_acc:.4f}"
        print(row)

    # - 2. Per-dataset ranking -
    print(f"\n{'=' * 75}")
    print("  2. Per-Dataset Ranking (by RF Accuracy, 1=best)")
    print(f"{'=' * 75}")

    rank_matrix = {}
    for ds in datasets:
        accs = [(alg, acc_matrix[ds][alg]) for alg in algorithms]
        accs.sort(key=lambda x: -x[1])
        rank_matrix[ds] = {}
        for rank, (alg, _) in enumerate(accs, 1):
            rank_matrix[ds][alg] = rank

    header = f"  {'Dataset':14s}"
    for a in algorithms:
        header += f" | {a:>6s}"
    print(header)
    print("  " + "-" * (15 + 9 * len(algorithms)))

    for ds in datasets:
        row = f"  {ds:14s}"
        for alg in algorithms:
            r = rank_matrix[ds][alg]
            marker = " *" if r == 1 else ""
            row += f" | {r:4d}{marker:2s}"
        print(row)

    # Average rank
    avg_ranks = {}
    for alg in algorithms:
        avg_ranks[alg] = np.mean([rank_matrix[ds][alg] for ds in datasets])

    row = f"  {'Avg Rank':14s}"
    for alg in algorithms:
        row += f" | {avg_ranks[alg]:6.2f}"
    print(row)

    # - 3. Win/Tie/Loss of AMPA_FS vs each -
    print(f"\n{'=' * 75}")
    print("  3. AMPA_FS Win/Tie/Loss vs Each Algorithm (RF Accuracy)")
    print(f"{'=' * 75}")

    for alg in algorithms[1:]:
        wins, ties, losses = 0, 0, 0
        for ds in datasets:
            a = acc_matrix[ds]["AMPA_FS"]
            b = acc_matrix[ds][alg]
            if a > b + 0.001:
                wins += 1
            elif b > a + 0.001:
                losses += 1
            else:
                ties += 1
        print(f"  vs {alg:10s}: W={wins}  T={ties}  L={losses}")

    # - 4. Feature selection ratio -
    print(f"\n{'=' * 75}")
    print("  4. Feature Selection Ratio (selected / total)")
    print(f"{'=' * 75}")

    header = f"  {'Dataset':14s} | {'Total':>5s}"
    for a in algorithms:
        header += f" | {a:>7s}"
    print(header)
    print("  " + "-" * (23 + 10 * len(algorithms)))

    for ds in datasets:
        r0 = all_results[ds][algorithms[0]]
        # Get total features from any solution
        total_f = len(r0["best_solutions"][0])
        row = f"  {ds:14s} | {total_f:5d}"
        for alg in algorithms:
            r = all_results[ds][alg]
            sel = np.mean(r["n_selected"])
            ratio = sel / total_f * 100
            row += f" | {ratio:5.1f}%"
        print(row)

    # - 5. Fitness comparison -
    print(f"\n{'=' * 75}")
    print("  5. Fitness (mean, lower is better)")
    print(f"{'=' * 75}")

    header = f"  {'Dataset':14s}"
    for a in algorithms:
        header += f" | {a:>8s}"
    print(header)
    print("  " + "-" * (15 + 11 * len(algorithms)))

    for ds in datasets:
        row = f"  {ds:14s}"
        for alg in algorithms:
            r = all_results[ds][alg]
            fit = np.mean(r["fitness"])
            row += f" |   {fit:.4f}"
        print(row)

    # - 6. Wilcoxon signed-rank test -
    print(f"\n{'=' * 75}")
    print("  6. Wilcoxon Signed-Rank Test (AMPA_FS vs others, p-values)")
    print(f"{'=' * 75}")

    from scipy.stats import wilcoxon

    for ds in datasets:
        print(f"\n  {ds}:")
        ampa_accs = np.array(all_results[ds]["AMPA_FS"]["final_rf_acc"])
        for alg in algorithms[1:]:
            other_accs = np.array(all_results[ds][alg]["final_rf_acc"])
            diff = ampa_accs - other_accs
            if np.all(diff == 0):
                print(f"    vs {alg:10s}: identical (no test)")
                continue
            try:
                stat, p = wilcoxon(ampa_accs, other_accs)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                better = "AMPA+" if np.mean(diff) > 0 else "other+"
                print(f"    vs {alg:10s}: p={p:.4f} {sig:3s} ({better})")
            except Exception as e:
                print(f"    vs {alg:10s}: error ({e})")

    # ── 7. Friedman test ──
    print(f"\n{'=' * 75}")
    print("  7. Friedman Chi-Squared Test (RF Accuracy)")
    print("=" * 75)

    from scipy.stats import friedmanchisquare
    rank_lists = [[rank_matrix[ds][alg] for ds in datasets] for alg in algorithms]
    try:
        stat_f, p_f = friedmanchisquare(*rank_lists)
        sig_f = "***" if p_f < 0.001 else "**" if p_f < 0.01 else "*" if p_f < 0.05 else "ns"
        print(f"\n  Friedman chi2 = {stat_f:.4f}, p = {p_f:.6f} {sig_f}")
        print(f"  (Null hypothesis: all algorithms perform equally)")
    except Exception as e:
        print(f"  [ERROR] Friedman test failed: {e}")

    # ── 8. Fitness Ranking (lower = better) ──
    print(f"\n{'=' * 75}")
    print("  8. Fitness Rankings (lower is better = better search quality)")
    print(f"{'=' * 75}")

    fit_matrix = {ds: {} for ds in datasets}
    for ds in datasets:
        for alg in algorithms:
            fit_matrix[ds][alg] = np.mean(all_results[ds][alg]["fitness"])

    fit_rank_matrix = {ds: {} for ds in datasets}
    for ds in datasets:
        sorted_by_fit = sorted(algorithms, key=lambda a: fit_matrix[ds][a])
        for rank, alg in enumerate(sorted_by_fit, 1):
            fit_rank_matrix[ds][alg] = rank

    avg_fit_ranks = {alg: np.mean([fit_rank_matrix[ds][alg] for ds in datasets])
                     for alg in algorithms}

    header = f"  {'Dataset':14s}"
    for a in algorithms:
        header += f" | {a:>8s}"
    print(header)
    print("  " + "-" * (15 + 11 * len(algorithms)))
    for ds in datasets:
        row = f"  {ds:14s}"
        for alg in algorithms:
            row += f" |   {fit_matrix[ds][alg]:.4f}"
        print(row)

    print(f"\n  Average Fitness Ranks:")
    for alg in sorted(algorithms, key=lambda a: avg_fit_ranks[a]):
        marker = " <-- AMPA_FS" if alg == "AMPA_FS" else ""
        print(f"    {alg:12s}: {avg_fit_ranks[alg]:.2f}{marker}")

    # ── 9. Key findings ──
    print(f"\n{'=' * 75}")
    print("  KEY FINDINGS")
    print("=" * 75)

    best_rank_alg = min(avg_ranks, key=avg_ranks.get)
    ampa_rank = avg_ranks["AMPA_FS"]
    sorted_algs = sorted(algorithms, key=lambda a: avg_ranks[a])

    print(f"\n  Ranking by average Friedman rank:")
    for i, alg in enumerate(sorted_algs, 1):
        marker = " <-- AMPA_FS" if alg == "AMPA_FS" else ""
        print(f"    {i:2d}. {alg:12s}: {avg_ranks[alg]:.2f}{marker}")

    print(f"\n  AMPA_FS vs BMPA (baseline MPA):")
    for ds in datasets:
        diff = acc_matrix[ds]["AMPA_FS"] - acc_matrix[ds]["BMPA"]
        print(f"    {ds:14s}: {diff:+.4f} "
              f"({'AMPA_FS wins' if diff > 0.001 else 'BMPA wins' if diff < -0.001 else 'tie'})")

    # Count wins
    ampa_best_count = sum(1 for ds in datasets if rank_matrix[ds]["AMPA_FS"] == 1)
    print(f"\n  AMPA_FS rank-1 count:  {ampa_best_count}/{len(datasets)} datasets")

    # Total wins vs all competitors
    total_win = 0
    total_comp = 0
    for alg in algorithms[1:]:
        for ds in datasets:
            total_comp += 1
            if acc_matrix[ds]["AMPA_FS"] > acc_matrix[ds][alg] + 0.001:
                total_win += 1
    print(f"  AMPA_FS overall win rate: {total_win}/{total_comp} "
          f"({total_win/total_comp*100:.1f}%)")

    print()


if __name__ == "__main__":
    main()
