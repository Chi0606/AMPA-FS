"""
Revision experiment for R5-12 ("integrate other components to enhance
performance"): tests whether adding search components genuinely improves
AMPA-FS. Reported in the response letter.

Levers tested independently and combined:
  L1  Elite bit-flip local search (memetic step, budget-limited, cache-aware)
  L2  Module-1 fix: keep top N/2 elites from OBL + N/2 fresh chaotic (diversity)

Variants:
  AMPA_FS   current full model (baseline of comparison)
  AMPA_LS   current + L1
  AMPA_I    current with L2 replacing greedy OBL selection
  AMPA_PLUS L1 + L2
Baselines: BMPA, BPSO.

Outputs per (dataset, algo): wrapper fitness, RF 10-fold acc, n_selected, NFE.
Paired Wilcoxon vs AMPA_FS across runs (same seeds).
"""

import os
import sys
import time
import pickle
import warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.data_loader import load_dataset
from utils.fitness import evaluate_final
from utils.transfer_functions import adaptive_sigmoid
from algorithms.ampa_fs import AMPA_FS
from algorithms.bpso import BPSO
import config


class AMPA_Improved(AMPA_FS):
    def __init__(self, *args, use_local_search=False, use_init_fix=False,
                 ls_interval=10, ls_budget=30, ls_safe=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_local_search = use_local_search
        self.use_init_fix = use_init_fix
        self.ls_interval = ls_interval
        self.ls_budget = ls_budget
        self.ls_safe = ls_safe

    # ---- L2: diversity-preserving chaotic-OBL init ----
    def _initialize_population(self, X, y):
        if not (self.use_init_fix and self.use_chaotic_init):
            return super()._initialize_population(X, y)
        D, N = self.n_features, self.pop_size
        pos = self._tent_map(N, D)
        temp_bin = (np.random.random((N, D)) <
                    adaptive_sigmoid(pos, self.gamma_min)).astype(int)
        temp_fit = np.zeros(N)
        for i in range(N):
            temp_fit[i], _, _ = self.evaluator.evaluate(temp_bin[i])
        opp = pos.min(axis=0) + pos.max(axis=0) - pos + 0.1 * np.random.randn(N, D)
        opp = np.clip(opp, -self.clip_bound, self.clip_bound)
        merged = np.vstack([pos, opp])
        merged_bin = (np.random.random(merged.shape) <
                      adaptive_sigmoid(merged, self.gamma_min)).astype(int)
        merged_fit = np.zeros(2 * N)
        for i in range(2 * N):
            merged_fit[i], _, _ = self.evaluator.evaluate(merged_bin[i])
        order = np.argsort(merged_fit)
        elite_idx = order[: N // 2]
        rest = order[N // 2:]
        rand_idx = np.random.choice(rest, N - N // 2, replace=False)
        keep = np.concatenate([elite_idx, rand_idx])
        self.continuous_pos = merged[keep]
        self.population = (np.random.random((N, D)) <
                           adaptive_sigmoid(self.continuous_pos, self.gamma_min)).astype(int)
        self.stag_counter = np.zeros(N, dtype=int)
        self.prev_fitness = np.full(N, np.inf)

    # ---- L1: elite bit-flip local search ----
    def _local_search(self):
        best = self.best_solution.copy()
        f_best = self.best_fitness
        a_best = self.best_accuracy
        budget = self.ls_budget
        improved_any = False
        sel = np.where(best == 1)[0]
        unsel = np.where(best == 0)[0]
        np.random.shuffle(sel)
        np.random.shuffle(unsel)
        # removals first (compactness), then a few additions
        cands = list(sel) + list(unsel[: max(3, len(unsel) // 10)])
        for j in cands:
            if budget <= 0:
                break
            cand = best.copy()
            cand[j] = 1 - cand[j]
            if cand.sum() == 0:
                continue
            f, a, s = self.evaluator.evaluate(cand)
            budget -= 1
            if self.ls_safe and a < a_best:
                continue
            if f < f_best:
                best, f_best, a_best = cand, f, a
                improved_any = True
                self.best_accuracy, self.best_n_selected = a, s
        if improved_any:
            self.best_solution = best
            self.best_fitness = f_best
            # inject into population (replace worst)
            w = int(np.argmax(self.fitness))
            self.population[w] = best
            self.fitness[w] = f_best
            self.continuous_pos[w] = (2.0 * best - 1.0) * 2.0

    def _update_population(self, t, X, y):
        super()._update_population(t, X, y)
        if self.use_local_search and (t + 1) % self.ls_interval == 0:
            self._local_search()


def make_algo(name, seed):
    common = dict(pop_size=config.POP_SIZE, max_iter=config.MAX_ITER,
                  alpha=config.ALPHA, beta=config.BETA, seed=seed)
    ampa = dict(fads_prob=config.FADS_PROB, stag_threshold=config.STAG_THRESHOLD,
                mutation_f=config.MUTATION_F, lambda_max=config.LAMBDA_MAX,
                gamma_min=config.GAMMA_MIN, gamma_max=config.GAMMA_MAX,
                tent_z0=config.TENT_Z0, use_chaotic_init=True,
                use_adaptive_step=True, use_elite_mutation=True)
    if name == "AMPA_FS":
        return AMPA_Improved(**ampa, **common)
    if name == "AMPA_LS":
        return AMPA_Improved(**ampa, use_local_search=True, **common)
    if name == "AMPA_LSS":
        return AMPA_Improved(**ampa, use_local_search=True, ls_safe=True,
                             **common)
    if name == "AMPA_I":
        return AMPA_Improved(**ampa, use_init_fix=True, **common)
    if name == "AMPA_PLUS":
        return AMPA_Improved(**ampa, use_local_search=True, use_init_fix=True,
                             **common)
    if name == "AMPA_PLUSS":
        return AMPA_Improved(**ampa, use_local_search=True, use_init_fix=True,
                             ls_safe=True, **common)
    if name == "BMPA":
        return AMPA_Improved(fads_prob=config.FADS_PROB, use_chaotic_init=False,
                             use_adaptive_step=False, use_elite_mutation=False,
                             **common)
    if name == "BPSO":
        return BPSO(**common)
    raise ValueError(name)


def main(datasets, algos, num_runs, out_name):
    out_path = os.path.join(config.RESULTS_DIR, out_name)
    results = {}
    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            results = pickle.load(f)
        print(f"[resume] loaded {out_path}: " + ", ".join(
            f"{ds}({len(results[ds])})" for ds in results), flush=True)
    for ds in datasets:
        prescreen = 200 if ds in config.REVISION_DATASETS else None
        X, y = load_dataset(ds, prescreen_k=prescreen)
        results.setdefault(ds, {})
        print(f"\n== {ds}  ({X.shape[0]}x{X.shape[1]}) ==", flush=True)
        for alg in algos:
            done = results[ds].get(alg)
            if done and len(done["fitness"]) >= num_runs:
                print(f"  {alg:10s} [skip, done]", flush=True)
                continue
            rr = {"fitness": [], "rf_acc": [], "n_sel": [], "nfe": [], "time": []}
            t0 = time.time()
            for run in range(num_runs):
                seed = run * 100 + 42
                a = make_algo(alg, seed)
                res = a.optimize(X, y)
                fin, _ = evaluate_final(res["best_solution"], X, y,
                                        random_state=seed)
                rr["fitness"].append(res["best_fitness"])
                rr["rf_acc"].append(fin["RF"]["accuracy_mean"])
                rr["n_sel"].append(res["best_n_selected"])
                rr["nfe"].append(a.evaluator._eval_count)
                rr["time"].append(res["elapsed_time"])
            results[ds][alg] = rr
            print(f"  {alg:10s} fit={np.mean(rr['fitness']):.4f} "
                  f"rf={np.mean(rr['rf_acc']):.4f} sel={np.mean(rr['n_sel']):.1f} "
                  f"nfe={np.mean(rr['nfe']):.0f} t={time.time()-t0:.0f}s", flush=True)
            with open(out_path, "wb") as f:
                pickle.dump(results, f)
    # paired Wilcoxon vs AMPA_FS
    from scipy.stats import wilcoxon
    print("\n=== Paired Wilcoxon vs AMPA_FS (same seeds) ===")
    for ds in datasets:
        for alg in algos:
            if alg == "AMPA_FS" or alg not in results[ds]:
                continue
            for metric in ["fitness", "rf_acc", "n_sel"]:
                a = np.array(results[ds]["AMPA_FS"][metric], dtype=float)
                b = np.array(results[ds][alg][metric], dtype=float)
                if np.allclose(a, b):
                    p = 1.0
                else:
                    try:
                        p = wilcoxon(a, b).pvalue
                    except ValueError:
                        p = 1.0
                diff = np.mean(b) - np.mean(a)
                star = "*" if p < 0.05 else " "
                print(f"  {ds:12s} {alg:10s} {metric:7s} "
                      f"delta={diff:+.4f} p={p:.4f}{star}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["Sonar", "Ionosphere", "Parkinsons"])
    ap.add_argument("--algos", nargs="+",
                    default=["AMPA_FS", "AMPA_LS", "AMPA_I", "AMPA_PLUS",
                             "BMPA", "BPSO"])
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--out", default="proto_improve.pkl")
    a = ap.parse_args()
    main(a.datasets, a.algos, a.runs, a.out)
