"""
Revision experiment for R5-12: tests the root-cause fix (evaluator swap).
Reported in the response letter.

Hypothesis (paper Sec. 5.2): AMPA-FS's mediocre final RF ranking stems from
the search-phase evaluator (single 70/30 KNN hold-out) being misaligned with
the reported metric (10-fold RF).  Here we swap the search evaluator for a
lightweight RF surrogate (25 trees, same 70/30 split) and measure whether the
final 10-fold RF accuracy genuinely improves.

Variants (30 runs, seeds identical to proto_improve.py so results are
directly comparable / mergeable with proto_full.pkl):
  AMPA_RFL   full AMPA-FS with RF-lite search evaluator
  BMPA_RFL   BMPA baseline with RF-lite search evaluator (isolates evaluator
             effect from algorithm effect)

Compare against AMPA_FS / BMPA from proto_full.pkl (KNN evaluator, same seeds).
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
from utils.fitness import FitnessEvaluator, evaluate_final
from algorithms.ampa_fs import AMPA_FS
import config

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class RFLiteEvaluator(FitnessEvaluator):
    """Search-phase evaluator using a lightweight RF instead of KNN."""

    def __init__(self, *args, n_trees=25, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_trees = n_trees

    def evaluate(self, solution):
        key = solution.tobytes()
        if key in self._cache:
            return self._cache[key]

        selected = np.where(solution > 0.5)[0]
        if len(selected) == 0:
            result = (1.0, 0.0, 0)
            self._cache[key] = result
            return result

        X_tr = self.X_train[:, selected]
        X_te = self.X_test[:, selected]
        clf = RandomForestClassifier(n_estimators=self.n_trees,
                                     random_state=self.random_state,
                                     n_jobs=1)
        try:
            clf.fit(X_tr, self.y_train)
            accuracy = accuracy_score(self.y_test, clf.predict(X_te))
        except Exception:
            accuracy = 0.0

        feature_ratio = len(selected) / self.n_features
        fitness = self.alpha * (1.0 - accuracy) + self.beta * feature_ratio
        result = (fitness, accuracy, len(selected))
        self._cache[key] = result
        self._eval_count += 1
        return result


class _RFLiteMixin:
    """Swap the search-phase evaluator built inside optimize() for RF-lite."""

    def optimize(self, X, y):
        import algorithms.base_optimizer as bo
        orig = bo.FitnessEvaluator
        bo.FitnessEvaluator = RFLiteEvaluator
        try:
            return super().optimize(X, y)
        finally:
            bo.FitnessEvaluator = orig


class AMPA_RFLite(_RFLiteMixin, AMPA_FS):
    pass


def make_algo(name, seed):
    common = dict(pop_size=config.POP_SIZE, max_iter=config.MAX_ITER,
                  alpha=config.ALPHA, beta=config.BETA, seed=seed)
    if name == "AMPA_RFL":
        return AMPA_RFLite(fads_prob=config.FADS_PROB,
                       stag_threshold=config.STAG_THRESHOLD,
                       mutation_f=config.MUTATION_F,
                       lambda_max=config.LAMBDA_MAX,
                       gamma_min=config.GAMMA_MIN, gamma_max=config.GAMMA_MAX,
                       tent_z0=config.TENT_Z0, use_chaotic_init=True,
                       use_adaptive_step=True, use_elite_mutation=True,
                       **common)
    if name == "BMPA_RFL":
        return AMPA_RFLite(fads_prob=config.FADS_PROB, use_chaotic_init=False,
                       use_adaptive_step=False, use_elite_mutation=False,
                       **common)
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
            rr = {"fitness": [], "rf_acc": [], "knn_acc": [], "n_sel": [],
                  "nfe": [], "time": []}
            t0 = time.time()
            for run in range(num_runs):
                seed = run * 100 + 42
                a = make_algo(alg, seed)
                res = a.optimize(X, y)
                fin, _ = evaluate_final(res["best_solution"], X, y,
                                        random_state=seed)
                rr["fitness"].append(res["best_fitness"])
                rr["rf_acc"].append(fin["RF"]["accuracy_mean"])
                rr["knn_acc"].append(fin["KNN"]["accuracy_mean"])
                rr["n_sel"].append(res["best_n_selected"])
                rr["nfe"].append(a.evaluator._eval_count)
                rr["time"].append(res["elapsed_time"])
            results[ds][alg] = rr
            print(f"  {alg:10s} fit={np.mean(rr['fitness']):.4f} "
                  f"rf={np.mean(rr['rf_acc']):.4f} sel={np.mean(rr['n_sel']):.1f} "
                  f"nfe={np.mean(rr['nfe']):.0f} t={time.time()-t0:.0f}s",
                  flush=True)
            with open(out_path, "wb") as f:
                pickle.dump(results, f)

    # paired Wilcoxon vs KNN-evaluator counterparts from proto_full.pkl
    from scipy.stats import wilcoxon
    ref_path = os.path.join(config.RESULTS_DIR, "proto_full.pkl")
    if os.path.exists(ref_path):
        with open(ref_path, "rb") as f:
            ref = pickle.load(f)
        pairs = [("AMPA_RFL", "AMPA_FS"), ("BMPA_RFL", "BMPA")]
        print("\n=== Paired Wilcoxon: RF-lite evaluator vs KNN evaluator "
              "(same seeds) ===")
        for ds in datasets:
            for new, old in pairs:
                if new not in results.get(ds, {}) or old not in ref.get(ds, {}):
                    continue
                for metric in ["rf_acc", "n_sel"]:
                    a = np.array(ref[ds][old][metric], dtype=float)
                    b = np.array(results[ds][new][metric], dtype=float)
                    n = min(len(a), len(b))
                    a, b = a[:n], b[:n]
                    if np.allclose(a, b):
                        p = 1.0
                    else:
                        try:
                            p = wilcoxon(a, b).pvalue
                        except ValueError:
                            p = 1.0
                    diff = np.mean(b) - np.mean(a)
                    star = "*" if p < 0.05 else " "
                    print(f"  {ds:12s} {new:9s} vs {old:8s} {metric:7s} "
                          f"delta={diff:+.4f} p={p:.4f}{star}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["Sonar", "Ionosphere", "Heart", "Parkinsons",
                             "Vehicle", "Colon"])
    ap.add_argument("--algos", nargs="+", default=["AMPA_RFL", "BMPA_RFL"])
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--out", default="proto_rfl.pkl")
    a = ap.parse_args()
    main(a.datasets, a.algos, a.runs, a.out)
