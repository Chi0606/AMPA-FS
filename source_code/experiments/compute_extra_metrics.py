"""
Compute the extended evaluation metrics requested in the revision
(specificity, sensitivity, precision, F1, ROC-AUC, #features, runtime) from
the *already stored* best feature subsets in ``results/main_results.pkl``.

This does NOT re-run the (expensive) wrapper search: it re-evaluates each
run's best subset with 10-fold CV, so it is cheap and fully reproducible.

Output: ``results/extra_metrics.csv`` with one row per (dataset, algorithm),
mean over the independent runs.
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from datasets.data_loader import load_dataset
from utils.fitness import evaluate_metrics


def main(pkl_name="main_results.pkl", classifier="RF", out_name="extra_metrics.csv"):
    path = os.path.join(config.RESULTS_DIR, pkl_name)
    if not os.path.exists(path):
        print(f"[ERROR] missing {path}; run the main experiment first.")
        return
    with open(path, "rb") as f:
        results = pickle.load(f)

    rows = []
    for ds_name, algs in results.items():
        prescreen = 200 if ds_name in getattr(config, "REVISION_DATASETS", []) else None
        X, y = load_dataset(ds_name, prescreen_k=prescreen)
        print(f"[{ds_name}] {X.shape[0]}x{X.shape[1]}")
        for alg_name, r in algs.items():
            sols = r.get("best_solutions", [])
            times = r.get("time", [])
            metric_runs = []
            for sol in sols:
                metric_runs.append(evaluate_metrics(sol, X, y, classifier=classifier))
            if not metric_runs:
                continue
            agg = {k: np.nanmean([m[k] for m in metric_runs])
                   for k in ("accuracy", "precision", "sensitivity",
                             "specificity", "f1", "auc", "n_selected")}
            agg.update({
                "Dataset": ds_name,
                "Algorithm": alg_name,
                "Runtime_s": float(np.mean(times)) if len(times) else np.nan,
            })
            rows.append(agg)
            print(f"  {alg_name:8s} acc={agg['accuracy']:.4f} f1={agg['f1']:.4f} "
                  f"sens={agg['sensitivity']:.4f} spec={agg['specificity']:.4f} "
                  f"auc={agg['auc']:.4f} nfeat={agg['n_selected']:.1f}")

    df = pd.DataFrame(rows)
    cols = ["Dataset", "Algorithm", "accuracy", "precision", "sensitivity",
            "specificity", "f1", "auc", "n_selected", "Runtime_s"]
    df = df[[c for c in cols if c in df.columns]]
    out = os.path.join(config.RESULTS_DIR, out_name)
    df.to_csv(out, index=False)
    print(f"\n[DONE] wrote {out}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
