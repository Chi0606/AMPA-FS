"""
Generate the additional revision figures/analyses (Week-2, tasks T2.1-T2.7).

  * T2.1  ROC curves (binary datasets) + learning curves    -> rev_roc_*.pdf, rev_learning_*.pdf
  * T2.2  RF accuracy vs. #selected features scatter        -> rev_acc_vs_nfeat.pdf
  * T2.3  Multi-metric radar charts                         -> rev_radar_*.pdf
  * T2.4  Feature-selection frequency analysis              -> rev_feature_freq.pdf + CSV
  * T2.5  Scalability / runtime vs. dimensionality          -> rev_scalability.pdf
  * T2.6  BPSO dominance: rank vs. dataset dimensionality   -> rev_bpso_dominance.pdf
  * T2.7  Clip-range [-6,6] vs [-2,2] comparison            -> rev_init_range.pdf
          (requires experiments/run_init_range.py output)

All inputs are read from ``results/``; missing files are skipped with a
warning so the script can be run incrementally.
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

OUT_DIR = os.path.join(config.BASE_DIR, "figures_generated")
os.makedirs(OUT_DIR, exist_ok=True)

BASE_DATASETS = list(config.DATASETS)
HIGHDIM_DATASETS = list(config.REVISION_DATASETS)
BINARY_DATASETS = ["BreastCancer", "Ionosphere", "Sonar", "Parkinsons",
                   "Colon", "Leukemia"]
RADAR_DATASETS = ["BreastCancer", "Sonar", "Vehicle", "Colon"]
RADAR_ALGS = ["AMPA_FS", "BMPA", "BPSO", "BHHO", "BGWO"]
LEARNING_DATASETS = ["BreastCancer", "Sonar", "Vehicle", "Colon"]


def _load_pkl(name):
    path = os.path.join(config.RESULTS_DIR, name)
    if not os.path.exists(path):
        print(f"  [SKIP] missing results file: {name}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_csv(name):
    path = os.path.join(config.RESULTS_DIR, name)
    if not os.path.exists(path):
        print(f"  [SKIP] missing results file: {name}")
        return None
    return pd.read_csv(path)


def _save(fig, stem):
    out = os.path.join(OUT_DIR, f"{stem}.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out}")


def _best_subset(results, ds, alg):
    """Best run's feature subset (lowest fitness) for (dataset, algorithm)."""
    r = results[ds][alg]
    idx = int(np.argmin(r["fitness"]))
    return np.asarray(r["best_solutions"][idx]).astype(int)


def _load_data(ds):
    from datasets.data_loader import load_dataset
    prescreen = 200 if ds in HIGHDIM_DATASETS else None
    return load_dataset(ds, prescreen_k=prescreen)


# ── T2.2 accuracy vs #features ─────────────────────────────────────────────
def fig_acc_vs_nfeat():
    df = _load_csv("main_summary.csv")
    if df is None:
        return
    datasets = [d for d in BASE_DATASETS + HIGHDIM_DATASETS
                if d in set(df["Dataset"])]
    n = len(datasets)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows))
    axes = np.atleast_2d(axes)
    for k, ds in enumerate(datasets):
        ax = axes[k // ncols][k % ncols]
        sub = df[df["Dataset"] == ds]
        for _, row in sub.iterrows():
            is_ampa = row["Algorithm"] == "AMPA_FS"
            ax.scatter(row["Selected_Mean"], row["RF_Acc_Mean"],
                       s=70 if is_ampa else 35,
                       marker="*" if is_ampa else "o",
                       color="crimson" if is_ampa else "steelblue",
                       zorder=3 if is_ampa else 2)
            if is_ampa:
                ax.annotate("AMPA-FS", (row["Selected_Mean"], row["RF_Acc_Mean"]),
                            fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_title(ds, fontsize=9)
        ax.set_xlabel("#features", fontsize=8)
        ax.set_ylabel("RF accuracy", fontsize=8)
        ax.tick_params(labelsize=7)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle("RF accuracy vs. mean number of selected features", y=1.02)
    fig.tight_layout()
    _save(fig, "rev_acc_vs_nfeat")


# ── T2.3 radar charts ──────────────────────────────────────────────────────
def fig_radar():
    df = _load_csv("extra_metrics.csv")
    if df is None:
        return
    metrics = ["accuracy", "precision", "sensitivity", "specificity", "f1", "auc"]
    datasets = [d for d in RADAR_DATASETS if d in set(df["Dataset"])]
    if not datasets:
        return
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 4.4),
                             subplot_kw=dict(polar=True))
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, datasets):
        sub = df[df["Dataset"] == ds].set_index("Algorithm")
        algs = [a for a in RADAR_ALGS if a in sub.index]
        # normalise each metric to [0.5, 1] band across shown algorithms
        for alg in algs:
            vals = []
            for m in metrics:
                col = sub[m]
                lo, hi = col.min(), col.max()
                v = 1.0 if hi - lo < 1e-12 else 0.5 + 0.5 * (sub.loc[alg, m] - lo) / (hi - lo)
                vals.append(v)
            vals += vals[:1]
            ax.plot(angles, vals, lw=1.6 if alg == "AMPA_FS" else 1.0,
                    label=alg.replace("_", "-"))
            ax.fill(angles, vals, alpha=0.06)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.upper() if m in ("f1", "auc") else m.capitalize()
                            for m in metrics], fontsize=7)
        ax.set_yticklabels([])
        ax.set_title(ds, fontsize=10)
    axes[-1].legend(loc="upper right", bbox_to_anchor=(1.45, 1.1), fontsize=7)
    fig.tight_layout()
    _save(fig, "rev_radar")


# ── T2.4 feature-selection frequency ───────────────────────────────────────
def fig_feature_freq():
    results = _load_pkl("main_results.pkl")
    if results is None:
        return
    datasets = [d for d in ["Wine", "Heart", "BreastCancer", "Ionosphere",
                            "Sonar", "Vehicle", "Parkinsons"] if d in results]
    fig, axes = plt.subplots(len(datasets), 1,
                             figsize=(9, 1.6 * len(datasets)), sharex=False)
    axes = np.atleast_1d(axes)
    rows = []
    for ax, ds in zip(axes, datasets):
        sols = np.array(results[ds]["AMPA_FS"]["best_solutions"], dtype=float)
        freq = sols.mean(axis=0)
        ax.bar(np.arange(len(freq)), freq, color="steelblue")
        ax.set_ylabel(ds, fontsize=8, rotation=0, ha="right", va="center")
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=7)
        top = np.argsort(freq)[::-1][:5]
        rows.append({"Dataset": ds,
                     "Top5_feature_idx": ",".join(map(str, top)),
                     "Top5_freq": ",".join(f"{freq[i]:.2f}" for i in top)})
    axes[-1].set_xlabel("feature index")
    fig.suptitle("AMPA-FS feature selection frequency over 30 runs", y=1.005)
    fig.tight_layout()
    _save(fig, "rev_feature_freq")
    out_csv = os.path.join(config.RESULTS_DIR, "feature_frequency_top5.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"  [OK] {out_csv}")


# ── T2.5 scalability ───────────────────────────────────────────────────────
def fig_scalability():
    df = _load_csv("main_summary.csv")
    if df is None:
        return
    dims = {"Iris": 4, "Wine": 13, "Heart": 13, "Zoo": 16, "Vehicle": 18,
            "Parkinsons": 22, "BreastCancer": 30, "Ionosphere": 34,
            "Dermatology": 34, "Sonar": 60, "Colon": 200, "SRBCT": 200,
            "Leukemia": 200}
    df = df[df["Dataset"].isin(dims)].copy()
    df["D"] = df["Dataset"].map(dims)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for alg in sorted(df["Algorithm"].unique()):
        sub = df[df["Algorithm"] == alg].sort_values("D")
        g = sub.groupby("D")["Time_Mean"].mean()
        ax.plot(g.index, g.values, marker="o", ms=3,
                lw=1.8 if alg == "AMPA_FS" else 0.9,
                label=alg.replace("_", "-"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("dataset dimensionality D (post pre-screening)")
    ax.set_ylabel("mean wall-clock time per run (s)")
    ax.legend(fontsize=6, ncol=2)
    ax.set_title("Runtime scalability vs. dimensionality")
    fig.tight_layout()
    _save(fig, "rev_scalability")


# ── T2.6 BPSO dominance analysis ───────────────────────────────────────────
def fig_bpso_dominance():
    df = _load_csv("main_summary.csv")
    if df is None:
        return
    piv = df.pivot(index="Dataset", columns="Algorithm", values="RF_Acc_Mean")
    ranks = piv.rank(axis=1, ascending=False)
    show = [a for a in ["AMPA_FS", "BMPA", "BPSO", "BHHO", "BGWO"]
            if a in ranks.columns]
    order = [d for d in BASE_DATASETS + HIGHDIM_DATASETS if d in ranks.index]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(order))
    w = 0.8 / len(show)
    for i, alg in enumerate(show):
        ax.bar(x + i * w, ranks.loc[order, alg], width=w,
               label=alg.replace("_", "-"))
    if any(d in order for d in HIGHDIM_DATASETS):
        first_hd = min(order.index(d) for d in HIGHDIM_DATASETS if d in order)
        ax.axvline(first_hd - 0.15, color="k", ls="--", lw=0.8)
        ax.text(first_hd - 0.05, ax.get_ylim()[1] * 0.95, "high-dim",
                fontsize=8, va="top")
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("RF-accuracy rank (lower better)")
    ax.legend(fontsize=7)
    ax.set_title("Per-dataset accuracy ranks: is BPSO dominance dataset-specific?")
    fig.tight_layout()
    _save(fig, "rev_bpso_dominance")


# ── T2.1 ROC + learning curves ─────────────────────────────────────────────
def fig_roc_and_learning():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, learning_curve
    from sklearn.metrics import roc_curve, auc

    results = _load_pkl("main_results.pkl")
    if results is None:
        return
    show_algs = ["AMPA_FS", "BMPA", "BPSO", "BHHO"]

    # ROC (binary datasets)
    ds_list = [d for d in BINARY_DATASETS if d in results]
    if ds_list:
        fig, axes = plt.subplots(1, len(ds_list),
                                 figsize=(3.4 * len(ds_list), 3.4))
        axes = np.atleast_1d(axes)
        for ax, ds in zip(axes, ds_list):
            X, y = _load_data(ds)
            if len(np.unique(y)) != 2:
                ax.axis("off")
                continue
            for alg in show_algs:
                if alg not in results[ds]:
                    continue
                mask = _best_subset(results, ds, alg).astype(bool)
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
                y_true, y_score = [], []
                for tr, te in skf.split(X[:, mask], y):
                    clf.fit(X[tr][:, mask], y[tr])
                    y_score.extend(clf.predict_proba(X[te][:, mask])[:, 1])
                    y_true.extend(y[te])
                fpr, tpr, _ = roc_curve(y_true, y_score)
                ax.plot(fpr, tpr, lw=1.6 if alg == "AMPA_FS" else 1.0,
                        label=f"{alg.replace('_', '-')} (AUC={auc(fpr, tpr):.3f})")
            ax.plot([0, 1], [0, 1], "k:", lw=0.7)
            ax.set_title(ds, fontsize=9)
            ax.set_xlabel("FPR", fontsize=8)
            ax.set_ylabel("TPR", fontsize=8)
            ax.legend(fontsize=6, loc="lower right")
            ax.tick_params(labelsize=7)
        fig.suptitle("ROC curves on best selected subsets (pooled 10-fold CV)",
                     y=1.03)
        fig.tight_layout()
        _save(fig, "rev_roc")

    # learning curves
    ds_list = [d for d in LEARNING_DATASETS if d in results]
    if ds_list:
        fig, axes = plt.subplots(1, len(ds_list),
                                 figsize=(3.4 * len(ds_list), 3.2))
        axes = np.atleast_1d(axes)
        for ax, ds in zip(axes, ds_list):
            X, y = _load_data(ds)
            for alg in ["AMPA_FS", "BMPA"]:
                if alg not in results[ds]:
                    continue
                mask = _best_subset(results, ds, alg).astype(bool)
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                n_cl = len(np.unique(y))
                cv = min(5, int(np.bincount(y).min())) if n_cl else 5
                sizes, _, test_sc = learning_curve(
                    clf, X[:, mask], y, cv=max(cv, 2),
                    train_sizes=np.linspace(0.2, 1.0, 5), random_state=42)
                m, s = test_sc.mean(axis=1), test_sc.std(axis=1)
                ax.plot(sizes, m, marker="o", ms=3,
                        lw=1.6 if alg == "AMPA_FS" else 1.0,
                        label=alg.replace("_", "-"))
                ax.fill_between(sizes, m - s, m + s, alpha=0.12)
            ax.set_title(ds, fontsize=9)
            ax.set_xlabel("training samples", fontsize=8)
            ax.set_ylabel("CV accuracy", fontsize=8)
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)
        fig.suptitle("Learning curves on best selected subsets", y=1.03)
        fig.tight_layout()
        _save(fig, "rev_learning")


# ── T2.7 clip-range comparison ─────────────────────────────────────────────
def fig_init_range():
    df = _load_csv("init_range_summary.csv")
    if df is None:
        return
    piv_acc = df.pivot(index="Dataset", columns="Bound", values="RF_Acc_Mean")
    piv_std = df.pivot(index="Dataset", columns="Bound", values="RF_Acc_Std")
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(piv_acc.index))
    w = 0.35
    for i, b in enumerate(piv_acc.columns):
        ax.bar(x + i * w, piv_acc[b], width=w, yerr=piv_std[b], capsize=3,
               label=f"[-{b[4:]}, {b[4:]}]")
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(piv_acc.index, fontsize=8)
    ax.set_ylabel("RF accuracy (mean ± std, 30 runs)")
    lo = float(np.nanmin(piv_acc.values)) - 0.05
    ax.set_ylim(max(0.0, lo), 1.0)
    ax.legend(fontsize=8, title="clip bound")
    ax.set_title("AMPA-FS: continuous-space clip range comparison (R3-5)")
    fig.tight_layout()
    _save(fig, "rev_init_range")


ALL = {
    "acc_vs_nfeat": fig_acc_vs_nfeat,
    "radar": fig_radar,
    "feature_freq": fig_feature_freq,
    "scalability": fig_scalability,
    "bpso_dominance": fig_bpso_dominance,
    "roc_learning": fig_roc_and_learning,
    "init_range": fig_init_range,
}


def main(only=None):
    for name, fn in ALL.items():
        if only and name != only:
            continue
        print(f"[revision-figs] {name}")
        try:
            fn()
        except Exception as e:
            print(f"  [WARN] {name} failed: {e}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
