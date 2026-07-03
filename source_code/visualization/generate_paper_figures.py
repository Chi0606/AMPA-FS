"""
Generate the paper figures (and a few LaTeX-ready tables) from the pickled
experiment results produced by the runners in ``experiments/``.

Figures (written to ``<RESULTS_DIR>/../figures_generated/`` and, for
convenience, mirrored to the manuscript figure names ``Fig1..Fig4.pdf``):

  * Fig1  -- Phase-aligned control-factor (CF) profile vs. the original MPA CF.
             (analytic; needs no results file)
  * Fig2  -- Ablation study: (a) heat-map of relative fitness improvement of
             each variant over BMPA; (b) average wrapper-fitness rank.
  * Fig3  -- Parameter sensitivity panels.
  * Fig4  -- Convergence curves on four representative datasets.

Usage
-----
    from visualization.generate_paper_figures import main
    main(only=None)              # generate every figure it has data for
    main(only="convergence")     # generate a single figure

Missing result files are skipped with a warning rather than raising, so the
module can be used incrementally while experiments are still running.
"""

import os
import sys
import pickle

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

OUT_DIR = os.path.join(config.BASE_DIR, "figures_generated")
os.makedirs(OUT_DIR, exist_ok=True)

# Datasets highlighted in the convergence figure (Fig4 in the manuscript).
CONVERGENCE_DATASETS = ["Wine", "BreastCancer", "Sonar", "Vehicle"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _load(name):
    path = os.path.join(config.RESULTS_DIR, name)
    if not os.path.exists(path):
        print(f"  [SKIP] missing results file: {name}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _save(fig, stem, mirror=None):
    out = os.path.join(OUT_DIR, f"{stem}.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    if mirror:
        fig.savefig(os.path.join(OUT_DIR, f"{mirror}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] wrote {out}")


def _mean_fitness_table(results, algorithms):
    """Return dict[ds][alg] -> mean fitness."""
    table = {}
    for ds, algs in results.items():
        table[ds] = {}
        for alg in algorithms:
            if alg in algs and len(algs[alg].get("fitness", [])):
                table[ds][alg] = float(np.mean(algs[alg]["fitness"]))
    return table


# ---------------------------------------------------------------------------
# Fig1 -- control-factor profile (analytic)
# ---------------------------------------------------------------------------
def fig_cf_profile():
    T = config.MAX_ITER
    t = np.arange(1, T + 1)
    frac = t / T

    # Original MPA CF: (1 - t/T)^(2 t/T)
    cf_orig = (1.0 - frac) ** (2.0 * frac)

    # Proposed phase-aligned CF (matches AMPA_FS._adaptive_cf).
    cf_new = np.empty_like(frac)
    p1 = frac <= 1.0 / 3.0
    p2 = (frac > 1.0 / 3.0) & (frac <= 2.0 / 3.0)
    p3 = frac > 2.0 / 3.0
    cf_new[p1] = 1.0
    seg = (frac[p2] - 1.0 / 3.0) / (1.0 / 3.0)          # 0..1 over phase 2
    cf_new[p2] = 0.25 + 0.75 * 0.5 * (1 + np.cos(np.pi * seg))
    seg3 = (frac[p3] - 2.0 / 3.0) / (1.0 / 3.0)         # 0..1 over phase 3
    cf_new[p3] = 0.25 * (1.0 - seg3)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(t, cf_new, "-", lw=2.2, color="#1f77b4", label="Proposed phase-aligned CF")
    ax.plot(t, cf_orig, "--", lw=2.0, color="#d62728", label="Original MPA CF")
    for b in (T / 3.0, 2 * T / 3.0):
        ax.axvline(b, color="gray", ls=":", lw=1)
    ax.set_xlabel("Iteration $t$")
    ax.set_ylabel("Control factor CF")
    ax.set_title("Phase-aligned adaptive control factor")
    ax.legend(frameon=False)
    ax.margins(x=0)
    _save(fig, "fig1_cf_profile", mirror="Fig1")


# ---------------------------------------------------------------------------
# Fig2 -- ablation
# ---------------------------------------------------------------------------
def fig_ablation():
    results = _load("ablation_results.pkl")
    if results is None:
        return
    variants = [v for v in config.ABLATION_VARIANTS]
    datasets = [d for d in config.DATASETS if d in results]
    table = _mean_fitness_table(results, variants)

    # Relative improvement over BMPA (%): positive = better (lower fitness).
    improv = np.full((len(datasets), len(variants)), np.nan)
    ranks = np.full((len(datasets), len(variants)), np.nan)
    for i, ds in enumerate(datasets):
        base = table[ds].get("BMPA")
        row_vals = [table[ds].get(v, np.nan) for v in variants]
        # rank (1 = lowest fitness / best)
        order = np.argsort(np.argsort(np.nan_to_num(row_vals, nan=np.inf)))
        for j, v in enumerate(variants):
            fv = table[ds].get(v)
            if base and fv is not None and base != 0:
                improv[i, j] = 100.0 * (base - fv) / base
            ranks[i, j] = order[j] + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                   gridspec_kw={"width_ratios": [1.6, 1]})

    im = ax1.imshow(improv, aspect="auto", cmap="RdYlGn", vmin=-5, vmax=5)
    ax1.set_xticks(range(len(variants)))
    ax1.set_xticklabels(variants, rotation=45, ha="right")
    ax1.set_yticks(range(len(datasets)))
    ax1.set_yticklabels(datasets)
    ax1.set_title("(a) Relative fitness improvement over BMPA (%)")
    for i in range(len(datasets)):
        for j in range(len(variants)):
            if not np.isnan(improv[i, j]):
                ax1.text(j, i, f"{improv[i, j]:.1f}", ha="center", va="center",
                         fontsize=7)
    # box the full algorithm column
    if "AMPA_FS" in variants:
        c = variants.index("AMPA_FS")
        ax1.add_patch(plt.Rectangle((c - 0.5, -0.5), 1, len(datasets),
                                    fill=False, edgecolor="black", lw=2))
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    mean_rank = np.nanmean(ranks, axis=0)
    order = np.argsort(mean_rank)
    ax2.barh([variants[k] for k in order], [mean_rank[k] for k in order],
             color="#4c72b0")
    ax2.invert_yaxis()
    ax2.set_xlabel("Average wrapper-fitness rank (lower is better)")
    ax2.set_title("(b) Mean rank across datasets")
    for k, v in enumerate([mean_rank[o] for o in order]):
        ax2.text(v + 0.05, k, f"{v:.2f}", va="center", fontsize=8)

    fig.tight_layout()
    _save(fig, "fig2_ablation", mirror="Fig2")


# ---------------------------------------------------------------------------
# Fig3 -- sensitivity
# ---------------------------------------------------------------------------
def fig_sensitivity():
    results = _load("sensitivity_results.pkl")
    if results is None:
        return
    params = [p for p in config.SENSITIVITY_PARAMS if p in results]
    if not params:
        print("  [SKIP] no sensitivity params present")
        return
    defaults = {
        "pop_size": config.POP_SIZE,
        "stag_threshold": config.STAG_THRESHOLD,
        "alpha": config.ALPHA,
        "gamma_max": config.GAMMA_MAX,
    }

    n = len(params)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 3.6 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for k, param in enumerate(params):
        ax = axes[k]
        # average over the sensitivity datasets
        per_val = {}
        for ds, vals in results[param].items():
            for val, res in vals.items():
                per_val.setdefault(val, []).extend(res.get("fitness_all", []))
        xs = sorted(per_val.keys())
        means = [np.mean(per_val[v]) for v in xs]
        stds = [np.std(per_val[v]) for v in xs]
        ax.errorbar(range(len(xs)), means, yerr=stds, marker="o", capsize=3,
                    color="#1f77b4")
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([str(v) for v in xs])
        ax.set_xlabel(param)
        ax.set_ylabel("Mean wrapper fitness")
        if param in defaults and defaults[param] in xs:
            ax.axvline(xs.index(defaults[param]), color="gray", ls="--", lw=1)
        ax.set_title(f"Sensitivity to {param}")

    for k in range(len(params), len(axes)):
        axes[k].axis("off")
    fig.tight_layout()
    _save(fig, "fig3_sensitivity", mirror="Fig3")


# ---------------------------------------------------------------------------
# Fig4 -- convergence
# ---------------------------------------------------------------------------
def fig_convergence():
    results = _load("main_results.pkl")
    if results is None:
        return
    datasets = [d for d in CONVERGENCE_DATASETS if d in results]
    if not datasets:
        print("  [SKIP] none of the convergence datasets present yet")
        return
    algorithms = [a for a in config.ALGORITHMS]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    for k, ds in enumerate(datasets):
        ax = axes[k]
        for alg in algorithms:
            if alg not in results[ds]:
                continue
            curves = results[ds][alg].get("convergence", [])
            curves = [c for c in curves if np.all(np.isfinite(c))]
            if not curves:
                continue
            L = min(len(c) for c in curves)
            arr = np.array([c[:L] for c in curves])
            mean = arr.mean(axis=0)
            if alg == "AMPA_FS":
                ax.plot(mean, lw=2.5, color="black", label=alg, zorder=5)
            else:
                ax.plot(mean, lw=1.2, alpha=0.8, label=alg)
        ax.set_title(ds)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best fitness")
    axes[0].legend(fontsize=7, ncol=2, frameon=False)
    for k in range(len(datasets), len(axes)):
        axes[k].axis("off")
    fig.tight_layout()
    _save(fig, "fig4_convergence", mirror="Fig4")


# ---------------------------------------------------------------------------
_FIGS = {
    "cf": fig_cf_profile,
    "ablation": fig_ablation,
    "sensitivity": fig_sensitivity,
    "convergence": fig_convergence,
}


def main(only=None):
    print(f"Generating figures into {OUT_DIR}")
    targets = _FIGS if only is None else {only: _FIGS[only]}
    for name, fn in targets.items():
        print(f"- {name}")
        try:
            fn()
        except Exception as exc:  # keep going; report which figure failed
            print(f"  [ERROR] {name}: {exc}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(only=arg)
