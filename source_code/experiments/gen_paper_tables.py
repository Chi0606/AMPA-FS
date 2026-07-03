"""
Generate LaTeX table fragments for the revised manuscript from the stored
13x13x30 results. Output: results/tex_fragments/*.tex plus a stats summary
printed to stdout (Friedman/Wilcoxon/Holm/A12 numbers to quote in prose).
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare, norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

ALGS = ["AMPA_FS", "BMPA", "BPSO", "BGA", "BGWO", "BWOA", "BSSA",
        "BHHO", "BSCA", "BGOA", "BWSO", "BEHO", "BCMAES"]
DS = ["Iris", "Wine", "Zoo", "Heart", "BreastCancer", "Ionosphere", "Sonar",
      "Dermatology", "Vehicle", "Parkinsons", "Colon", "SRBCT", "Leukemia"]
HD = ["Colon", "SRBCT", "Leukemia"]

OUT = os.path.join(config.RESULTS_DIR, "tex_fragments")
os.makedirs(OUT, exist_ok=True)


def texname(a):
    return a.replace("_", r"\_")


def main():
    m = pd.read_csv(os.path.join(config.RESULTS_DIR, "main_summary.csv"))
    with open(os.path.join(config.RESULTS_DIR, "main_results.pkl"), "rb") as f:
        raw = pickle.load(f)
    xm = pd.read_csv(os.path.join(config.RESULTS_DIR, "extra_metrics.csv"))

    acc = m.pivot(index="Dataset", columns="Algorithm", values="RF_Acc_Mean").loc[DS, ALGS]
    std = m.pivot(index="Dataset", columns="Algorithm", values="RF_Acc_Std").loc[DS, ALGS]
    fit = m.pivot(index="Dataset", columns="Algorithm", values="Fitness_Mean").loc[DS, ALGS]
    nsel = m.pivot(index="Dataset", columns="Algorithm", values="Selected_Mean").loc[DS, ALGS]
    tim = m.pivot(index="Dataset", columns="Algorithm", values="Time_Mean").loc[DS, ALGS]

    # ---------------- RF accuracy table (13 x 13) ----------------
    rank_acc = acc.rank(axis=1, ascending=False).mean(0)
    lines = []
    for ds in DS:
        best = acc.loc[ds].max()
        cells = []
        for a in ALGS:
            v, s = acc.loc[ds, a], std.loc[ds, a]
            cell = f".{v*1000:03.0f}$\\pm$.{s*1000:03.0f}"
            if abs(v - best) < 5e-4:
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(f"{ds:<13s} & " + " & ".join(cells) + r" \\")
    ranks = " & ".join(
        (f"\\textbf{{{rank_acc[a]:.2f}}}" if rank_acc[a] == rank_acc.min()
         else f"{rank_acc[a]:.2f}") for a in ALGS)
    lines.append(r"\midrule")
    lines.append(r"\textit{Rank} & " + ranks + r" \\")
    with open(os.path.join(OUT, "tab_rfacc_body.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---------------- fitness Friedman rank table ----------------
    rank_fit = fit.rank(axis=1, ascending=True).mean(0).sort_values()
    lines = []
    for i, (a, r) in enumerate(rank_fit.items(), 1):
        nm = texname(a)
        if a == "AMPA_FS":
            lines.append(f"\\textbf{{{i}}} & \\textbf{{{nm}}} & \\textbf{{{r:.2f}}} \\\\")
        else:
            lines.append(f"{i} & {nm} & {r:.2f} \\\\")
    with open(os.path.join(OUT, "tab_fitrank_body.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # Friedman omnibus tests
    fr_fit = friedmanchisquare(*[fit[a].values for a in ALGS])
    fr_acc = friedmanchisquare(*[acc[a].values for a in ALGS])
    print(f"Friedman fitness: chi2={fr_fit.statistic:.2f} p={fr_fit.pvalue:.3f}")
    print(f"Friedman rf-acc : chi2={fr_acc.statistic:.2f} p={fr_acc.pvalue:.3f}")
    print("rank_acc:", {a: round(rank_acc[a], 2) for a in ALGS})
    print("rank_fit:", dict(rank_fit.round(2)))

    # ---------------- #features table (D>=30 incl. high-dim) ----------------
    nsel_ds = ["BreastCancer", "Ionosphere", "Sonar", "Dermatology"] + HD
    dims = {"BreastCancer": 30, "Ionosphere": 34, "Sonar": 60, "Dermatology": 34,
            "Colon": 200, "SRBCT": 200, "Leukemia": 200}
    lines = []
    for ds in nsel_ds:
        cells = [f"{nsel.loc[ds, a]:.1f}" for a in ALGS]
        lines.append(f"{ds} ({dims[ds]}) & " + " & ".join(cells) + r" \\")
    avg = nsel.loc[nsel_ds].mean(0)
    lines.append(r"\midrule")
    lines.append(r"\textbf{Average} & " + " & ".join(
        (f"\\textbf{{{avg[a]:.1f}}}" if a == "AMPA_FS" else f"{avg[a]:.1f}")
        for a in ALGS) + r" \\")
    with open(os.path.join(OUT, "tab_nsel_body.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("nsel avg:", dict(avg.round(1)))
    red = 1 - avg["AMPA_FS"] / avg
    print("AMPA nsel reduction vs each:", dict((red * 100).round(1)))
    hd_red = 1 - nsel.loc[HD, "AMPA_FS"] / nsel.loc[HD, "BMPA"]
    print("high-dim reduction vs BMPA:", dict((hd_red * 100).round(1)))

    # ---------------- time table ----------------
    tmean = tim.mean(0)
    with open(os.path.join(OUT, "tab_time_body.tex"), "w") as f:
        f.write("Time (s) & " + " & ".join(f"{tmean[a]:.1f}" for a in ALGS) + " \\\\\n")
    print("time mean:", dict(tmean.round(2)),
          "AMPA/BMPA=", round(tmean["AMPA_FS"] / tmean["BMPA"], 3))

    # ---------------- Wilcoxon table (fitness, AMPA vs each) ----------------
    lines = []
    for ds in DS:
        cells = []
        a_runs = np.array(raw[ds]["AMPA_FS"]["fitness"])
        for a in ALGS[1:]:
            b_runs = np.array(raw[ds][a]["fitness"])
            diff = a_runs - b_runs
            if np.allclose(diff, 0):
                cells.append("--")
                continue
            p = wilcoxon(a_runs, b_runs).pvalue
            sign = "$+$" if a_runs.mean() < b_runs.mean() else "$-$"
            star = "$^{**}$" if p < 0.01 else ("$^{*}$" if p < 0.05 else "")
            cells.append(f"{p:.3f}{star}{sign}")
        lines.append(f"{ds} & " + " & ".join(cells) + r" \\")
    with open(os.path.join(OUT, "tab_wilcoxon_body.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # Wilcoxon AMPA vs BMPA on RF accuracy (headline safe-augmentation claim)
    n_sig_worse = n_sig_better = 0
    for ds in DS:
        a = np.array(raw[ds]["AMPA_FS"]["final_rf_acc"])
        b = np.array(raw[ds]["BMPA"]["final_rf_acc"])
        if np.allclose(a - b, 0):
            continue
        p = wilcoxon(a, b).pvalue
        if p < 0.05:
            if a.mean() > b.mean():
                n_sig_better += 1
            else:
                n_sig_worse += 1
        print(f"  RFacc AMPA vs BMPA {ds:13s} {a.mean():.4f} {b.mean():.4f} p={p:.3f}")
    print(f"sig better={n_sig_better} sig worse={n_sig_worse}")

    # ---------------- Holm post-hoc (control = best fitness rank) ----------------
    k, n = len(ALGS), len(DS)
    se = np.sqrt(k * (k + 1) / (6.0 * n))
    control = rank_fit.index[0]
    rows = []
    for a in rank_fit.index[1:]:
        z = (rank_fit[a] - rank_fit[control]) / se
        p = 2 * (1 - norm.cdf(abs(z)))
        rows.append((a, rank_fit[a], z, p))
    rows.sort(key=lambda r: -r[3])
    m_ = len(rows)
    lines = []
    for i, (a, r, z, p) in enumerate(rows):
        alpha_holm = 0.05 / (m_ - i)
        sig = "Yes" if p < alpha_holm else "No"
        nm = texname(a)
        if a == "AMPA_FS":
            nm = f"\\textbf{{{nm}}}"
        lines.append(f"{nm} & {r:.2f} & {z:.3f} & {p:.4f} & {sig} \\\\")
    with open(os.path.join(OUT, "tab_holm_body.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("Holm control:", control)
    for a, r, z, p in rows:
        print(f"  {a:8s} rank={r:.2f} z={z:.3f} p={p:.4f}")

    # ---------------- Vargha-Delaney A12 (fitness, AMPA vs each) ----------------
    def a12(x, y):
        gt = sum((xi < yi) for xi in x for yi in y)  # lower fitness better
        eq = sum(np.isclose(xi, yi) for xi in x for yi in y)
        return (gt + 0.5 * eq) / (len(x) * len(y))

    a12_mean = {}
    for a in ALGS[1:]:
        vals = []
        for ds in DS:
            x = np.array(raw[ds]["AMPA_FS"]["fitness"])
            y = np.array(raw[ds][a]["fitness"])
            vals.append(a12(x, y))
        a12_mean[a] = np.mean(vals)
    print("A12 (AMPA vs alg, >0.5 = AMPA better):",
          {a: round(v, 3) for a, v in a12_mean.items()})

    # ---------------- extended metrics: per-algorithm mean over 13 ds ----------
    agg = xm.groupby("Algorithm")[["accuracy", "precision", "sensitivity",
                                   "specificity", "f1", "auc"]].mean().loc[ALGS]
    lines = []
    for a in ALGS:
        r = agg.loc[a]
        nm = texname(a)
        if a == "AMPA_FS":
            nm = f"\\textbf{{{nm}}}"
        lines.append(f"{nm} & " + " & ".join(f"{r[c]:.4f}" for c in agg.columns) + r" \\")
    with open(os.path.join(OUT, "tab_extmetrics_body.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("ext metrics mean:\n", agg.round(4).to_string())

    # ---------------- high-dim table: acc + nfeat per dataset ----------------
    lines = []
    for ds in HD:
        best = acc.loc[ds].max()
        cells = []
        for a in ALGS:
            v = acc.loc[ds, a]
            c = f".{v*1000:03.0f}"
            if abs(v - best) < 5e-4:
                c = f"\\textbf{{{c}}}"
            cells.append(f"{c}/{nsel.loc[ds, a]:.0f}")
        lines.append(f"{ds} & " + " & ".join(cells) + r" \\")
    with open(os.path.join(OUT, "tab_highdim_body.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---------------- init-range table ----------------
    ir = pd.read_csv(os.path.join(config.RESULTS_DIR, "init_range_summary.csv"))
    piv_a = ir.pivot(index="Dataset", columns="Bound", values="RF_Acc_Mean")
    piv_s = ir.pivot(index="Dataset", columns="Bound", values="RF_Acc_Std")
    piv_n = ir.pivot(index="Dataset", columns="Bound", values="Selected_Mean")
    lines = []
    for ds in piv_a.index:
        lines.append(
            f"{ds} & .{piv_a.loc[ds,'clip6']*1000:03.0f}$\\pm$.{piv_s.loc[ds,'clip6']*1000:03.0f}"
            f" & .{piv_a.loc[ds,'clip2']*1000:03.0f}$\\pm$.{piv_s.loc[ds,'clip2']*1000:03.0f}"
            f" & {piv_a.loc[ds,'clip6']-piv_a.loc[ds,'clip2']:+.4f}"
            f" & {piv_n.loc[ds,'clip6']:.1f} & {piv_n.loc[ds,'clip2']:.1f} \\\\")
    with open(os.path.join(OUT, "tab_initrange_body.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n[DONE] fragments in {OUT}")


if __name__ == "__main__":
    main()
