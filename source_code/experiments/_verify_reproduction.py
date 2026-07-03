"""Compare reproduced main_summary.csv against the paper's Table tab:rfacc,
compute Friedman ranks and the AMPA_FS/BMPA time ratio per dataset."""
import pandas as pd

ALGS = ["AMPA_FS", "BMPA", "BPSO", "BGA", "BGWO", "BWOA", "BSSA", "BHHO", "BSCA", "BGOA"]
DATASETS = ["Iris", "Wine", "Zoo", "Heart", "BreastCancer", "Ionosphere",
            "Sonar", "Dermatology", "Vehicle", "Parkinsons"]

PAPER_RF = {  # from paper.tex tab:rfacc (3 decimals)
 "Iris":        [.948]*10,
 "Wine":        [.957,.950,.948,.958,.955,.956,.956,.957,.957,.963],
 "Zoo":         [.944,.942,.946,.940,.951,.946,.942,.946,.945,.942],
 "Heart":       [.795,.794,.794,.792,.802,.799,.794,.799,.795,.797],
 "BreastCancer":[.958,.959,.960,.959,.958,.957,.959,.960,.962,.960],
 "Ionosphere":  [.926,.926,.921,.921,.929,.927,.925,.929,.926,.925],
 "Sonar":       [.815,.813,.806,.810,.816,.812,.817,.816,.816,.815],
 "Dermatology": [.961,.961,.963,.959,.967,.965,.969,.968,.965,.968],
 "Vehicle":     [.743,.741,.739,.741,.740,.740,.739,.742,.743,.742],
 "Parkinsons":  [.909,.909,.911,.914,.913,.908,.909,.914,.903,.907],
}
PAPER_RANK = [5.80,6.85,6.60,7.00,4.30,6.00,5.75,2.80,4.80,5.10]

df = pd.read_csv("results/main_summary.csv")
piv = df.pivot(index="Dataset", columns="Algorithm", values="RF_Acc_Mean")
tpiv = df.pivot(index="Dataset", columns="Algorithm", values="Time_Mean")

print("=== RF accuracy: reproduced (3dp) vs paper | diff ===")
max_diff = 0.0
for ds in DATASETS:
    row = []
    for i, a in enumerate(ALGS):
        rep = round(piv.loc[ds, a], 3)
        pap = PAPER_RF[ds][i]
        d = rep - pap
        max_diff = max(max_diff, abs(d))
        row.append(f"{a}:{rep:.3f}/{pap:.3f}{'' if abs(d) < 0.0005 else f'({d:+.3f})'}")
    print(f"{ds:13s} " + " ".join(row))
print(f"\nmax |diff| = {max_diff:.3f}")

print("\n=== Friedman ranks on RF accuracy (reproduced vs paper) ===")
ranks = piv[ALGS].rank(axis=1, ascending=False).loc[DATASETS].mean()
for i, a in enumerate(ALGS):
    print(f"{a:8s} repro={ranks[a]:.2f}  paper={PAPER_RANK[i]:.2f}")

print("\n=== AMPA_FS / BMPA wall-clock time ratio ===")
rat = (tpiv["AMPA_FS"] / tpiv["BMPA"]).loc[DATASETS]
for ds in DATASETS:
    print(f"{ds:13s} AMPA={tpiv.loc[ds,'AMPA_FS']:.2f}s BMPA={tpiv.loc[ds,'BMPA']:.2f}s ratio={rat[ds]:.3f}")
print(f"\nmean ratio = {rat.mean():.3f}, median = {rat.median():.3f}")
