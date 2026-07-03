import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.data_loader import load_dataset

datasets = [
    "Iris", "Wine", "BreastCancer", "Zoo", "Heart",
    "Ionosphere", "Sonar", "Dermatology", "Vehicle", "GermanCredit",
    "Parkinsons", "Lymphography", "Musk1", "Arrhythmia", "SpectEW",
]

ok = []
for d in datasets:
    try:
        X, y = load_dataset(d)
        print(f"  OK   {d:15s} | {X.shape[0]:5d} x {X.shape[1]:4d} | classes={len(set(y))}")
        ok.append(d)
    except Exception as e:
        print(f"  FAIL {d:15s} | {str(e)[:60]}")

print(f"\nAvailable: {len(ok)}/{len(datasets)}")
print(f"List: {ok}")
