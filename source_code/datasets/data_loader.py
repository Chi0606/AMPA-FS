"""Dataset loading for the wrapper feature-selection experiments.

load_dataset(name) returns (X, y): X is float, min-max scaled to [0, 1]
(manuscript Sec. 4.1.1), y is integer labels 0..C-1. Iris/Wine/BreastCancer
ship with scikit-learn; the other seven UCI sets and the three revision
gene-expression sets (D >= 200) are pulled from OpenML on first use and
cached under datasets/data/<name>.npz so later runs need no network.
"""

import os
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_HERE, "data")
os.makedirs(CACHE_DIR, exist_ok=True)


def _load_sklearn(loader_name):
    from sklearn import datasets as skd

    loader = getattr(skd, loader_name)
    bunch = loader()
    X = np.asarray(bunch.data, dtype=float)
    y = np.asarray(bunch.target)
    return X, y


def _load_openml(candidates, drop_columns=None):
    """Try each (name, version) candidate in turn until one fetches; an
    optional drop_columns removes identifier-style columns."""
    from sklearn.datasets import fetch_openml

    last_err = None
    for name, version in candidates:
        try:
            kwargs = {"as_frame": True}
            if version is not None:
                kwargs["version"] = version
            bunch = fetch_openml(name, **kwargs)
        except Exception as exc:  # pragma: no cover - network dependent
            last_err = exc
            continue

        frame = bunch.frame.copy()
        target_col = bunch.target_names[0]
        if drop_columns:
            frame = frame.drop(columns=[c for c in drop_columns if c in frame.columns])

        y_raw = frame[target_col]
        X_frame = frame.drop(columns=[target_col])

        # keep only complete rows, matching the paper's clean subsets
        mask = X_frame.notna().all(axis=1) & y_raw.notna()
        X_frame = X_frame[mask]
        y_raw = y_raw[mask]

        X_frame = _numeric_encode(X_frame)

        X = X_frame.to_numpy(dtype=float)
        y = _encode_labels(y_raw.to_numpy())
        return X, y

    raise RuntimeError(
        f"Could not load any OpenML candidate {candidates}: {last_err}"
    )


def _numeric_encode(frame):
    import pandas as pd

    out = {}
    for col in frame.columns:
        s = frame[col]
        if s.dtype.kind in "biufc":
            out[col] = s.astype(float)
        else:
            out[col] = s.astype("category").cat.codes.astype(float)
    return pd.DataFrame(out)


def _encode_labels(y):
    from sklearn.preprocessing import LabelEncoder

    return LabelEncoder().fit_transform(np.asarray(y).ravel()).astype(int)


# name -> zero-argument loader returning (X_raw, y_int)
_REGISTRY = {
    # the ten paper benchmarks (D = 4..60)
    "Iris": lambda: _load_sklearn("load_iris"),
    "Wine": lambda: _load_sklearn("load_wine"),
    "BreastCancer": lambda: _load_sklearn("load_breast_cancer"),
    "Zoo": lambda: _load_openml([("zoo", 1)], drop_columns=["animal"]),
    "Heart": lambda: _load_openml([("heart-c", None), ("heart-statlog", 1)]),
    "Ionosphere": lambda: _load_openml([("ionosphere", 1)]),
    "Sonar": lambda: _load_openml([("sonar", 1)]),
    "Dermatology": lambda: _load_openml([("dermatology", 1)]),
    "Vehicle": lambda: _load_openml([("vehicle", 1)]),
    "Parkinsons": lambda: _load_openml(
        [("parkinsons", 1)], drop_columns=["name"]
    ),
    # revision gene-expression sets (D >= 200)
    # Colon: 62 x 2000, 2 classes
    "Colon": lambda: _load_openml([("colon", None), ("Colon", None)]),
    # Leukemia micro-array
    "Leukemia": lambda: _load_openml([("leukemia", None), ("Leukemia", None)]),
    # SRBCT: 83 x 2308, 4 classes
    "SRBCT": lambda: _load_openml([("SRBCT", None), ("srbct", None)]),
}


def available_datasets():
    return list(_REGISTRY.keys())


def load_dataset(name, normalize=True, use_cache=True, prescreen_k=None):
    """Load a dataset as (X float, y int).

    normalize min-max scales each column to [0, 1]; use_cache reads/writes a
    local .npz. prescreen_k, when set below the feature count, keeps the top-k
    features by ANOVA F-score so the wrapper stays tractable on the
    high-dimensional revision sets; the ten base benchmarks leave it None.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown dataset '{name}'. Available: {available_datasets()}"
        )

    cache_path = os.path.join(CACHE_DIR, f"{name}.npz")
    if use_cache and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=False)
        X, y = data["X"].astype(float), data["y"].astype(int)
    else:
        X, y = _REGISTRY[name]()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if use_cache:
            np.savez_compressed(cache_path, X=X, y=y)

    if normalize:
        X = _min_max_scale(X)
    if prescreen_k is not None and prescreen_k < X.shape[1]:
        X = _filter_prescreen(X, y, prescreen_k)
    return X, y


def _filter_prescreen(X, y, k):
    """Keep the top-k features by univariate ANOVA F-score."""
    from sklearn.feature_selection import SelectKBest, f_classif

    selector = SelectKBest(score_func=f_classif, k=int(k))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_new = selector.fit_transform(X, y)
    return np.asarray(X_new, dtype=float)


def _min_max_scale(X):
    """Min-max scale each column to [0, 1]; constant columns map to 0."""
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    span = hi - lo
    span[span == 0] = 1.0
    return (X - lo) / span


if __name__ == "__main__":
    for ds in available_datasets():
        try:
            X, y = load_dataset(ds)
            print(f"  OK   {ds:15s} | {X.shape[0]:5d} x {X.shape[1]:5d} "
                  f"| classes={len(set(y.tolist()))}")
        except Exception as exc:
            print(f"  FAIL {ds:15s} | {str(exc)[:70]}")
