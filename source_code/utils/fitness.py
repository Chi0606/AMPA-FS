"""Wrapper fitness evaluation.

Two-phase protocol, standard in the metaheuristic FS literature: the search
phase scores subsets with KNN(k=5) on a single stratified 70/30 hold-out
split (about 1 ms per call, no k-fold CV), and the report phase re-evaluates
the final subset with RF and KNN under 10-fold stratified CV (evaluate_final).
"""

import warnings
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class FitnessEvaluator:
    """Search-phase evaluator with result caching and a pre-computed split.

    fitness = alpha * error + beta * (n_selected / n_features), evaluated by
    KNN(k) on the stored 70/30 stratified split.
    """

    def __init__(self, X, y, alpha=0.99, beta=0.01, k=5,
                 test_ratio=0.3, random_state=42):
        self.X = X
        self.y = y
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.n_features = X.shape[1]
        self.random_state = random_state
        self._cache = {}
        self._eval_count = 0

        # A single stratified split instead of 5-fold CV keeps a run at
        # pop_size=30 / max_iter=100 roughly 4x cheaper (~1 ms per eval).
        # Final reported accuracy still uses 10-fold CV, see evaluate_final().
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=test_ratio,
                             stratify=y, random_state=random_state)

    def evaluate(self, solution):
        """Return (fitness, accuracy, n_selected) for a binary mask."""
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

        clf = KNeighborsClassifier(
            n_neighbors=min(self.k, len(self.y_train) - 1),
            algorithm='brute'
        )
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

    def clear_cache(self):
        self._cache.clear()


def evaluate_final(solution, X, y, random_state=42):
    """Report-phase evaluation: 10-fold stratified CV with RF and KNN.

    Returns ({'KNN': {...}, 'RF': {...}}, n_selected), each inner dict
    holding accuracy_mean and accuracy_std.
    """
    selected = np.where(solution > 0.5)[0]
    if len(selected) == 0:
        return {"KNN": {"accuracy_mean": 0, "accuracy_std": 0},
                "RF":  {"accuracy_mean": 0, "accuracy_std": 0}}, 0

    X_sel = X[:, selected]
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)

    classifiers = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "RF":  RandomForestClassifier(n_estimators=100, random_state=random_state,
                                      n_jobs=-1),   # all cores for speed
    }

    results = {}
    for name, clf in classifiers.items():
        try:
            scores = cross_val_score(clf, X_sel, y, cv=cv, scoring="accuracy",
                                     n_jobs=1)   # sequential folds (avoids nested parallelism)
            results[name] = {
                "accuracy_mean": np.mean(scores),
                "accuracy_std": np.std(scores),
            }
        except Exception:
            results[name] = {"accuracy_mean": 0.0, "accuracy_std": 0.0}

    return results, len(selected)


def evaluate_metrics(solution, X, y, classifier="RF", random_state=42):
    """Extended metrics for a feature subset via 10-fold cross_val_predict:
    accuracy, macro precision, sensitivity (macro recall), macro specificity,
    macro F1, and ROC-AUC when predicted probabilities are available."""
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, confusion_matrix, roc_auc_score)

    selected = np.where(np.asarray(solution) > 0.5)[0]
    empty = {"accuracy": 0.0, "precision": 0.0, "sensitivity": 0.0,
             "specificity": 0.0, "f1": 0.0, "auc": np.nan,
             "n_selected": int(len(selected))}
    if len(selected) == 0:
        return empty

    X_sel = X[:, selected]
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    if classifier == "KNN":
        clf = KNeighborsClassifier(n_neighbors=5)
    else:
        clf = RandomForestClassifier(n_estimators=100,
                                     random_state=random_state, n_jobs=-1)

    try:
        y_pred = cross_val_predict(clf, X_sel, y, cv=cv, n_jobs=1)
    except Exception:
        return empty

    # macro specificity from the multiclass confusion matrix
    cm = confusion_matrix(y, y_pred)
    spec_per_class = []
    total = cm.sum()
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        spec_per_class.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    auc = np.nan
    try:
        y_proba = cross_val_predict(clf, X_sel, y, cv=cv, n_jobs=1,
                                    method="predict_proba")
        classes = np.unique(y)
        if len(classes) == 2:
            auc = roc_auc_score(y, y_proba[:, 1])
        else:
            auc = roc_auc_score(y, y_proba, multi_class="ovr", average="macro")
    except Exception:
        pass

    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, average="macro",
                                            zero_division=0)),
        "sensitivity": float(recall_score(y, y_pred, average="macro",
                                           zero_division=0)),
        "specificity": float(np.mean(spec_per_class)),
        "f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "auc": float(auc) if auc == auc else np.nan,
        "n_selected": int(len(selected)),
    }
