"""
BEHO: Binary Elk Herd Optimizer for Feature Selection.

Continuous Elk Herd Optimizer (EHO):
    Al-Betar, Awadallah, Braik, Makhadmeh, Doush (2024),
    "Elk herd optimizer: a novel nature-inspired metaheuristic algorithm",
    Artificial Intelligence Review 57:48.

EHO divides the herd into families led by bulls (rutting season) and updates
calves toward their family's bull and the global best (calving season). The
continuous positions are binarized with the shared S-shaped transfer function
(added for Reviewer 5.11).
"""

import numpy as np

from algorithms.base_optimizer import BinaryOptimizer
from utils.transfer_functions import sigmoid


class BEHO(BinaryOptimizer):
    """Binary Elk Herd Optimizer."""

    def __init__(self, pop_size=30, max_iter=100, alpha=0.99, beta=0.01,
                 bull_ratio=0.2, seed=None, **kwargs):
        super().__init__(pop_size, max_iter, alpha, beta, seed=seed)
        self.bull_ratio = bull_ratio
        self.lo, self.hi = -6.0, 6.0

    def _initialize_population(self, X, y):
        N, D = self.pop_size, self.n_features
        self.continuous_pos = np.random.uniform(self.lo, self.hi, (N, D))
        prob = sigmoid(self.continuous_pos)
        self.population = (np.random.random((N, D)) < prob).astype(int)

    def _update_population(self, t, X, y):
        N, D = self.pop_size, self.n_features

        # --- Rutting season: rank elks, top fraction become bulls ---
        order = np.argsort(self.fitness)          # ascending (best first)
        n_bulls = max(1, int(self.bull_ratio * N))
        bulls = order[:n_bulls]

        # assign each non-bull elk to a bull family (harem) proportional to
        # bull fitness quality
        bull_fit = self.fitness[bulls]
        quality = 1.0 / (bull_fit - bull_fit.min() + 1e-9 + 1.0)
        quality = quality / quality.sum()
        family = {int(b): [] for b in bulls}
        for i in order[n_bulls:]:
            b = int(np.random.choice(bulls, p=quality))
            family[b].append(int(i))

        new_pos = self.continuous_pos.copy()
        gbest = self.continuous_pos[order[0]]

        # --- Calving season: update calves toward bull + herd average ---
        for b, calves in family.items():
            bull_pos = self.continuous_pos[b]
            for i in calves:
                r1, r2 = np.random.random(D), np.random.random(D)
                herd_mean = self.continuous_pos[[b] + calves].mean(axis=0)
                new_pos[i] = (self.continuous_pos[i]
                              + r1 * (bull_pos - self.continuous_pos[i])
                              + r2 * (herd_mean - self.continuous_pos[i]))

        # --- Bulls explore around the global best ---
        for b in bulls:
            r = np.random.random(D)
            new_pos[b] = self.continuous_pos[b] + r * (gbest - self.continuous_pos[b]) \
                + 0.1 * np.random.randn(D)

        self.continuous_pos = np.clip(new_pos, self.lo, self.hi)
        prob = sigmoid(self.continuous_pos)
        self.population = (np.random.random((N, D)) < prob).astype(int)
        self._evaluate_all(X, y)
