"""
BSCA: Binary Sine Cosine Algorithm for Feature Selection.

Reference: Mirjalili (2016), SCA: A Sine Cosine Algorithm.
"""

import numpy as np
from algorithms.base_optimizer import BinaryOptimizer
from utils.transfer_functions import sigmoid


class BSCA(BinaryOptimizer):
    """Binary Sine Cosine Algorithm."""

    def __init__(self, pop_size=30, max_iter=100, alpha=0.99, beta=0.01,
                 a_max=2.0, seed=None, **kwargs):
        super().__init__(pop_size, max_iter, alpha, beta, seed=seed)
        self.a_max = a_max

    def _initialize_population(self, X, y):
        self.population = self._random_binary_population(self.pop_size, self.n_features)
        self.continuous_pos = np.random.random((self.pop_size, self.n_features))

    def _update_population(self, t, X, y):
        N, D = self.continuous_pos.shape
        a = self.a_max - self.a_max * t / self.max_iter
        best_pos = self.continuous_pos[np.argmin(self.fitness)].copy()

        # SCA update: agents oscillate around best via sine/cosine (Mirjalili 2016, Eq. 3.3)
        for i in range(N):
            r1 = a                                    # decreasing amplitude
            r2 = 2 * np.pi * np.random.random(D)     # random phase
            r3 = 2 * np.random.random(D)              # random weight for destination
            r4 = np.random.random()                   # sine vs cosine switch

            if r4 < 0.5:
                self.continuous_pos[i] += r1 * np.sin(r2) * np.abs(
                    r3 * best_pos - self.continuous_pos[i]
                )
            else:
                self.continuous_pos[i] += r1 * np.cos(r2) * np.abs(
                    r3 * best_pos - self.continuous_pos[i]
                )

            self.continuous_pos[i] = np.clip(self.continuous_pos[i], 0, 1)
            prob = sigmoid(self.continuous_pos[i])
            self.population[i] = (np.random.random(D) < prob).astype(int)

        self._evaluate_all(X, y)
