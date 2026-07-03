"""
BWOA: Binary Whale Optimization Algorithm for Feature Selection.

Reference: Mafarja & Mirjalili (2018).
"""

import numpy as np
from algorithms.base_optimizer import BinaryOptimizer
from utils.transfer_functions import sigmoid


class BWOA(BinaryOptimizer):
    """Binary Whale Optimization Algorithm."""

    def __init__(self, pop_size=30, max_iter=100, alpha=0.99, beta=0.01,
                 b=1.0, seed=None, **kwargs):
        super().__init__(pop_size, max_iter, alpha, beta, seed=seed)
        self.b = b

    def _initialize_population(self, X, y):
        self.population = self._random_binary_population(self.pop_size, self.n_features)
        self.continuous_pos = np.random.random((self.pop_size, self.n_features))

    def _update_population(self, t, X, y):
        N, D = self.continuous_pos.shape
        a = 2 - 2 * t / self.max_iter
        a2 = -1 - t / self.max_iter
        best_pos = self.continuous_pos[np.argmin(self.fitness)].copy()

        # WOA update: 50% chance of shrinking-encircle vs spiral (Mirjalili & Lewis 2016)
        for i in range(N):
            r = np.random.random()
            A = 2 * a * np.random.random(D) - a
            C = 2 * np.random.random(D)
            l = (a2 - 1) * np.random.random() + 1
            p = np.random.random()

            if p < 0.5:
                if np.abs(A).mean() < 1:
                    # Encircling prey (exploitation, Eq. 2.1)
                    D_vec = np.abs(C * best_pos - self.continuous_pos[i])
                    self.continuous_pos[i] = best_pos - A * D_vec
                else:
                    # Search for prey with random agent (exploration, |A|>=1)
                    rand_idx = np.random.randint(N)
                    rand_pos = self.continuous_pos[rand_idx]
                    D_vec = np.abs(C * rand_pos - self.continuous_pos[i])
                    self.continuous_pos[i] = rand_pos - A * D_vec
            else:
                # Logarithmic spiral approach (Eq. 2.5)
                D_vec = np.abs(best_pos - self.continuous_pos[i])
                self.continuous_pos[i] = (
                    D_vec * np.exp(self.b * l) * np.cos(2 * np.pi * l) + best_pos
                )

            self.continuous_pos[i] = np.clip(self.continuous_pos[i], 0, 1)
            prob = sigmoid(self.continuous_pos[i])
            self.population[i] = (np.random.random(D) < prob).astype(int)

        self._evaluate_all(X, y)
