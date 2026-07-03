"""
BGOA: Binary Grasshopper Optimization Algorithm for Feature Selection.

Reference: Saremi et al. (2017), Grasshopper Optimisation Algorithm.
"""

import numpy as np
from algorithms.base_optimizer import BinaryOptimizer
from utils.transfer_functions import sigmoid


class BGOA(BinaryOptimizer):
    """Binary Grasshopper Optimization Algorithm."""

    def __init__(self, pop_size=30, max_iter=100, alpha=0.99, beta=0.01,
                 c_min=0.00004, c_max=1.0, seed=None, **kwargs):
        super().__init__(pop_size, max_iter, alpha, beta, seed=seed)
        self.c_min = c_min
        self.c_max = c_max

    def _initialize_population(self, X, y):
        self.population = self._random_binary_population(self.pop_size, self.n_features)
        self.continuous_pos = np.random.random((self.pop_size, self.n_features))

    def _s_function(self, r, f=0.5, l=1.5):
        """Social interaction function."""
        return f * np.exp(-r / l) - np.exp(-r)

    def _update_population(self, t, X, y):
        """GOA position update via pairwise social forces (Saremi et al. 2017, Eq. 2.7)."""
        N, D = self.continuous_pos.shape
        c = self.c_max - (self.c_max - self.c_min) * t / self.max_iter  # comfort zone shrinks
        best_pos = self.continuous_pos[np.argmin(self.fitness)].copy()

        new_pos = np.zeros_like(self.continuous_pos)

        for i in range(N):
            S_i = np.zeros(D)
            for j in range(N):
                if i == j:
                    continue
                dist = np.abs(self.continuous_pos[j] - self.continuous_pos[i])
                dist = np.clip(dist, 1e-10, None)
                r_ij = dist
                direction = (self.continuous_pos[j] - self.continuous_pos[i]) / (r_ij + 1e-10)
                S_i += c * ((1 - 0) / 2) * self._s_function(r_ij) * direction

            new_pos[i] = c * S_i + best_pos

        self.continuous_pos = np.clip(new_pos, 0, 1)
        for i in range(N):
            prob = sigmoid(self.continuous_pos[i])
            self.population[i] = (np.random.random(D) < prob).astype(int)

        self._evaluate_all(X, y)
