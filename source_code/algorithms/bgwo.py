"""
BGWO: Binary Grey Wolf Optimizer for Feature Selection.

Reference: Emary et al. (2016), Binary grey wolf optimization approaches.
"""

import numpy as np
from algorithms.base_optimizer import BinaryOptimizer
from utils.transfer_functions import sigmoid


class BGWO(BinaryOptimizer):
    """Binary Grey Wolf Optimizer."""

    def __init__(self, pop_size=30, max_iter=100, alpha=0.99, beta=0.01,
                 seed=None, **kwargs):
        super().__init__(pop_size, max_iter, alpha, beta, seed=seed)

    def _initialize_population(self, X, y):
        self.population = self._random_binary_population(self.pop_size, self.n_features)
        self.continuous_pos = np.random.random((self.pop_size, self.n_features))

    def _update_population(self, t, X, y):
        N, D = self.continuous_pos.shape
        a = 2 - 2 * t / self.max_iter  # Linearly decreased from 2 to 0

        # Identify the three best wolves (hierarchy: alpha > beta > delta)
        sorted_idx = np.argsort(self.fitness)
        alpha_pos = self.continuous_pos[sorted_idx[0]].copy()
        beta_pos = self.continuous_pos[sorted_idx[1]].copy()
        delta_pos = self.continuous_pos[sorted_idx[2]].copy()

        # Each wolf updates its position guided by alpha, beta, delta (Eq. 3.7 in Mirjalili 2014)
        for i in range(N):
            for j in range(D):
                # Alpha encircling
                r1, r2 = np.random.random(), np.random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - self.continuous_pos[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha

                # Beta encircling
                r1, r2 = np.random.random(), np.random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - self.continuous_pos[i, j])
                X2 = beta_pos[j] - A2 * D_beta

                # Delta encircling
                r1, r2 = np.random.random(), np.random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - self.continuous_pos[i, j])
                X3 = delta_pos[j] - A3 * D_delta

                self.continuous_pos[i, j] = (X1 + X2 + X3) / 3

            # Sigmoid-based stochastic binarization (Emary et al. 2016)
            prob = sigmoid(self.continuous_pos[i])
            self.population[i] = (np.random.random(D) < prob).astype(int)

        self._evaluate_all(X, y)
