"""
BHHO: Binary Harris Hawks Optimization for Feature Selection.

Reference: Heidari et al. (2019), Harris hawks optimization.
"""

import math
import numpy as np
from algorithms.base_optimizer import BinaryOptimizer
from utils.transfer_functions import sigmoid


class BHHO(BinaryOptimizer):
    """Binary Harris Hawks Optimization."""

    def __init__(self, pop_size=30, max_iter=100, alpha=0.99, beta=0.01,
                 seed=None, **kwargs):
        super().__init__(pop_size, max_iter, alpha, beta, seed=seed)

    def _initialize_population(self, X, y):
        self.population = self._random_binary_population(self.pop_size, self.n_features)
        self.continuous_pos = np.random.random((self.pop_size, self.n_features))

    def _levy_flight(self, d):
        """Mantegna's algorithm for Lévy flight."""
        beta_val = 1.5
        sigma = (math.gamma(1 + beta_val) * np.sin(np.pi * beta_val / 2)
                 / (math.gamma((1 + beta_val) / 2) * beta_val
                    * 2 ** ((beta_val - 1) / 2))) ** (1 / beta_val)
        u = np.random.randn(d) * sigma
        v = np.random.randn(d)
        return u / (np.abs(v) ** (1 / beta_val))

    def _update_population(self, t, X, y):
        N, D = self.continuous_pos.shape
        best_pos = self.continuous_pos[np.argmin(self.fitness)].copy()

        E0 = 2 * np.random.random() - 1   # Initial energy
        E = 2 * E0 * (1 - t / self.max_iter)  # Escaping energy

        # Four-branch HHO update (Heidari et al. 2019, Figs 1-2)
        for i in range(N):
            q = np.random.random()
            r = np.random.random()

            if abs(E) >= 1:
                # Exploration phase: hawks perch randomly or approach mean
                if q >= 0.5:
                    rand_idx = np.random.randint(N)
                    X_rand = self.continuous_pos[rand_idx]
                    self.continuous_pos[i] = X_rand - np.random.random(D) * np.abs(
                        X_rand - 2 * np.random.random(D) * self.continuous_pos[i]
                    )
                else:
                    X_mean = np.mean(self.continuous_pos, axis=0)
                    self.continuous_pos[i] = (
                        best_pos - X_mean
                    ) - np.random.random(D) * np.random.uniform(0, 1, D)
            else:
                # Exploitation phase: four siege variants based on |E| and r
                J = 2 * (1 - np.random.random())  # jump strength
                if r >= 0.5 and abs(E) >= 0.5:
                    # Soft besiege (Eq. 5)
                    self.continuous_pos[i] = (
                        best_pos - self.continuous_pos[i]
                    ) - E * np.abs(
                        J * best_pos - self.continuous_pos[i]
                    )
                elif r >= 0.5 and abs(E) < 0.5:
                    # Hard besiege (Eq. 6)
                    self.continuous_pos[i] = best_pos - E * np.abs(
                        best_pos - self.continuous_pos[i]
                    )
                elif r < 0.5 and abs(E) >= 0.5:
                    # Soft besiege with Lévy-flight rapid dives (Eq. 7)
                    Y = best_pos - E * np.abs(
                        J * best_pos - self.continuous_pos[i]
                    )
                    Z = Y + np.random.random(D) * self._levy_flight(D)
                    self.continuous_pos[i] = Y if np.random.random() < 0.5 else Z
                else:
                    # Hard besiege with Lévy-flight rapid dives (Eq. 10)
                    X_mean = np.mean(self.continuous_pos, axis=0)
                    Y = best_pos - E * np.abs(
                        J * best_pos - X_mean
                    )
                    Z = Y + np.random.random(D) * self._levy_flight(D)
                    self.continuous_pos[i] = Y if np.random.random() < 0.5 else Z

            self.continuous_pos[i] = np.clip(self.continuous_pos[i], 0, 1)
            prob = sigmoid(self.continuous_pos[i])
            self.population[i] = (np.random.random(D) < prob).astype(int)

        self._evaluate_all(X, y)
