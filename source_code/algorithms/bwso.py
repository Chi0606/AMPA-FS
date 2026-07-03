"""
BWSO: Binary White Shark Optimizer for Feature Selection.

Continuous White Shark Optimizer (WSO):
    Braik, Hammouri, Atwan, Al-Betar, Awadallah (2022),
    "White Shark Optimizer: A novel bio-inspired meta-heuristic algorithm
    for global optimization problems", Knowledge-Based Systems 243:108457.

The continuous velocity/position dynamics of WSO are kept intact and the
resulting real-valued positions are mapped to a binary feature mask with the
same S-shaped transfer function used by the other binary optimizers in this
package, so the comparison is apples-to-apples (added for Reviewer 5.11).
"""

import numpy as np

from algorithms.base_optimizer import BinaryOptimizer
from utils.transfer_functions import sigmoid


class BWSO(BinaryOptimizer):
    """Binary White Shark Optimizer."""

    def __init__(self, pop_size=30, max_iter=100, alpha=0.99, beta=0.01,
                 fmin=0.07, fmax=0.75, tau=4.125, a0=6.25, a1=100.0, a2=5e-4,
                 seed=None, **kwargs):
        super().__init__(pop_size, max_iter, alpha, beta, seed=seed)
        self.fmin = fmin
        self.fmax = fmax
        self.tau = tau
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2
        self.velocity = None
        self.gbest_cont = None

    def _initialize_population(self, X, y):
        N, D = self.pop_size, self.n_features
        self.continuous_pos = np.random.uniform(-6, 6, (N, D))
        self.velocity = np.zeros((N, D))
        self.population = (sigmoid(self.continuous_pos) > np.random.random((N, D))).astype(int)
        self.gbest_cont = self.continuous_pos[0].copy()

    def _update_population(self, t, X, y):
        N, D = self.pop_size, self.n_features
        T = self.max_iter

        # keep a continuous global-best proxy (best individual this iteration)
        best_idx = int(np.argmin(self.fitness))
        if self.gbest_cont is None:
            self.gbest_cont = self.continuous_pos[best_idx].copy()
        else:
            self.gbest_cont = self.continuous_pos[best_idx].copy()

        # WSO control schedules
        p1 = self.a0 + np.exp(-(t / T) * self.a1)  # exploration weight
        p2 = self.a1 + np.exp(-t / T) * self.a2    # (unused magnitude term)
        mu = 2.0 / abs(2.0 - self.tau - np.sqrt(self.tau ** 2 - 4 * self.tau))
        ss = np.abs(1.0 - np.exp(-self.a2 * t / T))
        freq = self.fmin + (self.fmax - self.fmin) / (self.fmin + self.fmax)

        lo, hi = -6.0, 6.0
        for i in range(N):
            c1, c2 = np.random.random(D), np.random.random(D)
            # velocity update towards global best (Eq. 4 of WSO)
            self.velocity[i] = (mu * self.velocity[i]
                                + p1 * c1 * (self.gbest_cont - self.continuous_pos[i])
                                + p2 * c2 * (self.continuous_pos[best_idx]
                                             - self.continuous_pos[i]) * freq)

            if np.random.random() < ss:
                # move towards the best shark's position
                self.continuous_pos[i] = self.continuous_pos[i] + self.velocity[i] / (freq + 1e-9)
            else:
                # fish-schooling towards global best with random drift
                self.continuous_pos[i] = (self.gbest_cont
                                          + np.random.random(D) * (hi - lo) * 0.02
                                          * np.sign(np.random.random(D) - 0.5))

        self.continuous_pos = np.clip(self.continuous_pos, lo, hi)
        prob = sigmoid(self.continuous_pos)
        self.population = (np.random.random((N, D)) < prob).astype(int)
        self._evaluate_all(X, y)
