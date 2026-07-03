"""
BSSA: Binary Salp Swarm Algorithm for Feature Selection.

Reference: Faris et al. (2018), "An efficient binary Salp Swarm Algorithm
           with crossover scheme for feature selection problems."
"""

import numpy as np
from algorithms.base_optimizer import BinaryOptimizer
from utils.transfer_functions import sigmoid


class BSSA(BinaryOptimizer):
    """Binary Salp Swarm Algorithm (Faris et al. 2018)."""

    def __init__(self, pop_size=30, max_iter=100, alpha=0.99, beta=0.01,
                 leader_ratio=0.2, boundary_ratio=0.1, update_threshold=0.8,
                 seed=None, **kwargs):
        super().__init__(pop_size, max_iter, alpha, beta, seed=seed)
        self.leader_ratio = leader_ratio        # Fraction of population acting as leaders
        self.boundary_ratio = boundary_ratio    # Fraction subject to boundary-awareness
        self.update_threshold = update_threshold  # Threshold for leader exploration mode

    def _initialize_population(self, X, y):
        self.population = self._random_binary_population(self.pop_size, self.n_features)
        self.continuous_pos = np.random.random((self.pop_size, self.n_features))

    def _update_population(self, t, X, y):
        N, D = self.continuous_pos.shape
        sorted_idx = np.argsort(self.fitness)
        n_leaders = max(1, int(N * self.leader_ratio))
        n_boundary = max(1, int(N * self.boundary_ratio))

        best_pos = self.continuous_pos[sorted_idx[0]].copy()
        worst_pos = self.continuous_pos[sorted_idx[-1]].copy()

        R2 = np.random.random()  # Alarm value

        # Leader salps: explore toward food source or random walk
        for idx in range(n_leaders):
            i = sorted_idx[idx]
            if R2 < self.update_threshold:
                alpha_val = np.random.random()
                self.continuous_pos[i] *= np.exp(-i / (alpha_val * self.max_iter + 1e-10))
            else:
                Q = np.random.randn(D)
                self.continuous_pos[i] += Q

        # Follower salps: chain-follow the leader or approach food source
        for idx in range(n_leaders, N):
            i = sorted_idx[idx]
            A = np.random.choice([-1, 1], size=D)
            Ap = A / (A @ A + 1e-10)
            if idx > N / 2:
                Q = np.random.randn(D)
                self.continuous_pos[i] = Q * np.exp(
                    (worst_pos - self.continuous_pos[i]) / (idx ** 2 + 1e-10)
                )
            else:
                leader_idx = sorted_idx[np.random.randint(n_leaders)]
                self.continuous_pos[i] = (
                    self.continuous_pos[leader_idx]
                    + np.abs(self.continuous_pos[i] - self.continuous_pos[leader_idx])
                    * Ap
                )

        # Boundary-awareness salps: escape local optima near worst/best
        danger_indices = np.random.choice(N, n_boundary, replace=False)
        for i in danger_indices:
            if self.fitness[i] > np.mean(self.fitness):
                self.continuous_pos[i] = best_pos + np.random.randn(D) * np.abs(
                    self.continuous_pos[i] - best_pos
                )
            else:
                K = np.random.uniform(-1, 1)
                self.continuous_pos[i] += K * (
                    np.abs(self.continuous_pos[i] - worst_pos)
                    / (self.fitness[i] - self.fitness[sorted_idx[-1]] + 1e-10)
                )

        self.continuous_pos = np.clip(self.continuous_pos, 0, 1)
        for i in range(N):
            prob = sigmoid(self.continuous_pos[i])
            self.population[i] = (np.random.random(D) < prob).astype(int)

        self._evaluate_all(X, y)
