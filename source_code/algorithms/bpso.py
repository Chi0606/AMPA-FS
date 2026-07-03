"""
BPSO: Binary Particle Swarm Optimization for Feature Selection.

Reference: Kennedy & Eberhart (1997).
"""

import numpy as np
from algorithms.base_optimizer import BinaryOptimizer
from utils.transfer_functions import sigmoid


class BPSO(BinaryOptimizer):
    """Binary Particle Swarm Optimization."""

    def __init__(self, pop_size=30, max_iter=100, alpha=0.99, beta=0.01,
                 w_max=0.9, w_min=0.4, c1=2.0, c2=2.0, seed=None, **kwargs):
        super().__init__(pop_size, max_iter, alpha, beta, seed=seed)
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.velocity = None
        self.pbest = None
        self.pbest_fitness = None

    def _initialize_population(self, X, y):
        D = self.n_features
        N = self.pop_size
        self.population = self._random_binary_population(N, D)
        self.velocity = np.random.uniform(-3, 3, (N, D))
        self.pbest = self.population.copy()
        self.pbest_fitness = np.full(N, np.inf)

    def _update_population(self, t, X, y):
        N, D = self.population.shape
        w = self.w_max - (self.w_max - self.w_min) * t / self.max_iter  # inertia weight
        gbest = self.best_solution

        for i in range(N):
            # Update personal best (cognitive component)
            if self.fitness[i] < self.pbest_fitness[i]:
                self.pbest[i] = self.population[i].copy()
                self.pbest_fitness[i] = self.fitness[i]

            r1, r2 = np.random.random(D), np.random.random(D)
            self.velocity[i] = (w * self.velocity[i]
                                + self.c1 * r1 * (self.pbest[i] - self.population[i])
                                + self.c2 * r2 * (gbest - self.population[i]))
            self.velocity[i] = np.clip(self.velocity[i], -6, 6)

            # Sigmoid transfer: P(x=1) = sigmoid(v) (Kennedy & Eberhart 1997)
            prob = sigmoid(self.velocity[i])
            self.population[i] = (np.random.random(D) < prob).astype(int)

        self._evaluate_all(X, y)
