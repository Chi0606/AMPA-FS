"""
BGA: Binary Genetic Algorithm for Feature Selection.
"""

import numpy as np
from algorithms.base_optimizer import BinaryOptimizer


class BGA(BinaryOptimizer):
    """Binary Genetic Algorithm."""

    def __init__(self, pop_size=30, max_iter=100, alpha=0.99, beta=0.01,
                 crossover_rate=0.8, mutation_rate=0.01, seed=None, **kwargs):
        super().__init__(pop_size, max_iter, alpha, beta, seed=seed)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def _initialize_population(self, X, y):
        self.population = self._random_binary_population(self.pop_size, self.n_features)

    def _update_population(self, t, X, y):
        """Standard GA cycle: elitism -> selection -> crossover -> mutation."""
        N, D = self.population.shape
        new_pop = np.zeros_like(self.population)

        # Elitism: copy the fittest individual unchanged
        best_idx = np.argmin(self.fitness)
        new_pop[0] = self.population[best_idx].copy()

        for k in range(1, N):
            # Tournament selection (k=3) for both parents
            p1 = self._tournament_select()
            p2 = self._tournament_select()

            # Single-point crossover (Holland 1975)
            if np.random.random() < self.crossover_rate:
                cp = np.random.randint(1, D)
                child = np.concatenate([p1[:cp], p2[cp:]])
            else:
                child = p1.copy()

            # Bit-flip mutation
            mask = np.random.random(D) < self.mutation_rate
            child[mask] = 1 - child[mask]

            new_pop[k] = child

        self.population = new_pop
        self._evaluate_all(X, y)

    def _tournament_select(self, k=3):
        """Tournament selection with size k."""
        indices = np.random.choice(self.pop_size, k, replace=False)
        best = indices[np.argmin(self.fitness[indices])]
        return self.population[best].copy()
