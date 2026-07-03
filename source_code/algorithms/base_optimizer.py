"""Shared base class for the binary metaheuristic wrappers."""

import time
import numpy as np
from utils.fitness import FitnessEvaluator


class BinaryOptimizer:
    """Subclasses implement _initialize_population(X, y) and
    _update_population(t, X, y); everything else (evaluation, best tracking,
    convergence history) is handled here."""

    def __init__(self, pop_size=30, max_iter=100, alpha=0.99, beta=0.01, seed=None,
                 **kwargs):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.seed = seed

        # set during optimize()
        self.n_features = None
        self.evaluator = None
        self.population = None       # (pop_size, n_features) binary
        self.continuous_pos = None   # (pop_size, n_features) continuous
        self.fitness = None
        self.accuracy = None
        self.n_selected = None

        self.best_solution = None
        self.best_fitness = np.inf
        self.best_accuracy = 0.0
        self.best_n_selected = 0

        self.convergence_curve = []
        self.accuracy_curve = []

    def optimize(self, X, y):
        """Run the search; returns a dict with the best solution, its
        fitness/accuracy/subset size, both convergence curves, and the
        elapsed wall-clock time."""
        if self.seed is not None:
            np.random.seed(self.seed)

        self.n_features = X.shape[1]
        self.evaluator = FitnessEvaluator(
            X, y, alpha=self.alpha, beta=self.beta,
            random_state=self.seed if self.seed else 42
        )
        start_time = time.time()

        self._initialize_population(X, y)
        self._evaluate_all(X, y)
        self._update_best()

        for t in range(self.max_iter):
            self._update_population(t, X, y)
            self._update_best()
            self.convergence_curve.append(self.best_fitness)
            self.accuracy_curve.append(self.best_accuracy)

        elapsed = time.time() - start_time

        return {
            "best_solution": self.best_solution.copy(),
            "best_fitness": self.best_fitness,
            "best_accuracy": self.best_accuracy,
            "best_n_selected": self.best_n_selected,
            "convergence_curve": list(self.convergence_curve),
            "accuracy_curve": list(self.accuracy_curve),
            "elapsed_time": elapsed,
        }

    def _evaluate_all(self, X, y):
        self.fitness = np.zeros(self.pop_size)
        self.accuracy = np.zeros(self.pop_size)
        self.n_selected = np.zeros(self.pop_size, dtype=int)
        for i in range(self.pop_size):
            f, a, s = self.evaluator.evaluate(self.population[i])
            self.fitness[i] = f
            self.accuracy[i] = a
            self.n_selected[i] = s

    def _update_best(self):
        idx = np.argmin(self.fitness)
        if self.fitness[idx] < self.best_fitness:
            self.best_fitness = self.fitness[idx]
            self.best_solution = self.population[idx].copy()
            self.best_accuracy = self.accuracy[idx]
            self.best_n_selected = self.n_selected[idx]

    def _initialize_population(self, X, y):
        raise NotImplementedError

    def _update_population(self, t, X, y):
        raise NotImplementedError

    def _random_binary_population(self, n, d):
        return (np.random.random((n, d)) > 0.5).astype(int)
