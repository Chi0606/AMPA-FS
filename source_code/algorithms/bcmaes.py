"""
BCMAES: Binary CMA-ES for Feature Selection.

Covariance Matrix Adaptation Evolution Strategy:
    Hansen & Ostermeier (2001), "Completely Derandomized Self-Adaptation in
    Evolution Strategies", Evolutionary Computation 9(2):159-195.

A separable (diagonal-covariance) CMA-ES is used so the per-iteration cost
stays O(N*D) and the method scales to the high-dimensional datasets added in
the revision (D up to several thousand). Continuous samples are mapped to a
binary feature mask with the shared S-shaped transfer function (added for
Reviewer 5.11).
"""

import numpy as np

from algorithms.base_optimizer import BinaryOptimizer
from utils.transfer_functions import sigmoid


class BCMAES(BinaryOptimizer):
    """Separable Binary CMA-ES."""

    def __init__(self, pop_size=30, max_iter=100, alpha=0.99, beta=0.01,
                 sigma0=2.0, seed=None, **kwargs):
        super().__init__(pop_size, max_iter, alpha, beta, seed=seed)
        self.sigma0 = sigma0
        # strategy state
        self.mean = None
        self.sigma = None
        self.C = None          # diagonal of covariance
        self.pc = None
        self.psigma = None
        self._weights = None
        self._mu = None
        self._mueff = None

    def _init_strategy(self, D):
        N = self.pop_size
        mu = N // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        self._weights = w
        self._mu = mu
        self._mueff = 1.0 / np.sum(w ** 2)

        self.mean = np.zeros(D)
        self.sigma = self.sigma0
        self.C = np.ones(D)
        self.pc = np.zeros(D)
        self.psigma = np.zeros(D)

        # adaptation constants (separable CMA-ES)
        self.cc = 4.0 / (D + 4.0)
        self.cs = (self._mueff + 2.0) / (D + self._mueff + 5.0)
        self.c1 = 2.0 / ((D + 1.3) ** 2 + self._mueff)
        self.cmu = min(1 - self.c1,
                       2.0 * (self._mueff - 2 + 1.0 / self._mueff)
                       / ((D + 2.0) ** 2 + self._mueff))
        self.damps = 1.0 + self.cs + 2.0 * max(0.0, np.sqrt((self._mueff - 1) / (D + 1)) - 1)
        self.chiN = np.sqrt(D) * (1 - 1.0 / (4 * D) + 1.0 / (21 * D ** 2))
        self._eigeval = 0
        self._samples = None

    def _initialize_population(self, X, y):
        N, D = self.pop_size, self.n_features
        self._init_strategy(D)
        self._sample(0)

    def _sample(self, t):
        N, D = self.pop_size, self.n_features
        z = np.random.randn(N, D)
        std = np.sqrt(self.C)
        self._z = z
        self._samples = self.mean + self.sigma * z * std        # (N, D)
        self.continuous_pos = np.clip(self._samples, -6, 6)
        prob = sigmoid(self.continuous_pos)
        self.population = (np.random.random((N, D)) < prob).astype(int)

    def _update_population(self, t, X, y):
        N, D = self.pop_size, self.n_features

        # rank current samples by fitness (ascending = best first)
        order = np.argsort(self.fitness)
        sel = order[:self._mu]
        old_mean = self.mean.copy()

        # weighted recombination in continuous space
        y_sel = (self._samples[sel] - old_mean) / self.sigma     # (mu, D)
        y_w = np.sum(self._weights[:, None] * y_sel, axis=0)
        self.mean = old_mean + self.sigma * y_w

        # cumulation for sigma (separable: use C^-1/2 ~ 1/sqrt(C))
        inv_sqrt_C = 1.0 / np.sqrt(self.C)
        self.psigma = ((1 - self.cs) * self.psigma
                       + np.sqrt(self.cs * (2 - self.cs) * self._mueff)
                       * inv_sqrt_C * y_w)
        ps_norm = np.linalg.norm(self.psigma)

        hsig = (ps_norm / np.sqrt(1 - (1 - self.cs) ** (2 * (t + 1))) / self.chiN
                < 1.4 + 2.0 / (D + 1))
        self.pc = ((1 - self.cc) * self.pc
                   + hsig * np.sqrt(self.cc * (2 - self.cc) * self._mueff) * y_w)

        # rank-mu update of diagonal covariance
        rank_mu = np.sum(self._weights[:, None] * (y_sel ** 2), axis=0)
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (self.pc ** 2 + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * rank_mu)
        self.C = np.maximum(self.C, 1e-12)

        # step-size update
        self.sigma *= np.exp((self.cs / self.damps) * (ps_norm / self.chiN - 1))
        self.sigma = float(np.clip(self.sigma, 1e-3, 1e3))

        # sample next generation
        self._sample(t)
        self._evaluate_all(X, y)
