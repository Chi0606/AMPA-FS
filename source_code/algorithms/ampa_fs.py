"""AMPA-FS: adaptive multi-strategy binary MPA for wrapper feature selection.

Three modules on top of BMPA:
  1. Tent chaotic map + elite opposition-based learning initialization
  2. phase-aligned adaptive step-size control factor
  3. stagnation-triggered elite-guided differential mutation

Base algorithm: Faramarzi et al. (2020), Marine Predators Algorithm, ESWA.
"""

import math
import numpy as np
from algorithms.base_optimizer import BinaryOptimizer
from utils.transfer_functions import binarize, adaptive_sigmoid


class AMPA_FS(BinaryOptimizer):
    """AMPA-FS optimizer.

    Notes on the non-obvious parameters:
      stag_threshold: iterations without improvement before Module 3 fires.
          Started at 5, moved to 10 after tau=5 proved too aggressive on
          the low-dimensional datasets.
      lambda_max: max Gaussian perturbation in the mutation; 0.1 was too
          weak to escape on Sonar/Ionosphere, so 0.5.
      gamma_min/gamma_max: adaptive sigmoid slope range (paper Eq. 12);
          gamma_max=10 binarized too hard late in the run, hence 8.
      tent_z0: Tent map seed, anything in (0,1) away from {0, 0.5, 1}.
      use_chaotic_init / use_adaptive_step / use_elite_mutation: module
          switches, used by the ablation study.
    The fitness weights alpha=0.99 / beta=0.01 follow Emary et al. (2016),
    and fads_prob=0.2 is unchanged from the original MPA.
    """

    def __init__(self, pop_size=30, max_iter=100, alpha=0.99, beta=0.01,
                 fads_prob=0.2, stag_threshold=10, mutation_f=0.5,
                 lambda_max=0.5, gamma_min=2.0, gamma_max=8.0,
                 tent_z0=0.7,
                 use_chaotic_init=True, use_adaptive_step=True,
                 use_elite_mutation=True, clip_bound=6.0, seed=None, **kwargs):
        super().__init__(pop_size, max_iter, alpha, beta, seed=seed)
        self.clip_bound = clip_bound
        self.fads_prob = fads_prob
        self.stag_threshold = stag_threshold
        self.mutation_f = mutation_f
        self.lambda_max = lambda_max
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.tent_z0 = tent_z0
        self.use_chaotic_init = use_chaotic_init
        self.use_adaptive_step = use_adaptive_step
        self.use_elite_mutation = use_elite_mutation

        self.stag_counter = None
        self.prev_fitness = None

    # ---- Module 1: initialization ----

    def _tent_map(self, n, d):
        """n x d Tent-map sequence, mapped to [-2, 2]."""
        Z = np.zeros((n, d))
        z = self.tent_z0
        for i in range(n):
            for j in range(d):
                z = 2 * z if z < 0.5 else 2 * (1 - z)
                if z == 0.0 or z == 0.5 or z == 1.0:  # kick off fixed points
                    z = np.random.random() * 0.5 + 0.25
                Z[i, j] = z
        return 4 * Z - 2

    def _elite_obl(self, population, fitness, X, y):
        """Elite opposition-based learning: build opposites w.r.t. the dynamic
        per-dimension bounds, then keep the best N of the merged 2N pool."""
        N, D = population.shape

        lb = population.min(axis=0)
        ub = population.max(axis=0)

        opp_population = np.zeros_like(population)
        for i in range(N):
            opp_population[i] = lb + ub - population[i]
            # small jitter so an opposite never lands exactly on an existing point
            opp_population[i] += 0.1 * np.random.randn(D)
        opp_population = np.clip(opp_population, -self.clip_bound, self.clip_bound)

        merged = np.vstack([population, opp_population])
        # binarize with gamma_min, same slope the main loop uses at t=0
        merged_bin = (np.random.random(merged.shape) <
                      adaptive_sigmoid(merged, self.gamma_min)).astype(int)
        merged_fit = np.zeros(2 * N)
        for i in range(2 * N):
            f, _, _ = self.evaluator.evaluate(merged_bin[i])
            merged_fit[i] = f

        top_indices = np.argsort(merged_fit)[:N]
        return merged[top_indices], merged_fit[top_indices]

    def _initialize_population(self, X, y):
        D = self.n_features
        N = self.pop_size

        if self.use_chaotic_init:
            self.continuous_pos = self._tent_map(N, D)
            temp_bin = (np.random.random((N, D)) <
                        adaptive_sigmoid(self.continuous_pos, self.gamma_min)).astype(int)
            temp_fit = np.zeros(N)
            for i in range(N):
                f, _, _ = self.evaluator.evaluate(temp_bin[i])
                temp_fit[i] = f
            self.continuous_pos, _ = self._elite_obl(
                self.continuous_pos, temp_fit, X, y
            )
        else:
            self.continuous_pos = 4 * np.random.random((N, D)) - 2

        self.population = (np.random.random((N, D)) <
                           adaptive_sigmoid(self.continuous_pos, self.gamma_min)).astype(int)

        self.stag_counter = np.zeros(N, dtype=int)
        self.prev_fitness = np.full(N, np.inf)

    # ---- Module 2: step-size control ----

    def _adaptive_cf(self, t):
        """Phase-aligned step-size control factor.

        Plateau at 1.0 through phase 1, cosine-anneal down to 0.25 through
        phase 2, then decay linearly to ~0 in phase 3. The 0.25 landing value
        matches the original CF = (1-r)^(2r), which is about 0.228 at
        r = 2/3, so the handover into phase 3 stays smooth.
        """
        T = self.max_iter
        ratio = t / T

        if not self.use_adaptive_step:
            return max((1 - ratio) ** (2 * ratio), 1e-6)  # original MPA CF

        if ratio <= 1 / 3:
            cf = 1.0
        elif ratio <= 2 / 3:
            local_r = (ratio - 1 / 3) * 3
            cf = 0.25 + 0.75 * 0.5 * (1 + np.cos(np.pi * local_r))
        else:
            local_r = (ratio - 2 / 3) * 3
            cf = 0.25 * (1 - local_r)

        return max(cf, 1e-4)

    # ---- Module 3: elite-guided mutation ----

    def _elite_guided_mutation(self, t):
        """Mutate stagnated individuals:
        mutant = x_best + F * (x_r1 - x_r2) + lambda(t) * N(0,1)."""
        if not self.use_elite_mutation:
            return

        T = self.max_iter
        N, D = self.continuous_pos.shape

        # perturbation shrinks with a Gaussian schedule over the run
        lam = self.lambda_max * np.exp(-5 * (t / T) ** 2)

        for i in range(N):
            if self.stag_counter[i] > self.stag_threshold:
                candidates = [j for j in range(N) if j != i]
                r1, r2 = np.random.choice(candidates, 2, replace=False)

                mutant = (self.continuous_pos[np.argmin(self.fitness)]
                          + self.mutation_f * (self.continuous_pos[r1] - self.continuous_pos[r2])
                          + lam * np.random.randn(D))
                mutant = np.clip(mutant, -self.clip_bound, self.clip_bound)

                mutant_bin = binarize(mutant, t, T, self.gamma_min, self.gamma_max)
                f_mut, _, _ = self.evaluator.evaluate(mutant_bin)

                # greedy replacement only
                if f_mut < self.fitness[i]:
                    self.continuous_pos[i] = mutant
                    self.population[i] = mutant_bin
                    self.fitness[i] = f_mut
                    self.stag_counter[i] = 0

    # ---- MPA core ----

    def _update_population(self, t, X, y):
        """Three-phase MPA update with adaptive CF, then Module 3."""
        T = self.max_iter
        N, D = self.continuous_pos.shape

        CF = self._adaptive_cf(t)
        Elite = self.continuous_pos[np.argmin(self.fitness)].copy()

        for i in range(N):
            if self.fitness[i] >= self.prev_fitness[i]:
                self.stag_counter[i] += 1
            else:
                self.stag_counter[i] = 0
        self.prev_fitness = self.fitness.copy()

        # Position updates follow Faramarzi et al. (2020): phase 1 uses
        # Eq. 5, phase 2 splits the population between Eq. 7 and Eq. 8,
        # phase 3 applies Eq. 9 to everyone.
        # P = 0.5 * Rand, R_B ~ N(0,1), R_L ~ Levy(1.5).

        if t < T / 3:  # exploration, Brownian-dominated
            for i in range(N):
                R_B = np.random.randn(D)
                stepsize = R_B * (Elite - R_B * self.continuous_pos[i])  # Eq. 5
                self.continuous_pos[i] += 0.5 * np.random.random(D) * stepsize * CF

        elif t < 2 * T / 3:  # transition, half Brownian / half Levy
            half = N // 2
            for i in range(half):
                R_B = np.random.randn(D)
                stepsize = R_B * (Elite - R_B * self.continuous_pos[i])  # Eq. 7
                self.continuous_pos[i] += 0.5 * np.random.random(D) * stepsize * CF

            for i in range(half, N):
                R_L = self._levy_flight(D)
                stepsize = R_L * (R_L * Elite - self.continuous_pos[i])  # Eq. 8
                self.continuous_pos[i] = Elite + 0.5 * CF * stepsize

        else:  # exploitation, Levy-dominated
            for i in range(N):
                R_L = self._levy_flight(D)
                stepsize = R_L * (R_L * Elite - self.continuous_pos[i])  # Eq. 9
                self.continuous_pos[i] = Elite + 0.5 * CF * stepsize

        # FADs effect (Faramarzi et al. 2020, Eqs. 12-13): with prob 0.2 a
        # random dimensional reset, from the other tail a DE-like crossover.
        # The factor 4 scales the reset to the [-2,2] init range.
        for i in range(N):
            r = np.random.random()
            if r < self.fads_prob:
                U = (np.random.random(D) < self.fads_prob).astype(float)
                self.continuous_pos[i] += CF * (2 * np.random.random(D) - 1) * 4 * U
            elif r > 1 - self.fads_prob:
                r1, r2 = np.random.choice(N, 2, replace=False)
                U = (np.random.random(D) < self.fads_prob).astype(float)
                self.continuous_pos[i] += (
                    self.fads_prob * (1 - r) + r
                ) * (self.continuous_pos[r1] - self.continuous_pos[r2]) * U

        # keep the sigmoid out of overflow territory
        self.continuous_pos = np.clip(self.continuous_pos, -self.clip_bound, self.clip_bound)

        self.population = binarize(
            self.continuous_pos, t, T, self.gamma_min, self.gamma_max
        )

        self._evaluate_all(X, y)

        self._elite_guided_mutation(t)

    def _levy_flight(self, d):
        """Levy(1.5) step vector of length d, Mantegna's algorithm."""
        beta_levy = 1.5
        sigma_u = (
            math.gamma(1 + beta_levy)
            * np.sin(np.pi * beta_levy / 2)
            / (math.gamma((1 + beta_levy) / 2) * beta_levy * 2 ** ((beta_levy - 1) / 2))
        ) ** (1 / beta_levy)

        u = np.random.randn(d) * sigma_u
        v = np.random.randn(d)
        step = u / (np.abs(v) ** (1 / beta_levy))

        return step


def create_ampa_variant(variant_name, **kwargs):
    """Build one of the eight ablation variants (BMPA .. AMPA_FS) by
    switching modules on or off: C = chaotic init, A = adaptive CF,
    E = elite mutation."""
    module_config = {
        "BMPA":    (False, False, False),
        "AMPA_C":  (True,  False, False),
        "AMPA_A":  (False, True,  False),
        "AMPA_E":  (False, False, True),
        "AMPA_CA": (True,  True,  False),
        "AMPA_CE": (True,  False, True),
        "AMPA_AE": (False, True,  True),
        "AMPA_FS": (True,  True,  True),
    }

    if variant_name not in module_config:
        raise ValueError(f"Unknown variant: {variant_name}")

    chaotic, adaptive, elite = module_config[variant_name]

    return AMPA_FS(
        use_chaotic_init=chaotic,
        use_adaptive_step=adaptive,
        use_elite_mutation=elite,
        **kwargs
    )
