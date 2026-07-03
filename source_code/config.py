"""Experiment configuration: hyperparameters, datasets, algorithm lists."""

import os

# paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "datasets", "data")

# search protocol
POP_SIZE = 30           # population size N
MAX_ITER = 100          # iterations T
NUM_RUNS = 30           # independent runs per (algorithm, dataset)
FADS_PROB = 0.2         # FADs effect probability (original MPA value)

# fitness weights, alpha*error + beta*ratio
ALPHA = 0.99
BETA = 0.01

# Module 1
TENT_Z0 = 0.7           # Tent map seed

# Module 2 has no free parameter here; the three-segment CF schedule
# lives in AMPA_FS._adaptive_cf().

# Module 3
STAG_THRESHOLD = 10     # stagnation patience; 5 was too trigger-happy
MUTATION_F = 0.5        # DE-style scale factor
LAMBDA_MAX = 0.5        # max Gaussian perturbation; 0.1 escaped too weakly

# adaptive sigmoid slope range
GAMMA_MIN = 2.0
GAMMA_MAX = 8.0         # 10 binarized too hard late in the run

# report-phase classifier
RF_N_ESTIMATORS = 100
RF_RANDOM_STATE = 42
CV_FOLDS = 5

# the ten UCI benchmarks
DATASETS = [
    "Iris", "Wine", "Zoo", "Heart", "BreastCancer",
    "Ionosphere", "Sonar", "Dermatology", "Vehicle", "Parkinsons",
]

# comparison algorithms
ALGORITHMS = [
    "AMPA_FS",   # Proposed
    "BMPA",      # Binary MPA baseline
    "BPSO",      # Binary PSO
    "BGA",       # Binary GA
    "BGWO",      # Binary GWO
    "BWOA",      # Binary WOA
    "BSSA",      # Binary SSA
    "BHHO",      # Binary HHO
    "BSCA",      # Binary SCA
    "BGOA",      # Binary GOA
]

# baselines added for the major revision (Reviewer 5.11)
REVISION_ALGORITHMS = [
    "BWSO",      # Binary White Shark Optimizer
    "BEHO",      # Binary Elk Herd Optimizer
    "BCMAES",    # Binary (separable) CMA-ES
]

# high-dimensional datasets added for the revision (Reviewers 3.4/4.11/5.4);
# ANOVA pre-screening keeps the wrapper KNN tractable on these
REVISION_DATASETS = ["Colon", "SRBCT", "Leukemia"]

# ablation variants: C = chaotic init, A = adaptive CF, E = elite mutation
ABLATION_VARIANTS = [
    "BMPA",
    "AMPA_C",
    "AMPA_A",
    "AMPA_E",
    "AMPA_CA",
    "AMPA_CE",
    "AMPA_AE",
    "AMPA_FS",
]

# parameter sensitivity
SENSITIVITY_DATASETS = ["BreastCancer", "Ionosphere", "Sonar"]
SENSITIVITY_RUNS = 10   # 30 runs per grid point would be prohibitively slow
SENSITIVITY_PARAMS = {
    "pop_size": [10, 20, 30, 40, 50],
    "stag_threshold": [3, 5, 10, 15, 20],
    "alpha": [0.90, 0.95, 0.99],
    "gamma_max": [4, 6, 8, 10, 12],
}
