# AMPA-FS: Supplementary Material (Source Code and Raw Results)

## Overview

This package contains the source code, experiment scripts, raw results, and instructions for reproducing the results reported in the manuscript:

> *AMPA-FS: An Adaptive Multi-strategy Marine Predators Algorithm for Wrapper-based Feature Selection*

## Directory Structure

```
source_code/
├── algorithms/          # All 13 algorithm implementations
│   ├── ampa_fs.py       # AMPA-FS (proposed algorithm)
│   ├── base_optimizer.py# Abstract base class for all optimizers
│   ├── bcmaes.py        # Binary CMA-ES
│   ├── beho.py          # Binary Elk Herd Optimizer
│   ├── bga.py           # Binary Genetic Algorithm
│   ├── bgoa.py          # Binary Grasshopper Optimization Algorithm
│   ├── bgwo.py          # Binary Grey Wolf Optimizer
│   ├── bhho.py          # Binary Harris Hawks Optimization
│   ├── bpso.py          # Binary Particle Swarm Optimization
│   ├── bsca.py          # Binary Sine Cosine Algorithm
│   ├── bssa.py          # Binary Salp Swarm Algorithm
│   ├── bwoa.py          # Binary Whale Optimization Algorithm
│   └── bwso.py          # Binary White Shark Optimizer
├── datasets/            # Dataset loaders (automatic download; ANOVA pre-screen for gene data)
├── experiments/         # Experiment runner scripts (main, ablation, sensitivity, bounds, metrics)
├── results/             # Raw results (.pkl) and summary tables (.csv)
├── utils/               # Utility functions (fitness, binarization, evaluation)
├── visualization/       # Plotting scripts
├── config.py            # Hyperparameter configuration
├── run_pipeline.py      # Main pipeline entry point
└── requirements.txt     # Python dependencies
```

## Requirements

- Python >= 3.10
- NumPy >= 1.26
- scikit-learn >= 1.4
- SciPy >= 1.12

Install dependencies:
```bash
pip install -r source_code/requirements.txt
```

## Reproducing Results

1. **Main comparison** (13 algorithms x 13 datasets x 30 runs; Tables 6-11):
   ```bash
   python source_code/run_pipeline.py
   ```

2. **Ablation study** (8 module variants x 10 UCI datasets x 30 runs; Tables 12-14):
   ```bash
   python source_code/experiments/run_ablation.py
   ```
   Note: the accuracy recorded by the ablation runner is the search-stage KNN
   hold-out accuracy (the accuracy component of the wrapper fitness), matching
   the manuscript's Table 14 caption.

3. **Parameter sensitivity** and **continuous-space bounds robustness**:
   ```bash
   python source_code/experiments/run_sensitivity.py
   python source_code/experiments/run_init_range.py
   ```

4. **Extended metrics** (precision, sensitivity, specificity, F1, ROC-AUC):
   ```bash
   python source_code/experiments/compute_extra_metrics.py
   ```

5. The ten UCI datasets are automatically downloaded from the UCI Machine
   Learning Repository; the three gene-expression datasets (Colon, SRBCT,
   Leukemia) are fetched by the loaders in `datasets/` and pre-screened with
   ANOVA F-score (top 200 features) before the wrapper search.

6. Pre-computed raw results are included in `source_code/results/`:
   `main_results.pkl` is a nested dictionary
   `data[dataset_name][algorithm_name][metric_name]` -> list of 30 values;
   `*_summary.csv` files back every table in the manuscript.

## License

This code is provided for academic reproducibility purposes accompanying the manuscript submission.
