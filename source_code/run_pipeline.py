"""
AMPA-FS Full Experiment Pipeline.

Runs in order:
  1. Main experiment (already running if started separately)
  2. Ablation study
  3. Parameter sensitivity analysis
  4. Generate all figures and LaTeX tables

Usage:
  python run_pipeline.py                        # run steps 2-4 (assume main done)
  python run_pipeline.py --step ablation        # only ablation
  python run_pipeline.py --step sensitivity     # only sensitivity
  python run_pipeline.py --step figures         # only figures/tables
  python run_pipeline.py --all                  # run all 4 steps
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def step_main(datasets=None, runs=None):
    from experiments.run_main import run_experiment
    datasets = datasets or config.DATASETS
    runs = runs or config.NUM_RUNS
    print(f"\n[STEP 1] Main Experiment  ({len(datasets)} datasets, {runs} runs)")
    run_experiment(datasets, config.ALGORITHMS, runs)


def step_ablation(datasets=None, runs=None):
    from experiments.run_ablation import run_ablation
    datasets = datasets or config.DATASETS
    runs = runs or config.NUM_RUNS
    print(f"\n[STEP 2] Ablation Study   ({len(datasets)} datasets, {runs} runs)")
    run_ablation(datasets, runs)


def step_sensitivity(datasets=None, runs=None):
    from experiments.run_sensitivity import run_sensitivity
    datasets = datasets or config.SENSITIVITY_DATASETS
    runs = runs or config.SENSITIVITY_RUNS
    print(f"\n[STEP 3] Sensitivity Analysis ({len(datasets)} datasets, {runs} runs)")
    run_sensitivity(datasets, runs)


def step_figures():
    from visualization.generate_paper_figures import main as gen_figures
    print(f"\n[STEP 4] Generate Figures & Tables")
    gen_figures(only=None)


def main():
    parser = argparse.ArgumentParser(description="AMPA-FS Pipeline Runner")
    parser.add_argument("--all", action="store_true",
                        help="Run all 4 steps (main + ablation + sensitivity + figures)")
    parser.add_argument("--step", choices=["main", "ablation", "sensitivity", "figures"],
                        help="Run only one step")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--runs", type=int, default=None)
    args = parser.parse_args()

    if args.all:
        step_main(args.datasets, args.runs)
        step_ablation(args.datasets, args.runs)
        step_sensitivity()
        step_figures()
    elif args.step == "main":
        step_main(args.datasets, args.runs)
    elif args.step == "ablation":
        step_ablation(args.datasets, args.runs)
    elif args.step == "sensitivity":
        step_sensitivity()
    elif args.step == "figures":
        step_figures()
    else:
        # Default: assume main is done, run steps 2-4
        step_ablation(args.datasets, args.runs)
        step_sensitivity()
        step_figures()


if __name__ == "__main__":
    main()
