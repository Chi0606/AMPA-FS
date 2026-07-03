#!/usr/bin/env bash
# Week-0 reproduction orchestrator: waits for any running main experiment to
# finish, then runs ablation, sensitivity, and figure generation in sequence.
# Safe to run unattended; every stage checkpoints to results/.
set -u
cd "$(dirname "$0")"
LOG=results/week0_orchestrator.log
mkdir -p results
echo "[orchestrator] start $(date)" | tee -a "$LOG"

# 1) wait for an already-running main experiment (if any)
while pgrep -f "run_main.py" >/dev/null; do
  echo "[orchestrator] main still running... $(date)" >> "$LOG"
  sleep 60
done

# 2) main results present? if not, run it
if [ ! -f results/main_results.pkl ]; then
  echo "[orchestrator] running main $(date)" | tee -a "$LOG"
  python3 experiments/run_main.py --runs 30 >> results/run_main.log 2>&1
fi

# 3) ablation
echo "[orchestrator] running ablation $(date)" | tee -a "$LOG"
python3 experiments/run_ablation.py >> results/run_ablation.log 2>&1

# 4) sensitivity
echo "[orchestrator] running sensitivity $(date)" | tee -a "$LOG"
python3 experiments/run_sensitivity.py >> results/run_sensitivity.log 2>&1

# 5) figures
echo "[orchestrator] generating figures $(date)" | tee -a "$LOG"
python3 visualization/generate_paper_figures.py >> results/figures.log 2>&1

echo "[orchestrator] DONE $(date)" | tee -a "$LOG"
