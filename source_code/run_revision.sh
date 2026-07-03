#!/usr/bin/env bash
# Revision experiments orchestrator. Runs AFTER the Week-0 reproduction
# (run_week0.sh) completes: adds the new comparison algorithms on the base
# datasets, runs all algorithms on the high-dimensional revision datasets
# (with filter pre-screening), computes the extended metrics, and regenerates
# figures. Safe to run unattended; everything checkpoints to results/.
set -u
cd "$(dirname "$0")"
LOG=results/revision_orchestrator.log
mkdir -p results
echo "[revision] start $(date)" | tee -a "$LOG"

# 1) wait for the Week-0 orchestrator to finish
while ! grep -q "\[orchestrator\] DONE" results/week0_orchestrator.log 2>/dev/null; do
  sleep 120
done
echo "[revision] week0 done, starting revision runs $(date)" | tee -a "$LOG"

ALL_ALGS="AMPA_FS BMPA BPSO BGA BGWO BWOA BSSA BHHO BSCA BGOA BWSO BEHO BCMAES"
ALL_DATA="Iris Wine Zoo Heart BreastCancer Ionosphere Sonar Dermatology Vehicle Parkinsons Colon SRBCT Leukemia"

# 2) all 13 algorithms x all 13 datasets, resuming from the Week-0 checkpoint.
#    Already-computed (base dataset, base algorithm) cells skip instantly; this
#    fills in the 3 new algorithms on the base datasets and every algorithm on
#    the high-dim revision datasets (which get filter pre-screening).
echo "[revision] full 13x13 run (resume) $(date)" | tee -a "$LOG"
python3 experiments/run_main.py --datasets $ALL_DATA \
    --algorithms $ALL_ALGS --runs 30 >> results/run_main_revision.log 2>&1

# 4) extended metrics from stored best subsets
echo "[revision] extended metrics $(date)" | tee -a "$LOG"
python3 experiments/compute_extra_metrics.py >> results/extra_metrics.log 2>&1

# 5) regenerate figures with the fuller result set
echo "[revision] figures $(date)" | tee -a "$LOG"
python3 visualization/generate_paper_figures.py >> results/figures.log 2>&1

echo "[revision] DONE $(date)" | tee -a "$LOG"
