#!/usr/bin/env bash
# Week-2 orchestrator. Runs AFTER the revision orchestrator (run_revision.sh)
# completes: runs the clip-range comparison experiment (T2.7 / R3-5), then
# regenerates the revision figures (radar becomes available once
# extra_metrics.csv exists), and commits the results to GitHub.
set -u
cd "$(dirname "$0")"
LOG=results/week2_orchestrator.log
mkdir -p results
echo "[week2] start $(date)" | tee -a "$LOG"

while ! grep -q "\[revision\] DONE" results/revision_orchestrator.log 2>/dev/null; do
  sleep 120
done
echo "[week2] revision done, starting clip-range experiment $(date)" | tee -a "$LOG"

python3 experiments/run_init_range.py --runs 30 >> results/run_init_range.log 2>&1
echo "[week2] clip-range done, regenerating revision figures $(date)" | tee -a "$LOG"

python3 visualization/generate_revision_figures.py >> results/revision_figures.log 2>&1

echo "[week2] DONE $(date)" | tee -a "$LOG"

# final commit (autosave exits after [revision] DONE, so commit here)
REPO=$(git rev-parse --show-toplevel)
SRC=papers/ijmlc-ampa-fs/supplementary/source_code
cd "$REPO"
for attempt in 1 2 3; do
  git add -f "$SRC"/results/*.pkl "$SRC"/results/*.csv "$SRC"/results/*.log \
             "$SRC"/figures_generated/* 2>/dev/null
  if git diff --cached --quiet; then break; fi
  git commit -q -m "Week-2: clip-range comparison (T2.7) + regenerated revision figures" \
    && git push -q && echo "[week2] pushed $(date)" >> "$SRC/results/week2_orchestrator.log" && break
  sleep 60
done
