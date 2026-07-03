#!/usr/bin/env bash
# Periodically commit in-progress experiment results/figures to GitHub so work
# is never lost if the session stops. Runs until the revision orchestrator
# writes its DONE marker (then does a final commit).
set -u
REPO=/home/ubuntu/repos/A03
SRC=papers/ijmlc-ampa-fs/supplementary/source_code
cd "$REPO"

commit_push() {
  git add -f "$SRC"/results/*.pkl "$SRC"/results/*.csv "$SRC"/results/*.log \
             "$SRC"/figures_generated/* 2>/dev/null
  if ! git diff --cached --quiet; then
    git commit -q -m "Autosave: experiment results/figures checkpoint $(date -u +%H:%M)" \
      && git push -q 2>/dev/null \
      && echo "[autosave] pushed $(date -u)"
  fi
}

while true; do
  commit_push
  if grep -q "\[revision\] DONE" "$SRC"/results/revision_orchestrator.log 2>/dev/null; then
    sleep 30; commit_push; echo "[autosave] final commit done"; break
  fi
  sleep 900   # every 15 minutes
done
