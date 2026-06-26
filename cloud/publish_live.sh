#!/bin/bash
# Cloud version of the site publisher: push the latest live plot to the
# planarian-live GitHub Pages repo. Single rolling commit (amend + force-push)
# keeps the repo tiny. Needs ~/planarian-live cloned with push auth (deploy key).
set -u
SITE="$HOME/planarian-live"
SRC="$HOME/planarian-research/realtime_runs/worm_run_01_live.png"
cd "$SITE" || exit 1
while true; do
  if [ -f "$SRC" ]; then
    cp "$SRC" "$SITE/live.png"
    git add -A
    if ! git diff --cached --quiet; then
      git commit --amend -m "live plot" >/dev/null 2>&1
      git push --force origin main >/dev/null 2>&1 || echo "[publish] push failed $(date)"
    fi
  fi
  sleep 360
done
