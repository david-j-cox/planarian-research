#!/bin/bash
# Daily batch catch-up on this Mac (Apple MPS, full quality). Run this when the
# Mac comes online. It:
#   1. pulls clips that have accumulated in Drive (and deletes them there -> tidy)
#   2. processes location + behavior at stride 2 (~5s/clip on MPS; ~2h clears a
#      full day) -- sharper than the cloud CPU box could manage
#   3. renders + publishes the live plot to the GitHub Pages site
# Clips arrive 1/min into the endicott Drive from the recording Mac, independent
# of this machine; with 68 TB free they can buffer indefinitely between runs.
set -u
export RT_LOCAL_FS=1     # clips are local (rclone/scp), not Drive File-Provider placeholders
REPO="/Users/davidjcox/Documents/ResearchRepos/behavioral-pharmacology/planarian-research"
SITE="/Users/davidjcox/Documents/ResearchRepos/behavioral-pharmacology/planarian-live"
PY="$REPO/venv/bin/python"
WATCH="$REPO/realtime_runs/catchup_in"
LOG="$REPO/realtime_runs/catch_up.log"
mkdir -p "$WATCH"
cd "$REPO/scripts_notebooks" || exit 1

echo "[catch_up] $(date) pulling settled clips from Drive (--min-age 2m)..."
rclone move "drive:Planarian Research/planarian_clips" "$WATCH" \
    --include "*.mkv" --min-age 2m --transfers 4 --checkers 8 2>>"$LOG"

# one-time: fold in the clips brought back from the retired cloud box
if ls "$REPO/realtime_runs/box_backlog"/*.mkv >/dev/null 2>&1; then
  mv "$REPO/realtime_runs/box_backlog"/*.mkv "$WATCH"/ 2>/dev/null
fi

N=$(ls "$WATCH"/*.mkv 2>/dev/null | wc -l | tr -d ' ')
echo "[catch_up] processing $N clips on MPS (stride 2, full quality)..."
"$PY" -u rt_watch.py --watch_dir "$WATCH" --session_id worm_run_01 \
    --out_dir "$REPO/realtime_runs" --device mps --imgsz 1024 --stride 2 \
    --stable_s 1 --min_frames 560 --delete_after \
    --behavior_model "$REPO/realtime_runs/behavior_clf.joblib" \
    --label_queue "$REPO/realtime_runs/label_queue" --keep_per_day 150 --once

echo "[catch_up] rendering + publishing plot..."
"$PY" rt_plot.py --session worm_run_01 --out "$REPO/realtime_runs/worm_run_01_live.png"
if [ -f "$REPO/realtime_runs/worm_run_01_live.png" ] && [ -d "$SITE" ]; then
  cp "$REPO/realtime_runs/worm_run_01_live.png" "$SITE/live.png"
  ( cd "$SITE" && git add -A && git commit -q --amend -m "live plot" \
      && git push --force origin main >/dev/null 2>&1 && echo "[catch_up] site published" )
fi
echo "[catch_up] done $(date)"
