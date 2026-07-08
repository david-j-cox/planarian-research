#!/bin/zsh
# Supervisor for the real-time worm watcher.
#
# WHY THIS EXISTS: the watcher reads clips out of a Google Drive File Provider
# folder. Opening those files requires the running process to have macOS Full
# Disk Access. A Terminal / login shell inherits it; a bare `launchd` agent does
# NOT, and its open() on a Drive file hangs forever. So for unattended runs we
# supervise from a Drive-capable session instead of launchd.
#
# Run it (keeps the Mac awake AND restarts the watcher if it ever exits):
#   caffeinate -dimsu zsh run_watch_supervised.sh
# Leave that Terminal open for the duration of the capture.

cd "$(dirname "$0")" || exit 1

WD="/Users/davidjcox/Library/CloudStorage/GoogleDrive-dcox@endicott.edu/My Drive/Planarian Research/planarian_clips"
LOG="../realtime_runs/rt_watch.supervisor.log"

echo "[supervisor] start $(date)" >> "$LOG"
while true; do
  ../venv/bin/python -u rt_watch.py \
      --watch_dir "$WD" \
      --session_id worm_run_01 \
      --stable_s 6 \
      --min_frames 560 \
      --delete_after \
      --keep_total 60 \
      --behavior_model ../realtime_runs/behavior_clf.joblib \
      --label_queue ../realtime_runs/label_queue \
      --keep_per_day 150 >> "$LOG" 2>&1
  code=$?
  echo "[supervisor] watcher exited code=$code; restarting in 5s $(date)" >> "$LOG"
  sleep 5
done
