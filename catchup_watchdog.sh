#!/bin/bash
# DEPLOYMENT NOTE: like catch_up.sh, macOS TCC blocks launchd from executing
# scripts inside ~/Documents, so the LaunchAgent runs a copy OUTSIDE Documents:
#       ~/bin/catchup_watchdog.sh
# THIS file (in the repo) is the version-controlled source of truth. After editing
# redeploy:   cp catchup_watchdog.sh ~/bin/catchup_watchdog.sh
# The LaunchAgent (com.planarian.watchdog.plist) points at ~/bin/catchup_watchdog.sh
# and fires it every 120s (StartInterval).
#
# SELF-HEALING WATCHDOG for the catch_up pipeline. On 2026-06-29 a single 0-byte
# clip wedged rt_watch's loop: the clip never passed materialized(), never got
# quarantined, so --once never terminated and one catch_up.sh spun for 6+ hours
# while the site froze. rt_watch.py now quarantines such clips, but this is the
# defense-in-depth layer: if a cycle is RUNNING yet making no progress, kill it so
# the launchd StartInterval restarts it clean within 90s. No alerting -- it simply
# recovers. No-op when nothing is running (idle) or when work is advancing.
#
# PROGRESS SIGNAL: every healthy phase touches the filesystem within seconds --
# the rclone pull adds clips to catchup_in, processing removes/moves them out and
# appends to the processed-list. A wedged spin-loop touches neither. So if the
# NEWEST of (catchup_in dir mtime, processed-list mtime) is older than STALL_SECS
# while catch_up.sh is alive, the cycle is stuck. STALL_SECS is far longer than any
# single clip (~5s) or normal pull, so a slow-but-healthy backlog drain is safe.
set -u
REPO="/Users/davidjcox/Documents/ResearchRepos/behavioral-pharmacology/planarian-research"
WATCH="$REPO/realtime_runs/catchup_in"
PROC="$REPO/realtime_runs/worm_run_01_processed.txt"
LOG="$HOME/Library/Logs/com.planarian.watchdog.log"
STALL_SECS=600          # 10 min of zero FS activity while running == wedged

# Nothing running -> healthy idle (rt_watch --once exits when the queue drains).
pgrep -f "catch_up.sh" >/dev/null 2>&1 || exit 0

now=$(date +%s)
newest=0
for p in "$WATCH" "$PROC"; do
    if [ -e "$p" ]; then
        m=$(stat -f %m "$p" 2>/dev/null || echo 0)
        [ "$m" -gt "$newest" ] && newest=$m
    fi
done
age=$(( now - newest ))

if [ "$newest" -gt 0 ] && [ "$age" -ge "$STALL_SECS" ]; then
    echo "[watchdog] $(date) cycle running but no activity for ${age}s (>= ${STALL_SECS}s) -> killing for clean restart"
    pkill -f "rt_watch.py"
    pkill -f "rclone move drive:Planarian"
    pkill -f "catch_up.sh"
    # Scrub anything that could re-wedge or race the restart: 0-byte clips (the
    # original poison pill) and rclone .partial leftovers from the killed transfer.
    find "$WATCH" -maxdepth 1 -name "*.mkv" -size 0 -delete 2>/dev/null
    rm -f "$WATCH"/*.partial 2>/dev/null
    echo "[watchdog] $(date) killed + scrubbed; launchd will restart catch_up within 90s"
fi
