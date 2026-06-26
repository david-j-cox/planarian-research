#!/bin/bash
# Pull new 1-minute clips from Google Drive to a local watch dir for rt_watch.
# Replaces the macOS File Provider: on Linux we use rclone (API), which gives
# clean complete files -- so rt_watch's materialized()/prefetch() gates are no-ops
# and clip_ready() just confirms the download finished.
#
# `rclone move` downloads each clip AND removes it from Drive (keeps Drive flat,
# same as the Mac's --delete_after did). rclone writes to a .partial temp and
# renames on completion, so the watch dir only ever sees whole files.
set -u
REMOTE="${RCLONE_REMOTE:-drive}"
DRIVE_DIR="${DRIVE_DIR:-Planarian Research/planarian_clips}"
WATCH="${WATCH_DIR:-$HOME/clips_in}"
mkdir -p "$WATCH"

while true; do
  rclone move "${REMOTE}:${DRIVE_DIR}" "$WATCH" \
      --include "*.mkv" --transfers 4 --checkers 8 \
      --no-traverse --drive-skip-gdocs 2>>"$HOME/drive_pull.log"
  sleep 15
done
