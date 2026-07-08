#!/usr/bin/env bash
# run_session.sh — One-shot offline pipeline for a single recording session.
#
# Track -> measure accuracy -> make report figures, for one labeled session.
# Assumes you've already clicked labels with:
#     python label_setup.py --clips_dir ../live_capture --session S1 --n_worm 40
# which wrote ../realtime_runs/S1_labels.json.
#
# Usage (from scripts_notebooks/):
#     ./run_session.sh S1
#     CLIPS_DIR=../live_capture OUT_DIR=../realtime_runs ./run_session.sh S2
#
set -euo pipefail

SESSION="${1:-}"
if [[ -z "$SESSION" ]]; then
  echo "usage: $0 <session>   e.g. $0 S1" >&2
  exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

CLIPS_DIR="${CLIPS_DIR:-../live_capture}"
OUT_DIR="${OUT_DIR:-../realtime_runs}"
PY="${PYTHON:-../venv/bin/python}"
# Channel for worm detection. The white/neutral final rig (a dark planarian on a
# bright white background) separates well on luminance, so gray is the default.
# NOTE: amber/dark-chamber test footage (e.g. live_capture S1/S4) instead needs
# CHANNEL=red, because there the worm is invisible on the auto-picked blue
# channel. Override per run: CHANNEL=red ./run_session.sh S1
CHANNEL="${CHANNEL:-gray}"
LABELS="$OUT_DIR/${SESSION}_labels.json"
CSV="$OUT_DIR/${SESSION}_tracks.csv"

if [[ ! -f "$LABELS" ]]; then
  echo "No labels for $SESSION at $LABELS" >&2
  echo "Run: $PY label_setup.py --clips_dir $CLIPS_DIR --session $SESSION --n_worm 40" >&2
  exit 1
fi

# Fresh CSV each run so re-running a session doesn't append to a stale one.
rm -f "$CSV" "$OUT_DIR/${SESSION}_processed.txt"

echo "==== [1/3] TRACK $SESSION (human labels override auto-detect, channel=$CHANNEL) ===="
"$PY" watch_folder_tracker.py \
  --watch_dir "$CLIPS_DIR" \
  --session "$SESSION" \
  --labels "$LABELS" \
  --channel "$CHANNEL" \
  --process_existing --once

echo
echo "==== [2/4] ACCURACY vs human ground truth ===="
"$PY" measure_accuracy.py --session "$SESSION" --output_dir "$OUT_DIR"

echo
echo "==== [3/4] FLAG impossible-speed steps (>${MAX_SPEED_MM_S:-7} mm/s) ===="
# Positions are trusted; this only NaNs untrustworthy per-frame speeds so the
# movement stats aren't poisoned by capture-stutter velocity spikes.
"$PY" filter_jumps.py "$CSV" ${MAX_SPEED_MM_S:+--max_speed_mm_s "$MAX_SPEED_MM_S"}

echo
echo "==== [4/4] REPORT figures ===="
"$PY" pilot_report.py --csv "$CSV"

echo
echo "Done. Outputs in $OUT_DIR (CSV, accuracy above, *.png report figures)."
