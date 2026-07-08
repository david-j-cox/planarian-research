# Planarian Tracking — Compute Node Handoff

**For:** the Claude running on the *second (faster) Mac* that does the tracking + analysis.
**Created:** 2026-06-02
**Role of this machine:** pull 1-minute video clips that arrive via Google Drive, track the worm in each, accumulate one rolling CSV, and run analysis. The capture Mac only records + uploads; this machine does all compute.

> **UPDATE 2026-06-26 — superseded by `rt_watch.py` + a cloud plan.** Live compute
> is now `rt_watch.py` (supervised via `run_watch_supervised.sh`), not
> `watch_folder_tracker.py`, and it does location + behavior + ethogram + active
> clip retention. It currently runs on the capture Mac itself. **MIGRATION PLAN**
> (to retire the Mac): a 24/7 cloud Linux box reads Drive via **rclone** (the
> macOS Drive File Provider is Mac-only and hangs under launchd) and runs the
> pipeline as systemd services. Target: **Oracle Cloud Always-Free** (Ampere ARM
> 2 OCPU/12GB, $0) for LOCATION at imgsz 640/stride 3; the BEHAVIOR layer ~doubles
> compute and needs a **paid box** (~$16/mo Hetzner CAX31 ARM, EU). CPU-only, no
> GPU. If the endicott Workspace blocks rclone OAuth, share the clips folder to a
> personal Gmail. Canonical status: SESSION_HANDOFF.md.

---

## 1. The experiment

A single planarian (flatworm) in a petri dish on a 1 cm grid, filmed top-down through an AmScope MU1003 microscope camera (via the AmLite app → OBS screen capture). The dish sits inside a **light-blocking box with a constant LED** — so lighting is fixed 24/7, no day/night variation. The goal: track the worm's position and movement continuously for **2–3 weeks**.

Clips are **1-minute MKVs**, named like `2026-06-02_15-01-32.mkv` (OBS timestamp). They arrive in a Google Drive folder synced to this machine.

## 2. Pipeline overview

```
Capture Mac:  OBS → live_capture/ → clip_uploader.py → Google Drive (planarian_clips/)
                                                              │  (Drive sync)
                                                              ▼
This Mac:     Drive folder (planarian_clips/) → watch_folder_tracker.py → rolling CSV → analysis
```

Each minute, one new clip appears in the Drive folder on this machine. `watch_folder_tracker.py` notices it, tracks the worm, appends to a single rolling `*_tracks.csv`, and (optionally) deletes the clip.

## 3. Setup on this machine

```bash
# 1. Get the code (copy these from the capture Mac's repo, scripts_notebooks/):
#      open_dish_tracker.py      (detection building blocks — REQUIRED dependency)
#      watch_folder_tracker.py   (the watcher you run)
#      diagnose_detection.py     (visual sanity-check tool)
#      session_analysis.py, habituation_analysis.py  (analysis)
#      requirements.txt

# 2. Python env
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt        # numpy, pandas, scipy, opencv-python, av, scikit-image, etc.

# 3. Install Google Drive desktop app, sign into cox.david.j@gmail.com,
#    and confirm the synced folder exists, e.g.:
#      ~/Library/CloudStorage/GoogleDrive-cox.david.j@gmail.com/My Drive/planarian_clips
```

## 4. Run the tracker (the main command)

```bash
DRIVE="$HOME/Library/CloudStorage/GoogleDrive-cox.david.j@gmail.com/My Drive/planarian_clips"

python watch_folder_tracker.py \
    --watch_dir "$DRIVE" \
    --output_dir ./realtime_runs \
    --session_id worm_run_01 \
    --channel auto \
    --process_existing
```

It will:
- Calibrate **once** from the first clip (dish circle, 1 cm grid → mm/px, best detection channel), save `worm_run_01_calibration.json`, and reuse it for every clip. The rig is fixed, so this is correct.
- For each clip: rebuild the background from that clip's own median frame, detect the worm, append rows to `worm_run_01_tracks.csv` with a continuous `time_s` clock across clips.
- Poll the folder for new clips forever (Ctrl-C to stop; safe to restart — processed clips are tracked in `worm_run_01_processed.txt`).

## 5. Key design points (so you can reason about it)

- **Calibrate once, reuse:** the box + constant LED means dish, grid, and lighting never change. Don't recalibrate per clip.
- **Per-clip background baseline:** each clip's median frame is the background for that minute (the worm moves, so it averages out). The worm = what deviates from it. This is robust even if lighting drifts slightly.
- **`--channel auto`:** detection doesn't assume grayscale. It measures per-channel worm-vs-background contrast on the calibration frames and picks the best (gray / blue / green / red / LAB L,a,b / `colordist`). `colordist` = per-pixel distance from the background color, which uses all color information and is the most lighting-robust. We confirmed on real clips that this gives a clean, unambiguous worm signal (the worm is the single brightest blob in the baseline-subtracted diff map).
- **Worm is DARKER than background** in this rig — the detection polarity matches the offline tracker.
- **`detect_worm` lives in `open_dish_tracker.py`** — the watcher imports it. Keep both files together.

## 6. Verify detection before a long run

Use the diagnostic on one empty-ish clip + one worm clip:
```bash
python diagnose_detection.py \
    --baseline_clip "$DRIVE/<some_clip>.mkv" \
    --worm_clip "$DRIVE/<another_clip>.mkv" \
    --out_dir ./diag
# Look at diag/detection.jpg (dish circle correct? worm marked?) and
# diag/diffmap.jpg (worm should be the brightest blob, background near-black).
```
If detection is poor, tune `--detect_thresh` (lower = catch fainter worms) and
`--min_area`/`--max_area` (the worm blob size in px), then re-run the diagnostic.

## 7. CSV schema (feeds the analysis scripts)

`worm_run_01_tracks.csv` — 2 metadata rows (`# mm_per_px=...`, `# session_id=...`) then a header. Read with `pandas.read_csv(path, skiprows=2)`. Columns:

`video_file, frame, time_s, centroid_x_px, centroid_y_px, centroid_x_mm, centroid_y_mm, area_px, speed_px_s, speed_mm_s, confidence, is_lost`

This is close to the offline tracker's schema, so `session_analysis.py` and `habituation_analysis.py` work on it (they read `skiprows=2`, compute distance, speed, time-to-stop, etc.). Filter implausible speeds (>15 mm/s) as tracker error — a planarian glides ~0.5–3 mm/s.

## 8. Disk

Decide with the user whether to delete clips after processing. For a 2–3 week run at 4K (~46 MB/min ≈ 66 GB/day) you will fill the disk fast. Options: lower OBS resolution on the capture side (720p is plenty for tracking), and/or delete each clip after its rows are written. The rolling CSV is tiny (~MB for the whole run).

## 9. Analysis ideas (your job, with the user)

- Distance traveled over time (does the worm habituate / slow down?).
- Position heatmap / dwell maps (does it prefer dish edge vs. center? thigmotaxis).
- Activity rhythms across days (even under constant light, is there periodicity?).
- Speed distribution, bout structure (move/pause cycles).
- The user has prior analysis in `session_analysis.py` / `habituation_analysis.py` to build on.
```
