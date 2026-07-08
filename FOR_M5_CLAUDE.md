# Context handoff for Claude on the M5 Max

> **HISTORICAL (2026-06-20 migration snapshot) — DO NOT treat as current.** Since
> this was written, the YOLO localizer AND a behavior model were trained, and a
> full live real-time system was built. For the current state, read
> **SESSION_HANDOFF.md** ("CURRENT STATE (2026-06-26)"). The "no trained model /
> no torch" notes below are obsolete.

Read this first. It carries context the originating machine had that this one won't.
Goal: finetune a vision model to track worm location + behavior. See also
`M5_PULL_INSTRUCTIONS.md` (how to pull data) and `MIGRATION_MANIFEST.md` (dataset inventory).

## Where things stand
- **No trained model exists yet.** Tracking today is classical OpenCV (`open_dish_tracker.py`)
  plus a rule-based behavior classifier (`behavior_rules.py` + `behavior_features.py`). The
  finetune pipeline is net-new work to build here.
- **No deep-learning deps installed.** `requirements.txt` has no torch. Install PyTorch with the
  MPS (Metal) backend + Ultralytics (YOLO). AVOID Detectron2 — poor Apple Silicon support.
- **The dataset still needs extraction.** Labels are annotations referencing video frames, not an
  extracted-frame dataset. Step 1: pull labeled frames from the 15 tier1 videos and convert to
  YOLO format.
  - `realtime_runs/S1_labels.json`, `S3_labels.json`: worm-position points (video, frame, x_px, y_px) -> boxes/keypoints
  - `realtime_runs/S3_labels_blind/human_labels.csv`: per-window behavior labels (gliding/turning, multi-label with `;`)
  - `scripts_notebooks/calibration*.json`: pixel<->mm scale + dish geometry (needed to interpret tracks)

## Non-obvious constraints (these bit us before)
- **Area bounds must be in mm^2, not fixed pixels.** A fixed `max_area=3000` broke the 7MP rig
  (caused 0.49mm error). Scale area thresholds by the rig's mm-per-pixel.
- **Midlines gap:** the behavior labeler expects `*_midlines.npz` that the white-rig tracker does
  NOT yet emit. None exist on disk. Relevant only if midlines become a training input.
- **Rig/channel notes:** S3 = white final rig (use gray channel); S1/S4 are amber (red channel);
  auto channel-selection picks blue badly. Calibration differs per rig.
- **Planarian speed sanity check:** gliding is ~1.5-2 mm/s (range 1-5). Speed spikes are usually
  capture-stutter, not real; existing code caps jumps at 7 mm/s.

## Data layout after you pull (per M5_PULL_INSTRUCTIONS.md)
- `tier1_training/` — 15 labeled videos + label/calibration files (~625 MB). Enough to start.
- `tier2_corpus/` — full unlabeled corpus (~39 GB): OpenDishWork (692), additional_videos (297),
  live_capture (24), planarian_social_interactions (9). For expanding the dataset later.

## Repo
- GitHub: `david-j-cox/planarian-research`, branch `dev` (not main — recent work is on dev).
- Data/videos are gitignored; they come from Google Drive (`PlanarianVideos/`), not git.

## How the user likes to work (please honor)
- No emojis.
- One question at a time; don't assume defaults; do the legwork rather than over-asking.
