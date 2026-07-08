# Migration Manifest — model training move to M5 Max

Generated 2026-06-20. Target: Apple M5 Max, 128 GB, 2 TB. Transfer via cloud (S3/Drive).
Code travels via GitHub (`dev` branch, pushed). Everything below is **gitignored** and must
move out-of-band.

## How to bring the repo over (on the M5 Max)
```
git clone https://github.com/david-j-cox/planarian-research.git
cd planarian-research && git checkout dev
```
Then drop the data files below back into the same relative paths.

---

## TIER 1 — Minimal training set (move this first: ~625 MB + a few KB)
Everything needed to start finetuning a worm-localization / behavior model.

### Label & calibration files (training labels — tiny)
| Path | Size | What it is |
|------|------|-----------|
| realtime_runs/S1_labels.json | 8 KB | Ground-truth worm positions (video, frame, x_px, y_px) + calibration |
| realtime_runs/S3_labels.json | 8 KB | Same, session 3 |
| realtime_runs/S3_labels_blind/human_labels.csv | 4 KB | Per-window behavior labels (gliding/turning, multi-label) |
| realtime_runs/S3_signals.npz | 812 KB | Precomputed behavior feature signals for S3 |
| scripts_notebooks/calibration.json | 108 KB | Rig calibration (pixel<->mm, dish geometry) |
| scripts_notebooks/calibration_2worms.json | 124 KB | Two-worm calibration |

### Videos referenced by the label files (all in live_capture/, 624 MB total)
These are the ONLY videos the current labels point at. Detection/pose labels in
S1/S3_labels.json reference these exact files by name.
```
live_capture/2026-06-02_14-31-03.mkv   45M
live_capture/2026-06-02_14-32-04.mkv   44M
live_capture/2026-06-02_14-33-05.mkv   45M
live_capture/2026-06-02_14-34-06.mkv   44M
live_capture/2026-06-02_14-35-06.mkv   45M
live_capture/2026-06-02_14-36-07.mkv   44M
live_capture/2026-06-02_14-37-08.mkv   6.6M
live_capture/2026-06-02_14-58-12.mkv   46M
live_capture/2026-06-02_14-59-24.mkv   45M
live_capture/2026-06-02_15-00-25.mkv   45M
live_capture/2026-06-02_15-01-32.mkv   48M
live_capture/2026-06-02_15-02-47.mkv   44M
live_capture/2026-06-02_15-04-08.mkv   47M
live_capture/2026-06-02_15-05-45.mkv   45M
live_capture/2026-06-02_15-07-26.mkv   29M
```

---

## TIER 2 — Full video corpus (only if expanding the dataset later: ~40 GB)
Not needed to start. Move only if you want to label more footage on the M5.
| Directory | Size | Videos |
|-----------|------|--------|
| OpenDishWork/ | 26 GB | 623 |
| additional_videos/ | 12 GB | 297 |
| live_capture/ | 1.7 GB | 39 (includes the 15 Tier-1 files) |
| planarian_social_interactions/ | 251 MB | 8 |
Total corpus: 968 video files, ~40 GB.

---

## Readiness notes (not migration blockers, but know before you train)
1. **No training code or model artifacts exist in the repo yet.** Tracking today is
   classical OpenCV (open_dish_tracker.py) + rule-based behavior (behavior_rules.py).
   The finetune pipeline is net-new work to build on the M5.
2. **No deep-learning deps.** requirements.txt has no torch. On the M5: install PyTorch
   with MPS (Metal) backend + Ultralytics for YOLO. CUDA-only tools (Detectron2) are a
   poor fit on Apple Silicon — avoid for the weekend.
3. **Dataset still needs extraction.** Labels are point/window annotations referencing
   .mkv frames; no extracted-frame dataset exists. Step 1 on the M5 is: pull labeled
   frames from the 15 videos and convert points -> boxes/keypoints (YOLO format).
4. **Midlines gap:** the behavior labeler wants *_midlines.npz that the white-rig tracker
   does not yet emit. None exist on disk. Relevant only if midlines are a training input.
