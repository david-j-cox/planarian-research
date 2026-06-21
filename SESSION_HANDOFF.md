# Session handoff — worm localizer (M5), 2026-06-21

Resume point for the next session. Everything below is on disk; data/labels are
gitignored (live under `realtime_runs/` and `live_capture/`).

## Where we landed
Goal: a near-perfect worm localizer on the **go-to rig = latest white 7MP**
(2026-06-02 15:54–16:08 series, 3360x2100, warm-white). These videos are now
local in `live_capture/` (15 files).

Key result chain (all measured against 600 human GT frames we labeled this
session, `realtime_runs/label_white7mp/`):

| Approach | Worm found | Median center err | p90 |
|---|---|---|---|
| YOLO nano (S3-trained), per-frame | 76.5% | 23.5 mm | 62 mm |
| **bg-sub + dish mask** (per-frame) | **100%** | **1.0 mm** | 4.6 mm |
| fusion tracker v1 (greedy nearest) | 100% | 1.3 mm | 31 mm (REGRESSED) |
| fusion tracker v2 (Hampel) | not yet measured (eval killed, exit 137) | — | — |

Takeaways that are settled:
- The YOLO model was doing blind per-frame detection and **58% of its boxes were
  outside the dish**. Adding the priors (background subtraction + dish ROI) cut
  median localization error **23x (23.5 -> 1.0 mm)** with zero training.
- Background subtraction also normalizes lighting -> likely the key to cross-rig
  robustness (white/blue/amber look alike after subtraction).
- Greedy temporal tracking (follow nearest-to-previous) HURTS: it locks onto
  debris / falls behind. The safe design is per-frame largest-blob-in-dish +
  post-hoc Hampel spike rejection (already implemented in fusion_tracker.py).

## IMMEDIATE next step (resume here)
Re-run the fusion (Hampel) eval — last run was killed by the machine
disconnect, not a bug:
```
cd scripts_notebooks
../venv/bin/python fusion_tracker.py eval        # ~3-5 min over 15 local videos
```
Compare median/p90 to the bg-sub-only baseline (1.0 / 4.6 mm). Hoped-for: p90
drops below 4.6. If p90 is still driven by a few clips, those are the hard cases
for the learned model.
- Watch memory: `build_background` does `np.median` over 50 full-res frames per
  video. If it OOMs, lower `n` or compute in float32 / streaming.

## Then (in priority order)
1. If a tail of hard frames remains (meniscus, debris): add the trained YOLO
   detector as a *disambiguation* layer, constrained to dish + near the tracked
   position — used only when bg-sub is ambiguous.
2. Retrain the detector on **bg-subtracted, dish-cropped** inputs (lighting-
   invariant) using S3 frames + the 600 new GT labels + negatives. This is the
   path to one model that works across white/blue/amber rigs. Hardware is ample
   (M5 Max, 40-GPU-core, 128 GB) — YOLO11l @ imgsz 1280 is the target if needed.
3. Confirm `mm_per_px` for the 16:xx videos. We used S3's 0.02657; if the rig
   geometry shifted, the mm numbers need the right calibration (pixel metrics
   are unaffected).
4. Behavior model (Stage C) is done but separate (RandomForest, macro-F1 0.67);
   revisit feature reconciliation later.

## Files written this session (scripts_notebooks/, tracked)
- `build_yolo_dataset.py`, `train_yolo.py`, `eval_localizer.py`,
  `eval_rigorous.py` — original YOLO pipeline (S3).
- `behavior_classifier.py` — behavior model.
- `infer_video.py` — end-to-end deploy (YOLO + mask-PCA + behavior).
- `label_localization.py` — model-in-the-loop GT labeler (prep/label/export).
- `bgsub_tracker.py` — bg-sub + dish detector + eval (the 1.0 mm result).
- `fusion_tracker.py` — fusion tracker (bg-sub + dish + Hampel temporal) +
  track/eval. **Resume by running its eval.**

## Artifacts (gitignored, on disk)
- `realtime_runs/label_white7mp/` — manifest + `labels.json` (600 GT frames) +
  cached frame images. THE precious artifact this session.
- `scripts_notebooks/runs/pose/worm_s3_n/weights/best.pt` (+ .onnx) — nano YOLO.
- `realtime_runs/behavior_clf.joblib` — behavior classifier.
- `live_capture/` — 8 S3 + 15 white-7MP go-to videos (local).
