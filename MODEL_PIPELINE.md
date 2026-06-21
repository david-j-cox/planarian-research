# Worm localization + behavior model (M5)

Finetuned vision pipeline that tracks worm location and classifies behavior,
built on the M5 Max from the migrated S3 session. Trains/runs on Apple MPS.

## What it does
- **Localization (YOLO):** detects the worm box per frame. A single feed-forward
  net, validated against the 73 hand-marked GT points at **~0.5 mm median error,
  100% detection on held-out video** (worm body ~8.2 mm).
- **Body axis/length:** PCA on the worm's dark pixels inside the detected box
  (recovers ~8 mm length). The trained head/tail keypoints are NOT used — they
  collapse to the centroid because the pseudo-label head/tail assignment is
  anatomically inconsistent.
- **Behavior (v1):** RandomForest on windowed track features (LOO-CV
  macro-F1 0.67, micro-F1 0.80 vs rule baseline 0.24). Approximate in
  deployment because inference features come from the PCA axis, not the
  classical 20-point midline it trained on. Location is the solid part;
  behavior is a usable v1.

## Environment
Python 3.12 venv at `venv/` with torch 2.12 (MPS), ultralytics 8.4, opencv.

## Pipeline (run from `scripts_notebooks/`)
```
# 1. Build YOLO dataset from the classical tracker's dense output (pseudo-labels)
python build_yolo_dataset.py        # -> ../realtime_runs/yolo_dataset/

# 2. Finetune the localizer on MPS
python train_yolo.py                # -> runs/pose/worm_s3_n/weights/best.pt

# 3. Evaluate vs the 73 hand-marked GT points (mm error, train vs held-out)
python eval_localizer.py

# 4. Train the behavior classifier on the 80 blind-labeled windows
python behavior_classifier.py       # -> ../realtime_runs/behavior_clf.joblib

# 5. Deploy: video -> per-frame location + body_len + behavior (CSV + overlay)
python infer_video.py --video ../live_capture/<file>.mkv \
    --mm_per_px 0.02657 --overlay
```

## Models
- `runs/pose/worm_s3_n/weights/best.pt`  — native PyTorch (MPS), primary
- `runs/pose/worm_s3_n/weights/best.onnx` — portable
- CoreML export currently blocked by a torch 2.12 / coremltools version
  conflict; revisit in a pinned env for Neural Engine acceleration.

## Known limitations / next steps
- **Calibration is per-rig.** `mm_per_px` must match the rig of the input video
  (S3 white rig = 0.02657). The repo `calibration.json` is a different rig
  (0.094) — pass `--mm_per_px` explicitly until per-rig calibration is wired in.
- **Trained on S3 only** (white 7MP rig). The amber S1 rig differs in
  resolution/scale; add S1 frames to generalize across rigs.
- **Behavior**: rare classes (wig_wag n=5, scrunching n=4) are label-limited;
  inference features should be reconciled with training features (emit a real
  midline at inference, or retrain the classifier on PCA-axis features).
- Dataset is 339/65 train/val frames subsampled by displacement; expand from the
  full 13k tracked frames or tier-2 corpus to push accuracy further.
