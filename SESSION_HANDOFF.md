# Session handoff — worm localizer (M5)

Resume point for the next session. Everything below is on disk; data/labels/
weights are gitignored (live under `realtime_runs/`, `live_capture/`, and
`scripts_notebooks/runs/`).

## Where we landed (updated 2026-06-22)
Goal: a near-perfect worm localizer on the **go-to rig = latest white 7MP**
(2026-06-02 15:54–16:08 series, 3360x2100, warm-white; 15 clips local in
`live_capture/`). Scored against the 600 human GT frames in
`realtime_runs/label_white7mp/` (400 'active', 200 held-out 'eval').

**A rig-trained YOLO is now the primary localizer and clearly wins.** Progress
chain on the SAME 200 held-out GT frames (box-center error, mm):

| Approach | Median | p90 | Max | Coverage |
|---|---|---|---|---|
| S3-YOLO, per-frame (old) | 23.5 | 62 | — | 76.5% |
| bg-sub + dish (fusion_tracker) | 0.978 | 3.79 | 50.9 | 100% |
| **white-7MP YOLO + temporal (yolo_tracker)** | **0.678** | **1.57** | **3.66** | 198/200 |

Settled this session:
- **The fusion eval's "kills" were a bug, not the machine.** `pca_axis` used
  `np.linalg.svd` with `full_matrices=True` → an M×M matrix on noisy contours
  (44 GB, multi-minute hangs). Fixed to a 2×2 covariance eigenvector. The eval
  is now resumable (`fusion_eval_checkpoint.json`).
- fusion (Hampel) tracker: median 0.978 / p90 3.79, 100% localized — the v1
  greedy-tracker p90 regression (31 mm) is gone. But a tail remained (max
  50.9 mm) from wrong-blob frames where the worm was momentarily invisible to
  bg-sub (noisy/illumination frames with 200–900 candidate blobs).
- **Segmented (time-windowed) backgrounds were tried and REVERTED** — net
  regression (median 0.978→1.35, p90 3.79→6.31): short per-segment windows let
  a dwelling worm contaminate its own background. Do not revisit without
  fixing that.
- The S3-trained YOLO cannot localize on white-7MP even with a crop prior
  (conf 0.15–0.42, errors 13–58 mm) → it was a domain-shift problem, fixed by
  fine-tuning on the rig.
- **Fine-tuning the S3 nano on 360 white-7MP frames** (build_white7mp_dataset
  → train_yolo --name worm_white7mp_n): median 0.672 mm per-frame, **no
  50 mm tail** (max 3.66), 100% within 5 mm. ~35× better than S3.

## Current best tracker: `scripts_notebooks/yolo_tracker.py`
- Position from `runs/pose/worm_white7mp_n/weights/best.pt` (dish gate ×1.10).
- Morphology (head/tail/body-len via PCA) from the bg-sub foreground INSIDE the
  YOLO box (`box_axis`, box expanded 0.35×), anchored THROUGH the YOLO
  position. This excludes the dish rim (which used to make the axis arc across
  the dish on edge worms) and caps body length to the box. Position never
  depends on bg-sub.
- Shared `fusion_tracker.hampel_clean` temporal pass (spike reject + interp/
  hold + smooth) fills no-detection frames and jitter.
- `python yolo_tracker.py eval` (windowed, ~20 min) | `... track --video ... --overlay`
  (overlay is downscaled 0.4× for smooth playback; --overlay_scale to change).

## Validated end-to-end (2026-06-22)
Full-video track on 2026-06-02_16-00-37 (1890 frames): 100% detection, 0
interp; body-length tight [5.06, 9.03]mm (median 7.87), present every frame;
green head/tail axis stays on the worm at the dish edge. Overlay in
`realtime_runs/yolo_out/`. KNOWN open item: ~1.2% of frames show 2–6mm
single-frame position jumps (mostly box-center wobble as the worm bends; the
box center is a noisier position proxy than a blob centroid). Hampel didn't
catch them (sub-threshold / not isolated). Revisit by gating on physical worm
speed if it matters downstream.

## IMMEDIATE next step (resume here): behavior-model integration
Wire yolo_tracker's per-frame output (position + head/tail/body-len + state)
into the behavior model (Stage C, `behavior_classifier.py`, RandomForest
macro-F1 0.67). Reconcile the feature set the classifier expects against what
yolo_tracker emits in its CSV. See also `behavior_rules` (rule-based v1) and
the blind-labeling tools for behavior GT.

## Then (priority order)
1. **Cross-rig generalization**: the white-7MP model is rig-specific (eval
   frames are other frames from the same 15 videos). For blue/amber rigs, train
   on bg-subtracted / dish-cropped (lighting-invariant) inputs, or add labels
   from other rigs. This is the path to ONE model.
2. Position jitter: optional Hampel/speed-gate tightening for the ~1.2% jump
   frames noted above.
3. Speed: if real-time needed, export to CoreML/ONNX or lower imgsz; current
   ~3 fps is fine for offline analysis only.
4. Confirm `mm_per_px` (used S3's 0.02657) for the 16:xx rig geometry. Pixel
   metrics are unaffected; only mm scaling.

## Files this session (scripts_notebooks/, tracked)
- `fusion_tracker.py` — bg-sub fusion tracker; pca_axis SVD fix; resumable
  eval; `hampel_clean` extracted for reuse.
- `build_white7mp_dataset.py` — YOLO pose dataset from label_white7mp (eval
  pool held out).
- `eval_white7mp_localizer.py` — per-frame YOLO box-center mm eval.
- `yolo_tracker.py` — YOLO-primary tracker (track/eval). **Current best.**

## Artifacts (gitignored, on disk)
- `realtime_runs/label_white7mp/` — manifest + labels.json (600 GT) + cached
  frames. THE precious artifact.
- `scripts_notebooks/runs/pose/worm_white7mp_n/weights/best.pt` — rig localizer.
- `scripts_notebooks/runs/pose/worm_s3_n/weights/best.pt` — S3 nano (source).
- `realtime_runs/yolo_white7mp/` — fine-tune dataset (360 train / 40 val).
- `live_capture/` — 8 S3 + 15 white-7MP go-to videos (local).
