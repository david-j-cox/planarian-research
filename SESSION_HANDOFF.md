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
speed if it matters downstream. NOTE: these jumps no longer corrupt head/tail
identity (orientation is now centroid-relative, jump-invariant) -- they only
wobble the position dot.

## Behavior-model integration — DONE (2026-06-22), unvalidated on this rig
- `yolo_to_signals.py` (new) runs yolo_tracker over clips and writes the
  `<session>_signals.npz` schema that `behavior_features` / `behavior_rules` /
  the trained classifier consume. The behavior model now runs unchanged on the
  YOLO-primary localizer. End-to-end verified on 2026-06-02_16-00-37 (1890
  frames, 0 lost): features compute, rule + S3-trained classifiers both run.
- Morphology upgraded from a straight PCA axis to a CURVED head->tail midline:
  `box_morphology` skeletonizes the box-constrained worm contour via
  `open_dish_tracker.extract_midline` (straight-axis PCA kept only as fallback).
  This decouples head oscillation from body heading -- `head_osc_deg` 12.4->28.9,
  `head_reversals` p90 0->3 (was flat zero). The rule classifier then surfaces
  all 7 behaviors (wig_wag 1.1%->26.4%, reversing 0%->7.5%) vs only 5 before.
- Head/tail ORIENTATION made robust (`orient_midlines`): centroid-relative EMA
  chaining for stability + a physical-speed-gated global velocity vote to name
  the head (planaria lead with the head when gliding). The two 180-deg head
  flips first seen in the overlay were NOT body folds -- they sat on box-center
  position jumps (~30 mm/s, worm max ~7), where an absolute-position reference
  degenerates. Centroid-relative matching is jump-invariant; residual head
  reversals 2->0 on the validated clip. Confirmed visually 2026-06-22.

### Behavior model VALIDATED + retrained on white-7MP (2026-06-23)
- Built a blind label set: tracked all 15 go-to clips into one
  `realtime_runs/white7mp_signals.npz` (27441 frames, 100% curved midline), then
  `behavior_label_tool.py sample --n 130 --window_s 3` -> 126 windows stratified
  18/behavior, in `realtime_runs/white7mp_labels_blind/` (manifest + hidden rule
  preds tracked; `human_labels.csv` tracked).
- Labeled all 126 blind. Distribution: gliding 87, wig_wag 27, turning 23,
  scrunching 5; ZERO resting/peristalsis/reversing (the rule's rare predictions
  were systematically wrong on this rig).
- Rule classifier vs blind labels (`behavior_accuracy.py`): 21% / macro-F1
  0.208 (low, expected -- sample is balanced by rule pred, adversarial to it).
- Retrained RandomForest (`behavior_classifier.py`, LOO-CV): macro-F1 0.681 /
  micro-F1 0.811. gliding F1 0.94, wig_wag 0.75, turning 0.43, scrunch 0.60.
  3.3x the rule; first behavior model validated ON this rig (not borrowed S3).
  Saved `realtime_runs/behavior_clf_white7mp.joblib` (gitignored, regenerable).
- NOTE during labeling: the label GUI playback was choppy (wall-clock frame
  indexing skipped frames under load + near-native crop render). Fixed in
  `behavior_label_tool.py` (paced one-frame-per-period + tiny persistent display
  buffer; new --disp_w). Source clips are clean 30fps (verified all 15).

### Improving the behavior model (2026-06-23): 3 levers worked
Finding: at n=126 with scrunch=5/turning=22, macro-F1 bounces +/-0.05 from
feature/model variants -- the binding constraint is per-class DATA + behavior
DEFINITIONS, not the model (RF is right at this size). Levers pursued:
1. MOTION FEATURES (`ang_vel_p90_deg_s`, `path_curv_deg_mm`, `body_curv_deg`,
   jitter-robust) -- NEUTRAL on current data (RF 0.681->0.669). The raw
   (non-robust) variant scored 0.718 but that was jitter overfit (won't
   generalize); rejected. Kept robust versions for when classes grow.
2. ETHOGRAM (`docs/behavior_ethogram.md`) -- measurable per-gait definitions,
   explicit gliding-vs-turning boundary (the main label-noise source). Use it
   for the next labeling round; consider a 2nd-labeler inter-rater check.
3. ACTIVE LEARNING (`behavior_label_tool.py active`) -- DONE round 2: labeled
   the 95-window batch, merged to 221 via `merge_label_sets.py`
   (`white7mp_labels_combined`), retrained. Active learning enriched the hard
   classes (turning 23->48, scrunch 5->15). Combined RF LOO-CV (4 classes,
   min_support=5): macro-F1 0.637; TURNING 0.43->0.53 (the targeted gain),
   gliding 0.90, wig_wag 0.72, scrunch 0.40. Macro looks flat vs the first
   batch's 0.681 only because the combined set is harder/representative (active
   learning adds boundary cases) -- 0.637 is the more trustworthy estimate.
   The label->train->active->label loop is reproducible; run it again for more
   turning/scrunch. Tool now has an 'x' no-worm skip + resume-at-first-unlabeled.

### Corpus scope (checked 2026-06-23)
Drive `My Drive/PlanarianVideos/` has ~968 videos across 4 dates, but only
2026-06-02 is the CURRENT white rig (OpenDishWork/additional_videos = deprecated
old setups, per user -- do NOT use). Of 39 white-rig clips: 14:31-14:49 (16
clips) are EMPTY pre-worm setup footage (0-10% detection -- don't re-pull);
14:58-16:08 (23 clips) hold a worm. Usable pool = `white7mp_signals_pool.npz`
(23 clips, 40,134 detected frames). 16 empty clips sit in live_capture
(gitignored), deletable.

### Still open
- scrunch/peristalsis/reversing are scarce in this baseline session -- they are
  EVOKED gaits. NOT obtainable from existing footage (all 2026-06-02 baseline;
  old rigs unusable). Biggest remaining lever = record evoked behavior on the
  WHITE rig (drug/stimulus) and label that. Data-collection decision (lab).
- Active batch 3 ready to label: `realtime_runs/white7mp_labels_active2` (93
  windows). Label -> merge_label_sets (blind+active+active2) -> retrain.
- A few clips show 1-8 residual head-flips (position-jump frames); only affects
  features on those windows. Tighten position jitter (speed gate) if needed.
- Apply `behavior_clf_white7mp.joblib` across full clips for habituation/
  pharmacology readouts (`habituation_analysis.py`, `infer_video.py`).
- Label-tool playback was choppy mid-session; fixed (paced playback; --disp_w).

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
