#!/usr/bin/env python3
"""
diagnose_detection.py — See what the tracker sees, on the NEW rig.

Given a worm-free baseline clip and a worm clip, this dumps annotated images so
you can visually confirm (not guess from a %) whether:
  - the dish circle is detected in the right place,
  - the chosen channel actually separates the worm,
  - the baseline-subtraction lights up on the worm,
  - detect_worm lands on the worm.

Outputs into --out_dir:
  channels.jpg     — the worm frame split into gray/blue/green/red/L/a/b
  baseline.jpg     — the worm-free baseline (chosen channel) + detected dish
  detection.jpg    — a worm frame with dish circle + detected worm marked
  diffmap.jpg      — baseline-minus-frame (what the threshold sees)

Usage:
  cd scripts_notebooks
  source ../venv/bin/activate
  python diagnose_detection.py \
      --baseline_clip ../live_capture/EMPTY.mkv \
      --worm_clip ../live_capture/WORM.mkv \
      --out_dir /tmp/diag

Then look at the JPEGs in --out_dir.
"""

import os
import sys
import argparse
import cv2
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
from open_dish_tracker import auto_detect_dish, auto_grid_calibration, circle_mask  # noqa: E402
from watch_folder_tracker import frame_to_channel, pick_best_channel, detect_worm  # noqa: E402

MIN_AREA, MAX_AREA, ROI_PX, MAX_JUMP_PX, DETECT_THRESH = 80, 3000, 120, 100, 0.05


def sample_frames(path, n):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 600
    idxs = np.linspace(0, max(0, total - 1), min(n, total), dtype=int)
    out = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok and f is not None:
            out.append(f)
    cap.release()
    return out


def u8(ch01):
    return np.clip(ch01 * 255, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline_clip", required=True, help="Worm-free clip.")
    ap.add_argument("--worm_clip", required=True, help="Clip with the worm.")
    ap.add_argument("--out_dir", default="/tmp/diag")
    ap.add_argument("--channel", default="auto")
    ap.add_argument("--detect_thresh", type=float, default=DETECT_THRESH)
    ap.add_argument("--min_area", type=int, default=MIN_AREA)
    ap.add_argument("--max_area", type=int, default=MAX_AREA)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    base_frames = sample_frames(args.baseline_clip, 30)
    worm_frames = sample_frames(args.worm_clip, 30)
    if not base_frames or not worm_frames:
        sys.exit("Could not read frames from one of the clips.")

    # 1) Channel panel on a worm frame (middle of the clip).
    wf = worm_frames[len(worm_frames) // 2]
    panels = []
    for ch in ("gray", "blue", "green", "red", "lab_l", "lab_a", "lab_b"):
        img = u8(frame_to_channel(wf, ch))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.putText(img, ch, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0), 2)
        panels.append(cv2.resize(img, (420, 263)))
    rows = [np.hstack(panels[i:i + 4]) for i in range(0, len(panels), 4)]
    # pad last row
    if rows and rows[-1].shape[1] < rows[0].shape[1]:
        pad = np.zeros((rows[-1].shape[0],
                        rows[0].shape[1] - rows[-1].shape[1], 3), np.uint8)
        rows[-1] = np.hstack([rows[-1], pad])
    cv2.imwrite(os.path.join(args.out_dir, "channels.jpg"), np.vstack(rows))
    print(f"Wrote channels.jpg — compare which channel shows the worm clearest")

    # 2) Channel choice + baseline + dish.
    chosen = pick_best_channel(base_frames) if args.channel == "auto" else args.channel
    base = np.median(np.stack([frame_to_channel(f, chosen) for f in base_frames]),
                     axis=0).astype(np.float32)
    h, w = base.shape
    dish_ok = True
    try:
        dc, dr = auto_detect_dish(base)
    except Exception as e:
        print(f"Dish detection FAILED: {e}; using full frame.")
        dc, dr = (w / 2, h / 2), 0.48 * min(h, w)
        dish_ok = False
    try:
        mmpp, sp = auto_grid_calibration(base, dc, dr)
        print(f"Grid: {mmpp:.5f} mm/px (spacing {sp:.1f}px)")
    except Exception as e:
        mmpp = None
        print(f"Grid calibration failed: {e}")

    base_bgr = cv2.cvtColor(u8(base), cv2.COLOR_GRAY2BGR)
    cv2.circle(base_bgr, (int(dc[0]), int(dc[1])), int(dr),
               (0, 0, 255) if dish_ok else (0, 165, 255), 3)
    cv2.circle(base_bgr, (int(dc[0]), int(dc[1])), 6, (0, 0, 255), -1)
    cv2.putText(base_bgr, f"channel={chosen} dish r={dr:.0f} "
                f"{'OK' if dish_ok else 'FALLBACK'}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(args.out_dir, "baseline.jpg"), base_bgr)
    print(f"Wrote baseline.jpg — red circle should sit ON the dish rim")

    dish_mask_bool = circle_mask((h, w), dc, dr) > 0

    # 3) Diff map + detection on the worm frame.
    wch = frame_to_channel(wf, chosen)
    bl_mean = base[dish_mask_bool].mean()
    gr_mean = wch[dish_mask_bool].mean()
    norm = wch * (bl_mean / gr_mean) if gr_mean > 1e-6 else wch
    diff = np.clip(base - norm, 0, 1)
    diff[~dish_mask_bool] = 0
    cv2.imwrite(os.path.join(args.out_dir, "diffmap.jpg"),
                u8(diff / max(1e-6, diff.max())))
    print(f"Wrote diffmap.jpg — the worm should be the brightest blob; "
          f"max diff={diff.max():.3f}, thresh={args.detect_thresh}")

    centroid, area, contour, conf = detect_worm(
        wch, None, dish_mask_bool, None,
        args.min_area, args.max_area, ROI_PX, MAX_JUMP_PX, 0,
        grid_baseline=base, detect_thresh=args.detect_thresh)
    det = cv2.cvtColor(u8(wch), cv2.COLOR_GRAY2BGR)
    cv2.circle(det, (int(dc[0]), int(dc[1])), int(dr), (0, 0, 255), 2)
    if centroid is not None:
        cv2.circle(det, (int(centroid[0]), int(centroid[1])), 12, (0, 255, 0), 3)
        if contour is not None:
            cv2.drawContours(det, [contour], -1, (0, 255, 0), 2)
        msg = f"DETECTED area={area} conf={conf:.2f}"
    else:
        msg = "NO DETECTION"
    cv2.putText(det, msg, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 0), 2)
    cv2.imwrite(os.path.join(args.out_dir, "detection.jpg"), det)
    print(f"Wrote detection.jpg — {msg}")
    print(f"\nAll images in {args.out_dir}/")


if __name__ == "__main__":
    main()
