#!/usr/bin/env python3
"""
tune_tracker.py — Interactive tuner for the planarian tracker.

Use this when the tracker's detection-rate gate trips (i.e., the first
video of a session detects below the floor, usually 90%). Common cause:
a new recording rig (different lighting, dish size, IR vs visible light)
needs different threshold/area/ROI than the defaults.

What it does:
    1. Opens the first MKV in --session_dir.
    2. Asks you to click on the worm in the first frame (manual seed).
    3. Builds the grid baseline from frames before the seed.
    4. Runs detect_worm in a tight loop with 4 trackbars exposed:
           threshold, min_area, max_area, roi_px.
    5. Shows live overlay: contour + centroid + a rolling detection-rate
       gauge over the last 100 frames.
    6. On [s] save: writes tracker_params.json to --output_dir.
    7. On [q] quit: exits without saving.

The tracker (open_dish_tracker.py) loads tracker_params.json from its
output_dir automatically, so a saved tune sticks for that rig.

Usage:
    python scripts_notebooks/tune_tracker.py \\
        --session_dir OpenDishWork/yuja_scratch/yuja_20251019_132514 \\
        --output_dir OpenDishWork/tracker_results
"""
from __future__ import annotations
import argparse
import collections
import glob
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# Load helpers from open_dish_tracker.py without re-importing main().
SCRIPT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "odt", SCRIPT_DIR / "open_dish_tracker.py")
odt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(odt)


WINDOW = "Tracker Tuner"
DETECTION_HISTORY = 100


def _click_seed(frame_bgr) -> tuple[int, int] | None:
    """Show a frame and return the (x, y) the user clicks. ESC = cancel."""
    seed = {"pos": None}

    def on_click(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            seed["pos"] = (x, y)

    win = "Click the worm (ESC=cancel)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    h, w = frame_bgr.shape[:2]
    cv2.resizeWindow(win, min(w, 1400), min(h, 900))
    cv2.setMouseCallback(win, on_click)
    while True:
        vis = frame_bgr.copy()
        cv2.putText(vis, "Click the worm, then press [Enter]. ESC to cancel.",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (40, 220, 40), 2)
        if seed["pos"]:
            cv2.circle(vis, seed["pos"], 12, (40, 220, 40), 2)
        cv2.imshow(win, vis)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            seed["pos"] = None
            break
        if k in (13, 10) and seed["pos"]:
            break
    cv2.destroyWindow(win)
    return seed["pos"]


def _build_baseline(video_path, seed_frame, dish_mask_bool):
    try:
        return odt.build_grid_baseline(video_path, seed_frame, dish_mask_bool)
    except Exception as e:
        print(f"  WARN: pre-seed baseline failed ({e}); trying median.")
        return odt.build_grid_baseline_median(video_path, dish_mask_bool,
                                              n_samples=30)


def tune(session_dir: str, output_dir: str,
         start_thresh: float, start_min_area: int,
         start_max_area: int, start_roi: int) -> dict | None:
    """Open the first MKV, run the loop, return params to save (or None)."""
    videos = sorted(glob.glob(os.path.join(session_dir, "*.mkv")))
    if not videos:
        raise SystemExit(f"No *.mkv in {session_dir}")
    video_path = videos[0]
    print(f"Tuning on: {video_path}")

    cap = odt.open_video(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    # First-frame seed (skip a few to let exposure settle)
    seed_frame_idx = min(10, max(0, total_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, seed_frame_idx)
    ok, first_bgr = cap.read()
    if not ok:
        raise SystemExit("Cannot read seed frame")

    # Dish detection
    gray = odt.to_gray01(first_bgr)
    H, W = gray.shape
    try:
        dish_center, dish_radius = odt.auto_detect_dish(gray)
    except Exception as e:
        raise SystemExit(f"Dish detection failed: {e}")
    dish_mask_bool = odt.circle_mask((H, W), dish_center, dish_radius)
    print(f"  Dish: center={dish_center} radius={dish_radius:.0f}px")

    seed_pos = _click_seed(first_bgr)
    if seed_pos is None:
        print("Cancelled.")
        return None
    print(f"  Seed at {seed_pos} (frame {seed_frame_idx})")

    baseline = _build_baseline(video_path, seed_frame=15,
                               dish_mask_bool=dish_mask_bool)
    if baseline is None:
        raise SystemExit("Could not build a usable baseline.")
    bg_max = odt.build_background(cap, dish_mask_bool, n_samples=30)

    # Trackbars
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1400, 900)
    # cv2 trackbars are integers, so scale thresh to per-mille.
    cv2.createTrackbar("thresh x1000", WINDOW, int(start_thresh * 1000),
                       300, lambda v: None)
    cv2.createTrackbar("min_area", WINDOW, start_min_area, 2000, lambda v: None)
    cv2.createTrackbar("max_area", WINDOW, start_max_area, 8000, lambda v: None)
    cv2.createTrackbar("roi_px", WINDOW, start_roi, 400, lambda v: None)

    last_centroid = (float(seed_pos[0]), float(seed_pos[1]))
    history = collections.deque(maxlen=DETECTION_HISTORY)
    prev_area = None
    lost_count = 0
    saved = None

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    print()
    print("  [s] save tracker_params.json  [r] restart video  [q] quit")
    print()

    while True:
        ok, bgr = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            continue
        frame_idx += 1
        gray = odt.to_gray01(bgr)

        thresh = max(0.001, cv2.getTrackbarPos("thresh x1000", WINDOW) / 1000.0)
        min_a = max(1, cv2.getTrackbarPos("min_area", WINDOW))
        max_a = max(min_a + 1, cv2.getTrackbarPos("max_area", WINDOW))
        roi = max(10, cv2.getTrackbarPos("roi_px", WINDOW))

        centroid, area, contour, conf = odt.detect_worm(
            gray, bg_max, dish_mask_bool, last_centroid,
            min_a, max_a, roi, max_jump_px=roi,
            lost_count=lost_count, prev_area=prev_area,
            grid_baseline=baseline, detect_thresh=thresh)

        if centroid is not None:
            last_centroid = centroid
            lost_count = 0
            history.append(1)
            prev_area = (float(area) if prev_area is None
                         else 0.8 * prev_area + 0.2 * float(area))
        else:
            lost_count += 1
            history.append(0)

        rate = (sum(history) / len(history) * 100) if history else 0.0

        # Render
        vis = bgr.copy()
        if contour is not None:
            cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)
        if centroid is not None:
            cv2.circle(vis, (int(centroid[0]), int(centroid[1])),
                       6, (40, 220, 40), -1)
        # ROI ring around last position
        cv2.circle(vis, (int(last_centroid[0]), int(last_centroid[1])),
                   roi, (0, 200, 200), 1)
        # HUD
        color = (50, 220, 50) if rate >= 90 else \
                (50, 200, 220) if rate >= 75 else (60, 60, 240)
        cv2.rectangle(vis, (5, 5), (560, 105), (30, 30, 30), -1)
        cv2.putText(vis,
                    f"det rate: {rate:5.1f}%  ({sum(history)}/{len(history)})",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(vis,
                    f"thresh={thresh:.3f}  area={min_a}-{max_a}  roi={roi}",
                    (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (220, 220, 220), 1)
        cv2.putText(vis,
                    "[s] save   [r] restart   [q] quit",
                    (15, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (180, 180, 180), 1)
        cv2.imshow(WINDOW, vis)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("r"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            history.clear()
            last_centroid = (float(seed_pos[0]), float(seed_pos[1]))
            prev_area = None
            lost_count = 0
            continue
        if k == ord("s"):
            saved = {
                "params": {
                    "detect_thresh": thresh,
                    "min_area": min_a,
                    "max_area": max_a,
                    "roi_px": roi,
                    "_tuned_on_video": os.path.basename(video_path),
                    "_tuned_rate_pct": round(rate, 1),
                    "_tuned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                               time.gmtime()),
                },
                "seed": {
                    "seed_frame": int(seed_frame_idx),
                    "seed_position": [float(seed_pos[0]), float(seed_pos[1])],
                    "source_video": os.path.basename(video_path),
                    "source": "tune_tracker_click",
                },
            }
            break

    cap.release()
    cv2.destroyAllWindows()
    return saved


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session_dir", required=True,
                    help="Session folder containing the *.mkv to tune on.")
    ap.add_argument("--output_dir", required=True,
                    help="Where to write tracker_params.json (same dir the "
                         "main tracker writes its CSVs).")
    ap.add_argument("--detect_thresh", type=float, default=0.05)
    ap.add_argument("--min_area", type=int, default=80)
    ap.add_argument("--max_area", type=int, default=3000)
    ap.add_argument("--roi_px", type=int, default=120)
    args = ap.parse_args()

    if not os.path.isdir(args.session_dir):
        raise SystemExit(f"Session dir not found: {args.session_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    saved = tune(args.session_dir, args.output_dir,
                 args.detect_thresh, args.min_area,
                 args.max_area, args.roi_px)
    if saved is None:
        print("\nQuit without saving.")
        return

    # 1. Params -> output_dir (tracker auto-loads from here)
    params_path = os.path.join(args.output_dir, "tracker_params.json")
    with open(params_path, "w") as f:
        json.dump(saved["params"], f, indent=2)

    # 2. Seed -> session_dir, in the *_seed.json shape the tracker
    #    already understands. Filename pattern: <session_name>_seed.json
    session_name = os.path.basename(os.path.normpath(args.session_dir))
    seed_path = os.path.join(args.session_dir, f"{session_name}_seed.json")
    with open(seed_path, "w") as f:
        json.dump(saved["seed"], f, indent=2)

    print(f"\nSaved:")
    print(f"  params: {params_path}")
    print(f"  seed:   {seed_path}")
    print(f"\nNext run of open_dish_tracker.py with --output_dir "
          f"{args.output_dir!r} will:")
    print(f"  - load the tuned threshold/area/ROI from tracker_params.json")
    print(f"  - use your clicked seed as the worm's starting position")
    print(f"  - chain state across one-minute video boundaries as usual.")


if __name__ == "__main__":
    main()
