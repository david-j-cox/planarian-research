#!/usr/bin/env python3
"""
realtime_tracker.py — Near real-time single-worm tracker for a live webcam.

End-to-end loop: capture → detect → display + log. Reuses the proven detection
code from open_dish_tracker.py (dish detection, grid calibration, grid-baseline
subtraction, detect_worm) rather than reimplementing it.

Pipeline (mirrors docs/realtime_monitoring_architecture.md §4.2):
  1. Open the camera, warm it up.
  2. Calibrate: capture N "empty dish" frames → median background →
     auto_detect_dish + auto_grid_calibration (mm_per_px). Build the grid
     baseline from the same empty frames.
  3. Auto-seed: wait until a worm-sized dark blob appears inside the dish.
  4. Per-frame: detect_worm() threading last_centroid/lost_count/prev_area,
     compute speed, draw a live overlay window, and log one row per frame to
     CSV (and optionally SQLite).

I/O:
  - CSV columns match the offline open_dish_tracker schema closely enough to
    feed straight into session_analysis.py / habituation_analysis.py.
  - Output lands in ../realtime_runs/ by default (gitignored — it's data).

Keys (in the live window):
  q = quit   r = re-seed   c = recalibrate   space = pause

Usage:
  cd scripts_notebooks
  source ../venv/bin/activate
  python realtime_tracker.py                       # built-in/USB cam (index 0)
  python realtime_tracker.py --camera 1            # a different camera
  python realtime_tracker.py --video ../additional_videos/some.mkv  # replay
  python realtime_tracker.py --sqlite              # also log to SQLite

Python 3.9+. Requires: opencv-python, numpy (same as the tracker).
"""

import os
import sys
import csv
import time
import math
import sqlite3
import argparse
from collections import deque
from datetime import datetime

import cv2
import numpy as np

# Reuse the offline tracker's vetted building blocks.
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
from open_dish_tracker import (   # noqa: E402
    open_video,
    to_gray01,
    circle_mask,
    auto_detect_dish,
    auto_grid_calibration,
    detect_worm,
)

# ──────────────────────────────────────────────────────────────────────
# Defaults (match open_dish_tracker.py argparse defaults)
# ──────────────────────────────────────────────────────────────────────
MIN_AREA = 80
MAX_AREA = 3000
ROI_PX = 120
MAX_JUMP_PX = 100
DETECT_THRESH = 0.05
LOST_THRESHOLD = 30

CALIB_FRAMES = 30          # empty-dish frames for background/baseline
WARMUP_SEC = 1.0           # let camera auto-exposure settle
SEED_STABLE_FRAMES = 3     # consecutive in-dish detections before we lock on
TRAIL_LEN = 60             # centroid trail length in the overlay

# Overlay colors (BGR)
COL_CONTOUR = (0, 220, 0)
COL_CENTROID = (0, 0, 255)
COL_TRAIL = (255, 200, 0)
COL_TEXT = (255, 255, 255)
COL_LOST = (0, 0, 255)
COL_DISH = (120, 120, 120)


# ──────────────────────────────────────────────────────────────────────
# Camera / source abstraction
# ──────────────────────────────────────────────────────────────────────

def open_source(camera_index, video_path):
    """Return an opened capture for either a live camera or a video file."""
    if video_path:
        cap = open_video(video_path)
        is_live = False
        print(f"Source: replaying video {video_path}")
    else:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera index {camera_index}. "
                f"Try --camera 1, or check camera permissions "
                f"(macOS: System Settings → Privacy → Camera).")
        # Ask for a reasonable resolution; the camera may override.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        is_live = True
        print(f"Source: live camera index {camera_index}")
    return cap, is_live


def grab(cap):
    """Read one BGR frame; return None at end-of-stream / read failure."""
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


# ──────────────────────────────────────────────────────────────────────
# Calibration
# ──────────────────────────────────────────────────────────────────────

def calibrate(cap, n_frames=CALIB_FRAMES):
    """Capture empty-dish frames → dish mask, mm_per_px, grid baseline.

    Returns a dict: dish_center, dish_radius, dish_mask_bool, mm_per_px,
    grid_baseline (float32 [0,1] mean of the empty frames).

    Degrades gracefully: if the grid can't be read, mm_per_px is None
    (metrics fall back to pixels). If the dish can't be found, the whole
    frame is treated as the dish.
    """
    print(f"\nCalibrating — keep the worm OUT of the dish for "
          f"{n_frames} frames...")
    grays = []
    while len(grays) < n_frames:
        frame = grab(cap)
        if frame is None:
            raise RuntimeError("Stream ended during calibration.")
        grays.append(to_gray01(frame))
        # Show progress so the user knows it's working.
        prog = frame.copy()
        cv2.putText(prog, f"Calibrating {len(grays)}/{n_frames} "
                    f"(empty dish)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imshow(WIN, prog)
        cv2.waitKey(1)

    median_gray = np.median(np.stack(grays), axis=0).astype(np.float32)
    h, w = median_gray.shape

    # Dish detection (fall back to full frame).
    try:
        dish_center, dish_radius = auto_detect_dish(median_gray)
        print(f"  Dish: center=({dish_center[0]:.0f},{dish_center[1]:.0f}) "
              f"r={dish_radius:.0f}px")
    except Exception as e:
        print(f"  Dish detection failed ({e}); using full frame.")
        dish_center = (w / 2.0, h / 2.0)
        dish_radius = 0.48 * min(h, w)

    dish_mask = circle_mask((h, w), dish_center, dish_radius)
    dish_mask_bool = dish_mask.astype(bool)

    # Grid calibration (fall back to pixels-only).
    mm_per_px = None
    try:
        mm_per_px, spacing_px = auto_grid_calibration(
            median_gray, dish_center, dish_radius)
        print(f"  Grid: {mm_per_px:.5f} mm/px (spacing {spacing_px:.1f}px). "
              f"Dish ≈ {2 * dish_radius * mm_per_px:.1f} mm across.")
    except Exception as e:
        print(f"  Grid calibration failed ({e}); reporting pixels only.")

    return {
        "dish_center": dish_center,
        "dish_radius": dish_radius,
        "dish_mask_bool": dish_mask_bool,
        "mm_per_px": mm_per_px,
        "grid_baseline": median_gray,   # mean/median of empty frames
    }


# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────

CSV_HEADER = [
    "frame", "time_s",
    "centroid_x_px", "centroid_y_px", "centroid_x_mm", "centroid_y_mm",
    "area_px", "speed_px_s", "speed_mm_s", "confidence", "is_lost",
]


class Logger:
    """Writes one row per frame to CSV, and optionally SQLite."""

    def __init__(self, out_dir, session_id, mm_per_px, use_sqlite):
        os.makedirs(out_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, f"{session_id}_tracks.csv")
        self._fh = open(self.csv_path, "w", newline="")
        self._w = csv.writer(self._fh)
        # Two metadata rows then header — same shape session_analysis expects
        # (it reads with skiprows=2).
        self._w.writerow([f"# mm_per_px={mm_per_px if mm_per_px else ''}"])
        self._w.writerow([f"# session_id={session_id} "
                          f"created={datetime.now().isoformat(timespec='seconds')}"])
        self._w.writerow(CSV_HEADER)

        self.db = None
        if use_sqlite:
            db_path = os.path.join(out_dir, f"{session_id}.db")
            self.db = sqlite3.connect(db_path)
            self.db.execute("PRAGMA journal_mode=WAL;")
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS tracking (
                    frame INTEGER, time_s REAL,
                    x_px REAL, y_px REAL, x_mm REAL, y_mm REAL,
                    area_px REAL, speed_px_s REAL, speed_mm_s REAL,
                    confidence REAL, is_lost INTEGER)""")
            self.db_path = db_path

    def write(self, row):
        # row is a dict keyed by CSV_HEADER names.
        self._w.writerow([row[k] for k in CSV_HEADER])
        if self.db is not None:
            self.db.execute(
                "INSERT INTO tracking VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                tuple(row[k] if row[k] != "" else None for k in CSV_HEADER))

    def close(self):
        self._fh.flush()
        self._fh.close()
        if self.db is not None:
            self.db.commit()
            self.db.close()


# ──────────────────────────────────────────────────────────────────────
# Overlay
# ──────────────────────────────────────────────────────────────────────

def draw_overlay(frame, calib, contour, centroid, trail, hud_lines, lost):
    out = frame.copy()
    c = calib["dish_center"]
    cv2.circle(out, (int(c[0]), int(c[1])), int(calib["dish_radius"]),
               COL_DISH, 1)
    # Trail
    pts = [p for p in trail if p is not None]
    for i in range(1, len(pts)):
        cv2.line(out, pts[i - 1], pts[i], COL_TRAIL, 1, cv2.LINE_AA)
    if contour is not None and not lost:
        cv2.drawContours(out, [contour], -1, COL_CONTOUR, 2)
    if centroid is not None and not lost:
        cv2.circle(out, (int(centroid[0]), int(centroid[1])), 4,
                   COL_CENTROID, -1)
    # HUD
    y = 28
    for line in hud_lines:
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, COL_TEXT, 2, cv2.LINE_AA)
        y += 26
    if lost:
        cv2.putText(out, "LOST", (out.shape[1] - 110, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COL_LOST, 2)
    cv2.putText(out, "q quit   r re-seed   c recalibrate   space pause",
                (12, out.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (180, 180, 180), 1, cv2.LINE_AA)
    return out


# ──────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────

WIN = "Planarian Real-Time Tracker"


def run(args):
    cap, is_live = open_source(args.camera, args.video)
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    if is_live and args.warmup > 0:
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.warmup:
            f = grab(cap)
            if f is not None:
                cv2.imshow(WIN, f)
                cv2.waitKey(1)

    calib = calibrate(cap, args.calib_frames)
    mm_per_px = calib["mm_per_px"]
    dish_mask_bool = calib["dish_mask_bool"]
    grid_baseline = calib["grid_baseline"]

    session_id = args.session_id or f"realtime_{datetime.now():%Y%m%d_%H%M%S}"
    logger = Logger(args.output_dir, session_id, mm_per_px, args.sqlite)
    print(f"\nLogging to {logger.csv_path}")
    if args.sqlite:
        print(f"SQLite:    {logger.db_path}")

    # Per-frame state threaded through detect_worm.
    last_centroid = None
    lost_count = 0
    prev_area = None
    seeded = False
    seed_streak = 0

    trail = deque(maxlen=TRAIL_LEN)
    frame_idx = 0
    start_t = time.monotonic()
    last_pos = None
    last_t = None
    frame_interval = 1.0 / args.fps if args.fps > 0 else 0.0
    paused = False

    print("\nWaiting for the worm to enter the dish (auto-seed)...")
    while True:
        loop_t0 = time.monotonic()
        if not paused:
            frame = grab(cap)
            if frame is None:
                print("\nStream ended.")
                break
            gray = to_gray01(frame)

            centroid, area, contour, confidence = detect_worm(
                gray, None, dish_mask_bool, last_centroid,
                args.min_area, args.max_area, args.roi_px,
                args.max_jump_px, lost_count, prev_area=prev_area,
                lost_threshold=LOST_THRESHOLD, grid_baseline=grid_baseline,
                detect_thresh=args.detect_thresh)

            # ── Pre-seed: wait for a stable in-dish detection ──
            if not seeded:
                if centroid is not None:
                    seed_streak += 1
                    if seed_streak >= SEED_STABLE_FRAMES:
                        seeded = True
                        last_centroid = centroid
                        prev_area = area
                        start_t = time.monotonic()  # t=0 at seed
                        print(f"Seeded at ({centroid[0]:.0f},{centroid[1]:.0f}).")
                else:
                    seed_streak = 0
                hud = ["SEEDING — worm not yet locked",
                       f"streak {seed_streak}/{SEED_STABLE_FRAMES}"]
                view = draw_overlay(frame, calib, contour, centroid,
                                    trail, hud, lost=(centroid is None))
                cv2.imshow(WIN, view)
                if _handle_keys(cv2.waitKey(1), locals()):
                    break
                continue

            # ── Tracking ──
            now = time.monotonic()
            t_s = now - start_t
            lost = centroid is None
            speed_px_s = 0.0

            if not lost:
                lost_count = 0
                if last_pos is not None and last_t is not None and now > last_t:
                    d = math.dist(centroid, last_pos)
                    speed_px_s = d / (now - last_t)
                last_pos = centroid
                last_t = now
                last_centroid = centroid
                # Smooth the running area estimate.
                prev_area = area if prev_area is None else \
                    0.8 * prev_area + 0.2 * area
                trail.append((int(centroid[0]), int(centroid[1])))
            else:
                lost_count += 1
                trail.append(None)

            speed_mm_s = speed_px_s * mm_per_px if mm_per_px else ""
            x_mm = centroid[0] * mm_per_px if (not lost and mm_per_px) else ""
            y_mm = centroid[1] * mm_per_px if (not lost and mm_per_px) else ""

            logger.write({
                "frame": frame_idx,
                "time_s": round(t_s, 3),
                "centroid_x_px": round(centroid[0], 2) if not lost else "",
                "centroid_y_px": round(centroid[1], 2) if not lost else "",
                "centroid_x_mm": round(x_mm, 4) if x_mm != "" else "",
                "centroid_y_mm": round(y_mm, 4) if y_mm != "" else "",
                "area_px": area if not lost else "",
                "speed_px_s": round(speed_px_s, 3) if not lost else "",
                "speed_mm_s": round(speed_mm_s, 4) if speed_mm_s != "" else "",
                "confidence": round(confidence, 4),
                "is_lost": int(lost),
            })

            # HUD
            elapsed = now - start_t
            fps = (frame_idx + 1) / max(1e-6, elapsed)
            if mm_per_px and not lost:
                pos_str = f"pos ({x_mm:.1f}, {y_mm:.1f}) mm"
                spd_str = f"speed {speed_mm_s:.2f} mm/s"
            elif not lost:
                pos_str = f"pos ({centroid[0]:.0f}, {centroid[1]:.0f}) px"
                spd_str = f"speed {speed_px_s:.1f} px/s"
            else:
                pos_str = "pos --"
                spd_str = "speed --"
            hud = [
                f"t {elapsed:6.1f}s   frame {frame_idx}   {fps:4.1f} fps",
                pos_str, spd_str,
                f"conf {confidence:.2f}   {'LOST ' + str(lost_count) if lost else 'tracking'}",
            ]
            view = draw_overlay(frame, calib, contour, centroid, trail,
                                hud, lost)
            cv2.imshow(WIN, view)
            frame_idx += 1

        # Rate-limit to target fps (live only; replay runs as fast as decode).
        wait_ms = 1
        if is_live and frame_interval > 0:
            elapsed = time.monotonic() - loop_t0
            wait_ms = max(1, int((frame_interval - elapsed) * 1000))
        key = cv2.waitKey(wait_ms) & 0xFF
        action = _handle_keys(key, None)
        if action == "quit":
            break
        elif action == "pause":
            paused = not paused
        elif action == "reseed":
            seeded = False
            seed_streak = 0
            last_centroid = None
            last_pos = None
            last_t = None
            trail.clear()
            print("Re-seeding...")
        elif action == "recalibrate":
            print("Recalibrating...")
            calib = calibrate(cap, args.calib_frames)
            mm_per_px = calib["mm_per_px"]
            dish_mask_bool = calib["dish_mask_bool"]
            grid_baseline = calib["grid_baseline"]
            seeded = False
            seed_streak = 0
            last_centroid = None

    logger.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. {frame_idx} frames logged to {logger.csv_path}")


def _handle_keys(key, _ctx):
    """Map a waitKey result to an action string (or None)."""
    if key in (ord('q'), 27):       # q or Esc
        return "quit"
    if key == ord(' '):
        return "pause"
    if key == ord('r'):
        return "reseed"
    if key == ord('c'):
        return "recalibrate"
    return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0,
                    help="Camera index for live capture (default: 0).")
    ap.add_argument("--video", default=None,
                    help="Replay a video file instead of the live camera "
                         "(validates the loop end-to-end).")
    ap.add_argument("--output_dir", default="../realtime_runs",
                    help="Where to write *_tracks.csv (default: ../realtime_runs).")
    ap.add_argument("--session_id", default=None,
                    help="Session name (default: realtime_<timestamp>).")
    ap.add_argument("--fps", type=float, default=10.0,
                    help="Target tracking fps for live capture (default: 10).")
    ap.add_argument("--calib_frames", type=int, default=CALIB_FRAMES,
                    help=f"Empty-dish frames for calibration (default: {CALIB_FRAMES}).")
    ap.add_argument("--warmup", type=float, default=WARMUP_SEC,
                    help=f"Camera warm-up seconds before calibration (default: {WARMUP_SEC}).")
    ap.add_argument("--sqlite", action="store_true",
                    help="Also log to a SQLite DB alongside the CSV.")
    ap.add_argument("--min_area", type=int, default=MIN_AREA)
    ap.add_argument("--max_area", type=int, default=MAX_AREA)
    ap.add_argument("--roi_px", type=int, default=ROI_PX)
    ap.add_argument("--max_jump_px", type=float, default=MAX_JUMP_PX)
    ap.add_argument("--detect_thresh", type=float, default=DETECT_THRESH)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
