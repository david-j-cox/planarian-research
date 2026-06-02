#!/usr/bin/env python3
"""
watch_folder_tracker.py — Near-real-time tracking of a watch folder.

Designed for the OBS capture pipeline: OBS records the AmLite microscope view
as one-minute MKV chunks into a folder; this script watches that folder and
processes each clip seconds after it lands, appending to a single rolling CSV.

Flow:
  1. Watch --watch_dir for new video files (mkv/mp4).
  2. Calibrate ONCE on the first clip (auto_detect_dish + auto_grid_calibration),
     save the calibration JSON, and reuse it for every later clip — the rig is
     assumed fixed (camera + dish don't move).
  3. For each new clip, run the per-frame detect_worm loop (same code path as
     realtime_tracker.py / open_dish_tracker.py), threading tracker state
     (last_centroid, lost_count, prev_area) ACROSS clips so movement
     continuity is preserved at clip boundaries.
  4. APPEND one row per frame to a single rolling <session>_tracks.csv with a
     continuous time_s clock. Raw videos are KEPT (not deleted).

A clip is considered "done writing" once its size is stable for --settle
seconds (OBS is no longer appending to it).

CSV schema matches the offline tracker closely enough to feed straight into
session_analysis.py / habituation_analysis.py (read with skiprows=2).

Usage:
  cd scripts_notebooks
  source ../venv/bin/activate
  python watch_folder_tracker.py \
      --watch_dir "../live_capture" \
      --output_dir "../realtime_runs" \
      --session_id live_2026xxxx

  # Re-process everything already in the folder, then keep watching:
  python watch_folder_tracker.py --watch_dir ../live_capture --process_existing

Stop with Ctrl-C. Safe to restart: already-processed clips are skipped.

Python 3.9+. Requires: opencv-python, numpy (same as the tracker).
"""

import os
import sys
import csv
import time
import math
import glob
import json
import argparse
from datetime import datetime

import cv2
import numpy as np

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

# Defaults match open_dish_tracker.py.
MIN_AREA = 80
MAX_AREA = 3000
ROI_PX = 120
MAX_JUMP_PX = 100
DETECT_THRESH = 0.05
LOST_THRESHOLD = 30

VIDEO_EXTS = (".mkv", ".mp4", ".avi", ".mov")

# Channel/transform options for worm detection. Grayscale (luminance) is the
# offline tracker's default, but it weights green ~0.59 and blue ~0.07 — a poor
# projection when the rig has a strong color cast. Measuring per-channel
# worm-vs-background contrast and tracking on the best channel can be several×
# better than luminance. 'auto' picks the highest-variance channel from the
# calibration frames.
CHANNELS = ("auto", "gray", "blue", "green", "red", "lab_l", "lab_a", "lab_b",
            "colordist")

# Background color (BGR) for the 'colordist' channel, set at calibration time
# from the worm-free / median frame. Module-level so frame_to_channel can read
# it without threading it through every call.
_BG_COLOR_BGR = None


def frame_to_channel(bgr, channel):
    """BGR uint8 frame -> float32 [0,1] single channel per the chosen transform.

    All outputs are normalized so a worm reads as DARKER than background (the
    grid-baseline subtraction expects baseline - frame > 0 on the worm), matching
    to_gray01's polarity. For channels where the worm is brighter (e.g. blue
    under backlight), we invert so the convention holds.
    """
    if channel == "gray":
        return to_gray01(bgr)
    if channel == "colordist":
        # Distance (per pixel) from the background color, in [0,1]. The worm,
        # which differs from the white/grid background in ALL channels, lights
        # up strongly. Inverted so worm reads DARKER (baseline-subtraction
        # convention): closer-to-bg = bright, far-from-bg (worm) = dark.
        bg = _BG_COLOR_BGR if _BG_COLOR_BGR is not None \
            else np.array([200., 200., 200.], np.float32)
        f = bgr.astype(np.float32)
        dist = np.sqrt(((f - bg) ** 2).sum(axis=2))   # 0..~441
        d01 = np.clip(dist / 441.673, 0, 1)           # normalize to [0,1]
        return 1.0 - d01                              # worm -> dark
    b, g, r = cv2.split(bgr.astype(np.float32) / 255.0)
    if channel == "blue":
        ch = b
    elif channel == "green":
        ch = g
    elif channel == "red":
        ch = r
    else:  # lab_*
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
        L, A, B = cv2.split(lab)
        ch = {"lab_l": L, "lab_a": A, "lab_b": B}[channel]
    return ch


def set_bg_color(frames_bgr):
    """Set the module-level background color (median over frames) used by the
    'colordist' channel."""
    global _BG_COLOR_BGR
    med = np.median(np.stack(frames_bgr), axis=0)
    _BG_COLOR_BGR = np.median(med.reshape(-1, 3), axis=0).astype(np.float32)


def pick_best_channel(frames_bgr):
    """Choose the channel whose calibration-frame median has the most spatial
    contrast (std inside the frame). Returns a channel name from CHANNELS.

    Also sets the background color so 'colordist' is scored fairly.
    """
    set_bg_color(frames_bgr)
    med = np.median(np.stack(frames_bgr), axis=0).astype(np.uint8)
    scores = {}
    for ch in ("gray", "blue", "green", "red", "lab_l", "lab_a", "lab_b",
               "colordist"):
        scores[ch] = float(frame_to_channel(med, ch).std())
    best = max(scores, key=scores.get)
    print("  Channel contrast (std): "
          + "  ".join(f"{k}={v:.3f}" for k, v in scores.items()))
    print(f"  -> tracking on '{best}' channel")
    return best

CSV_HEADER = [
    "video_file", "frame", "time_s",
    "centroid_x_px", "centroid_y_px", "centroid_x_mm", "centroid_y_mm",
    "area_px", "speed_px_s", "speed_mm_s", "confidence", "is_lost",
]


# ──────────────────────────────────────────────────────────────────────
# Calibration (once, on the first clip)
# ──────────────────────────────────────────────────────────────────────

def calibrate_from_clip(video_path, n_calib=30, channel="auto"):
    """Auto-detect dish + grid + best channel from a clip's median frame.

    Best results when video_path is a WORM-FREE clip (dish + water + grid, no
    worm) — then the median is a true background baseline. Degrades to
    full-frame / pixels-only if dish or grid can't be read.
    """
    cap = open_video(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 600
    idxs = np.linspace(0, max(0, total - 1), min(n_calib, total), dtype=int)
    frames_bgr = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok and f is not None:
            frames_bgr.append(f)
    cap.release()
    if not frames_bgr:
        raise RuntimeError(f"Could not read frames from {video_path}")

    # Always set the background color (needed by the 'colordist' channel).
    set_bg_color(frames_bgr)
    # Pick the detection channel (the rig's color cast makes this matter).
    chosen = pick_best_channel(frames_bgr) if channel == "auto" else channel
    print(f"  Channel: {chosen}")

    # Baseline = median of the chosen channel across calibration frames.
    chans = [frame_to_channel(f, chosen) for f in frames_bgr]
    baseline = np.median(np.stack(chans), axis=0).astype(np.float32)
    h, w = baseline.shape

    try:
        dish_center, dish_radius = auto_detect_dish(baseline)
        print(f"  Dish: center=({dish_center[0]:.0f},{dish_center[1]:.0f}) "
              f"r={dish_radius:.0f}px")
    except Exception as e:
        print(f"  Dish detection failed ({e}); using full frame.")
        dish_center = (w / 2.0, h / 2.0)
        dish_radius = 0.48 * min(h, w)

    mm_per_px = None
    spacing_px = None
    try:
        mm_per_px, spacing_px = auto_grid_calibration(
            baseline, dish_center, dish_radius)
        print(f"  Grid: {mm_per_px:.5f} mm/px (spacing {spacing_px:.1f}px). "
              f"Dish ≈ {2 * dish_radius * mm_per_px:.1f} mm across.")
    except Exception as e:
        print(f"  Grid calibration failed ({e}); reporting pixels only.")

    return {
        "channel": chosen,
        "dish_center": list(dish_center),
        "dish_radius_px": float(dish_radius),
        "mm_per_px": mm_per_px,
        "grid_spacing_px": spacing_px,
        "frame_size": [w, h],
        "grid_baseline": baseline,   # not serialized; used in-memory
    }


# ──────────────────────────────────────────────────────────────────────
# Rolling state threaded across clips
# ──────────────────────────────────────────────────────────────────────

class TrackState:
    """Tracker state carried from one clip to the next so the worm's path is
    continuous across clip boundaries (no re-seed gap each minute)."""
    def __init__(self):
        self.last_centroid = None
        self.lost_count = 0
        self.prev_area = None
        self.last_pos = None       # for speed: previous centroid
        self.last_t = None         # for speed: previous timestamp (session s)
        self.session_t = 0.0       # cumulative seconds across all clips
        self.global_frame = 0      # cumulative frame index


def process_clip(video_path, calib, dish_mask_bool, state, args, writer):
    """Run the detect loop over one clip, append rows, update state.
    Returns (n_frames, n_detected)."""
    cap = open_video(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    if fps <= 0:
        fps = args.fps
    dt = 1.0 / fps
    mm_per_px = calib["mm_per_px"]
    grid_baseline = calib["grid_baseline"]
    channel = calib.get("channel", "gray")
    name = os.path.basename(video_path)

    n_frames = 0
    n_det = 0
    local_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if args.frame_skip > 1 and (local_idx % args.frame_skip) != 0:
            local_idx += 1
            state.session_t += dt
            continue
        local_idx += 1

        gray = frame_to_channel(frame, channel)
        centroid, area, contour, confidence = detect_worm(
            gray, None, dish_mask_bool, state.last_centroid,
            args.min_area, args.max_area, args.roi_px, args.max_jump_px,
            state.lost_count, prev_area=state.prev_area,
            lost_threshold=LOST_THRESHOLD, grid_baseline=grid_baseline,
            detect_thresh=args.detect_thresh)

        t_s = state.session_t
        lost = centroid is None
        speed_px_s = 0.0
        if not lost:
            n_det += 1
            state.lost_count = 0
            if state.last_pos is not None and state.last_t is not None \
                    and t_s > state.last_t:
                d = math.dist(centroid, state.last_pos)
                speed_px_s = d / (t_s - state.last_t)
            state.last_pos = centroid
            state.last_t = t_s
            state.last_centroid = centroid
            state.prev_area = area if state.prev_area is None else \
                0.8 * state.prev_area + 0.2 * area
        else:
            state.lost_count += 1

        speed_mm_s = speed_px_s * mm_per_px if mm_per_px else ""
        x_mm = centroid[0] * mm_per_px if (not lost and mm_per_px) else ""
        y_mm = centroid[1] * mm_per_px if (not lost and mm_per_px) else ""

        writer.writerow([
            name,
            state.global_frame,
            round(t_s, 3),
            round(centroid[0], 2) if not lost else "",
            round(centroid[1], 2) if not lost else "",
            round(x_mm, 4) if x_mm != "" else "",
            round(y_mm, 4) if y_mm != "" else "",
            area if not lost else "",
            round(speed_px_s, 3) if not lost else "",
            round(speed_mm_s, 4) if speed_mm_s != "" else "",
            round(confidence, 4),
            int(lost),
        ])
        state.global_frame += 1
        state.session_t += dt
        n_frames += 1

    cap.release()
    return n_frames, n_det


# ──────────────────────────────────────────────────────────────────────
# Folder watching
# ──────────────────────────────────────────────────────────────────────

def list_videos(watch_dir):
    out = []
    for ext in VIDEO_EXTS:
        out.extend(glob.glob(os.path.join(watch_dir, f"*{ext}")))
    return sorted(out)


def is_settled(path, settle_s):
    """True if the file's size has been stable for settle_s seconds (OBS done
    writing). We check size now, sleep, check again."""
    try:
        s1 = os.path.getsize(path)
    except OSError:
        return False
    time.sleep(settle_s)
    try:
        s2 = os.path.getsize(path)
    except OSError:
        return False
    return s1 == s2 and s2 > 0


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    session_id = args.session_id or f"live_{datetime.now():%Y%m%d_%H%M%S}"
    csv_path = os.path.join(args.output_dir, f"{session_id}_tracks.csv")
    calib_path = os.path.join(args.output_dir, f"{session_id}_calibration.json")
    done_path = os.path.join(args.output_dir, f"{session_id}_processed.txt")

    # Resume support: which clips are already done.
    processed = set()
    if os.path.exists(done_path):
        with open(done_path) as f:
            processed = set(l.strip() for l in f if l.strip())

    # Open CSV in append mode; write header + metadata only if new.
    new_csv = not os.path.exists(csv_path)
    csv_fh = open(csv_path, "a", newline="")
    writer = csv.writer(csv_fh)

    calib = None
    dish_mask_bool = None
    state = TrackState()

    def ensure_calibrated(first_clip):
        nonlocal calib, dish_mask_bool
        if calib is not None:
            return
        # Calibrate from the worm-free baseline clip if given, else first clip.
        cal_src = args.baseline_clip or first_clip
        if args.baseline_clip:
            print(f"Calibrating from worm-free baseline "
                  f"{os.path.basename(cal_src)} ...")
        else:
            print(f"Calibrating from {os.path.basename(cal_src)} "
                  f"(no --baseline_clip; worm may be present) ...")
        if os.path.exists(calib_path):
            with open(calib_path) as f:
                saved = json.load(f)
            print(f"Reusing saved calibration from {calib_path}")
            tmp = calibrate_from_clip(cal_src, args.calib_frames,
                                      channel=saved.get("channel", args.channel))
            saved["grid_baseline"] = tmp["grid_baseline"]
            calib = saved
        else:
            calib = calibrate_from_clip(cal_src, args.calib_frames,
                                        channel=args.channel)
            serial = {k: v for k, v in calib.items() if k != "grid_baseline"}
            with open(calib_path, "w") as f:
                json.dump(serial, f, indent=2)
        w, h = calib["frame_size"]
        dish_mask = circle_mask((h, w), tuple(calib["dish_center"]),
                                calib["dish_radius_px"])
        dish_mask_bool = dish_mask > 0
        if new_csv:
            writer.writerow([f"# mm_per_px={calib['mm_per_px'] or ''}"])
            writer.writerow([f"# session_id={session_id} "
                             f"created={datetime.now().isoformat(timespec='seconds')}"])
            writer.writerow(CSV_HEADER)
            csv_fh.flush()

    def handle(path):
        nonlocal calib
        name = os.path.basename(path)
        if name in processed:
            return
        if not is_settled(path, args.settle):
            return  # still being written; try again next poll
        ensure_calibrated(path)
        t0 = time.monotonic()
        nf, nd = process_clip(path, calib, dish_mask_bool, state, args, writer)
        csv_fh.flush()
        with open(done_path, "a") as f:
            f.write(name + "\n")
        processed.add(name)
        pct = 100 * nd / max(1, nf)
        print(f"[{datetime.now():%H:%M:%S}] {name}: {nf} frames, "
              f"{pct:.0f}% detected, {time.monotonic()-t0:.1f}s  "
              f"(session t={state.session_t:.0f}s)")

    print(f"Watching {os.path.abspath(args.watch_dir)}")
    print(f"Rolling CSV: {csv_path}")
    print(f"Poll every {args.poll}s; settle {args.settle}s. Ctrl-C to stop.\n")

    if args.process_existing:
        for p in list_videos(args.watch_dir):
            handle(p)

    try:
        while True:
            for p in list_videos(args.watch_dir):
                handle(p)
            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        csv_fh.close()
        print(f"Done. Rolling CSV at {csv_path} "
              f"({len(processed)} clips processed).")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch_dir", required=True,
                    help="Folder OBS writes clips into.")
    ap.add_argument("--output_dir", default="../realtime_runs",
                    help="Where the rolling CSV + calibration go (default: ../realtime_runs).")
    ap.add_argument("--session_id", default=None,
                    help="Session name (default: live_<timestamp>).")
    ap.add_argument("--process_existing", action="store_true",
                    help="Process clips already in the folder before watching.")
    ap.add_argument("--poll", type=float, default=5.0,
                    help="Seconds between folder scans (default: 5).")
    ap.add_argument("--settle", type=float, default=2.0,
                    help="Seconds a clip's size must be stable before "
                         "processing, so we don't read a half-written file "
                         "(default: 2).")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="Fallback fps if a clip doesn't report one (default: 30).")
    ap.add_argument("--frame_skip", type=int, default=1,
                    help="Process every Nth frame (default: 1 = all).")
    ap.add_argument("--calib_frames", type=int, default=30)
    ap.add_argument("--baseline_clip", default=None,
                    help="Path to a WORM-FREE clip (dish+water+grid, no worm) "
                         "to calibrate dish/grid/channel and build the "
                         "background baseline. Strongly recommended.")
    ap.add_argument("--channel", choices=CHANNELS, default="auto",
                    help="Detection channel/transform. 'auto' (default) picks "
                         "the highest-contrast channel from the calibration "
                         "frames — robust to the rig's color cast.")
    ap.add_argument("--min_area", type=int, default=MIN_AREA)
    ap.add_argument("--max_area", type=int, default=MAX_AREA)
    ap.add_argument("--roi_px", type=int, default=ROI_PX)
    ap.add_argument("--max_jump_px", type=float, default=MAX_JUMP_PX)
    ap.add_argument("--detect_thresh", type=float, default=DETECT_THRESH)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
