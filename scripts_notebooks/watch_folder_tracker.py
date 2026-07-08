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
from collections import deque
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
    extract_midline,
    _HAS_MIDLINE,
)
from sessions import list_session_clips  # noqa: E402

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
    "video_file", "frame", "native_frame", "time_s",
    "centroid_x_px", "centroid_y_px", "centroid_x_mm", "centroid_y_mm",
    "area_px", "speed_px_s", "speed_mm_s", "confidence", "is_lost",
    "body_length_mm",
]
# "frame" is the cumulative (session-wide) index; "native_frame" is the index
# WITHIN this clip (0-based), which is the correct key for joining to the source
# video or to human labels (label_setup.py stores per-clip frame numbers).


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
        self.trail = deque(maxlen=90)  # recent centroids for the overlay trail
        self.prev_midline = None   # for head/tail temporal consistency
        self.prev_gray = None      # previous frame's channel image, for motion
        # Per-frame behavior signals, accumulated across all clips of a session
        # and saved as <session>_signals.npz for the behavior feature extractor.
        self.sig = {"video": [], "native_frame": [], "time_s": [],
                    "cx_px": [], "cy_px": [], "midline": [], "body_len_px": [],
                    "head_angle_deg": [], "lost": []}


# Overlay colors (BGR)
OV_CONTOUR = (0, 180, 0)
OV_CENTROID = (0, 165, 255)   # orange centroid
OV_TRAIL = (255, 200, 0)
OV_DISH = (120, 120, 200)
OV_TEXT = (255, 255, 255)
OV_LOST = (0, 0, 255)
OV_MIDLINE = (0, 255, 255)    # yellow body axis
OV_HEAD = (0, 0, 255)         # red head
OV_TAIL = (255, 0, 0)         # blue tail


def draw_overlay(frame, calib, contour, centroid, midline, trail, lost,
                 hud_lines):
    """Draw detection overlay on a BGR frame copy and return it.

    midline: ordered Nx2 head->tail points (or None). Drawn as the body axis
    with a red head dot and blue tail dot so head/torso/tail motion is visible.
    """
    out = frame.copy()
    c = calib["dish_center"]
    cv2.circle(out, (int(c[0]), int(c[1])), int(calib["dish_radius_px"]),
               OV_DISH, 2)
    pts = [p for p in trail if p is not None]
    for i in range(1, len(pts)):
        cv2.line(out, pts[i - 1], pts[i], OV_TRAIL, 2, cv2.LINE_AA)
    if contour is not None and not lost:
        cv2.drawContours(out, [contour], -1, OV_CONTOUR, 2)
    if midline is not None and not lost and len(midline) >= 2:
        ml = midline.astype(np.int32)
        for i in range(len(ml) - 1):
            cv2.line(out, tuple(ml[i]), tuple(ml[i + 1]), OV_MIDLINE, 3,
                     cv2.LINE_AA)
        cv2.circle(out, tuple(ml[0]), 9, OV_HEAD, -1, cv2.LINE_AA)   # head
        cv2.circle(out, tuple(ml[-1]), 7, OV_TAIL, -1, cv2.LINE_AA)  # tail
    elif centroid is not None and not lost:
        cv2.circle(out, (int(centroid[0]), int(centroid[1])), 6,
                   OV_CENTROID, -1)
    y = 40
    for line in hud_lines:
        cv2.putText(out, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 0), 4, cv2.LINE_AA)        # black outline
        cv2.putText(out, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    OV_TEXT, 2, cv2.LINE_AA)
        y += 42
    if lost:
        cv2.putText(out, "LOST", (out.shape[1] - 180, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, OV_LOST, 3)
    return out


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

    # Midline needed for the overlay's body axis, the behavior signals npz, or if
    # explicitly requested. On by default unless --no_signals (behavior pipeline
    # needs the per-frame midline/head-angle/body-length).
    want_midline = (getattr(args, "save_overlay", False)
                    or getattr(args, "midline", False)
                    or not getattr(args, "no_signals", False)) and _HAS_MIDLINE

    # Optional overlay video writer (one per clip).
    ov_writer = None
    if getattr(args, "save_overlay", False):
        w, h = calib["frame_size"]
        ov_path = os.path.join(args.output_dir,
                               os.path.splitext(name)[0] + "_overlay.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps = fps / max(1, args.frame_skip)
        ov_writer = cv2.VideoWriter(ov_path, fourcc, out_fps, (w, h))

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

        # Frame-to-frame motion: |current - previous| on the channel image.
        # The worm moves and lights this up at its body; the static dish rim and
        # grid intersections stay dark, so the detector can demote them. None on
        # the first frame (and harmless then — selection falls back to score).
        motion_map = None
        if state.prev_gray is not None and not args.no_motion:
            motion_map = np.abs(gray - state.prev_gray)
        state.prev_gray = gray

        centroid, area, contour, confidence = detect_worm(
            gray, None, dish_mask_bool, state.last_centroid,
            args.min_area, args.max_area, args.roi_px, args.max_jump_px,
            state.lost_count, prev_area=state.prev_area,
            lost_threshold=LOST_THRESHOLD, motion_map=motion_map,
            grid_baseline=grid_baseline, detect_thresh=args.detect_thresh)

        t_s = state.session_t
        lost = centroid is None

        # Midline (head->tail body axis) for head/torso/tail behaviors.
        midline = None
        body_len_px = 0.0
        head_angle_deg = float("nan")
        if want_midline and not lost and contour is not None:
            midline, _curv, body_len_px = extract_midline(
                contour, state.prev_midline, args.midline_points, frame.shape)
            if midline is not None:
                state.prev_midline = midline
                # Head direction: the head-end segment (pt0 -> pt1), in degrees.
                # Wigwagging shows up as oscillation in this signal frame-to-frame.
                if len(midline) >= 2:
                    hv = midline[0] - midline[1]
                    head_angle_deg = math.degrees(math.atan2(hv[1], hv[0]))

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
            state.trail.append((int(centroid[0]), int(centroid[1])))
        else:
            state.lost_count += 1
            state.trail.append(None)

        speed_mm_s = speed_px_s * mm_per_px if mm_per_px else ""
        x_mm = centroid[0] * mm_per_px if (not lost and mm_per_px) else ""
        y_mm = centroid[1] * mm_per_px if (not lost and mm_per_px) else ""

        # Accumulate per-frame behavior signals (for <session>_signals.npz).
        if not getattr(args, "no_signals", False):
            s = state.sig
            s["video"].append(name)
            s["native_frame"].append(local_idx - 1)
            s["time_s"].append(round(t_s, 3))
            s["cx_px"].append(centroid[0] if not lost else float("nan"))
            s["cy_px"].append(centroid[1] if not lost else float("nan"))
            s["body_len_px"].append(body_len_px if not lost else float("nan"))
            s["head_angle_deg"].append(head_angle_deg)
            s["lost"].append(int(lost))
            # Midline padded to midline_points x 2; NaN when unavailable.
            mp = args.midline_points
            ml_row = np.full((mp, 2), np.nan, np.float32)
            if midline is not None and len(midline):
                k = min(len(midline), mp)
                ml_row[:k] = midline[:k]
            s["midline"].append(ml_row)

        writer.writerow([
            name,
            state.global_frame,
            local_idx - 1,          # native (per-clip) frame index just read
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
            round(body_len_px * mm_per_px, 4) if (body_len_px and mm_per_px)
            else "",
        ])

        if ov_writer is not None:
            if mm_per_px and not lost:
                pos = f"({x_mm:.1f}, {y_mm:.1f}) mm"
                spd = f"{speed_mm_s:.2f} mm/s"
            elif not lost:
                pos = f"({centroid[0]:.0f}, {centroid[1]:.0f}) px"
                spd = f"{speed_px_s:.1f} px/s"
            else:
                pos, spd = "--", "--"
            hud = [f"t {state.session_t:6.1f}s   frame {state.global_frame}",
                   f"pos {pos}", f"speed {spd}",
                   ("LOST " + str(state.lost_count)) if lost
                   else f"conf {confidence:.2f}"]
            ov_writer.write(draw_overlay(frame, calib, contour, centroid,
                                         midline, state.trail, lost, hud))

        state.global_frame += 1
        state.session_t += dt
        n_frames += 1

    cap.release()
    if ov_writer is not None:
        ov_writer.release()
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
    session_id = (args.session_id or args.session
                  or f"live_{datetime.now():%Y%m%d_%H%M%S}")

    # When a recording session is named, work off only that session's clips.
    def session_videos(d):
        if args.session:
            return list_session_clips(d, args.session)
        return list_videos(d)
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

        # Human labels override auto dish + scale (the rig is fixed and the
        # user clicked the true dish/grid — no reason to trust auto-detection).
        if args.labels and os.path.exists(args.labels):
            with open(args.labels) as f:
                lab = json.load(f)
            if "dish_center" in lab and "dish_radius_px" in lab:
                calib["dish_center"] = lab["dish_center"]
                calib["dish_radius_px"] = lab["dish_radius_px"]
                print(f"  [labels] dish center={lab['dish_center']} "
                      f"r={lab['dish_radius_px']:.0f}px (human)")
            if lab.get("mm_per_px"):
                calib["mm_per_px"] = lab["mm_per_px"]
                print(f"  [labels] mm_per_px={lab['mm_per_px']:.6f} (human)")

        # Resolution-aware area bounds. A planarian is a physical size (~0.5-25
        # mm^2), but the px area depends on the rig's resolution/zoom. The old
        # px defaults (80-3000) assumed a ~1MP frame; on the 7MP white rig the
        # worm is 4k-24k px and was being discarded by max_area=3000. So when we
        # know mm/px and the user didn't override the px caps, derive them from
        # mm^2. This makes the tracker resolution-independent.
        mmpp = calib.get("mm_per_px")
        if mmpp and not args.area_px_explicit:
            px_per_mm2 = 1.0 / (mmpp ** 2)
            args.min_area = int(args.min_area_mm2 * px_per_mm2)
            args.max_area = int(args.max_area_mm2 * px_per_mm2)
            print(f"  area bounds from scale: {args.min_area_mm2}-"
                  f"{args.max_area_mm2} mm^2 -> {args.min_area}-{args.max_area} px")

        w, h = calib["frame_size"]
        # Shrink the mask inward so the bright dish RIM and the grid lines that
        # ride along it are excluded — on white footage the rim is a strong dark
        # ring that the detector otherwise locks onto instead of the worm.
        eff_radius = calib["dish_radius_px"] * (1.0 - args.dish_margin)
        if args.dish_margin:
            print(f"  dish mask shrunk {args.dish_margin*100:.0f}% "
                  f"({calib['dish_radius_px']:.0f} -> {eff_radius:.0f}px) to drop the rim")
        dish_mask = circle_mask((h, w), tuple(calib["dish_center"]),
                                eff_radius)
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

    def save_signals():
        """Write per-frame behavior signals to <session>_signals.npz."""
        s = state.sig
        if getattr(args, "no_signals", False) or not s["time_s"]:
            return
        sig_path = os.path.join(args.output_dir, f"{session_id}_signals.npz")
        np.savez_compressed(
            sig_path,
            video=np.array(s["video"]),
            native_frame=np.array(s["native_frame"], dtype=np.int32),
            time_s=np.array(s["time_s"], dtype=np.float32),
            cx_px=np.array(s["cx_px"], dtype=np.float32),
            cy_px=np.array(s["cy_px"], dtype=np.float32),
            midline=np.array(s["midline"], dtype=np.float32),   # (N, mp, 2)
            body_len_px=np.array(s["body_len_px"], dtype=np.float32),
            head_angle_deg=np.array(s["head_angle_deg"], dtype=np.float32),
            lost=np.array(s["lost"], dtype=np.int8),
            mm_per_px=np.float32(calib["mm_per_px"] or 0.0),
            fps=np.float32(args.fps),
        )
        print(f"Behavior signals: {sig_path} ({len(s['time_s'])} frames)")

    print(f"Watching {os.path.abspath(args.watch_dir)}")
    print(f"Rolling CSV: {csv_path}")
    print(f"Poll every {args.poll}s; settle {args.settle}s. Ctrl-C to stop.\n")

    if args.process_existing:
        for p in session_videos(args.watch_dir):
            handle(p)
        if args.once:
            csv_fh.close()
            save_signals()
            print(f"Done (--once). Rolling CSV at {csv_path} "
                  f"({len(processed)} clips processed).")
            return

    try:
        while True:
            for p in session_videos(args.watch_dir):
                handle(p)
            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        csv_fh.close()
        save_signals()
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
                    help="Session name (default: the --session id, e.g. 'S2', "
                         "else live_<timestamp>).")
    ap.add_argument("--session", default=None,
                    help="Restrict to one recording session (S1, S2, ... or "
                         "'all'). Clips in --watch_dir are grouped by capture-"
                         "time gaps; only this session's clips are processed.")
    ap.add_argument("--process_existing", action="store_true",
                    help="Process clips already in the folder before watching.")
    ap.add_argument("--once", action="store_true",
                    help="With --process_existing, exit after the existing "
                         "clips are done instead of watching for new ones "
                         "(use for offline batch runs over recorded sessions).")
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
    ap.add_argument("--save_overlay", action="store_true",
                    help="Write a *_overlay.mp4 per clip with the detection "
                         "drawn on it (worm outline, midline head->tail axis, "
                         "trail, dish, HUD). Slower — for piloting.")
    ap.add_argument("--labels", default=None,
                    help="Path to a *_labels.json from label_setup.py. Its "
                         "human-clicked dish center/radius and mm/px override "
                         "auto-detection (the rig is fixed).")
    ap.add_argument("--midline", action="store_true",
                    help="Extract the head->tail midline and log body_length_mm "
                         "even without overlay (for head/torso/tail behaviors).")
    ap.add_argument("--midline_points", type=int, default=20,
                    help="Max midline sample points (default: 20).")
    ap.add_argument("--channel", choices=CHANNELS, default="auto",
                    help="Detection channel/transform. 'auto' (default) picks "
                         "the highest-contrast channel from the calibration "
                         "frames — robust to the rig's color cast.")
    ap.add_argument("--min_area", type=int, default=None,
                    help="Min blob area in PIXELS. Overrides --min_area_mm2. "
                         "Default: derived from mm/px (see --min_area_mm2).")
    ap.add_argument("--max_area", type=int, default=None,
                    help="Max blob area in PIXELS. Overrides --max_area_mm2.")
    ap.add_argument("--min_area_mm2", type=float, default=0.5,
                    help="Min worm area in mm^2 (default 0.5). Converted to px "
                         "via the calibration scale so the tracker is "
                         "resolution-independent.")
    ap.add_argument("--max_area_mm2", type=float, default=30.0,
                    help="Max worm area in mm^2 (default 30). A planarian is a "
                         "few mm^2; the generous cap allows shadow/stretch.")
    ap.add_argument("--roi_px", type=int, default=ROI_PX)
    ap.add_argument("--max_jump_px", type=float, default=MAX_JUMP_PX)
    ap.add_argument("--detect_thresh", type=float, default=DETECT_THRESH)
    ap.add_argument("--dish_margin", type=float, default=0.0,
                    help="Fraction of the dish radius to exclude at the rim "
                         "(default 0 = full dish). Testing showed the rim is not "
                         "the main false-positive source on the white rig; the "
                         "motion bonus handles rim/grid blobs instead.")
    ap.add_argument("--no_motion", action="store_true",
                    help="Disable the frame-to-frame motion bonus that demotes "
                         "static rim/grid blobs (for A/B comparison).")
    ap.add_argument("--no_signals", action="store_true",
                    help="Skip per-frame behavior signals + <session>_signals.npz "
                         "(midline/head-angle/body-length). On by default; the "
                         "behavior pipeline needs them.")
    args = ap.parse_args()
    # Did the user pin the px area caps explicitly? If so, honor them and skip
    # the mm^2->px derivation. Otherwise fall back to the legacy px constants
    # for runs with no calibration scale.
    args.area_px_explicit = (args.min_area is not None or args.max_area is not None)
    if args.min_area is None:
        args.min_area = MIN_AREA
    if args.max_area is None:
        args.max_area = MAX_AREA
    run(args)


if __name__ == "__main__":
    main()
