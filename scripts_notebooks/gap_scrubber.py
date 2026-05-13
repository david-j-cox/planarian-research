#!/usr/bin/env python3
"""
gap_scrubber.py — Interactive GUI to manually anchor or trace unresolved
LOST gaps in planarian tracking data.

Loads every gap from *_gaps.json sidecars in a results directory, filters
to medium/long gaps (or short gaps the auto-imputer flagged as implausible),
and presents each as a clickable video window:

  • Bisect mode  (default):     start at the gap midpoint. Linear-interp
                                prediction is overlaid as a dashed dot.
                                If the prediction matches the actual worm
                                (within half a body length), press [y] and
                                the whole branch is filled by linear interp.
                                Otherwise click the actual worm; the sub-gap
                                is split in half and the same question is
                                asked on each side, recursively, until every
                                sub-gap is short enough or matches.
                                A straight glide costs 1 keypress; a wiggly
                                gap costs ~one click per real direction change.
  • Anchor mode  (legacy):      click the worm at the gap's 50% point,
                                then again at the 75% point. Two-anchor
                                PCHIP fills the gap rows.
  • Trace mode   (legacy):      step through the gap in ~2-second jumps;
                                click the worm at each step. Linear
                                interpolation between clicks fills rows.

Each scrubbed gap rewrites the corresponding tracks.csv rows with
source=imputed_anchored or source=human_traced, and writes the raw clicks
to *_human_anchors.json alongside the CSV so we can recompute later if
the imputation algorithm changes.

Usage:
    python gap_scrubber.py --data_dir OpenDishWork/tracker_results \\
                           --sessions_root OpenDishWork

Keyboard shortcuts in the UI:
    [space]  step forward to next frame (trace mode)
    [,]/[.]  prev/next frame (fine scrub)
    [u]      undo last click in current gap
    [s]      save & advance to next gap
    [k]      mark gap as 'unrecoverable' (no clicks possible) & advance
    [b]      back to previous gap (unsaved clicks are lost)
    [q]      quit (current gap's clicks are not saved)
"""
from __future__ import annotations
import argparse
import glob
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

# Share the tracker's cv2-vs-PyAV video opener so the scrubber can read
# the same MKV files the tracker writes its rows from. (cv2 on macOS is
# often built without ffmpeg, so it can't decode MKV directly.)
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
from open_dish_tracker import open_video as _open_video  # noqa: E402


# ─── Tier thresholds (must match open_dish_tracker.py) ───────────────────
ANCHOR_MAX_S = 10.0
TRACE_CLICK_INTERVAL_S = 2.0

# Bisection mode params
BISECT_MIN_SUBGAP_S = 1.0        # stop recursing below this duration
BISECT_TOLERANCE_MM = 2.5        # ~half a planarian body length (~5 mm)

# Display geometry — main panel is the entire canvas (no PRE/POST sidebars).
DISP_W, DISP_H = 1600, 900
PRE_W = 0
POST_W = 0
MAIN_W = DISP_W
HUD_H = 80
MAIN_H = DISP_H - HUD_H

# Zoom-crop around the predicted position so the worm is big enough to click.
# CROP_SIDE_PX is the *initial* side length in source-video pixels; the user
# can pan (arrow keys) and zoom (z/x) interactively. `f` toggles full-frame.
CROP_SIDE_PX = 500
PAN_STEP_PX = 100         # arrow-key pan in source px
ZOOM_FACTOR = 1.4         # multiplicative zoom on z/x
CROP_MIN_PX = 120         # don't zoom in past ~one body length

COL_BG = (30, 30, 30)
COL_PANEL_BG = (40, 40, 40)
COL_TEXT = (220, 220, 220)
COL_DIM = (130, 130, 130)
COL_CLICK = (60, 220, 60)        # green crosshair
COL_ANCHOR_50 = (60, 60, 255)    # red
COL_ANCHOR_75 = (255, 120, 60)   # blue
COL_LAST_KNOWN = (255, 200, 60)  # orange
COL_NEXT_KNOWN = (200, 60, 255)  # magenta
COL_PREDICTION = (60, 220, 220)  # yellow — linear-interp prediction in bisect mode


# ─── Gap & session data structures ───────────────────────────────────────

@dataclass
class GapTask:
    session: str                  # e.g. "Bubba_0004_021526"
    gap_idx: int                  # index within that session's gaps list
    tier: str                     # 'medium' | 'long'
    start_row: int
    end_row: int
    start_frame: int
    end_frame: int
    start_video: str              # basename
    end_video: str
    start_time_s: float
    end_time_s: float
    duration_s: float
    n_frames: int
    last_known: dict
    next_known: dict
    csv_path: str
    anchors_path: str             # *_human_anchors.json (per session)
    session_dir: str              # absolute path to *.mkv folder
    mm_per_px: float = 0.0        # filled later from CSV header
    completed: bool = False       # already scrubbed in a previous run
    mode: str = ""                # 'anchor' | 'trace' — set on selection


def _read_csv_meta(csv_path: str) -> dict:
    """Pull mm_per_px and dish geometry out of the CSV header lines."""
    with open(csv_path) as f:
        l1, l2 = f.readline(), f.readline()
    mm_per_px = 0.10747665
    if "mm_per_px=" in l1:
        try:
            mm_per_px = float(l1.split("mm_per_px=")[1].strip().strip('"#'))
        except ValueError:
            pass
    return {"mm_per_px": mm_per_px}


def _load_anchors(anchors_path: str) -> dict:
    if os.path.exists(anchors_path):
        with open(anchors_path) as f:
            return json.load(f)
    return {"gaps": {}}


def _save_anchors(anchors_path: str, anchors: dict) -> None:
    tmp = anchors_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(anchors, f, indent=2)
    os.replace(tmp, anchors_path)


def discover_gaps(data_dir: str, sessions_root: str) -> list[GapTask]:
    """Walk *_gaps.json files and collect every medium/long gap."""
    tasks: list[GapTask] = []
    for gaps_path in sorted(glob.glob(os.path.join(data_dir, "*_gaps.json"))):
        session = os.path.basename(gaps_path).replace("_gaps.json", "")
        csv_path = os.path.join(data_dir, f"{session}_tracks.csv")
        anchors_path = os.path.join(data_dir, f"{session}_human_anchors.json")
        if not os.path.exists(csv_path):
            print(f"  skip {session}: no tracks.csv", file=sys.stderr)
            continue

        session_dir = os.path.join(sessions_root, session)
        if not os.path.isdir(session_dir):
            print(f"  skip {session}: no session dir at {session_dir}",
                  file=sys.stderr)
            continue

        meta = _read_csv_meta(csv_path)
        anchors = _load_anchors(anchors_path)

        with open(gaps_path) as f:
            gaps_doc = json.load(f)

        for gi, g in enumerate(gaps_doc["gaps"]):
            if g["tier"] not in ("medium", "long"):
                continue
            key = str(gi)
            completed = key in anchors["gaps"] and anchors["gaps"][key].get("status") in (
                "anchored", "traced", "bisected", "unrecoverable")
            tasks.append(GapTask(
                session=session,
                gap_idx=gi,
                tier=g["tier"],
                start_row=g["start_row"],
                end_row=g["end_row"],
                start_frame=g["start_frame"] or 0,
                end_frame=g["end_frame"] or 0,
                start_video=g["start_video"],
                end_video=g["end_video"],
                start_time_s=g["start_time_s"] or 0.0,
                end_time_s=g["end_time_s"] or 0.0,
                duration_s=g["duration_s"] or 0.0,
                n_frames=g["n_frames"],
                last_known=g["last_known"],
                next_known=g["next_known"],
                csv_path=csv_path,
                anchors_path=anchors_path,
                session_dir=session_dir,
                mm_per_px=meta["mm_per_px"],
                completed=completed,
                mode="anchor" if g["tier"] == "medium" else "trace",
            ))
    return tasks


# ─── Video frame fetch ───────────────────────────────────────────────────

class VideoCache:
    """Caches one open VideoCapture per video path (LRU-ish, small)."""

    def __init__(self, max_open: int = 3):
        self._caps: dict[str, cv2.VideoCapture] = {}
        self._order: list[str] = []
        self._max = max_open

    def get(self, path: str):
        if path in self._caps:
            self._order.remove(path)
            self._order.append(path)
            return self._caps[path]
        # Uses the tracker's helper, which falls back to PyAV when cv2's
        # ffmpeg backend isn't available (common on macOS opencv-python).
        cap = _open_video(path)
        self._caps[path] = cap
        self._order.append(path)
        while len(self._order) > self._max:
            old = self._order.pop(0)
            self._caps.pop(old).release()
        return cap

    def fetch(self, video_path: str, frame_idx: int) -> Optional[np.ndarray]:
        cap = self.get(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        return frame if ok else None

    def close(self):
        for c in self._caps.values():
            c.release()
        self._caps.clear()
        self._order.clear()


# ─── Frame walking inside a gap ──────────────────────────────────────────

def _build_frame_walk(task: GapTask, csv_path: str) -> list[dict]:
    """Return the per-row metadata for every frame in the gap, ordered.

    Each entry: {row_idx, frame, time_s, video_file}. The CSV is the source
    of truth — start_row..end_row are positional indices into the CSV's
    data rows (after the 2 metadata rows).
    """
    df = pd.read_csv(csv_path, skiprows=2)
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    sub = df.iloc[task.start_row:task.end_row + 1]
    return [
        {
            "row_idx": task.start_row + i,
            "frame": int(r["frame"]),
            "time_s": float(r["time_s"]),
            "video_file": str(r["video_file"]),
        }
        for i, r in enumerate(sub.to_dict("records"))
    ]


# ─── Rendering ───────────────────────────────────────────────────────────

def _composite(main_bgr: np.ndarray,
               pre_bgr: np.ndarray,           # kept for signature compat, ignored
               post_bgr: np.ndarray,          # kept for signature compat, ignored
               task: GapTask,
               cursor_xy: Optional[tuple],
               click_history: list[dict],
               status_text: str,
               cur_offset: int,
               total_offsets: int,
               prediction_xy_px: Optional[tuple] = None,
               crop_side_px: int = CROP_SIDE_PX,
               pan_offset_px: tuple = (0, 0),
               full_frame: bool = False) -> np.ndarray:
    """Build the 1600x900 composite shown in the cv2 window.

    The full canvas is now one big zoomed crop of the main frame, centered
    on the prediction. Returns (canvas, src_per_canvas_px, crop_origin_src)
    so the caller can map clicks/cursors in canvas coords back to source
    video pixel coords:
        src_x = crop_origin_src[0] + canvas_x * src_per_canvas_px
    """
    canvas = np.full((DISP_H, DISP_W, 3), COL_BG, dtype=np.uint8)

    # Choose the source-pixel window to crop. Default: center on prediction.
    # When prediction is None (legacy modes), fall back to centering on the
    # last_known position so the worm is still in the frame.
    if main_bgr is None:
        ih = iw = 0
    else:
        ih, iw = main_bgr.shape[:2]

    if prediction_xy_px is not None:
        cx_src, cy_src = float(prediction_xy_px[0]), float(prediction_xy_px[1])
    elif task.last_known.get("x_px") is not None:
        cx_src = float(task.last_known["x_px"])
        cy_src = float(task.last_known["y_px"])
    else:
        cx_src = iw / 2.0
        cy_src = ih / 2.0

    # Apply user pan (in source pixels).
    cx_src += pan_offset_px[0]
    cy_src += pan_offset_px[1]

    # Full-frame mode overrides the crop window to the entire source image.
    if full_frame and main_bgr is not None:
        x0, y0 = 0, 0
        x1, y1 = iw, ih
        crop_side_px_eff_w = iw
        crop_side_px_eff_h = ih
    else:
        half = crop_side_px / 2.0
        x0 = int(round(cx_src - half))
        y0 = int(round(cy_src - half))
        x1 = x0 + crop_side_px
        y1 = y0 + crop_side_px
        crop_side_px_eff_w = crop_side_px
        crop_side_px_eff_h = crop_side_px

    # Clamp to image bounds, but track the un-clamped origin so click→src
    # mapping stays correct (we'll pad with COL_PANEL_BG instead of shifting).
    if main_bgr is not None:
        crop = np.full((crop_side_px_eff_h, crop_side_px_eff_w, 3),
                       COL_PANEL_BG, dtype=np.uint8)
        sx0 = max(0, x0)
        sy0 = max(0, y0)
        sx1 = min(iw, x1)
        sy1 = min(ih, y1)
        if sx1 > sx0 and sy1 > sy0:
            crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = main_bgr[sy0:sy1, sx0:sx1]
    else:
        crop = np.full((crop_side_px_eff_h, crop_side_px_eff_w, 3),
                       COL_PANEL_BG, dtype=np.uint8)
        cv2.putText(crop, "(no frame)",
                    (crop_side_px_eff_w // 2 - 60, crop_side_px_eff_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_DIM, 2)

    # Stretch the crop into the main panel area (full canvas minus HUD).
    scale = min(MAIN_W / crop_side_px_eff_w, MAIN_H / crop_side_px_eff_h)
    nw = int(crop_side_px_eff_w * scale)
    nh = int(crop_side_px_eff_h * scale)
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
    main_offset = ((MAIN_W - nw) // 2, (MAIN_H - nh) // 2)
    canvas[main_offset[1]:main_offset[1] + nh,
           main_offset[0]:main_offset[0] + nw] = resized

    main_panel = canvas[0:MAIN_H, 0:MAIN_W]

    # Helper: source (video) pixel → canvas pixel
    def src_to_canvas(sx: float, sy: float) -> tuple[int, int]:
        cx = main_offset[0] + (sx - x0) * scale
        cy = main_offset[1] + (sy - y0) * scale
        return int(round(cx)), int(round(cy))

    # Click history drawn on the main panel
    for click in click_history:
        nx, ny = src_to_canvas(click["x_px"], click["y_px"])
        col = (COL_ANCHOR_50 if click.get("kind") == "a50"
               else COL_ANCHOR_75 if click.get("kind") == "a75"
               else COL_CLICK)
        cv2.drawMarker(main_panel, (nx, ny), col, cv2.MARKER_CROSS, 24, 2)
        if click.get("label"):
            cv2.putText(main_panel, click["label"], (nx + 10, ny - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

    # Prediction overlay
    if prediction_xy_px is not None and main_bgr is not None:
        nx, ny = src_to_canvas(prediction_xy_px[0], prediction_xy_px[1])
        cv2.circle(main_panel, (nx, ny), 20, COL_PREDICTION, 2)
        cv2.circle(main_panel, (nx, ny), 3, COL_PREDICTION, -1)
        cv2.putText(main_panel, "predicted", (nx + 24, ny - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_PREDICTION, 1)

    # Live cursor crosshair
    if cursor_xy is not None:
        cx, cy = cursor_xy
        if 0 <= cx < MAIN_W and 0 <= cy < MAIN_H:
            cv2.drawMarker(main_panel, (cx, cy), (180, 180, 180),
                           cv2.MARKER_CROSS, 18, 1)

    # Title bar
    if iw:
        zoom_x = (MAIN_W / iw) / (crop_side_px_eff_w / iw)  # = MAIN_W / crop_w
        zoom_x = MAIN_W / crop_side_px_eff_w
    else:
        zoom_x = 1.0
    mode_str = "FULL FRAME" if full_frame else f"crop {crop_side_px_eff_w}px"
    cv2.putText(canvas,
                f"CLICK WORM HERE  ({cur_offset + 1}/{total_offsets})  "
                f"— {mode_str}, {zoom_x:.1f}x",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_TEXT, 2)

    # HUD bar
    hud_y = MAIN_H
    canvas[hud_y:DISP_H, :] = COL_PANEL_BG
    cv2.putText(canvas,
                f"[{task.session}] gap #{task.gap_idx}  "
                f"tier={task.tier}  dur={task.duration_s:.1f}s  "
                f"mode={task.mode.upper()}  "
                f"clicks={len(click_history)}",
                (10, hud_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_TEXT, 1)
    if task.mode == "bisect":
        hud2 = ("[y]accept  [click]override  [u]undo  [k]unrec  [b]back  [q]quit  "
                "| [WASD/arrows]pan  [z]zoom-in  [x]zoom-out  [f]full-frame  [r]recenter")
    else:
        hud2 = ("[click]anchor  [u]undo  [s]save&next  [k]unrec  "
                "[,/.]prev/next frame  [b]back  [q]quit  | "
                "[arrows]pan  [z]zoom-in  [x]zoom-out  [f]full-frame  [r]recenter")
    cv2.putText(canvas, hud2, (10, hud_y + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_DIM, 1)
    if status_text:
        cv2.putText(canvas, status_text, (10, hud_y + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_CLICK, 1)

    # Transform dict: convert canvas pixel (cx, cy) to source video pixel:
    #   src_x = crop_origin[0] + (cx - canvas_offset[0]) / scale
    transform = {
        "scale": scale,              # canvas px per crop px (≈ canvas px per src px)
        "canvas_offset": main_offset,  # where crop starts on canvas (top-left)
        "crop_origin": (x0, y0),     # top-left of crop in source video coords
    }
    return canvas, transform


# ─── Persist back to CSV ─────────────────────────────────────────────────

def _apply_anchors_to_csv(task: GapTask,
                          clicks: list[dict],
                          frames: list[dict],
                          mode: str) -> None:
    """Compute interpolation from clicks and rewrite the CSV rows in-place."""
    with open(task.csv_path) as f:
        lines = f.readlines()
    meta = lines[:2]
    df = pd.read_csv(task.csv_path, skiprows=2)

    # Build the anchor time-series: last_known + clicks + next_known.
    lk = task.last_known
    nk = task.next_known
    anchor_ts = [lk["time_s"]] + [c["time_s"] for c in clicks] + [nk["time_s"]]
    anchor_xs_mm = [lk["x_mm"]] + [c["x_mm"] for c in clicks] + [nk["x_mm"]]
    anchor_ys_mm = [lk["y_mm"]] + [c["y_mm"] for c in clicks] + [nk["y_mm"]]
    anchor_xs_px = [lk["x_px"]] + [c["x_px"] for c in clicks] + [nk["x_px"]]
    anchor_ys_px = [lk["y_px"]] + [c["y_px"] for c in clicks] + [nk["y_px"]]

    ts = np.array(anchor_ts, dtype=float)
    # Strictly increasing (clicks should already be in time order)
    order = np.argsort(ts)
    ts = ts[order]
    keep = np.concatenate([[True], np.diff(ts) > 1e-9])
    ts = ts[keep]
    xs_mm = np.array(anchor_xs_mm)[order][keep]
    ys_mm = np.array(anchor_ys_mm)[order][keep]
    xs_px = np.array(anchor_xs_px)[order][keep]
    ys_px = np.array(anchor_ys_px)[order][keep]

    # Mode-dependent interpolator. Two-anchor (medium) gets PCHIP if 3+
    # points are present; trace (long) gets linear between clicks.
    if mode == "anchor" and len(ts) >= 3:
        fx_mm = PchipInterpolator(ts, xs_mm)
        fy_mm = PchipInterpolator(ts, ys_mm)
        fx_px = PchipInterpolator(ts, xs_px)
        fy_px = PchipInterpolator(ts, ys_px)
        eval_fn = lambda t: (float(fx_mm(t)), float(fy_mm(t)),
                             float(fx_px(t)), float(fy_px(t)))
    else:
        eval_fn = lambda t: (
            float(np.interp(t, ts, xs_mm)),
            float(np.interp(t, ts, ys_mm)),
            float(np.interp(t, ts, xs_px)),
            float(np.interp(t, ts, ys_px)),
        )

    source_tag = {
        "anchor": "imputed_anchored",
        "trace": "human_traced",
        "bisect": "imputed_bisect",
    }.get(mode, "imputed_anchored")

    if "source" not in df.columns:
        df["source"] = df["centroid_x_mm"].notna().map(
            {True: "tracked", False: ""})

    for fr in frames:
        x_mm, y_mm, x_px, y_px = eval_fn(fr["time_s"])
        df.at[fr["row_idx"], "centroid_x_mm"] = x_mm
        df.at[fr["row_idx"], "centroid_y_mm"] = y_mm
        df.at[fr["row_idx"], "centroid_x_px"] = x_px
        df.at[fr["row_idx"], "centroid_y_px"] = y_px
        df.at[fr["row_idx"], "confidence"] = 0.0
        df.at[fr["row_idx"], "source"] = source_tag

    # Recompute speed for the imputed rows + the row immediately after.
    # (Simple finite-difference; small cost vs the cleanliness of the data.)
    for i in range(task.start_row, min(task.end_row + 2, len(df))):
        if i == 0:
            continue
        x1, y1 = df.at[i, "centroid_x_mm"], df.at[i, "centroid_y_mm"]
        x0, y0 = df.at[i - 1, "centroid_x_mm"], df.at[i - 1, "centroid_y_mm"]
        t1, t0 = df.at[i, "time_s"], df.at[i - 1, "time_s"]
        if pd.isna(x1) or pd.isna(x0) or pd.isna(t1) or pd.isna(t0):
            continue
        dt = float(t1) - float(t0)
        if dt <= 0:
            continue
        dist_mm = math.hypot(float(x1) - float(x0), float(y1) - float(y0))
        df.at[i, "speed_mm_s"] = dist_mm / dt
        df.at[i, "speed_px_s"] = (dist_mm / task.mm_per_px) / dt if task.mm_per_px else 0.0

    tmp = task.csv_path + ".tmp"
    with open(tmp, "w") as f:
        for line in meta:
            f.write(line if line.endswith("\n") else line + "\n")
        df.to_csv(f, index=False)
    os.replace(tmp, task.csv_path)


# ─── UI loop ─────────────────────────────────────────────────────────────

WINDOW = "Gap Scrubber"


def _run_one_gap_bisect(task: GapTask, vcache: VideoCache,
                        tolerance_mm: float = BISECT_TOLERANCE_MM,
                        min_subgap_s: float = BISECT_MIN_SUBGAP_S
                        ) -> tuple[str, list[dict], list[dict]]:
    """Bisection-based scrubber for one gap.

    Maintains a queue of sub-gaps bounded by (lo_anchor, hi_anchor) where
    each anchor has time_s + x_px/y_px + x_mm/y_mm. For each sub-gap we
    surface the midpoint frame, overlay the linear-interp prediction, and
    ask the user:
      [y]    → accept prediction; this whole sub-gap is filled by linear
               interpolation between its bounding anchors. Done.
      click  → record an anchor at the clicked position. If the click is
               within tolerance of the prediction, treat as accept. Else
               push the two halves (lo, mid) and (mid, hi) onto the queue.
      [k]    → mark this sub-gap as unrecoverable; bounding anchors still
               cover it via linear interp (best we can do).
      [b]    → return 'back' (caller goes to previous gap).
      [q]    → return 'quit' (current gap not persisted).

    Returns (status, clicks, frames) — same shape as _run_one_gap so the
    caller can pass to _apply_anchors_to_csv unchanged.
    """
    frames = _build_frame_walk(task, task.csv_path)
    if not frames:
        return "unrecoverable", [], []

    lk_path = os.path.join(task.session_dir, task.start_video)
    nk_path = os.path.join(task.session_dir, task.end_video)
    pre_bgr = vcache.fetch(lk_path, task.last_known["frame"]) \
        if task.last_known.get("frame") is not None else None
    post_bgr = vcache.fetch(nk_path, task.next_known["frame"]) \
        if task.next_known.get("frame") is not None else None

    # Initial bounding anchors come from the tracker's last_known / next_known.
    lk = task.last_known
    nk = task.next_known
    # Bisection needs both anchors to interpolate. Open-ended gaps (gap runs
    # off the end of the session) are not recoverable here — skip cleanly.
    if (lk.get("time_s") is None or nk.get("time_s") is None
            or lk.get("x_px") is None or nk.get("x_px") is None):
        print(f"  skipping gap (open-ended: no {'lk' if lk.get('time_s') is None else 'nk'})")
        return "unrecoverable", [], frames
    queue: list[tuple[dict, dict]] = [(lk, nk)]
    clicks: list[dict] = []   # human-placed anchors, in time order

    cursor_xy = [None]
    # The mouse callback maps canvas-px to source-video-px using the
    # transform returned by _composite (set after each render).
    last_transform = {"scale": 1.0, "canvas_offset": (0, 0), "crop_origin": (0, 0)}
    current = {"frame": None, "lo": None, "hi": None,
               "predicted_px": None, "predicted_mm": None}
    pending_click: list[Optional[dict]] = [None]

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_MOUSEMOVE:
            cursor_xy[0] = (x, y) if 0 <= x < MAIN_W and y < MAIN_H else None
        elif event == cv2.EVENT_LBUTTONDOWN:
            if not (0 <= x < MAIN_W and y < MAIN_H):
                return
            t = last_transform
            px = t["crop_origin"][0] + (x - t["canvas_offset"][0]) / t["scale"]
            py = t["crop_origin"][1] + (y - t["canvas_offset"][1]) / t["scale"]
            fr = current["frame"]
            if fr is None:
                return
            pending_click[0] = {
                "row_idx": fr["row_idx"],
                "frame": fr["frame"],
                "time_s": fr["time_s"],
                "video_file": fr["video_file"],
                "x_px": float(px),
                "y_px": float(py),
                "x_mm": float(px) * task.mm_per_px,
                "y_mm": float(py) * task.mm_per_px,
                "kind": "bisect",
                "label": f"#{len(clicks) + 1}",
            }

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, DISP_W, DISP_H)
    cv2.setMouseCallback(WINDOW, on_mouse)

    status = ""
    accepted_branches = 0
    split_branches = 0
    unrecoverable_branches = 0
    y_presses = 0   # require at least one explicit confirmation before saving
    # User-controlled view state. Reset each new sub-gap so panning doesn't
    # carry over to the next frame.
    view = {"pan": [0, 0], "crop": CROP_SIDE_PX, "full_frame": False}
    while queue:
        lo, hi = queue.pop()
        sub_dur = float(hi["time_s"]) - float(lo["time_s"])
        if sub_dur <= min_subgap_s:
            # Too short to subdivide further — boundary anchors are enough.
            accepted_branches += 1
            continue

        # Pick the frame closest to the midpoint time inside this sub-gap.
        t_mid = (float(lo["time_s"]) + float(hi["time_s"])) / 2.0
        sub_frames = [f for f in frames
                      if lo["time_s"] < f["time_s"] < hi["time_s"]]
        if not sub_frames:
            accepted_branches += 1
            continue
        fr = min(sub_frames, key=lambda f: abs(f["time_s"] - t_mid))

        # Linear-interp prediction at this frame's time.
        alpha = (fr["time_s"] - lo["time_s"]) / (hi["time_s"] - lo["time_s"])
        pred_x_px = lo["x_px"] + alpha * (hi["x_px"] - lo["x_px"])
        pred_y_px = lo["y_px"] + alpha * (hi["y_px"] - lo["y_px"])
        pred_x_mm = lo["x_mm"] + alpha * (hi["x_mm"] - lo["x_mm"])
        pred_y_mm = lo["y_mm"] + alpha * (hi["y_mm"] - lo["y_mm"])

        current["frame"] = fr
        current["lo"] = lo
        current["hi"] = hi
        current["predicted_px"] = (pred_x_px, pred_y_px)
        current["predicted_mm"] = (pred_x_mm, pred_y_mm)
        pending_click[0] = None
        # Reset view to default zoom on each new sub-gap.
        view["pan"] = [0, 0]
        view["crop"] = CROP_SIDE_PX
        view["full_frame"] = False

        main_path = os.path.join(task.session_dir, fr["video_file"])

        # Inner loop: wait for a decision on this sub-gap.
        while True:
            main_bgr = vcache.fetch(main_path, fr["frame"])
            depth_left = len([q for q in queue if (q[1]["time_s"] - q[0]["time_s"]) > min_subgap_s])
            total_branches = (accepted_branches + split_branches +
                              unrecoverable_branches + 1 + depth_left)
            cur_idx = accepted_branches + split_branches + unrecoverable_branches
            sub_status = (f"sub-gap {sub_dur:.1f}s "
                          f"(accepted {accepted_branches}, "
                          f"split {split_branches}, "
                          f"unrec {unrecoverable_branches}, "
                          f"queue {len(queue)})  "
                          f"{status}")

            canvas, transform = _composite(
                main_bgr, pre_bgr, post_bgr, task,
                cursor_xy[0], clicks, sub_status, cur_idx, total_branches,
                prediction_xy_px=current["predicted_px"],
                crop_side_px=view["crop"],
                pan_offset_px=tuple(view["pan"]),
                full_frame=view["full_frame"])
            last_transform.update(transform)

            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(33) & 0xFF

            if key == ord("q"):
                return "quit", clicks, frames
            if key == ord("b"):
                return "back", clicks, frames
            if key == ord("k"):
                unrecoverable_branches += 1
                status = "sub-gap marked unrecoverable"
                break
            if key == ord("u"):
                if clicks:
                    last = clicks.pop()
                    status = f"undo click @ t={last['time_s']:.2f}s"
                else:
                    status = "nothing to undo"
            if key == ord("y"):
                accepted_branches += 1
                y_presses += 1
                status = "accepted prediction"
                break
            # View controls (don't break the inner loop).
            # Arrow keys (cv2 Linux/macOS codes vary, so accept WASD too).
            if key == 81 or key == ord("a"):
                view["pan"][0] -= PAN_STEP_PX
            elif key == 83 or key == ord("d"):
                view["pan"][0] += PAN_STEP_PX
            elif key == 82 or key == ord("w"):
                view["pan"][1] -= PAN_STEP_PX
            elif key == 84 or key == ord("s"):
                view["pan"][1] += PAN_STEP_PX
            elif key == ord("z"):
                view["crop"] = max(CROP_MIN_PX, int(view["crop"] / ZOOM_FACTOR))
                view["full_frame"] = False
            elif key == ord("x"):
                view["crop"] = int(view["crop"] * ZOOM_FACTOR)
                view["full_frame"] = False
            elif key == ord("f"):
                view["full_frame"] = not view["full_frame"]
            elif key == ord("r"):
                view["pan"] = [0, 0]
                view["crop"] = CROP_SIDE_PX
                view["full_frame"] = False
            if pending_click[0] is not None:
                click = pending_click[0]
                pending_click[0] = None
                # Distance between click and prediction (mm).
                dx = click["x_mm"] - pred_x_mm
                dy = click["y_mm"] - pred_y_mm
                dist_mm = math.hypot(dx, dy)
                if dist_mm <= tolerance_mm:
                    accepted_branches += 1
                    y_presses += 1  # explicit click counts as confirmation
                    status = (f"click within tolerance "
                              f"({dist_mm:.2f}mm ≤ {tolerance_mm:.2f}mm) → accepted")
                    break
                # Outside tolerance — record click, split sub-gap.
                clicks.append(click)
                queue.append((click, hi))
                queue.append((lo, click))
                split_branches += 1
                status = (f"click {dist_mm:.2f}mm from prediction → "
                          f"split sub-gap")
                break

    # Guard: never persist a gap unless the human explicitly confirmed at
    # least one sub-gap (via [y] or a within-tolerance click). Without
    # that, we'd silently fill rows with linear interp the user never saw.
    if y_presses == 0 and not clicks:
        return "unrecoverable", clicks, frames
    if not clicks and unrecoverable_branches and not accepted_branches:
        return "unrecoverable", clicks, frames
    return "saved", clicks, frames


def _run_one_gap(task: GapTask, vcache: VideoCache
                 ) -> tuple[str, list[dict], list[dict]]:
    """Run the UI for one gap.

    Returns (status, clicks, frames):
      status:  'saved' | 'unrecoverable' | 'quit' | 'back'
      clicks:  the list of human-placed anchors (empty when not 'saved')
      frames:  the per-row metadata for every gap frame (so the caller
               can apply interpolation without re-reading the CSV)
    """
    frames = _build_frame_walk(task, task.csv_path)
    if not frames:
        return "unrecoverable", [], []

    if task.mode == "anchor":
        # Anchor mode: two suggested offsets at 50% and 75% of the gap.
        offsets = [len(frames) // 2, (3 * len(frames)) // 4]
        anchor_labels = ["a50", "a75"]
    else:
        # Trace mode: aim for ~one click every TRACE_CLICK_INTERVAL_S.
        if task.duration_s <= 0:
            return "unrecoverable", [], frames
        step = max(1, int(round(TRACE_CLICK_INTERVAL_S *
                                (len(frames) / task.duration_s))))
        offsets = list(range(0, len(frames), step))
        if offsets[-1] != len(frames) - 1:
            offsets.append(len(frames) - 1)
        anchor_labels = [None] * len(offsets)

    # Pre/post bracketing frames
    lk_path = os.path.join(task.session_dir, task.start_video)
    nk_path = os.path.join(task.session_dir, task.end_video)
    pre_bgr = vcache.fetch(lk_path, task.last_known["frame"]) \
        if task.last_known.get("frame") is not None else None
    post_bgr = vcache.fetch(nk_path, task.next_known["frame"]) \
        if task.next_known.get("frame") is not None else None

    cur_offset = 0
    cursor_xy = [None]   # mutable for closure
    clicks: list[dict] = []
    last_transform = {"scale": 1.0, "canvas_offset": (0, 0), "crop_origin": (0, 0)}

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_MOUSEMOVE:
            cursor_xy[0] = (x, y) if 0 <= x < MAIN_W and y < MAIN_H else None
        elif event == cv2.EVENT_LBUTTONDOWN:
            if not (0 <= x < MAIN_W and y < MAIN_H):
                return
            t = last_transform
            px = t["crop_origin"][0] + (x - t["canvas_offset"][0]) / t["scale"]
            py = t["crop_origin"][1] + (y - t["canvas_offset"][1]) / t["scale"]
            fr = frames[offsets[cur_offset]]
            click = {
                "row_idx": fr["row_idx"],
                "frame": fr["frame"],
                "time_s": fr["time_s"],
                "video_file": fr["video_file"],
                "x_px": float(px),
                "y_px": float(py),
                "x_mm": float(px) * task.mm_per_px,
                "y_mm": float(py) * task.mm_per_px,
                "kind": anchor_labels[cur_offset] if cur_offset < len(anchor_labels) else None,
                "label": (anchor_labels[cur_offset] or f"#{len(clicks) + 1}"),
            }
            clicks.append(click)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, DISP_W, DISP_H)
    cv2.setMouseCallback(WINDOW, on_mouse)

    status = ""
    while True:
        cur_offset = max(0, min(cur_offset, len(offsets) - 1))
        fr = frames[offsets[cur_offset]]
        main_path = os.path.join(task.session_dir, fr["video_file"])
        main_bgr = vcache.fetch(main_path, fr["frame"])

        canvas, transform = _composite(
            main_bgr, pre_bgr, post_bgr, task,
            cursor_xy[0], clicks, status, cur_offset, len(offsets))
        last_transform.update(transform)

        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(33) & 0xFF

        if key == ord("q"):
            return "quit", clicks, frames
        elif key == ord("b"):
            return "back", clicks, frames
        elif key == ord("k"):
            return "unrecoverable", clicks, frames
        elif key == ord("u"):
            if clicks:
                clicks.pop()
                status = "undo"
        elif key == ord("s"):
            if not clicks:
                status = "no clicks placed; press [k] to mark unrecoverable"
                continue
            return "saved", clicks, frames
        elif key == ord(",") or key == 81:  # left arrow alt
            cur_offset -= 1
        elif key == ord(".") or key == 83:
            cur_offset += 1
        elif key == ord(" "):
            cur_offset += 1
        # Auto-advance after click in anchor mode
        if task.mode == "anchor" and len(clicks) > cur_offset:
            cur_offset = len(clicks)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True,
                    help="Directory containing *_tracks.csv + *_gaps.json.")
    ap.add_argument("--sessions_root", required=True,
                    help="Directory containing per-session video folders.")
    ap.add_argument("--only_session", default=None,
                    help="Restrict to a single session by name.")
    ap.add_argument("--show_completed", action="store_true",
                    help="Re-show gaps already scrubbed in a prior run.")
    ap.add_argument("--mode", choices=["bisect", "legacy"], default="bisect",
                    help="bisect (default): midpoint-first with linear-interp "
                         "prediction check. legacy: fixed 2-click for medium, "
                         "every-2s for long.")
    ap.add_argument("--tolerance_mm", type=float, default=BISECT_TOLERANCE_MM,
                    help=f"Bisect: click-vs-prediction tolerance "
                         f"(default {BISECT_TOLERANCE_MM} mm).")
    ap.add_argument("--min_subgap_s", type=float, default=BISECT_MIN_SUBGAP_S,
                    help=f"Bisect: stop recursing below this sub-gap duration "
                         f"(default {BISECT_MIN_SUBGAP_S} s).")
    args = ap.parse_args()

    tasks = discover_gaps(args.data_dir, args.sessions_root)
    if args.only_session:
        tasks = [t for t in tasks if t.session == args.only_session]
    if not args.show_completed:
        tasks = [t for t in tasks if not t.completed]
    if args.mode == "bisect":
        for t in tasks:
            t.mode = "bisect"

    if not tasks:
        print("No unresolved gaps. Either all done or none discovered.")
        return

    print(f"Loaded {len(tasks)} unresolved gap(s).")
    for t in tasks[:5]:
        print(f"  {t.session}  gap#{t.gap_idx}  {t.tier}  "
              f"{t.duration_s:.1f}s  mode={t.mode}")
    if len(tasks) > 5:
        print(f"  ... + {len(tasks) - 5} more")

    vcache = VideoCache(max_open=3)
    try:
        i = 0
        while i < len(tasks):
            t = tasks[i]
            print(f"\n[{i + 1}/{len(tasks)}] {t.session} gap#{t.gap_idx} "
                  f"({t.tier}, {t.duration_s:.1f}s, mode={t.mode})")
            if t.mode == "bisect":
                status, clicks, frames = _run_one_gap_bisect(
                    t, vcache,
                    tolerance_mm=args.tolerance_mm,
                    min_subgap_s=args.min_subgap_s)
            else:
                status, clicks, frames = _run_one_gap(t, vcache)

            if status == "quit":
                print("Quitting (current gap not saved).")
                break
            if status == "back":
                i = max(0, i - 1)
                continue

            anchors = _load_anchors(t.anchors_path)
            anchor_record = {
                "mode": t.mode,
                "duration_s": t.duration_s,
                "tier": t.tier,
                "clicks": clicks,
            }
            if status == "saved":
                # Rewrite the gap's rows in the CSV from the clicks.
                try:
                    _apply_anchors_to_csv(t, clicks, frames, mode=t.mode)
                    anchor_record["status"] = {
                        "anchor": "anchored",
                        "trace": "traced",
                        "bisect": "bisected",
                    }.get(t.mode, "anchored")
                    print(f"  saved {len(clicks)} click(s) → "
                          f"{t.n_frames} row(s) imputed as "
                          f"{anchor_record['status']}.")
                except Exception as e:
                    print(f"  ERROR writing anchors to CSV: {e}")
                    anchor_record["status"] = "error"
                    anchor_record["error"] = str(e)[:200]
            elif status == "unrecoverable":
                anchor_record["status"] = "unrecoverable"
                print(f"  marked unrecoverable.")
            else:
                anchor_record["status"] = status

            anchors["gaps"][str(t.gap_idx)] = anchor_record
            _save_anchors(t.anchors_path, anchors)
            i += 1
    finally:
        vcache.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
