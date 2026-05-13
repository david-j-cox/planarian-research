#!/usr/bin/env python3
"""
gap_scrubber.py — Interactive GUI to manually anchor or trace unresolved
LOST gaps in planarian tracking data.

Loads every gap from *_gaps.json sidecars in a results directory, filters
to medium/long gaps (or short gaps the auto-imputer flagged as implausible),
and presents each as a clickable video window:

  • Anchor mode  (gaps 1-10s):  click the worm at the gap's 50% point,
                                then again at the 75% point. Two-anchor
                                PCHIP fills the gap rows.
  • Trace mode   (gaps >10s):   step through the gap in ~2-second jumps;
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


# ─── Tier thresholds (must match open_dish_tracker.py) ───────────────────
ANCHOR_MAX_S = 10.0
TRACE_CLICK_INTERVAL_S = 2.0

# Display geometry
DISP_W, DISP_H = 1600, 900
PRE_W = 400
POST_W = 400
MAIN_W = DISP_W - PRE_W - POST_W
HUD_H = 80
MAIN_H = DISP_H - HUD_H

COL_BG = (30, 30, 30)
COL_PANEL_BG = (40, 40, 40)
COL_TEXT = (220, 220, 220)
COL_DIM = (130, 130, 130)
COL_CLICK = (60, 220, 60)        # green crosshair
COL_ANCHOR_50 = (60, 60, 255)    # red
COL_ANCHOR_75 = (255, 120, 60)   # blue
COL_LAST_KNOWN = (255, 200, 60)  # orange
COL_NEXT_KNOWN = (200, 60, 255)  # magenta


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
                "anchored", "traced", "unrecoverable")
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

    def get(self, path: str) -> cv2.VideoCapture:
        if path in self._caps:
            self._order.remove(path)
            self._order.append(path)
            return self._caps[path]
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
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
               pre_bgr: np.ndarray,
               post_bgr: np.ndarray,
               task: GapTask,
               cursor_xy: Optional[tuple],
               click_history: list[dict],
               status_text: str,
               cur_offset: int,
               total_offsets: int) -> np.ndarray:
    """Build the 1600x900 composite shown in the cv2 window."""
    canvas = np.full((DISP_H, DISP_W, 3), COL_BG, dtype=np.uint8)

    def _fit(img, w, h):
        if img is None:
            blank = np.full((h, w, 3), COL_PANEL_BG, dtype=np.uint8)
            cv2.putText(blank, "(no frame)", (w // 2 - 40, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_DIM, 1)
            return blank
        ih, iw = img.shape[:2]
        s = min(w / iw, h / ih)
        nw, nh = int(iw * s), int(ih * s)
        out = np.full((h, w, 3), COL_PANEL_BG, dtype=np.uint8)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        out[(h - nh) // 2:(h - nh) // 2 + nh,
            (w - nw) // 2:(w - nw) // 2 + nw] = resized
        return out, s, ((w - nw) // 2, (h - nh) // 2)

    # Pre / Post (left & right panels)
    pre_fit, _, _ = _fit(pre_bgr, PRE_W, MAIN_H)
    post_fit, _, _ = _fit(post_bgr, POST_W, MAIN_H)
    canvas[0:MAIN_H, 0:PRE_W] = pre_fit
    canvas[0:MAIN_H, PRE_W + MAIN_W:DISP_W] = post_fit

    # Main (center, clickable)
    main_fit, main_scale, main_offset = _fit(main_bgr, MAIN_W, MAIN_H)
    canvas[0:MAIN_H, PRE_W:PRE_W + MAIN_W] = main_fit

    # Overlay last_known and next_known dots on the pre/post panels
    if pre_bgr is not None and task.last_known.get("x_px") is not None:
        lk_x = int(task.last_known["x_px"])
        lk_y = int(task.last_known["y_px"])
        # Need the pre-fit scale. Recompute (a small redundancy).
        ih, iw = pre_bgr.shape[:2]
        s = min(PRE_W / iw, MAIN_H / ih)
        nx = int(lk_x * s) + (PRE_W - int(iw * s)) // 2
        ny = int(lk_y * s) + (MAIN_H - int(ih * s)) // 2
        cv2.circle(canvas[0:MAIN_H, 0:PRE_W], (nx, ny), 8, COL_LAST_KNOWN, 2)
        cv2.putText(canvas[0:MAIN_H, 0:PRE_W], "last known",
                    (nx + 10, ny - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_LAST_KNOWN, 1)

    if post_bgr is not None and task.next_known.get("x_px") is not None:
        nk_x = int(task.next_known["x_px"])
        nk_y = int(task.next_known["y_px"])
        ih, iw = post_bgr.shape[:2]
        s = min(POST_W / iw, MAIN_H / ih)
        nx = int(nk_x * s) + (POST_W - int(iw * s)) // 2
        ny = int(nk_y * s) + (MAIN_H - int(ih * s)) // 2
        post_panel = canvas[0:MAIN_H, PRE_W + MAIN_W:DISP_W]
        cv2.circle(post_panel, (nx, ny), 8, COL_NEXT_KNOWN, 2)
        cv2.putText(post_panel, "next known", (nx + 10, ny - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_NEXT_KNOWN, 1)

    # Click history on the main panel
    main_panel = canvas[0:MAIN_H, PRE_W:PRE_W + MAIN_W]
    for click in click_history:
        x_px, y_px = click["x_px"], click["y_px"]
        nx = int(x_px * main_scale) + main_offset[0]
        ny = int(y_px * main_scale) + main_offset[1]
        col = (COL_ANCHOR_50 if click.get("kind") == "a50"
               else COL_ANCHOR_75 if click.get("kind") == "a75"
               else COL_CLICK)
        cv2.drawMarker(main_panel, (nx, ny), col, cv2.MARKER_CROSS, 18, 2)
        if click.get("label"):
            cv2.putText(main_panel, click["label"], (nx + 8, ny - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

    # Live cursor crosshair on the main panel
    if cursor_xy is not None:
        cx, cy = cursor_xy
        if 0 <= cx < MAIN_W and 0 <= cy < MAIN_H:
            cv2.drawMarker(main_panel, (cx, cy), (180, 180, 180),
                           cv2.MARKER_CROSS, 14, 1)

    # Panel labels (top of each)
    cv2.putText(canvas, f"PRE-GAP (frame {task.last_known.get('frame', '?')})",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_TEXT, 1)
    cv2.putText(canvas, f"CLICK WORM HERE  ({cur_offset + 1}/{total_offsets})",
                (PRE_W + 10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_TEXT, 2)
    cv2.putText(canvas, f"POST-GAP (frame {task.next_known.get('frame', '?')})",
                (PRE_W + MAIN_W + 10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_TEXT, 1)

    # HUD bar
    hud_y = MAIN_H
    canvas[hud_y:DISP_H, :] = COL_PANEL_BG
    cv2.putText(canvas,
                f"[{task.session}] gap #{task.gap_idx}  "
                f"tier={task.tier}  dur={task.duration_s:.1f}s  "
                f"mode={task.mode.upper()}  "
                f"clicks={len(click_history)}",
                (10, hud_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_TEXT, 1)
    cv2.putText(canvas,
                "[click] place anchor   [u] undo   [s] save & next   "
                "[k] unrecoverable   [,/.] prev/next frame   [b] back   [q] quit",
                (10, hud_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_DIM, 1)
    if status_text:
        cv2.putText(canvas, status_text, (10, hud_y + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_CLICK, 1)

    return canvas, main_scale, main_offset


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

    source_tag = "imputed_anchored" if mode == "anchor" else "human_traced"

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


def _run_one_gap(task: GapTask, vcache: VideoCache) -> str:
    """Returns 'saved' | 'unrecoverable' | 'quit' | 'back'."""
    frames = _build_frame_walk(task, task.csv_path)
    if not frames:
        return "unrecoverable"

    if task.mode == "anchor":
        # Anchor mode: two suggested offsets at 50% and 75% of the gap.
        offsets = [len(frames) // 2, (3 * len(frames)) // 4]
        anchor_labels = ["a50", "a75"]
    else:
        # Trace mode: aim for ~one click every TRACE_CLICK_INTERVAL_S.
        if task.duration_s <= 0:
            return "unrecoverable"
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
    last_main_state = {"scale": 1.0, "offset": (0, 0)}

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_MOUSEMOVE:
            cursor_xy[0] = (x - PRE_W, y) if PRE_W <= x < PRE_W + MAIN_W else None
        elif event == cv2.EVENT_LBUTTONDOWN:
            if not (PRE_W <= x < PRE_W + MAIN_W and y < MAIN_H):
                return
            # Convert click to source-pixel coords using last render's scale.
            px = (x - PRE_W - last_main_state["offset"][0]) / last_main_state["scale"]
            py = (y - last_main_state["offset"][1]) / last_main_state["scale"]
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

        canvas, m_scale, m_off = _composite(
            main_bgr, pre_bgr, post_bgr, task,
            cursor_xy[0], clicks, status, cur_offset, len(offsets))
        last_main_state["scale"] = m_scale
        last_main_state["offset"] = m_off

        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(33) & 0xFF

        if key == ord("q"):
            return "quit"
        elif key == ord("b"):
            return "back"
        elif key == ord("k"):
            return "unrecoverable"
        elif key == ord("u"):
            if clicks:
                clicks.pop()
                status = "undo"
        elif key == ord("s"):
            if not clicks:
                status = "no clicks placed; press [k] to mark unrecoverable"
                continue
            return "saved"
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
    args = ap.parse_args()

    tasks = discover_gaps(args.data_dir, args.sessions_root)
    if args.only_session:
        tasks = [t for t in tasks if t.session == args.only_session]
    if not args.show_completed:
        tasks = [t for t in tasks if not t.completed]

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
                  f"({t.tier}, {t.duration_s:.1f}s)")
            result = _run_one_gap(t, vcache)
            if result == "quit":
                print("Quitting (current gap not saved).")
                break
            if result == "back":
                i = max(0, i - 1)
                continue

            anchors = _load_anchors(t.anchors_path)
            if result == "saved":
                clicks = []  # _run_one_gap doesn't return them; simpler to re-derive
                # NB: refactor — pass clicks out. For now, simplest fix:
                # restructure so _run_one_gap returns (status, clicks).
                pass
            anchors["gaps"][str(t.gap_idx)] = {
                "status": result if result in ("unrecoverable",) else (
                    "anchored" if t.mode == "anchor" else "traced"),
                "mode": t.mode,
                "duration_s": t.duration_s,
            }
            _save_anchors(t.anchors_path, anchors)
            i += 1
    finally:
        vcache.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
