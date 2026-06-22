#!/usr/bin/env python3
"""
yolo_to_signals.py — Bridge yolo_tracker output into a behavior-features npz.

behavior_features.compute_features (and the classifier/rule models on top of it)
consume a <session>_signals.npz with this per-frame schema:

  video           (N,)        clip id (str)
  native_frame    (N,) int32  frame index within the clip
  time_s          (N,) f32
  cx_px, cy_px    (N,) f32    worm centroid (pixels)
  midline         (N, mp, 2)  head->tail body curve (NaN-padded)
  body_len_px     (N,) f32
  head_angle_deg  (N,) f32    direction of the head-end segment
  lost            (N,) int8   1 = no detection that frame
  mm_per_px       scalar f32
  fps             scalar f32

yolo_tracker.track() already emits, per frame: cx, cy, head[x,y], tail[x,y],
blen (px), state. This script runs the tracker over one or more clips and writes
that npz, so the behavior model runs unchanged on the YOLO-primary localizer.

Two reconciliation points are handled here, both consequences of how the YOLO
morphology is built (a straight PCA axis through the box-constrained bg-sub
foreground), not of the behavior code:

  1. HEAD/TAIL SIGN CONSISTENCY. box_axis takes the axis from an eigenvector,
     whose sign is arbitrary per frame, so the head/tail labels can flip 180 deg
     between frames and inject spurious heading reversals. We re-align each
     frame's head to the previous frame's head (nearest endpoint) before writing.
     This removes the spurious flips; it does NOT recover the true biological
     head (no curvature/motion cue is used) — heading is therefore axis-direction
     consistent but head/tail identity is provisional.

  2. STRAIGHT AXIS vs CURVED MIDLINE. The YOLO axis is 2 points, so head_angle
     equals body heading and the head-oscillation features collapse onto
     heading_change. The midline is written as the 2 endpoints (mp=2); the
     position/speed/body-length features are unaffected. Porting a curved-midline
     extractor onto the box foreground would restore wigwag independence — left
     as a follow-up.

Usage:
  python yolo_to_signals.py --videos ../live_capture/2026-06-02_16-00-37.mkv \
      --out ../realtime_runs/white7mp_signals.npz
  python yolo_to_signals.py --glob '../live_capture/2026-06-02_16-*.mkv' --out ...
"""
import argparse
import glob as globmod
import os

import numpy as np

import yolo_tracker as yt


def _consistent_head_tail(recs):
    """Re-align head/tail so head identity is temporally stable.

    Walks frames in order; for each frame with a morphology axis, swaps head and
    tail if the tail endpoint is closer to the previous frame's head than the
    head endpoint is. Returns parallel lists (heads, tails) of [x,y] or None, and
    the number of frames whose labels were swapped (a diagnostic on how unstable
    the raw eigenvector sign was).
    """
    heads, tails = [], []
    prev_head = None
    swaps = 0
    for r in recs:
        h, t = r["head"], r["tail"]
        if h is None or t is None:
            heads.append(None)
            tails.append(None)
            continue
        h = np.asarray(h, np.float64)
        t = np.asarray(t, np.float64)
        if prev_head is not None:
            if np.hypot(*(t - prev_head)) < np.hypot(*(h - prev_head)):
                h, t = t, h
                swaps += 1
        prev_head = h
        heads.append(h.tolist())
        tails.append(t.tolist())
    return heads, tails, swaps


def video_to_signal(video, model, mm_per_px, conf, imgsz, device):
    """Run yolo_tracker over one clip -> per-frame signal columns for that clip."""
    recs, fps, _dish = yt.track(video, model, mm_per_px, conf=conf,
                                imgsz=imgsz, device=device)
    heads, tails, swaps = _consistent_head_tail(recs)
    vid = os.path.basename(video)

    cols = {k: [] for k in ("video", "native_frame", "time_s", "cx_px", "cy_px",
                            "midline", "body_len_px", "head_angle_deg", "lost")}
    for r, h, t in zip(recs, heads, tails):
        lost = 0 if r["state"] == "detected" else 1
        cols["video"].append(vid)
        cols["native_frame"].append(int(r["frame"]))
        cols["time_s"].append(r["frame"] / fps)
        cols["cx_px"].append(r["cx"])
        cols["cy_px"].append(r["cy"])
        # 2-point midline [head, tail]; NaN when no axis this frame.
        if h is not None and t is not None:
            cols["midline"].append([h, t])
            hv = np.asarray(h) - np.asarray(t)
            cols["head_angle_deg"].append(np.degrees(np.arctan2(hv[1], hv[0])))
        else:
            cols["midline"].append([[np.nan, np.nan], [np.nan, np.nan]])
            cols["head_angle_deg"].append(np.nan)
        cols["body_len_px"].append(r["blen"] if np.isfinite(r["blen"]) else np.nan)
        cols["lost"].append(lost)
    return cols, fps, swaps


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="*", default=[], help="explicit clip paths")
    ap.add_argument("--glob", default=None, help="glob of clip paths (quote it)")
    ap.add_argument("--model", default=yt.DEFAULT_MODEL)
    ap.add_argument("--mm_per_px", type=float, default=0.02657)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=os.path.join(here, "..", "realtime_runs", "white7mp_signals.npz"))
    a = ap.parse_args()

    videos = list(a.videos)
    if a.glob:
        videos += sorted(globmod.glob(a.glob))
    videos = [v for v in videos if os.path.exists(v)]
    if not videos:
        ap.error("no existing videos given (use --videos and/or --glob)")

    from ultralytics import YOLO
    model = YOLO(a.model)   # load once, reuse across clips

    merged = {k: [] for k in ("video", "native_frame", "time_s", "cx_px", "cy_px",
                              "midline", "body_len_px", "head_angle_deg", "lost")}
    fps = None
    for v in videos:
        cols, fps, swaps = video_to_signal(v, model, a.mm_per_px, a.conf, a.imgsz, a.device)
        n = len(cols["time_s"])
        nlost = int(np.sum(cols["lost"]))
        print(f"  {os.path.basename(v)}: {n} frames, {nlost} lost, "
              f"{swaps} head/tail sign corrections")
        for k in merged:
            merged[k].extend(cols[k])

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(
        a.out,
        video=np.array(merged["video"]),
        native_frame=np.array(merged["native_frame"], dtype=np.int32),
        time_s=np.array(merged["time_s"], dtype=np.float32),
        cx_px=np.array(merged["cx_px"], dtype=np.float32),
        cy_px=np.array(merged["cy_px"], dtype=np.float32),
        midline=np.array(merged["midline"], dtype=np.float32),   # (N, 2, 2)
        body_len_px=np.array(merged["body_len_px"], dtype=np.float32),
        head_angle_deg=np.array(merged["head_angle_deg"], dtype=np.float32),
        lost=np.array(merged["lost"], dtype=np.int8),
        mm_per_px=np.float32(a.mm_per_px),
        fps=np.float32(fps or 30.0),
    )
    print(f"wrote {a.out}  ({len(merged['time_s'])} frames, {len(videos)} clips)")


if __name__ == "__main__":
    main()
