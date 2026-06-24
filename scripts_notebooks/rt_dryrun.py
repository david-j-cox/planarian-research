#!/usr/bin/env python3
"""
rt_dryrun.py — throughput dry run for the continuous (clips-on-Drive) pipeline.

Streams existing clips through a LEAN location+movement tracker (YOLO box ->
worm blob centroid -> Hampel-cleaned position + speed; behavior deferred until
the model is retrained on clean capture) and reports whether it keeps up with
the real-time budget (clips arrive 1/min -> must process each in < 60s), where
the time goes, and peak memory. This is for finding bottlenecks before going
live, not for production tracking.

Two efficiency levers it can measure:
  --reuse_bg : build the dish background once and reuse it (the rig is fixed);
               avoids rebuilding the per-clip median (~28% of a clip's time).
  --stride N : run YOLO every Nth frame (the worm is slow; 10-15 fps is plenty
               for location/movement), cutting the dominant cost ~N x.

Usage:
  python rt_dryrun.py --glob '../live_capture/2026-06-02_16-*.mkv' --reuse_bg --stride 2
"""
import argparse
import glob as globmod
import os
import resource
import time

import numpy as np
import cv2
from ultralytics import YOLO

import fusion_tracker as ft
import yolo_tracker as yt


def track_location(video, model, bg, dish, fps, mm_per_px, conf, imgsz, device, stride):
    """Lean per-frame position (mm) + speed. Returns (n_samples, stage_times)."""
    cap = cv2.VideoCapture(video)
    raw = []
    t_dec = t_yolo = t_cent = 0.0
    fi = 0
    while True:
        a = time.monotonic(); ok, frame = cap.read(); t_dec += time.monotonic() - a
        if not ok:
            break
        if stride > 1 and (fi % stride):
            fi += 1
            continue
        a = time.monotonic()
        r = model.predict(frame, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        t_yolo += time.monotonic() - a
        det = yt.yolo_box_in_dish(r, dish)
        a = time.monotonic()
        if det is not None:
            bx, by, box, cf = det
            px, py = bx, by
            th, origin = yt._box_foreground(frame, bg, box, 0.35)
            if th is not None:
                cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    c = max(cnts, key=cv2.contourArea); M = cv2.moments(c)
                    if M["m00"] > 0:
                        px = M["m10"] / M["m00"] + origin[0]
                        py = M["m01"] / M["m00"] + origin[1]
            raw.append((fi, px, py))
        else:
            raw.append((fi, np.nan, np.nan))
        t_cent += time.monotonic() - a
        fi += 1
    cap.release()
    if raw:
        xs = np.array([r[1] for r in raw]); ys = np.array([r[2] for r in raw])
        ft.hampel_clean(xs, ys, fps / max(1, stride), mm_per_px,
                        floor_mult=2.0, abs_floor=20.0)   # the cleaned track we'd log
    return len(raw), {"decode": t_dec, "yolo": t_yolo, "centroid": t_cent}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="*", default=[])
    ap.add_argument("--glob", default=None)
    ap.add_argument("--max_clips", type=int, default=12)
    ap.add_argument("--model", default=yt.DEFAULT_MODEL)
    ap.add_argument("--mm_per_px", type=float, default=0.02657)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--reuse_bg", action="store_true")
    a = ap.parse_args()

    clips = sorted(a.videos) + (sorted(globmod.glob(a.glob)) if a.glob else [])
    clips = [c for c in clips if os.path.exists(c)][:a.max_clips]
    if not clips:
        raise SystemExit("no clips found (use --videos and/or --glob)")
    print(f"dry run: {len(clips)} clips | imgsz={a.imgsz} stride={a.stride} "
          f"reuse_bg={a.reuse_bg}")
    print("budget: clips arrive 1/min -> each must process in < 60s\n")

    model = YOLO(a.model)
    # warmup (first MPS call is slow)
    c0 = cv2.VideoCapture(clips[0]); ok, f0 = c0.read(); c0.release()
    if ok:
        model.predict(f0, imgsz=a.imgsz, device=a.device, verbose=False)

    shared = None
    per_clip = []
    for n, vp in enumerate(clips):
        t0 = time.monotonic()
        if a.reuse_bg and shared is not None:
            bg, dish, fps = shared; t_bg = 0.0
        else:
            b = time.monotonic(); bg, dish, fps = ft.build_background(vp)
            t_bg = time.monotonic() - b
            if a.reuse_bg:
                shared = (bg, dish, fps)
        if bg is None:
            print(f"  {os.path.basename(vp)}: no dish/bg -> skip"); continue
        nsmp, st = track_location(vp, model, bg, dish, fps, a.mm_per_px,
                                  a.conf, a.imgsz, a.device, a.stride)
        dt = time.monotonic() - t0
        per_clip.append(dt)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # macOS: bytes->MB
        flag = "" if dt < 60 else "  <-- OVER 60s"
        print(f"  [{n+1:2d}/{len(clips)}] {os.path.basename(vp):26s} {dt:5.1f}s "
              f"| bg {t_bg:4.1f} yolo {st['yolo']:4.1f} cent {st['centroid']:4.1f} "
              f"| {nsmp} samples peakRSS {rss:.0f}MB{flag}")

    pc = np.array(per_clip)
    print(f"\n=== summary ({len(pc)} clips) ===")
    print(f"  per-clip: median {np.median(pc):.1f}s  mean {pc.mean():.1f}s  max {pc.max():.1f}s")
    print(f"  real-time headroom vs 60s: {60/np.median(pc):.1f}x median, "
          f"{60/pc.max():.1f}x worst")
    print(f"  clips over 60s budget: {int((pc>60).sum())}/{len(pc)}")
    print(f"  peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.0f}MB")


if __name__ == "__main__":
    main()
