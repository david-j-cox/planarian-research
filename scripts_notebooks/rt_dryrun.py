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
    """Lean per-frame position (mm) + speed via YOLO box -> worm blob centroid ->
    Hampel-cleaned track. Returns (records, stage_times). Each record:
    (native_frame, time_s, x_mm, y_mm, speed_mm_s, conf, state)."""
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
            raw.append((fi, px, py, cf))
        else:
            raw.append((fi, np.nan, np.nan, 0.0))
        t_cent += time.monotonic() - a
        fi += 1
    cap.release()
    times = {"decode": t_dec, "yolo": t_yolo, "centroid": t_cent}
    if not raw:
        return [], times
    frames = np.array([r[0] for r in raw])
    xs = np.array([r[1] for r in raw]); ys = np.array([r[2] for r in raw])
    cx, cy, keep = ft.hampel_clean(xs, ys, fps / max(1, stride), mm_per_px,
                                   floor_mult=2.0, abs_floor=20.0)
    xmm, ymm = cx * mm_per_px, cy * mm_per_px
    recs = []
    for i in range(len(raw)):
        if i == 0:
            spd = 0.0
        else:
            dt = (frames[i] - frames[i - 1]) / fps
            spd = (np.hypot(xmm[i] - xmm[i - 1], ymm[i] - ymm[i - 1]) / dt
                   if dt > 0 else 0.0)
        recs.append((int(frames[i]), float(frames[i] / fps), float(xmm[i]),
                     float(ymm[i]), float(spd), float(raw[i][3]),
                     "detected" if keep[i] else "interp"))
    return recs, times


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
    ap.add_argument("--repeat", type=int, default=1,
                    help="loop the clip list N times (soak test for memory leaks)")
    a = ap.parse_args()

    base = sorted(a.videos) + (sorted(globmod.glob(a.glob)) if a.glob else [])
    base = [c for c in base if os.path.exists(c)][:a.max_clips]
    clips = base * a.repeat
    if not clips:
        raise SystemExit("no clips found (use --videos and/or --glob)")
    print(f"dry run: {len(clips)} clip-passes ({len(base)} unique x{a.repeat}) | "
          f"imgsz={a.imgsz} stride={a.stride} reuse_bg={a.reuse_bg}")
    print("budget: clips arrive 1/min -> each must process in < 60s\n")

    model = YOLO(a.model)
    # warmup (first MPS call is slow)
    c0 = cv2.VideoCapture(clips[0]); ok, f0 = c0.read(); c0.release()
    if ok:
        model.predict(f0, imgsz=a.imgsz, device=a.device, verbose=False)

    try:
        import psutil; _proc = psutil.Process()
        def cur_rss():
            return _proc.memory_info().rss / 1e6     # CURRENT RSS (MB), for leak trend
    except Exception:
        def cur_rss():
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # peak only

    shared = None
    per_clip = []; rss_series = []
    verbose = len(clips) <= 30
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
        recs, st = track_location(vp, model, bg, dish, fps, a.mm_per_px,
                                  a.conf, a.imgsz, a.device, a.stride)
        nsmp = len(recs)
        dt = time.monotonic() - t0
        per_clip.append(dt); rss_series.append(cur_rss())
        flag = "" if dt < 60 else "  <-- OVER 60s"
        if verbose or n % 25 == 0:
            print(f"  [{n+1:3d}/{len(clips)}] {os.path.basename(vp):26s} {dt:5.1f}s "
                  f"| bg {t_bg:4.1f} yolo {st['yolo']:4.1f} cent {st['centroid']:4.1f} "
                  f"| RSS {rss_series[-1]:.0f}MB{flag}")

    pc = np.array(per_clip); rs = np.array(rss_series)
    print(f"\n=== summary ({len(pc)} clip-passes) ===")
    print(f"  per-clip: median {np.median(pc):.1f}s  mean {pc.mean():.1f}s  max {pc.max():.1f}s")
    print(f"  real-time headroom vs 60s: {60/np.median(pc):.1f}x median, "
          f"{60/pc.max():.1f}x worst")
    print(f"  clips over 60s budget: {int((pc>60).sum())}/{len(pc)}")
    # Leak check: linear slope of current RSS over clips.
    if len(rs) >= 5:
        slope = float(np.polyfit(np.arange(len(rs)), rs, 1)[0])   # MB per clip
        print(f"  RSS: start {rs[0]:.0f}MB -> end {rs[-1]:.0f}MB  "
              f"trend {slope:+.2f} MB/clip  ({'FLAT/ok' if abs(slope) < 0.5 else 'GROWING -- possible leak'})")
        print(f"  extrapolated over a 7-day run (10080 clips): "
              f"{rs[-1] + slope * (10080 - len(rs)):.0f}MB")


if __name__ == "__main__":
    main()
