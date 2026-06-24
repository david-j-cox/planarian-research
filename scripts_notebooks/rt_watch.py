#!/usr/bin/env python3
"""
rt_watch.py — continuous folder watcher for the real-time worm pipeline.

Watches a folder (e.g. the Google-Drive-synced clips dir), and as each new
1-minute clip lands, tracks LOCATION + MOVEMENT (YOLO box -> worm blob centroid
-> Hampel-cleaned position + speed; behavior deferred until the model is
retrained on clean capture) and appends to one rolling CSV. Built for an
unattended multi-day run.

Robustness:
  - FILE-STABILITY GATE: a clip is processed only once its size has been
    unchanged for --stable_s (so a half-synced Drive file is never read).
  - RESUMABLE: processed clips are recorded in <session>_processed.txt and
    skipped on restart; the CSV is appended, never rewritten.
  - REUSE BACKGROUND: built once from the first clip (the rig is fixed) and
    reused -- ~28% faster per clip (see rt_dryrun findings). --rebuild_every N
    rebuilds periodically if lighting drifts.
  - BACKLOG ALARM: if unprocessed clips pile up (processing falling behind
    1/min), it warns.
  - Optional --delete_after to free disk on long runs.

Usage:
  python rt_watch.py --watch_dir "$DRIVE/planarian_clips" --session_id worm_run_01
  # dry/local test against existing clips:
  python rt_watch.py --watch_dir ../live_capture --once
"""
import argparse
import csv
import gc
import glob
import os
import sys
import time
from datetime import datetime

import numpy as np
from ultralytics import YOLO

import fusion_tracker as ft
import yolo_tracker as yt
from rt_dryrun import track_location

try:
    import torch
    _HAS_MPS = torch.backends.mps.is_available()
except Exception:
    torch = None; _HAS_MPS = False


def reclaim():
    """Per-clip reclaim: Python cycles + MPS cache. Halves the per-clip leak;
    the periodic re-exec caps whatever residual (native-lib) leak remains."""
    gc.collect()
    if _HAS_MPS:
        torch.mps.empty_cache()

CSV_HEADER = ["video_file", "native_frame", "time_s", "x_mm", "y_mm",
              "speed_mm_s", "conf", "state"]


def stable(path, stable_s, poll=0.5):
    """True once the file size has been unchanged for stable_s (sync complete)."""
    try:
        s0 = os.path.getsize(path)
    except OSError:
        return False
    waited = 0.0
    while waited < stable_s:
        time.sleep(poll); waited += poll
        try:
            s1 = os.path.getsize(path)
        except OSError:
            return False
        if s1 != s0:
            return False         # still being written/synced
        s0 = s1
    return s0 > 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch_dir", required=True)
    ap.add_argument("--out_dir", default="../realtime_runs")
    ap.add_argument("--session_id", default=None)
    ap.add_argument("--pattern", default="*.mkv")
    ap.add_argument("--model", default=yt.DEFAULT_MODEL)
    ap.add_argument("--mm_per_px", type=float, default=0.02657)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--stride", type=int, default=2, help="YOLO every Nth frame")
    ap.add_argument("--poll_s", type=float, default=5.0, help="folder poll interval")
    ap.add_argument("--stable_s", type=float, default=3.0, help="size-stable wait before processing")
    ap.add_argument("--rebuild_every", type=int, default=0, help="rebuild background every N clips (0=never)")
    ap.add_argument("--restart_every", type=int, default=200,
                    help="re-exec the process every N clips to cap a residual native "
                         "memory leak (resumes via the processed-list; 0=never)")
    ap.add_argument("--delete_after", action="store_true", help="delete each clip after processing")
    ap.add_argument("--once", action="store_true", help="process the current backlog and exit (test mode)")
    a = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)   # flush logs per line under a supervisor
    except Exception:
        pass
    sid = a.session_id or f"worm_run_{datetime.now():%Y%m%d_%H%M%S}"
    os.makedirs(a.out_dir, exist_ok=True)
    csv_path = os.path.join(a.out_dir, f"{sid}_tracks.csv")
    proc_path = os.path.join(a.out_dir, f"{sid}_processed.txt")

    processed = set()
    if os.path.exists(proc_path):
        processed = set(l.strip() for l in open(proc_path) if l.strip())
    new_csv = not os.path.exists(csv_path)
    fh = open(csv_path, "a", newline="")
    w = csv.writer(fh)
    if new_csv:
        w.writerow([f"# session={sid} mm_per_px={a.mm_per_px} created={datetime.now().isoformat(timespec='seconds')}"])
        w.writerow(CSV_HEADER)
        fh.flush()

    model = YOLO(a.model)
    shared = None; n_since_bg = 0; n_session = 0
    print(f"[{sid}] watching {os.path.abspath(a.watch_dir)}  -> {csv_path}")
    print(f"  resume: {len(processed)} clips already processed | stride={a.stride} reuse_bg=on")

    try:
        while True:
            pending = sorted(f for f in glob.glob(os.path.join(a.watch_dir, a.pattern))
                             if os.path.basename(f) not in processed)
            if pending and len(pending) > 5:
                print(f"  [backlog] {len(pending)} clips pending -- falling behind 1/min?")
            did = False
            for vp in pending:
                name = os.path.basename(vp)
                if not stable(vp, a.stable_s):
                    continue                      # still syncing; try next poll
                t0 = time.monotonic()
                if shared is None or (a.rebuild_every and n_since_bg >= a.rebuild_every):
                    bg, dish, fps = ft.build_background(vp); shared = (bg, dish, fps); n_since_bg = 0
                else:
                    bg, dish, fps = shared
                if bg is None:
                    print(f"  {name}: no dish/bg -> skip"); processed.add(name)
                    open(proc_path, "a").write(name + "\n"); continue
                recs, st = track_location(vp, model, bg, dish, fps, a.mm_per_px,
                                          a.conf, a.imgsz, a.device, a.stride)
                for (nf, ts, xmm, ymm, spd, cf, state) in recs:
                    w.writerow([name, nf, f"{ts:.3f}", f"{xmm:.3f}", f"{ymm:.3f}",
                                f"{spd:.3f}", f"{cf:.3f}", state])
                fh.flush()
                processed.add(name); n_since_bg += 1
                open(proc_path, "a").write(name + "\n")
                if a.delete_after:
                    try: os.remove(vp)
                    except OSError: pass
                dt = time.monotonic() - t0
                det = sum(1 for r in recs if r[6] == "detected")
                print(f"  {name}: {dt:4.1f}s  {len(recs)} samples ({det} detected)"
                      + ("  <-- OVER 60s" if dt > 60 else ""))
                did = True
                del recs; reclaim(); n_session += 1
                # Cap any residual native-library leak: re-exec the process every
                # restart_every clips. State lives on disk (processed-list + CSV
                # appended), so it resumes seamlessly; the OS frees all memory.
                if a.restart_every and n_session >= a.restart_every and not a.once:
                    print(f"  [restart] {n_session} clips this session -> re-exec to reset memory")
                    fh.flush(); fh.close()
                    os.execv(sys.executable, [sys.executable] + sys.argv)
            if a.once and not pending:
                break
            if not did:
                time.sleep(a.poll_s)
    except KeyboardInterrupt:
        print("\nstopping (Ctrl-C)")
    finally:
        fh.flush(); fh.close()
        print(f"done. {len(processed)} clips processed -> {csv_path}")


if __name__ == "__main__":
    main()
