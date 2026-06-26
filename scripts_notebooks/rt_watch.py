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
import shutil
import subprocess
import sys
import time
from datetime import datetime

import cv2
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
    """Cheap pre-filter: True once file SIZE has been unchanged for stable_s.

    NOTE: necessary but NOT sufficient for Google-Drive-synced files -- the Drive
    File Provider reports the FINAL size up front and materializes the bytes
    afterward, so size is 'stable' while content is still downloading. Always
    follow with clip_ready() (content-based) before processing."""
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


def prefetch(path):
    """Ask Drive to download an online-only clip WITHOUT blocking the watcher.

    materialized() makes us skip not-yet-downloaded clips (opening them would
    hang). But if we only ever wait, a clip that Drive doesn't auto-download
    never gets processed. So we nudge the download in a detached `cat` that reads
    the file through -- which forces the File Provider to fetch it -- in its own
    subprocess. If that read blocks, only the helper blocks; the watcher loop
    stays responsive and picks the clip up on a later poll once it materializes."""
    try:
        return subprocess.Popen(["cat", path], stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
    except OSError:
        return None


def materialized(path):
    """True only if the file is FULLY downloaded to local disk.

    Critical for Drive sources: an 'online-only' placeholder (or a mid-download
    partial) has fewer on-disk blocks than its apparent size. Calling open()/
    VideoCapture on such a file BLOCKS in the open() syscall until the File
    Provider materializes it -- which can hang the whole single-threaded watcher
    indefinitely if sync is slow or paused. os.stat() reads only metadata and
    never blocks, so we use the on-disk block count as the gate and never open a
    file that isn't fully present. (st_blocks is in 512-byte units.)"""
    try:
        st = os.stat(path)
    except OSError:
        return False
    return st.st_size > 0 and st.st_blocks * 512 >= st.st_size


def _decodable_frames(path):
    """Count frames that actually decode (grab() avoids pixel copy -> fast).
    Reflects how much of a Drive file has truly materialized, unlike file size."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return -1
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def clip_ready(path, min_frames, settle_s=3.0, checks=2):
    """True once the clip is FULLY materialized. Byte size lies for Drive files
    (final size reported up front), and a single short settle also lies (Drive
    downloads in bursts with multi-second pauses, so the frame count can plateau
    mid-transfer). So require BOTH:
      (1) decodable frames >= min_frames  -- near a full clip, not an early stub;
      (2) frame count unchanged across `checks` polls spaced settle_s apart
          (~checks*settle_s of no growth) -- the transfer has actually finished.
    A still-downloading clip keeps growing and fails (2); a tiny stub fails (1).
    NOTE: a genuinely short final clip (recording stopped mid-minute) stays below
    min_frames and is intentionally left pending -- flush it with `--once
    --min_frames <low>` after recording ends."""
    last = _decodable_frames(path)
    if last < min_frames:
        return False
    for _ in range(checks):
        time.sleep(settle_s)
        n = _decodable_frames(path)
        if n != last:
            return False          # still growing -> not done syncing
        last = n
    return True


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
    ap.add_argument("--min_frames", type=int, default=30,
                    help="minimum decodable frames before a clip is treated as fully "
                         "synced (set near a full clip's frame count for Drive sources; "
                         "e.g. ~580 for a 60s clip @10fps)")
    ap.add_argument("--delete_after", action="store_true",
                    help="delete each clip after processing (only deletes FULL clips -- "
                         "see min_delete_samples -- so a truncated clip is never lost)")
    ap.add_argument("--keep_total", type=int, default=0,
                    help="preserve up to N full clips for behavior-model training, "
                         "balanced across activity buckets (0 = keep none)")
    ap.add_argument("--keep_dir", default="../realtime_runs/keep_clips",
                    help="where preserved training clips are moved")
    ap.add_argument("--keep_edges", default="0.7,1.05",
                    help="mean-speed (mm/s) bucket edges: <e0=low, <e1=med, >=e1=high")
    ap.add_argument("--once", action="store_true", help="process the current backlog and exit (test mode)")
    # --- behavior layer (additive; off unless --behavior_model is given) ---
    ap.add_argument("--behavior_model", default=None,
                    help="path to behavior_clf.joblib; enables per-clip behavior + ethogram + active keep")
    ap.add_argument("--label_queue", default="../realtime_runs/label_queue",
                    help="dir where uncertain/rare clips are kept for labeling")
    ap.add_argument("--keep_per_day", type=int, default=150,
                    help="max clips/day to retain for labeling (active-learning budget)")
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
    # Transient build_background failures are almost always a not-yet-fully-synced
    # Drive file (stable size, partial contents). Retry on later polls instead of
    # poisoning the processed list; give up only after MAX_BG_RETRIES.
    fail_counts = {}; MAX_BG_RETRIES = 8

    # Behavior-training clip retention: keep up to keep_total full clips, balanced
    # across 3 activity buckets (by mean detected speed). Kept clips are MOVED out
    # of the Drive watch_dir into keep_dir (frees Drive, preserves locally). A
    # manifest makes the per-bucket counts resumable across restarts.
    keep_edges = [float(x) for x in a.keep_edges.split(",")]
    keep_quota = a.keep_total // 3 if a.keep_total else 0
    keep_counts = [0, 0, 0]
    keep_manifest = os.path.join(a.out_dir, f"{sid}_keep_manifest.csv")
    if a.keep_total:
        os.makedirs(a.keep_dir, exist_ok=True)
        if os.path.exists(keep_manifest):
            for ln in open(keep_manifest):
                b = ln.strip().split(",")[-1]
                for i, nm in enumerate(("low", "med", "high")):
                    if b == nm:
                        keep_counts[i] += 1
        new_mf = not os.path.exists(keep_manifest)
        mf = open(keep_manifest, "a")
        if new_mf:
            mf.write("clip,mean_speed_mm_s,bucket\n"); mf.flush()

    # --- behavior layer: per-clip states (ethogram) + uncertainty-based keep ---
    beh = None
    if a.behavior_model:
        import joblib
        import rt_behavior as rb
        beh_csv = os.path.join(a.out_dir, f"{sid}_behavior.csv")
        bnew = not os.path.exists(beh_csv)
        bfh = open(beh_csv, "a", newline=""); bw = csv.writer(bfh)
        if bnew:
            bw.writerow(["video_file", "native_frame", "time_s", "behavior", "conf"]); bfh.flush()
        os.makedirs(a.label_queue, exist_ok=True)
        lq_man = os.path.join(a.label_queue, "_kept.csv")
        lqnew = not os.path.exists(lq_man); lqfh = open(lq_man, "a")
        if lqnew:
            lqfh.write("clip,date,reason,frac_lowconf,dominant\n"); lqfh.flush()
        kept_day = {}                                   # date -> count (resumable)
        if not lqnew:
            for ln in open(lq_man):
                p = ln.strip().split(",")
                if len(p) >= 2 and p[1] != "date":
                    kept_day[p[1]] = kept_day.get(p[1], 0) + 1
        beh = dict(art=joblib.load(a.behavior_model), w=bw, fh=bfh, rb=rb,
                   lq=lqfh, kept=kept_day, rare={"wig_wag"})
        print(f"  behavior: ON -> {beh_csv} | label_queue {a.label_queue} (<={a.keep_per_day}/day)")

    print(f"[{sid}] watching {os.path.abspath(a.watch_dir)}  -> {csv_path}")
    print(f"  resume: {len(processed)} clips already processed | stride={a.stride} reuse_bg=on")

    prefetch_started = {}                 # name -> Popen, so we trigger each download once
    PREFETCH_AHEAD = 4                    # how many not-yet-local clips to pull at once

    try:
        while True:
            pending = sorted(f for f in glob.glob(os.path.join(a.watch_dir, a.pattern))
                             if os.path.basename(f) not in processed)
            if pending and len(pending) > 5:
                print(f"  [backlog] {len(pending)} clips pending -- falling behind 1/min?")
            # Actively pull down the oldest online-only clips (non-blocking) so the
            # backlog drains even if Drive isn't auto-downloading on its own.
            inflight = 0
            for vp in pending:
                if inflight >= PREFETCH_AHEAD:
                    break
                name = os.path.basename(vp)
                if materialized(vp):
                    continue
                already = prefetch_started.get(name)
                if already is None or already.poll() is not None:
                    prefetch_started[name] = prefetch(vp)
                    print(f"  [prefetch] pulling {name} from Drive")
                inflight += 1
            did = False
            for vp in pending:
                name = os.path.basename(vp)
                if not materialized(vp):
                    continue                      # Drive online-only / partial -> never open (open() blocks)
                if not stable(vp, a.stable_s):
                    continue                      # size still changing; try next poll
                if not clip_ready(vp, a.min_frames):
                    continue                      # belt-and-suspenders: frames decode & stable
                t0 = time.monotonic()
                if shared is None or (a.rebuild_every and n_since_bg >= a.rebuild_every):
                    bg, dish, fps = ft.build_background(vp); shared = (bg, dish, fps); n_since_bg = 0
                else:
                    bg, dish, fps = shared
                if bg is None:
                    fail_counts[name] = fail_counts.get(name, 0) + 1
                    if fail_counts[name] < MAX_BG_RETRIES:
                        print(f"  {name}: no dish/bg (attempt {fail_counts[name]}/{MAX_BG_RETRIES}) "
                              f"-> likely still syncing, retry later")
                        continue          # leave pending; do NOT mark processed
                    print(f"  {name}: no dish/bg after {MAX_BG_RETRIES} tries -> skip")
                    processed.add(name); open(proc_path, "a").write(name + "\n"); continue
                if beh is not None:
                    # UNIFIED single YOLO pass -> location recs + behavior in one go
                    recs, rbres = beh["rb"].track_and_behavior(
                        vp, model, bg, dish, fps, beh["art"], a.mm_per_px,
                        a.conf, a.imgsz, a.device, a.stride)
                else:
                    recs, _ = track_location(vp, model, bg, dish, fps, a.mm_per_px,
                                             a.conf, a.imgsz, a.device, a.stride)
                    rbres = None
                for (nf, ts, xmm, ymm, spd, cf, state) in recs:
                    w.writerow([name, nf, f"{ts:.3f}", f"{xmm:.3f}", f"{ymm:.3f}",
                                f"{spd:.3f}", f"{cf:.3f}", state])
                fh.flush()
                processed.add(name); n_since_bg += 1
                open(proc_path, "a").write(name + "\n")
                # Delete safety net: only remove a clip whose track is full-length
                # (>= min_frames/stride samples). If a truncated clip ever slips the
                # readiness gate, this keeps it on disk for reprocessing instead of
                # destroying the only copy. Decouples delete-safety from gate-perfection.
                full_samples = a.min_frames // max(a.stride, 1)
                is_full = len(recs) >= full_samples
                kept_for_train = False
                beh_kept = False
                # Behavior pass: per-clip states (-> ethogram CSV, ~1 row/s) and
                # uncertainty-based keep (low-confidence / rare-behavior clips are
                # moved to the label_queue, up to keep_per_day -- active learning).
                # rbres was already computed in the unified track_and_behavior pass.
                if beh is not None and is_full and rbres is not None:
                    try:
                        cls = rbres["classes"]; nf2 = rbres["native_frame"]
                        ts2 = rbres["time_s"]; dom = rbres["dom"]
                        cf2 = rbres["conf"]; ok2 = rbres["ok"]
                        last_sec = -1
                        for i in range(len(nf2)):
                            if ok2[i] and int(ts2[i]) != last_sec:
                                beh["w"].writerow([name, int(nf2[i]), f"{ts2[i]:.3f}",
                                                   cls[dom[i]], f"{cf2[i]:.3f}"])
                                last_sec = int(ts2[i])
                        beh["fh"].flush()
                        s = rbres["summary"]
                        if s.get("n_scored"):
                            flc = s.get("frac_lowconf", 0.0); dominant = s.get("dominant", "")
                            date = name[:10]
                            interesting = (flc >= 0.20 or dominant in beh["rare"]
                                           or s.get("mean_conf", 1.0) < 0.65)
                            if interesting and beh["kept"].get(date, 0) < a.keep_per_day:
                                try:
                                    shutil.move(vp, os.path.join(a.label_queue, name))
                                    beh_kept = True
                                    beh["kept"][date] = beh["kept"].get(date, 0) + 1
                                    reason = ("lowconf" if flc >= 0.20 else
                                              ("rare:" + dominant if dominant in beh["rare"] else "meanconf"))
                                    beh["lq"].write(f"{name},{date},{reason},{flc:.2f},{dominant}\n")
                                    beh["lq"].flush()
                                    print(f"  [label-queue] {name} ({reason}, lowconf {flc:.0%}) "
                                          f"-> {beh['kept'][date]}/{a.keep_per_day} today")
                                except OSError as e:
                                    print(f"  [label-queue] move failed {name}: {e}")
                    except Exception as e:
                        print(f"  [behavior] failed on {name}: {e}")
                # Behavior-training retention: bucket this clip by mean detected
                # speed and preserve it if its bucket isn't full yet.
                if a.keep_total and is_full and not beh_kept:
                    sps = [r[4] for r in recs if r[6] == "detected" and r[4] <= 6]
                    mspd = sum(sps) / len(sps) if sps else 0.0
                    b = 0 if mspd < keep_edges[0] else (2 if mspd >= keep_edges[1] else 1)
                    if keep_counts[b] < keep_quota:
                        bname = ("low", "med", "high")[b]
                        try:
                            shutil.move(vp, os.path.join(a.keep_dir, name))
                            keep_counts[b] += 1; kept_for_train = True
                            mf.write(f"{name},{mspd:.3f},{bname}\n"); mf.flush()
                            print(f"  [keep:{bname}] {name} ({mspd:.2f} mm/s) -> "
                                  f"{keep_counts[b]}/{keep_quota}  (total kept {sum(keep_counts)}/{a.keep_total})")
                        except OSError as e:
                            print(f"  [keep] move failed {name}: {e}")
                if not kept_for_train and not beh_kept and a.delete_after:
                    if is_full:
                        try: os.remove(vp)
                        except OSError: pass
                    else:
                        print(f"  {name}: KEPT (only {len(recs)} samples < {full_samples}); "
                              f"not deleting a short clip")
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
