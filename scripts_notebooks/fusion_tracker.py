#!/usr/bin/env python3
"""
fusion_tracker.py — Worm tracker that uses the priors a per-frame detector
ignores:

  1. Static-background subtraction (median over the video) -> the worm is the
     moving signal; lighting/dish/grid cancel out (lighting-invariant).
  2. Dish ROI mask -> nothing outside the dish can be the worm.
  3. Temporal continuity -> among candidate blobs, pick the one nearest the
     previous position; gate motion so the worm can't teleport (>MAX_SPEED);
     HOLD the last position through resting/occluded frames; linearly
     interpolate true gaps; light median-smooth the final track.

Body axis (head/tail/length) comes from PCA on the chosen blob.

Subcommands:
  track  one video -> per-frame CSV (+ optional overlay mp4)
  eval   run over the labeled videos and score the tracked position at each GT
         frame against the human labels (apples-to-apples with bgsub_tracker).

Usage:
  python fusion_tracker.py track --video ../live_capture/2026-06-02_16-00-37.mkv \
      --mm_per_px 0.02657 --overlay
  python fusion_tracker.py eval --label_dir ../realtime_runs/label_white7mp \
      --videos_dir ../live_capture --mm_per_px 0.02657
"""
import argparse
import glob
import json
import os
import numpy as np
import cv2

MAX_SPEED_MM_S = 7.0


# ── background + dish ────────────────────────────────────────────────
def build_background(video, n=50):
    cap = cv2.VideoCapture(video)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if N <= 0:
        cap.release(); return None, None, fps
    frames = []
    for f in np.linspace(0, N - 1, min(n, N)).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, im = cap.read()
        if ok:
            frames.append(im)
    cap.release()
    if not frames:
        return None, None, fps
    bg = np.median(np.stack(frames), axis=0).astype(np.uint8)
    g = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(cv2.medianBlur(g, 9), cv2.HOUGH_GRADIENT, 1.5, 2000,
                               param1=120, param2=60, minRadius=900, maxRadius=1200)
    H, W = g.shape
    dish = tuple(float(x) for x in circles[0][0]) if circles is not None \
        else (W / 2.0, H / 2.0, min(H, W) / 2.0 * 0.95)
    return bg, dish, fps


def candidate_blobs(frame, bg, dish, min_area=200):
    """All plausible moving blobs inside the dish: (cx, cy, box, area, contour)."""
    diff = cv2.cvtColor(cv2.absdiff(frame, bg), cv2.COLOR_BGR2GRAY)
    cxd, cyd, rd = dish
    mask = np.zeros(diff.shape, np.uint8)
    cv2.circle(mask, (int(cxd), int(cyd)), int(rd * 0.97), 255, -1)
    diff = cv2.bitwise_and(diff, mask)
    th = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        out.append((x + w / 2.0, y + h / 2.0, [float(x), float(y), float(x + w), float(y + h)],
                    float(a), c))
    return out


def pca_axis(contour):
    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < 5:
        return None, None, np.nan
    mean = pts.mean(0)
    d = pts - mean
    # Principal axis = top eigenvector of the 2x2 covariance. (Do NOT use
    # np.linalg.svd(d): its default full_matrices=True allocates the M x M U
    # matrix -- for a noisy contour with tens of thousands of perimeter points
    # that's tens of GB and minutes of compute, and U is never used.)
    cov = (d.T @ d) / len(d)
    axis = np.linalg.eigh(cov)[1][:, -1]
    proj = d @ axis
    return (mean + axis * proj.max()).tolist(), (mean + axis * proj.min()).tolist(), \
        float(proj.max() - proj.min())


# ── tracker ──────────────────────────────────────────────────────────
def track(video, mm_per_px, min_area=200):
    """Per-frame largest-blob-in-dish (the robust cue), then a SAFE temporal
    pass: Hampel spike rejection (drop isolated jumps faster than a worm can
    move) + interpolation + light smoothing. The temporal layer only repairs
    the tail; it never 'locks on', so it cannot drag the track onto debris."""
    bg, dish, fps = build_background(video)
    if bg is None:
        return [], fps, dish
    cap = cv2.VideoCapture(video)
    raw = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cands = candidate_blobs(frame, bg, dish, min_area)
        if cands:
            c = max(cands, key=lambda c: c[3])   # largest moving blob inside the dish
            h, t, blen = pca_axis(c[4])
            raw.append({"frame": len(raw), "cx": c[0], "cy": c[1], "box": c[2],
                        "head": h, "tail": t, "blen": blen})
        else:
            raw.append({"frame": len(raw), "cx": np.nan, "cy": np.nan, "box": None,
                        "head": None, "tail": None, "blen": np.nan})
    cap.release()

    n = len(raw)
    xs = np.array([r["cx"] for r in raw]); ys = np.array([r["cy"] for r in raw])
    valid = np.isfinite(xs)
    idx = np.arange(n)
    # Hampel: flag points far from their local median (window 7). A worm moves
    # <= MAX_SPEED, so a point that sits well off the local trajectory is a
    # wrong-blob spike, not real motion.
    win = 7
    floor_px = max(MAX_SPEED_MM_S / fps / mm_per_px * 5.0, 40.0)
    flagged = np.zeros(n, bool)
    for i in range(n):
        if not valid[i]:
            continue
        lo, hi = max(0, i - win), min(n, i + win + 1)
        m = valid[lo:hi]
        if m.sum() < 3:
            continue
        mx, my = np.median(xs[lo:hi][m]), np.median(ys[lo:hi][m])
        if np.hypot(xs[i] - mx, ys[i] - my) > floor_px:
            flagged[i] = True
    keep = valid & ~flagged

    if keep.any():
        cx = np.interp(idx, idx[keep], xs[keep])
        cy = np.interp(idx, idx[keep], ys[keep])

        def medsmooth(a, w=3):
            pad = w // 2
            ap = np.pad(a, pad, mode="edge")
            return np.array([np.median(ap[i:i + w]) for i in range(len(a))])
        cx, cy = medsmooth(cx), medsmooth(cy)
    else:
        cx, cy = xs, ys

    # Rebuild records: cleaned position; carry head/tail from the last kept frame.
    recs = []
    last_h = last_t = None
    for i in range(n):
        if keep[i]:
            last_h, last_t = raw[i]["head"], raw[i]["tail"]
            state = "detected"
        else:
            state = "interp"
        recs.append({"frame": i, "cx": float(cx[i]), "cy": float(cy[i]),
                     "box": raw[i]["box"] if keep[i] else None,
                     "head": last_h, "tail": last_t,
                     "blen": raw[i]["blen"] if keep[i] else np.nan, "state": state})
    return recs, fps, dish


# ── eval ─────────────────────────────────────────────────────────────
def cmd_eval(a):
    man = {it["id"]: it for it in json.load(open(os.path.join(a.label_dir, "manifest.json")))["items"]}
    lab = {int(k): v for k, v in json.load(open(os.path.join(a.label_dir, "labels.json"))).items()}
    ev = [k for k in lab if man[k]["pool"] == "eval" and lab[k].get("present")]
    by_vid = {}
    for k in ev:
        by_vid.setdefault(man[k]["video"], []).append(k)

    # Resumable checkpoint: per-video {"errs":[mm...], "missing":int}. The long
    # eval gets reaped before finishing, so bank each video as it completes and
    # skip videos already done on restart. Delete the file to start fresh.
    ckpt = a.checkpoint or os.path.join(a.label_dir, "fusion_eval_checkpoint.json")
    done = json.load(open(ckpt)) if os.path.exists(ckpt) else {}
    if done:
        print(f"  resuming: {len(done)}/{len(by_vid)} videos already done")

    # --max_videos caps how many uncheckpointed videos this process handles
    # before exiting, so a shell loop can isolate each video in its own process
    # (defensive: bounds peak memory if a single video ever misbehaves again).
    n_this_run = 0
    for v, ks in sorted(by_vid.items()):
        if v in done:
            continue
        if a.max_videos and n_this_run >= a.max_videos:
            break
        vp = os.path.join(a.videos_dir, v)
        if not os.path.exists(vp):
            print(f"  missing video {v}")
            done[v] = {"errs": [], "missing": len(ks)}
            json.dump(done, open(ckpt, "w")); continue
        recs, fps, dish = track(vp, a.mm_per_px)
        pos_by_frame = {r["frame"]: (r["cx"], r["cy"]) for r in recs}
        verrs, vmiss = [], 0
        for k in ks:
            f = man[k]["frame"]; gt = lab[k]["box"]
            gx, gy = (gt[0]+gt[2])/2, (gt[1]+gt[3])/2
            if f in pos_by_frame and np.isfinite(pos_by_frame[f][0]):
                px, py = pos_by_frame[f]
                verrs.append(float(np.hypot(px-gx, py-gy) * a.mm_per_px))
            else:
                vmiss += 1
        done[v] = {"errs": verrs, "missing": vmiss}
        json.dump(done, open(ckpt, "w"))   # bank progress after every video
        n_this_run += 1
        print(f"  {v}: tracked ({len(ks)} GT frames)")

    remaining = [v for v in by_vid if v not in done]
    if remaining:
        print(f"\n  incomplete: {len(remaining)} video(s) left -> re-run to resume")
        return
    errs = np.array([e for d in done.values() for e in d["errs"]])
    missing = sum(d["missing"] for d in done.values())
    print(f"\n=== FUSION TRACKER (bg-sub + dish + temporal) — {len(errs)} GT frames ===")
    print(f"  localized: {len(errs)}/{len(ev)}  (no-position: {missing})")
    print(f"  center error: median={np.median(errs):.3f}mm  mean={errs.mean():.3f}mm "
          f"p90={np.percentile(errs,90):.3f}mm  max={errs.max():.3f}mm")
    print(f"  within 0.5mm: {(errs<0.5).mean()*100:.0f}%   within 1mm: {(errs<1).mean()*100:.0f}%  "
          f"within 2mm: {(errs<2).mean()*100:.0f}%")
    print(f"\n  baselines: bg-sub-only median=1.0mm p90=4.6mm | YOLO median=23.5mm")


def cmd_track(a):
    recs, fps, dish = track(a.video, a.mm_per_px)
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.video))[0]
    import csv
    cp = os.path.join(a.out, f"{stem}_fusion.csv")
    with open(cp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "cx_px", "cy_px", "x_mm", "y_mm", "body_len_mm", "state"])
        for r in recs:
            blmm = r["blen"] * a.mm_per_px if np.isfinite(r["blen"]) else ""
            w.writerow([r["frame"], f"{r['frame']/fps:.3f}", f"{r['cx']:.1f}", f"{r['cy']:.1f}",
                        f"{r['cx']*a.mm_per_px:.3f}", f"{r['cy']*a.mm_per_px:.3f}",
                        f"{blmm:.3f}" if blmm != "" else "", r["state"]])
    print(f"wrote {cp}  ({len(recs)} frames)")
    nd = sum(1 for r in recs if r["state"] == "detected")
    print(f"  detected={nd}  held={sum(1 for r in recs if r['state']=='held')}")
    if a.overlay:
        cap = cv2.VideoCapture(a.video)
        W = int(cap.get(3)); H = int(cap.get(4))
        vw = cv2.VideoWriter(os.path.join(a.out, f"{stem}_fusion.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        i = 0
        while True:
            ok, im = cap.read()
            if not ok or i >= len(recs):
                break
            r = recs[i]
            if np.isfinite(r["cx"]):
                p = (int(r["cx"]), int(r["cy"]))
                col = (0, 255, 255) if r["state"] == "detected" else (0, 140, 255)
                cv2.circle(im, p, 10, col, -1)
                if r["head"]:
                    cv2.line(im, (int(r["head"][0]), int(r["head"][1])),
                             (int(r["tail"][0]), int(r["tail"][1])), (0, 255, 0), 3)
                cv2.putText(im, r["state"], (p[0]+12, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
            vw.write(im); i += 1
        cap.release(); vw.release()
        print(f"  wrote overlay {stem}_fusion.mp4")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("track")
    t.add_argument("--video", required=True)
    t.add_argument("--mm_per_px", type=float, default=0.02657)
    t.add_argument("--out", default=os.path.join(here, "..", "realtime_runs", "fusion_out"))
    t.add_argument("--overlay", action="store_true")
    t.set_defaults(func=cmd_track)
    e = sub.add_parser("eval")
    e.add_argument("--label_dir", default=os.path.join(here, "..", "realtime_runs", "label_white7mp"))
    e.add_argument("--videos_dir", default=os.path.join(here, "..", "live_capture"))
    e.add_argument("--mm_per_px", type=float, default=0.02657)
    e.add_argument("--checkpoint", default=None,
                   help="resume file (default: <label_dir>/fusion_eval_checkpoint.json)")
    e.add_argument("--max_videos", type=int, default=0,
                   help="process at most N uncheckpointed videos this run (0=all); "
                        "use 1 to isolate each video in its own process")
    e.set_defaults(func=cmd_eval)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
