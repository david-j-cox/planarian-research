#!/usr/bin/env python3
"""
yolo_tracker.py — YOLO-primary worm tracker for the white-7MP rig.

Position comes from the rig-trained YOLO localizer (runs/pose/worm_white7mp_n),
which on 200 held-out GT frames scored median 0.67mm / p90 1.59mm / max 3.66mm
-- crucially with NO wrong-blob tail (bg-sub's max was 50.9mm). The bg-sub
contour nearest the YOLO box supplies morphology (head/tail/body-length via PCA)
for the behavior model; position never depends on bg-sub, so noisy frames no
longer corrupt the track. A safe temporal pass (shared hampel_clean: Hampel
spike rejection + interpolate/hold + light smoothing) fills the rare
no-detection frame and removes jitter.

Subcommands:
  track  one video -> per-frame CSV (+ optional overlay mp4)
  eval   score the tracked position at each GT frame vs the human labels,
         apples-to-apples with fusion_tracker eval (200 white-7MP GT frames).

Usage:
  python yolo_tracker.py track --video ../live_capture/2026-06-02_16-00-37.mkv --overlay
  python yolo_tracker.py eval
"""
import argparse
import csv
import json
import os
import numpy as np
import cv2
from ultralytics import YOLO

import fusion_tracker as ft   # build_background, hampel_clean
from open_dish_tracker import extract_midline   # curved head->tail midline

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "runs", "pose", "worm_white7mp_n", "weights", "best.pt")
MIDLINE_POINTS = 20   # max ordered points along the body curve


def yolo_box_in_dish(result, dish, margin=1.10):
    """Highest-confidence YOLO box whose center lies within the dish (+margin).
    The margin matters: a hard cut at the Hough radius clips worms near the
    meniscus edge (and a stretch of edge frames defeats temporal interp too).
    YOLO rarely fires outside the dish anyway (per-frame max error 3.66mm), so
    the gate is just a light guard against the occasional far false positive.
    Returns (cx, cy, [x0,y0,x1,y1], conf) or None."""
    if result.boxes is None or len(result.boxes) == 0:
        return None
    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    cxd, cyd, rd = dish
    best = None
    for (x0, y0, x1, y1), cf in zip(xyxy, conf):
        bx, by = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        if np.hypot(bx - cxd, by - cyd) > rd * margin:   # reject only far-outside boxes
            continue
        if best is None or cf > best[3]:
            best = (float(bx), float(by), [float(x0), float(y0), float(x1), float(y1)], float(cf))
    return best


def _box_foreground(frame, bg, box, pad):
    """bg-sub foreground (binary) and its full-frame pixel coords INSIDE the YOLO
    box expanded by `pad`. Constraining to the box excludes the dish rim/meniscus
    -- which otherwise merges with an edge worm -- and caps morphology to the box.
    Returns (th_roi, (rx0, ry0)) or (None, None)."""
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    rx0, ry0 = max(0, int(x0 - pad * bw)), max(0, int(y0 - pad * bh))
    rx1, ry1 = min(W, int(x1 + pad * bw)), min(H, int(y1 + pad * bh))
    if rx1 - rx0 < 3 or ry1 - ry0 < 3:
        return None, None
    diff = cv2.cvtColor(cv2.absdiff(frame[ry0:ry1, rx0:rx1], bg[ry0:ry1, rx0:rx1]),
                        cv2.COLOR_BGR2GRAY)
    th = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return th, (rx0, ry0)


def box_morphology(frame, bg, box, pos, prev_midline, pad=0.35, n_points=MIDLINE_POINTS):
    """Body morphology from the bg-sub foreground INSIDE the YOLO box.

    Preferred path: skeletonize the worm's largest contour into an ordered,
    curved head->tail midline (open_dish_tracker.extract_midline), which makes
    the head-end direction independent of the overall body heading -- so the
    head-oscillation (wigwag) features are real signal, not a copy of the turning
    feature. extract_midline also keeps head/tail identity stable across frames
    via prev_midline (third-centroid alignment), so no separate sign-fix is
    needed downstream.

    Fallback (skeleton too short / fails): a straight PCA axis anchored THROUGH
    the YOLO position, same as before -- always yields a head/tail so the track
    never has a hole. Returns (midline_pts, head, tail, blen_px); midline_pts is
    an ordered (k,2) array (>=2 rows) or None only when there is no foreground.
    """
    th, origin = _box_foreground(frame, bg, box, pad)
    if th is None:
        return None, None, None, np.nan
    rx0, ry0 = origin
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None, None, None, np.nan
    cnt = max(cnts, key=cv2.contourArea) + np.array([rx0, ry0])   # -> full-frame

    midline, _curv, blen = extract_midline(cnt, prev_midline, n_points, frame.shape)
    if midline is not None and len(midline) >= 3:
        head = midline[0].tolist()
        tail = midline[-1].tolist()
        return np.asarray(midline, np.float32), head, tail, float(blen)

    # Fallback: straight PCA axis through the YOLO position.
    ys, xs = np.where(th > 0)
    if len(xs) < 5:
        return None, None, None, np.nan
    pts = np.column_stack([xs + rx0, ys + ry0]).astype(np.float32)
    d = pts - pts.mean(0)
    axis = np.linalg.eigh((d.T @ d) / len(d))[1][:, -1]
    proj = (pts - np.asarray(pos, np.float32)) @ axis
    head = (np.asarray(pos) + axis * proj.max()).tolist()
    tail = (np.asarray(pos) + axis * proj.min()).tolist()
    return np.asarray([head, tail], np.float32), head, tail, float(proj.max() - proj.min())


def _thirds(ml):
    """Head-third and tail-third centroids of an ordered midline."""
    k = max(1, len(ml) // 3)
    return ml[:k].mean(0), ml[-k:].mean(0)


def orient_midlines(mids, cx, cy, fps, mm_per_px, alpha=0.25,
                    move_thresh_mm_s=0.3, vsmooth=5):
    """Orient every midline so midline[0] is the biological HEAD, robustly.

    Two stages, chosen because the worm is mostly stationary (~85% of frames
    move < 0.3 mm/s) yet still has one true head:

    1. STABILITY by CENTROID-RELATIVE EMA chaining. Keep pt0 on one physical end
       by matching each frame's head/tail OFFSET FROM THE WORM CENTROID (not its
       absolute pixel position) to an exponential moving average of recent
       offsets. This is translation-invariant, so the box-center position jumps
       (the known jitter -- up to ~30 mm/s, far above the worm's ~7 mm/s max)
       cannot break the reference: the head stays "up-left of center" regardless
       of where the box leaps. Only a real, gradual body rotation moves the
       offset, which the EMA tracks. (Absolute-position matching flipped here:
       on a jump the reference sat far from both new endpoints and the alignment
       became a coin-flip.)

    2. SENSE by global velocity vote, gated to PHYSICAL speeds. Planaria lead
       with the head when translating, so we sum speed-weighted votes over the
       clip for whether pt0 leads the motion, and reverse the whole clip once if
       the tail wins. Frames with non-physical speed (> MAX_SPEED_MM_S, i.e.
       jitter jumps) are excluded so a spurious leap cannot dominate the vote.
       A global vote is also robust to brief tail-first reversal/scrunch bouts.

    Returns (oriented_mids, n_flip_events): residual frame-to-frame head-end
    swaps, the quantity we drive toward 0.
    """
    n = len(mids)
    cx = np.asarray(cx, float); cy = np.asarray(cy, float)

    def _sm(a, w):
        if w <= 1:
            return a
        return np.convolve(np.nan_to_num(a), np.ones(w) / w, mode="same")
    vx = np.gradient(_sm(cx, vsmooth)); vy = np.gradient(_sm(cy, vsmooth))
    spd = np.hypot(vx, vy) * fps * mm_per_px

    # Stage 1: centroid-relative EMA-chained orientation (jump-invariant).
    oriented = [None] * n
    ema_rh = ema_rt = None
    for i in range(n):
        ml = mids[i]
        if ml is None or len(ml) < 3:
            continue
        c = np.array([cx[i], cy[i]])
        h, t = _thirds(ml)
        rh, rt = h - c, t - c               # offsets from the worm centroid
        if ema_rh is not None:
            keep_d = np.linalg.norm(rh - ema_rh) + np.linalg.norm(rt - ema_rt)
            flip_d = np.linalg.norm(rh - ema_rt) + np.linalg.norm(rt - ema_rh)
            if flip_d < keep_d:
                ml = ml[::-1].copy(); rh, rt = rt, rh
        oriented[i] = ml
        ema_rh = rh if ema_rh is None else (1 - alpha) * ema_rh + alpha * rh
        ema_rt = rt if ema_rt is None else (1 - alpha) * ema_rt + alpha * rt

    # Stage 2: global head sense from speed-weighted velocity votes (physical
    # speeds only -- jitter jumps above MAX_SPEED_MM_S are not real travel).
    vote = 0.0
    for i in range(n):
        ml = oriented[i]
        if ml is None or spd[i] < move_thresh_mm_s or spd[i] > ft.MAX_SPEED_MM_S:
            continue
        c = np.array([cx[i], cy[i]]); v = np.array([vx[i], vy[i]])
        h, t = _thirds(ml)
        vote += spd[i] * np.sign((h - c) @ v - (t - c) @ v)
    if vote < 0:                       # pt0 trails motion on average -> it's tail
        oriented = [ml[::-1].copy() if ml is not None else None for ml in oriented]

    # Residual head-direction reversals: frames where the head-end segment angle
    # swings > 150 deg vs the previous present frame -- physically impossible for
    # a real head (it cannot teleport across the body in 1/30 s), so each one is
    # a genuine orientation error. Position-jump invariant (uses the head-segment
    # ANGLE, not endpoint pixel distance), unlike the earlier crude metric.
    flips, prev_ang = 0, None
    for ml in oriented:
        if ml is None:
            continue
        hv = ml[0] - ml[1]
        ang = np.degrees(np.arctan2(hv[1], hv[0]))
        if prev_ang is not None and abs((ang - prev_ang + 180) % 360 - 180) > 150:
            flips += 1
        prev_ang = ang
    return oriented, flips


def track(video, model, mm_per_px, conf=0.10, imgsz=1024, device="mps", min_area=200):
    bg, dish, fps = ft.build_background(video)
    if bg is None:
        return [], fps, dish
    if isinstance(model, str):
        model = YOLO(model)
    cap = cv2.VideoCapture(video)
    raw = []
    prev_midline = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        det = yolo_box_in_dish(r, dish)
        if det is not None:
            bx, by, box, cf = det
            ml, h, t, blen = box_morphology(frame, bg, box, (bx, by), prev_midline)
            if ml is not None and len(ml) >= 3:
                prev_midline = ml   # only the curved midline seeds H/T consistency
            raw.append({"cx": bx, "cy": by, "box": box, "conf": cf,
                        "midline": ml, "head": h, "tail": t, "blen": blen})
        else:
            raw.append({"cx": np.nan, "cy": np.nan, "box": None, "conf": 0.0,
                        "midline": None, "head": None, "tail": None, "blen": np.nan})
    cap.release()

    xs = np.array([r["cx"] for r in raw]); ys = np.array([r["cy"] for r in raw])
    cx, cy, keep = ft.hampel_clean(xs, ys, fps, mm_per_px)

    # Global head/tail orientation from velocity (replaces fragile chaining).
    oriented, nflip = orient_midlines([r["midline"] for r in raw], cx, cy, fps, mm_per_px)
    for i, r in enumerate(raw):
        ml = oriented[i]
        r["midline"] = ml
        if ml is not None:
            r["head"] = ml[0].tolist(); r["tail"] = ml[-1].tolist()
    if nflip:
        print(f"  head/tail orientation: {nflip} residual flip events")

    recs, last_ml, last_h, last_t = [], None, None, None
    for i, r in enumerate(raw):
        if keep[i]:
            last_ml, last_h, last_t = r["midline"], r["head"], r["tail"]
            state = "detected"
        else:
            state = "interp"
        recs.append({"frame": i, "cx": float(cx[i]), "cy": float(cy[i]),
                     "box": r["box"] if keep[i] else None, "conf": r["conf"],
                     "midline": last_ml, "head": last_h, "tail": last_t,
                     "blen": r["blen"] if keep[i] else np.nan, "state": state})
    return recs, fps, dish


# ── eval ─────────────────────────────────────────────────────────────
def cmd_eval(a):
    man = {it["id"]: it for it in json.load(open(os.path.join(a.label_dir, "manifest.json")))["items"]}
    lab = {int(k): v for k, v in json.load(open(os.path.join(a.label_dir, "labels.json"))).items()}
    ev = [k for k in lab if man[k]["pool"] == "eval" and lab[k].get("present")]
    by_vid = {}
    for k in ev:
        by_vid.setdefault(man[k]["video"], []).append(k)

    model = YOLO(a.model)
    W = a.window
    errs, missing = [], 0
    for v, ks in sorted(by_vid.items()):
        vp = os.path.join(a.videos_dir, v)
        if not os.path.exists(vp):
            print(f"  missing video {v}"); missing += len(ks); continue
        bg, dish, fps = ft.build_background(vp)
        cap = cv2.VideoCapture(vp)
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for k in ks:
            f = man[k]["frame"]; gt = lab[k]["box"]
            gx, gy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
            # local YOLO track over [f-W, f+W], cleaned, sampled at f
            lo, hi = max(0, f - W), min(nframes, f + W + 1)
            xs, ys = [], []
            for fi in range(lo, hi):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, im = cap.read()
                if not ok:
                    xs.append(np.nan); ys.append(np.nan); continue
                det = yolo_box_in_dish(
                    model.predict(im, imgsz=a.imgsz, conf=a.conf, device=a.device, verbose=False)[0], dish)
                xs.append(det[0] if det else np.nan); ys.append(det[1] if det else np.nan)
            cx, cy, _ = ft.hampel_clean(np.array(xs), np.array(ys), fps, a.mm_per_px)
            px, py = cx[f - lo], cy[f - lo]
            if np.isfinite(px):
                errs.append(float(np.hypot(px - gx, py - gy) * a.mm_per_px))
            else:
                missing += 1
        cap.release()
        print(f"  {v}: scored ({len(ks)} GT frames)")

    errs = np.array(errs)
    print(f"\n=== YOLO TRACKER (YOLO pos + temporal) — {len(errs)}/{len(ev)} localized ===")
    print(f"  model: {a.model}  (imgsz={a.imgsz} conf={a.conf} window=±{W})")
    print(f"  no-position frames: {missing}")
    if len(errs):
        print(f"  center error: median={np.median(errs):.3f}mm  mean={errs.mean():.3f}mm "
              f"p90={np.percentile(errs,90):.3f}mm  max={errs.max():.3f}mm")
        print(f"  within 0.5mm: {(errs<0.5).mean()*100:.0f}%   within 1mm: {(errs<1).mean()*100:.0f}%  "
              f"within 2mm: {(errs<2).mean()*100:.0f}%   within 5mm: {(errs<5).mean()*100:.0f}%")
    print(f"\n  baselines: bg-sub/fusion median=0.978mm p90=3.79mm max=50.9mm (100%) | "
          f"per-frame YOLO median=0.672mm p90=1.59mm max=3.66mm (199/200)")


def cmd_track(a):
    recs, fps, dish = track(a.video, a.model, a.mm_per_px, conf=a.conf,
                            imgsz=a.imgsz, device=a.device)
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.video))[0]
    cp = os.path.join(a.out, f"{stem}_yolo.csv")
    with open(cp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "cx_px", "cy_px", "x_mm", "y_mm",
                    "body_len_mm", "conf", "state"])
        for r in recs:
            blmm = r["blen"] * a.mm_per_px if np.isfinite(r["blen"]) else ""
            w.writerow([r["frame"], f"{r['frame']/fps:.3f}", f"{r['cx']:.1f}", f"{r['cy']:.1f}",
                        f"{r['cx']*a.mm_per_px:.3f}", f"{r['cy']*a.mm_per_px:.3f}",
                        f"{blmm:.3f}" if blmm != "" else "", f"{r['conf']:.3f}", r["state"]])
    nd = sum(1 for r in recs if r["state"] == "detected")
    print(f"wrote {cp}  ({len(recs)} frames, {nd} detected, {len(recs)-nd} interp)")
    if a.overlay:
        cap = cv2.VideoCapture(a.video)
        W = int(cap.get(3)); H = int(cap.get(4))
        # Downscale for smooth playback: full-res 3360x2100 frames stutter in
        # most players. Draw at full res, then resize the written frame.
        s = a.overlay_scale
        sW, sH = int(W * s), int(H * s)
        vw = cv2.VideoWriter(os.path.join(a.out, f"{stem}_yolo.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (sW, sH))
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
                ml = r["midline"]
                if ml is not None and len(ml) >= 2:
                    cv2.polylines(im, [np.asarray(ml, np.int32)], False, (0, 255, 0), 3)
                    cv2.circle(im, (int(ml[0][0]), int(ml[0][1])), 7, (0, 0, 255), -1)  # head
                elif r["head"]:
                    cv2.line(im, (int(r["head"][0]), int(r["head"][1])),
                             (int(r["tail"][0]), int(r["tail"][1])), (0, 255, 0), 3)
            vw.write(cv2.resize(im, (sW, sH)) if s != 1.0 else im); i += 1
        cap.release(); vw.release()
        print(f"  wrote overlay {stem}_yolo.mp4  ({sW}x{sH})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("track")
    t.add_argument("--video", required=True)
    t.add_argument("--model", default=DEFAULT_MODEL)
    t.add_argument("--mm_per_px", type=float, default=0.02657)
    t.add_argument("--conf", type=float, default=0.10)
    t.add_argument("--imgsz", type=int, default=1024)
    t.add_argument("--device", default="mps")
    t.add_argument("--out", default=os.path.join(HERE, "..", "realtime_runs", "yolo_out"))
    t.add_argument("--overlay", action="store_true")
    t.add_argument("--overlay_scale", type=float, default=0.4,
                   help="downscale factor for the overlay mp4 (1.0 = full res)")
    t.set_defaults(func=cmd_track)
    e = sub.add_parser("eval")
    e.add_argument("--model", default=DEFAULT_MODEL)
    e.add_argument("--label_dir", default=os.path.join(HERE, "..", "realtime_runs", "label_white7mp"))
    e.add_argument("--videos_dir", default=os.path.join(HERE, "..", "live_capture"))
    e.add_argument("--mm_per_px", type=float, default=0.02657)
    e.add_argument("--conf", type=float, default=0.10)
    e.add_argument("--imgsz", type=int, default=1024)
    e.add_argument("--device", default="mps")
    e.add_argument("--window", type=int, default=8, help="frames each side for the local temporal pass")
    e.set_defaults(func=cmd_eval)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
