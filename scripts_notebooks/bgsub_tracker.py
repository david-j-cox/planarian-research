#!/usr/bin/env python3
"""
bgsub_tracker.py — Localize the worm using the priors the per-frame detector
ignored: (1) a static-background model (median over the video) subtracted from
each frame, and (2) a dish ROI mask. The worm becomes the dominant blob inside
the dish. (Temporal continuity is added in the full tracker; here we score
per-frame so it can be measured against the independent eval labels.)

eval subcommand scores this against the human GT and prints a head-to-head with
the YOLO baseline.
"""
import argparse
import json
import os
import numpy as np
import cv2


def build_background(video, n=40):
    cap = cv2.VideoCapture(video)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if N <= 0:
        cap.release(); return None, None
    frames = []
    for f in np.linspace(0, N - 1, min(n, N)).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, im = cap.read()
        if ok:
            frames.append(im)
    cap.release()
    if not frames:
        return None, None
    bg = np.median(np.stack(frames), axis=0).astype(np.uint8)
    g = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(cv2.medianBlur(g, 9), cv2.HOUGH_GRADIENT, 1.5, 2000,
                               param1=120, param2=60, minRadius=900, maxRadius=1200)
    H, W = g.shape
    dish = tuple(circles[0][0]) if circles is not None else (W/2, H/2, min(H, W)/2*0.95)
    return bg, dish


def detect(frame, bg, dish, min_area=300):
    """Largest blob in the background-subtracted, dish-masked image."""
    diff = cv2.cvtColor(cv2.absdiff(frame, bg), cv2.COLOR_BGR2GRAY)
    cxd, cyd, rd = dish
    mask = np.zeros(diff.shape, np.uint8)
    cv2.circle(mask, (int(cxd), int(cyd)), int(rd * 0.97), 255, -1)
    diff = cv2.bitwise_and(diff, mask)
    th = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
    if not cnts:
        return None
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return [float(x), float(y), float(x + w), float(y + h)]


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0


def ctr(b):
    return ((b[0]+b[2])/2, (b[1]+b[3])/2)


def cmd_eval(a):
    d = a.label_dir
    man = {it["id"]: it for it in json.load(open(os.path.join(d, "manifest.json")))["items"]}
    lab = {int(k): v for k, v in json.load(open(os.path.join(d, "labels.json"))).items()}
    frames_dir = os.path.join(d, "frames")
    MMPP = a.mm_per_px

    ev = [k for k in lab if man[k]["pool"] == "eval" and lab[k].get("present")]
    videos = sorted({man[k]["video"] for k in ev})
    bgcache = {}
    for v in videos:
        vp = os.path.join(a.videos_dir, v)
        if os.path.exists(vp):
            bgcache[v] = build_background(vp)
        else:
            bgcache[v] = (None, None)
    print(f"built backgrounds for {sum(1 for v in bgcache if bgcache[v][0] is not None)}/{len(videos)} videos")

    det = miss = 0
    errs, ious = [], []
    yolo_err, yolo_good = [], 0
    for k in ev:
        it = man[k]; gt = lab[k]["box"]
        bg, dish = bgcache[it["video"]]
        img = cv2.imread(os.path.join(frames_dir, it["image"]))
        if bg is None or img is None:
            continue
        box = detect(img, bg, dish)
        if box is None:
            miss += 1
        else:
            det += 1
            errs.append(np.hypot(*(np.subtract(ctr(box), ctr(gt)))) * MMPP)
            ious.append(iou(box, gt))
        # YOLO baseline from stored proposal
        if it["ndet"] >= 1 and it["proposal"]:
            yolo_err.append(np.hypot(*(np.subtract(ctr(it["proposal"]), ctr(gt)))) * MMPP)
            if iou(it["proposal"], gt) > 0.5:
                yolo_good += 1

    N = len(ev)
    errs = np.array(errs); ious = np.array(ious)
    good = int((ious > 0.5).sum())
    print(f"\n=== BG-SUB + DISH MASK detector — {N} eval frames (worm present in all) ===")
    print(f"  detection rate: {det}/{N} = {det/N*100:.1f}%   missed: {miss}/{N} = {miss/N*100:.1f}%")
    print(f"  good localization (IoU>0.5): {good}/{N} = {good/N*100:.1f}%")
    if len(errs):
        print(f"  localization err: median={np.median(errs):.3f}mm  p90={np.percentile(errs,90):.3f}mm")
        print(f"  within 0.5mm: {(errs<0.5).mean()*100:.0f}%   within 1mm: {(errs<1).mean()*100:.0f}%")
    print(f"\n=== YOLO baseline (nano, S3-trained) for comparison ===")
    print(f"  good localization (IoU>0.5): {yolo_good}/{N} = {yolo_good/N*100:.1f}%   "
          f"median err (when it fires): {np.median(yolo_err):.1f}mm")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("eval")
    p.add_argument("--label_dir", default=os.path.join(here, "..", "realtime_runs", "label_white7mp"))
    p.add_argument("--videos_dir", default="/Users/davidjcox/Library/CloudStorage/GoogleDrive-cox.david.j@gmail.com/My Drive/PlanarianVideos/tier2_corpus/live_capture")
    p.add_argument("--mm_per_px", type=float, default=0.02657)
    p.set_defaults(func=cmd_eval)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
