#!/usr/bin/env python3
"""
eval_rigorous.py — Adversarial evaluation of the worm localizer, designed to
break the circularity of "model matches the tracker it learned from".

Run over fully held-out videos and report:
  1. Dense agreement vs the classical tracker on non-lost frames (N ~ thousands).
  2. RECOVERY on frames the tracker LOST: does YOLO find a worm the teacher
     could not? (Copying cannot explain recovery.) + a visual montage.
  3. Detection-count distribution: false positives / missed frames (want ~1/frame).
  4. Physical plausibility: track speed distribution, % implausible jumps (>7 mm/s).
"""
import argparse
import json
import os
import numpy as np
import cv2
from ultralytics import YOLO

MMPP_S3 = 0.0265731
FPS = 30.0
MAX_SPEED = 7.0  # mm/s, per filter_jumps


def run_video(model, path, imgsz, conf, device, iou=0.5):
    cap = cv2.VideoCapture(path)
    centers, ndet, confs = {}, {}, {}
    fno = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        r = model.predict(img, imgsz=imgsz, conf=conf, device=device,
                          iou=iou, agnostic_nms=True, verbose=False)[0]
        k = len(r.boxes) if r.boxes is not None else 0
        ndet[fno] = k
        if k:
            bi = int(np.argmax(r.boxes.conf.cpu().numpy()))
            x0, y0, x1, y1 = r.boxes.xyxy.cpu().numpy()[bi]
            centers[fno] = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
            confs[fno] = float(r.boxes.conf.cpu().numpy()[bi])
        fno += 1
    cap.release()
    return centers, ndet, confs, fno


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(here, "runs", "pose", "worm_s3_n", "weights", "best.pt"))
    ap.add_argument("--signals", default=os.path.join(here, "..", "realtime_runs", "S3_signals.npz"))
    ap.add_argument("--videos_dir", default=os.path.join(here, "..", "live_capture"))
    ap.add_argument("--videos", nargs="*", default=["2026-06-02_15-05-45", "2026-06-02_15-07-26"])
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--montage", default=os.path.join(here, "..", "realtime_runs", "recovery_montage.png"))
    a = ap.parse_args()

    z = np.load(a.signals, allow_pickle=True)
    svid = np.array([str(v) for v in z["video"]]); sfr = z["native_frame"].astype(int)
    scx, scy, slost = z["cx_px"], z["cy_px"], z["lost"]

    model = YOLO(a.model)
    agree, recov_hit, recov_tot = [], 0, 0
    counts = []
    speeds_all = []
    recov_examples = []  # (video, frame, (cx,cy))

    for stem in a.videos:
        path = os.path.join(a.videos_dir, stem + ".mkv")
        if not os.path.exists(path):
            print(f"  skip missing {path}"); continue
        centers, ndet, confs, n = run_video(model, path, a.imgsz, a.conf, a.device, a.iou)
        counts.extend(ndet.values())

        # tracker rows for this video
        m = svid == (stem + ".mkv")
        tfr = sfr[m]; tcx = scx[m]; tcy = scy[m]; tlost = slost[m]
        tmap = {int(f): (cx, cy, int(l)) for f, cx, cy, l in zip(tfr, tcx, tcy, tlost)}

        for f, (px, py) in centers.items():
            if f in tmap:
                cx, cy, l = tmap[f]
                if l == 0 and np.isfinite(cx):
                    agree.append(float(np.hypot(px - cx, py - cy) * MMPP_S3))
        # recovery on tracker-lost frames
        for f, (cx, cy, l) in tmap.items():
            if l == 1:
                recov_tot += 1
                if ndet.get(f, 0) >= 1:
                    recov_hit += 1
                    if len(recov_examples) < 12:
                        recov_examples.append((stem, f, centers[f]))
        # speed plausibility from YOLO track
        fs = sorted(centers)
        for i in range(1, len(fs)):
            if fs[i] - fs[i - 1] == 1:
                (x0, y0), (x1, y1) = centers[fs[i - 1]], centers[fs[i]]
                speeds_all.append(np.hypot(x1 - x0, y1 - y0) * MMPP_S3 * FPS)
        print(f"{stem}: frames={n}  detected={sum(1 for v in ndet.values() if v>=1)}/{n}")

    agree = np.array(agree); counts = np.array(counts); speeds = np.array(speeds_all)
    print("\n=== 1. DENSE agreement vs tracker (non-lost frames) ===")
    print(f"  N={len(agree)}  median={np.median(agree):.3f}mm  p90={np.percentile(agree,90):.3f}mm "
          f"max={agree.max():.3f}mm  within1mm={(agree<1).mean()*100:.1f}%")
    print("\n=== 2. RECOVERY on tracker-LOST frames (beats its teacher) ===")
    print(f"  tracker-lost frames={recov_tot}  YOLO recovered={recov_hit} ({recov_hit/max(1,recov_tot)*100:.1f}%)")
    print("\n=== 3. Detection-count distribution (want ~1/frame) ===")
    for c in range(0, max(counts.max(), 2) + 1):
        print(f"  {c} detections: {(counts==c).sum()} frames ({(counts==c).mean()*100:.1f}%)")
    print("\n=== 4. Physical plausibility (track speed) ===")
    print(f"  N={len(speeds)}  median={np.median(speeds):.2f}mm/s  p90={np.percentile(speeds,90):.2f}mm/s "
          f"  implausible(>{MAX_SPEED}mm/s)={ (speeds>MAX_SPEED).mean()*100:.2f}%")

    # montage of recovered worms on tracker-lost frames
    if recov_examples:
        tiles = []
        for stem, f, (cx, cy) in recov_examples:
            cap = cv2.VideoCapture(os.path.join(a.videos_dir, stem + ".mkv"))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, img = cap.read(); cap.release()
            if not ok:
                continue
            cx, cy = int(cx), int(cy)
            cv2.circle(img, (cx, cy), 14, (0, 255, 255), 3)
            crop = img[max(0, cy-160):cy+160, max(0, cx-160):cx+160]
            if crop.size:
                tiles.append(cv2.resize(crop, (220, 220)))
        if tiles:
            while len(tiles) % 4:
                tiles.append(np.zeros((220, 220, 3), np.uint8))
            rows = [np.hstack(tiles[i:i+4]) for i in range(0, len(tiles), 4)]
            cv2.imwrite(a.montage, np.vstack(rows))
            print(f"\nwrote recovery montage ({len(tiles)} tracker-lost frames YOLO recovered) -> {a.montage}")


if __name__ == "__main__":
    main()
