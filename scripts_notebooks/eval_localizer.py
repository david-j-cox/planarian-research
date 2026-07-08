#!/usr/bin/env python3
"""
eval_localizer.py — Evaluate a trained YOLO worm localizer against the
hand-marked ground-truth points (the 73 independent GT labels), reporting
localization error in millimeters.

This is the metric that matters for the science (mm error), not YOLO's
internal mAP. Results are split by whether the GT point's video was in the
training set or held out for validation, so the held-out number is the honest
generalization estimate.

Usage:
  python eval_localizer.py --model runs/pose/worm_s3_n/weights/best.pt \
      --labels ../realtime_runs/S3_labels.json --videos_dir ../live_capture \
      --val_videos 2026-06-02_15-05-45 2026-06-02_15-07-26
"""
import argparse
import json
import os
import cv2
import numpy as np
from ultralytics import YOLO


def grab_frame(videos_dir, video, frame):
    path = os.path.join(videos_dir, video)
    if not os.path.exists(path):
        return None
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(here, "runs", "pose", "worm_s3_n", "weights", "best.pt"))
    ap.add_argument("--labels", default=os.path.join(here, "..", "realtime_runs", "S3_labels.json"))
    ap.add_argument("--videos_dir", default=os.path.join(here, "..", "live_capture"))
    ap.add_argument("--val_videos", nargs="*", default=["2026-06-02_15-05-45", "2026-06-02_15-07-26"])
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    lab = json.load(open(a.labels))
    mmpp = float(lab["mm_per_px"])
    gt = lab["worm_truth"]
    val_set = set(a.val_videos)
    model = YOLO(a.model)

    rows = []  # (split, err_mm, head_err_mm or nan, detected)
    for g in gt:
        img = grab_frame(a.videos_dir, g["video"], g["frame"])
        if img is None:
            continue
        stem = os.path.splitext(g["video"])[0]
        split = "val" if stem in val_set else "train"
        r = model.predict(img, imgsz=a.imgsz, conf=a.conf, device=a.device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            rows.append((split, np.nan, np.nan, 0))
            continue
        # highest-confidence detection
        bi = int(np.argmax(r.boxes.conf.cpu().numpy()))
        x0, y0, x1, y1 = r.boxes.xyxy.cpu().numpy()[bi]
        pcx, pcy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        err = float(np.hypot(pcx - g["x_px"], pcy - g["y_px"]) * mmpp)
        head_err = np.nan
        if r.keypoints is not None and r.keypoints.xy is not None and len(r.keypoints.xy) > bi:
            kp = r.keypoints.xy.cpu().numpy()[bi]
            if kp.shape[0] >= 1:
                head_err = float(np.hypot(kp[0, 0] - g["x_px"], kp[0, 1] - g["y_px"]) * mmpp)
        rows.append((split, err, head_err, 1))

    def report(split):
        sub = [r for r in rows if r[0] == split]
        if not sub:
            print(f"  {split}: no GT points")
            return
        det = np.array([r[3] for r in sub])
        errs = np.array([r[1] for r in sub if r[3] == 1])
        print(f"  {split}: n={len(sub)}  detected={int(det.sum())}/{len(sub)} ({det.mean()*100:.0f}%)")
        if len(errs):
            print(f"    box-center err mm: median={np.median(errs):.3f}  mean={errs.mean():.3f} "
                  f"p90={np.percentile(errs,90):.3f}  max={errs.max():.3f}")
            print(f"    within 0.5mm: {(errs<0.5).mean()*100:.0f}%   within 1mm: {(errs<1.0).mean()*100:.0f}%")

    print(f"model={a.model}")
    print(f"labels={os.path.basename(a.labels)}  mm_per_px={mmpp:.5f}  GT points={len(gt)}")
    for split in ("train", "val"):
        report(split)


if __name__ == "__main__":
    main()
