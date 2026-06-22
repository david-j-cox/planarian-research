#!/usr/bin/env python3
"""
eval_white7mp_localizer.py — Score a YOLO localizer on the held-out 200
white-7MP eval frames, in mm of box-center error, apples-to-apples with the
fusion/bg-sub tracker eval (which reports against the same 200 GT frames).

Usage:
  python eval_white7mp_localizer.py \
      --model runs/pose/worm_white7mp_n/weights/best.pt \
      --label_dir ../realtime_runs/label_white7mp \
      --videos_dir ../live_capture --imgsz 1024 --conf 0.25
"""
import argparse
import json
import os
import cv2
import numpy as np
from ultralytics import YOLO


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(here, "runs", "pose", "worm_white7mp_n", "weights", "best.pt"))
    ap.add_argument("--label_dir", default=os.path.join(here, "..", "realtime_runs", "label_white7mp"))
    ap.add_argument("--videos_dir", default=os.path.join(here, "..", "live_capture"))
    ap.add_argument("--mm_per_px", type=float, default=0.02657)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    man = {it["id"]: it for it in json.load(open(os.path.join(a.label_dir, "manifest.json")))["items"]}
    lab = {int(k): v for k, v in json.load(open(os.path.join(a.label_dir, "labels.json"))).items()}
    ev = [k for k in lab if man[k]["pool"] == "eval" and lab[k].get("present")]
    by_vid = {}
    for k in ev:
        by_vid.setdefault(man[k]["video"], []).append(k)

    model = YOLO(a.model)
    errs, missing = [], 0
    for v, ks in sorted(by_vid.items()):
        vp = os.path.join(a.videos_dir, v)
        if not os.path.exists(vp):
            print(f"  missing video {v}"); missing += len(ks); continue
        cap = cv2.VideoCapture(vp)
        for k in ks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(man[k]["frame"]))
            ok, im = cap.read()
            gt = lab[k]["box"]; gx, gy = (gt[0]+gt[2])/2, (gt[1]+gt[3])/2
            if not ok:
                missing += 1; continue
            r = model.predict(im, imgsz=a.imgsz, conf=a.conf, device=a.device, verbose=False)[0]
            if r.boxes is None or len(r.boxes) == 0:
                missing += 1; continue
            bi = int(np.argmax(r.boxes.conf.cpu().numpy()))
            x0, y0, x1, y1 = r.boxes.xyxy.cpu().numpy()[bi]
            px, py = (x0+x1)/2, (y0+y1)/2
            errs.append(float(np.hypot(px-gx, py-gy) * a.mm_per_px))
        cap.release()
        print(f"  {v}: scored ({len(ks)} GT frames)")

    errs = np.array(errs)
    print(f"\n=== WHITE-7MP YOLO LOCALIZER — {len(errs)}/{len(ev)} detected ===")
    print(f"  model: {a.model}  (imgsz={a.imgsz} conf={a.conf})")
    print(f"  no-detection frames: {missing}")
    if len(errs):
        print(f"  center error: median={np.median(errs):.3f}mm  mean={errs.mean():.3f}mm "
              f"p90={np.percentile(errs,90):.3f}mm  max={errs.max():.3f}mm")
        print(f"  within 0.5mm: {(errs<0.5).mean()*100:.0f}%   within 1mm: {(errs<1).mean()*100:.0f}%  "
              f"within 2mm: {(errs<2).mean()*100:.0f}%   within 5mm: {(errs<5).mean()*100:.0f}%")
    print(f"\n  baselines: bg-sub/fusion median=0.978mm p90=3.79mm max=50.9mm (100% localized) | "
          f"S3-YOLO full-frame median=23.5mm")


if __name__ == "__main__":
    main()
