#!/usr/bin/env python3
"""
train_yolo.py — Finetune a YOLO pose model (worm box + head/tail) on Apple MPS.

Trains on the pseudo-labeled S3 dataset built by build_yolo_dataset.py.
Worms are small relative to the 3360x2100 frame, so we train at higher imgsz.

Usage:
  python train_yolo.py --data ../realtime_runs/yolo_dataset/data.yaml \
      --model yolo11n-pose.pt --imgsz 1024 --epochs 120 --batch 8
"""
import argparse
import os
from ultralytics import YOLO


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(here, "..", "realtime_runs", "yolo_dataset", "data.yaml"))
    ap.add_argument("--model", default="yolo11n-pose.pt")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--project", default=os.path.join(here, "runs", "pose"))
    ap.add_argument("--name", default="worm_s3_n")
    a = ap.parse_args()

    model = YOLO(a.model)
    model.train(
        data=a.data,
        imgsz=a.imgsz,
        epochs=a.epochs,
        batch=a.batch,
        device=a.device,
        project=a.project,
        name=a.name,
        patience=30,
        # single large worm; keep geometric aug modest, lean on photometric
        degrees=180.0, fliplr=0.5, flipud=0.5, mosaic=0.0, scale=0.3,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, translate=0.1,
        close_mosaic=0, plots=True, verbose=True,
    )
    print("TRAIN_DONE", model.trainer.best)


if __name__ == "__main__":
    main()
