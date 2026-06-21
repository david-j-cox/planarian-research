#!/usr/bin/env python3
"""
infer_video.py — End-to-end deployment: video -> per-frame worm location +
head/tail + behavior, written to CSV (and optional overlay video).

Pipeline:
  1. YOLO model (best.pt) detects the worm box in each frame (box center =
     cx, cy). The trained head/tail keypoints collapse to the centroid because
     the pseudo-label head/tail assignment is anatomically inconsistent, so we
     do NOT use them.
  2. Body axis + length come from PCA on the worm's dark pixels inside the
     detected box (a real measurement: recovers ~8 mm body length). The two
     axis endpoints are disambiguated frame-to-frame by temporal continuity so
     head_angle stays smooth.
  3. behavior_features.compute_features + the trained behavior_clf.joblib give a
     multi-label behavior per frame.

Usage:
  python infer_video.py --video ../live_capture/2026-06-02_15-07-26.mkv \
      --model runs/pose/worm_s3_n/weights/best.pt \
      --behavior ../realtime_runs/behavior_clf.joblib \
      --calibration calibration.json --out ../realtime_runs/infer_demo --overlay
"""
import argparse
import json
import os
import csv
import numpy as np
import cv2
from ultralytics import YOLO

from behavior_features import compute_features


def axis_from_box(img, box, pad=10):
    """PCA on the worm's dark pixels inside the detected box.
    Returns (endpoint_a, endpoint_b, body_len_px) in full-image coords, or
    (None, None, nan) if no contour is found."""
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = x1 + pad, y1 + pad
    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        return None, None, np.nan
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None, None, np.nan
    pts = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    mean = pts.mean(0)
    axis = np.linalg.svd(pts - mean)[2][0]
    proj = (pts - mean) @ axis
    pa = mean + axis * proj.max() + [x0, y0]
    pb = mean + axis * proj.min() + [x0, y0]
    return tuple(pa), tuple(pb), float(proj.max() - proj.min())


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default=os.path.join(here, "runs", "pose", "worm_s3_n", "weights", "best.pt"))
    ap.add_argument("--behavior", default=os.path.join(here, "..", "realtime_runs", "behavior_clf.joblib"))
    ap.add_argument("--calibration", default=os.path.join(here, "calibration.json"))
    ap.add_argument("--mm_per_px", type=float, default=None,
                    help="Override calibration mm_per_px (must match the rig of --video).")
    ap.add_argument("--out", default=os.path.join(here, "..", "realtime_runs", "infer_demo"))
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--overlay", action="store_true", help="Also write an annotated mp4.")
    ap.add_argument("--max_frames", type=int, default=0, help="0 = all frames.")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.video))[0]

    mmpp = a.mm_per_px
    if mmpp is None and os.path.exists(a.calibration):
        try:
            cal = json.load(open(a.calibration))
            mmpp = float(cal.get("mm_per_px") or cal.get("mmpp"))
        except Exception:
            mmpp = None

    model = YOLO(a.model)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if a.max_frames:
        n_total = min(n_total, a.max_frames)

    # Pass 1: detect + PCA body axis (head/tail by temporal continuity).
    rec = {k: [] for k in ("frame", "cx", "cy", "hx", "hy", "tx", "ty", "conf", "lost")}
    frames_for_overlay = []
    prev_head = None
    fno = 0
    while True:
        ok, img = cap.read()
        if not ok or (a.max_frames and fno >= a.max_frames):
            break
        r = model.predict(img, imgsz=a.imgsz, conf=a.conf, device=a.device,
                          iou=0.5, agnostic_nms=True, verbose=False)[0]
        if r.boxes is not None and len(r.boxes):
            bi = int(np.argmax(r.boxes.conf.cpu().numpy()))
            box = r.boxes.xyxy.cpu().numpy()[bi]
            x0, y0, x1, y1 = box
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            conf = float(r.boxes.conf.cpu().numpy()[bi])
            pa, pb, _ = axis_from_box(img, box)
            if pa is None:
                hx = hy = tx = ty = np.nan
            else:
                # Keep "head" near the previous head so head_angle stays smooth.
                if prev_head is not None and (np.hypot(pb[0]-prev_head[0], pb[1]-prev_head[1])
                                              < np.hypot(pa[0]-prev_head[0], pa[1]-prev_head[1])):
                    pa, pb = pb, pa
                (hx, hy), (tx, ty) = pa, pb
                prev_head = pa
            lost = 0
        else:
            cx = cy = hx = hy = tx = ty = np.nan
            conf, lost = 0.0, 1
        rec["frame"].append(fno); rec["cx"].append(cx); rec["cy"].append(cy)
        rec["hx"].append(hx); rec["hy"].append(hy); rec["tx"].append(tx); rec["ty"].append(ty)
        rec["conf"].append(conf); rec["lost"].append(lost)
        if a.overlay:
            frames_for_overlay.append(img)
        fno += 1
    cap.release()
    n = len(rec["frame"])
    print(f"detected worm in {n - int(np.sum(rec['lost']))}/{n} frames")

    # Assemble signals dict for compute_features (chord approx for body_len/head_angle).
    head = np.array([rec["hx"], rec["hy"]]).T
    tail = np.array([rec["tx"], rec["ty"]]).T
    body_len = np.hypot(head[:, 0] - tail[:, 0], head[:, 1] - tail[:, 1])
    head_ang = np.degrees(np.arctan2(head[:, 1] - tail[:, 1], head[:, 0] - tail[:, 0]))
    midline = np.stack([head, tail], axis=1)  # (n, 2, 2)
    sig = {
        "video": np.array([os.path.basename(a.video)] * n),
        "native_frame": np.array(rec["frame"], int),
        "time_s": np.array(rec["frame"], float) / fps,
        "cx_px": np.array(rec["cx"], float), "cy_px": np.array(rec["cy"], float),
        "body_len_px": body_len, "head_angle_deg": head_ang,
        "midline": midline, "lost": np.array(rec["lost"], np.int8),
        "mm_per_px": np.float32(mmpp if mmpp else 1.0), "fps": np.float32(fps),
    }

    # Behavior.
    behaviors = ["unknown"] * n
    if os.path.exists(a.behavior):
        import joblib
        bundle = joblib.load(a.behavior)
        feats = compute_features(sig, bundle["window_s"])
        Xcols = [feats[k] for k in bundle["features"]]
        X = np.array(Xcols, float).T
        ok = np.all(np.isfinite(X), axis=1)
        pred = np.zeros((n, len(bundle["classes"])), int)
        if ok.any():
            pred[ok] = bundle["model"].predict(X[ok])
        behaviors = [";".join(c for c, v in zip(bundle["classes"], row) if v) or "none"
                     for row in pred]
    else:
        print("WARN: behavior model not found, skipping behavior column")

    # Write CSV.
    csv_path = os.path.join(a.out, f"{stem}_track.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "cx_px", "cy_px", "x_mm", "y_mm",
                    "head_x", "head_y", "tail_x", "tail_y", "body_len_mm",
                    "conf", "lost", "behavior"])
        for i in range(n):
            xmm = rec["cx"][i] * mmpp if mmpp else ""
            ymm = rec["cy"][i] * mmpp if mmpp else ""
            blmm = body_len[i] * mmpp if mmpp else ""
            w.writerow([rec["frame"][i], f"{i/fps:.3f}", f"{rec['cx'][i]:.1f}", f"{rec['cy'][i]:.1f}",
                        f"{xmm:.3f}" if mmpp else "", f"{ymm:.3f}" if mmpp else "",
                        f"{rec['hx'][i]:.1f}", f"{rec['hy'][i]:.1f}", f"{rec['tx'][i]:.1f}", f"{rec['ty'][i]:.1f}",
                        f"{blmm:.3f}" if mmpp else "", f"{rec['conf'][i]:.3f}", rec["lost"][i], behaviors[i]])
    print(f"wrote {csv_path}")

    # Optional overlay.
    if a.overlay and frames_for_overlay:
        vw = cv2.VideoWriter(os.path.join(a.out, f"{stem}_overlay.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        for i, img in enumerate(frames_for_overlay):
            if not rec["lost"][i]:
                p = (int(rec["cx"][i]), int(rec["cy"][i]))
                cv2.circle(img, p, 8, (0, 255, 255), -1)
                if np.isfinite(rec["hx"][i]):
                    cv2.circle(img, (int(rec["hx"][i]), int(rec["hy"][i])), 7, (0, 255, 0), -1)
                    cv2.circle(img, (int(rec["tx"][i]), int(rec["ty"][i])), 7, (0, 0, 255), -1)
                cv2.putText(img, behaviors[i], (p[0] + 12, p[1]), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (255, 255, 255), 3)
            vw.write(img)
        vw.release()
        print(f"wrote overlay mp4 in {a.out}")


if __name__ == "__main__":
    main()
