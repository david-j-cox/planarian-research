#!/usr/bin/env python3
"""
rt_behavior.py -- behavior classification on a clip.

`track_and_behavior` is the UNIFIED per-clip pass: ONE YOLO inference per frame
yields BOTH the location track (worm blob centroid -> Hampel-cleaned mm + speed,
identical format/quality to rt_dryrun.track_location) AND the morphology that the
behavior features need. This replaces the old two-pass approach (a location pass
+ a separate morphology pass) -- the second YOLO pass was the redundant cost, not
the RandomForest. `clip_behavior` is the standalone behavior-only path (kept for
offline use); both share `_behavior_from_sig`.
"""
import math
import os

import cv2
import numpy as np

import yolo_tracker as yt
import fusion_tracker as ft
from yolo_to_signals import video_to_signal
from behavior_features import compute_features

MP = yt.MIDLINE_POINTS


def _behavior_from_sig(sig, clf_art):
    """Per-frame behavior from an in-memory signal dict (same schema as the npz),
    so live predictions match the trained model exactly."""
    model = clf_art["model"]; FEATURES = clf_art["features"]; classes = list(clf_art["classes"])
    window_s = float(clf_art.get("window_s", 3.0))
    feats = compute_features(sig, window_s)

    X = np.column_stack([feats[k] for k in FEATURES])
    nf = np.asarray(sig["native_frame"]); ts = np.asarray(sig["time_s"], float)
    lost = np.asarray(sig["lost"])
    ok = np.all(np.isfinite(X), axis=1) & (lost == 0)

    proba = np.full((len(X), len(classes)), np.nan)
    if ok.any():
        proba[ok] = model.predict_proba(X[ok])

    dom = np.full(len(X), -1, int)
    conf_top = np.full(len(X), np.nan)
    if ok.any():
        dom[ok] = np.argmax(proba[ok], axis=1)
        conf_top[ok] = np.max(proba[ok], axis=1)

    n_ok = int(ok.sum())
    summary = {"n_frames": len(X), "n_scored": n_ok}
    if n_ok:
        topc = conf_top[ok]; domc = dom[ok]
        summary["mean_conf"] = float(np.mean(topc))
        summary["frac_lowconf"] = float(np.mean(topc < 0.6))
        frac = {classes[i]: float(np.mean(domc == i)) for i in range(len(classes))}
        summary["behavior_frac"] = frac
        summary["dominant"] = max(frac, key=frac.get)
    return {"native_frame": nf, "time_s": ts, "dom": dom, "conf": conf_top,
            "classes": classes, "proba": proba, "ok": ok, "summary": summary}


def _sig_from_cols(cols, fps, mm_per_px):
    return {
        "video": np.array(cols["video"]),
        "native_frame": np.array(cols["native_frame"], dtype=np.int32),
        "time_s": np.array(cols["time_s"], dtype=np.float32),
        "cx_px": np.array(cols["cx_px"], dtype=np.float32),
        "cy_px": np.array(cols["cy_px"], dtype=np.float32),
        "midline": np.array(cols["midline"], dtype=np.float32),
        "body_len_px": np.array(cols["body_len_px"], dtype=np.float32),
        "head_angle_deg": np.array(cols["head_angle_deg"], dtype=np.float32),
        "lost": np.array(cols["lost"], dtype=np.int8),
        "fps": np.float32(fps), "mm_per_px": np.float32(mm_per_px),
    }


def clip_behavior(video, yolo_model, clf_art, mm_per_px=0.02657, conf=0.10,
                  imgsz=1024, device="mps"):
    """Standalone behavior-only path (runs its own YOLO pass). For live use prefer
    track_and_behavior, which produces location too from a single pass."""
    cols, fps, _ = video_to_signal(video, yolo_model, mm_per_px, conf, imgsz, device)
    return _behavior_from_sig(_sig_from_cols(cols, fps, mm_per_px), clf_art)


def track_and_behavior(video, yolo_model, bg, dish, fps, clf_art, mm_per_px=0.02657,
                       conf=0.10, imgsz=1024, device="mps", stride=2):
    """ONE YOLO pass -> (location_recs, behavior). location_recs match
    rt_dryrun.track_location: (native_frame, time_s, x_mm, y_mm, speed_mm_s, conf,
    state). Position = worm blob centroid (yt.box_morphology), Hampel-cleaned."""
    vid = os.path.basename(video)
    cap = cv2.VideoCapture(video)
    cols = {k: [] for k in ("video", "native_frame", "time_s", "cx_px", "cy_px",
                            "midline", "body_len_px", "head_angle_deg", "lost")}
    confs = []
    prev_mid = None; fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if stride > 1 and (fi % stride):
            fi += 1
            continue
        r = yolo_model.predict(frame, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        det = yt.yolo_box_in_dish(r, dish)
        row = np.full((MP, 2), np.nan, np.float32); ha = np.nan; blen = np.nan
        if det is not None:
            bx, by, box, cf = det
            centroid, midline, _h, _t, blen_px = yt.box_morphology(frame, bg, box, (bx, by), prev_mid)
            px, py = (centroid if centroid is not None else (bx, by))
            if midline is not None and len(midline) >= 2:
                ml = np.asarray(midline, np.float32); k = min(len(ml), MP); row[:k] = ml[:k]
                hv = ml[0] - ml[1]; ha = math.degrees(math.atan2(hv[1], hv[0]))
                prev_mid = midline
            blen = blen_px if np.isfinite(blen_px) else np.nan
            lost = 0
        else:
            px = py = np.nan; cf = 0.0; lost = 1
        cols["video"].append(vid); cols["native_frame"].append(fi)
        cols["time_s"].append(fi / fps); cols["cx_px"].append(px); cols["cy_px"].append(py)
        cols["midline"].append(row); cols["body_len_px"].append(blen)
        cols["head_angle_deg"].append(ha); cols["lost"].append(lost); confs.append(cf)
        fi += 1
    cap.release()
    if not cols["native_frame"]:
        return [], None

    frames = np.array(cols["native_frame"]); xs = np.array(cols["cx_px"]); ys = np.array(cols["cy_px"])
    cxc, cyc, keep = ft.hampel_clean(xs, ys, fps / max(1, stride), mm_per_px,
                                     floor_mult=2.0, abs_floor=20.0)
    xmm, ymm = cxc * mm_per_px, cyc * mm_per_px
    recs = []
    for i in range(len(frames)):
        if i == 0:
            spd = 0.0
        else:
            dt = (frames[i] - frames[i - 1]) / fps
            spd = (np.hypot(xmm[i] - xmm[i - 1], ymm[i] - ymm[i - 1]) / dt if dt > 0 else 0.0)
        recs.append((int(frames[i]), float(frames[i] / fps), float(xmm[i]), float(ymm[i]),
                     float(spd), float(confs[i]), "detected" if keep[i] else "interp"))

    beh = _behavior_from_sig(_sig_from_cols(cols, fps, mm_per_px), clf_art)
    return recs, beh
