#!/usr/bin/env python3
"""
rt_behavior.py -- run the trained behavior classifier on a single clip.

Reuses the SAME signal + feature path as training (yolo_to_signals.video_to_signal
-> behavior_features.compute_features -> the joblib model) so live predictions
match the trained model exactly. Returns, per frame, the dominant behavior and its
confidence, plus a clip-level summary used for (a) the ethogram and (b) deciding
whether a clip is worth keeping for labeling (low-confidence / rare-class = keep).
"""
import numpy as np

from yolo_to_signals import video_to_signal
from behavior_features import compute_features


def clip_behavior(video, yolo_model, clf_art, mm_per_px=0.02657, conf=0.10,
                  imgsz=1024, device="mps"):
    cols, fps, _ = video_to_signal(video, yolo_model, mm_per_px, conf, imgsz, device)
    # assemble the in-memory signal dict in the same schema the npz uses, so
    # compute_features sees exactly what it saw at training time.
    sig = {
        "video": np.array(cols["video"]),
        "native_frame": np.array(cols["native_frame"], dtype=np.int32),
        "time_s": np.array(cols["time_s"], dtype=np.float32),
        "cx_px": np.array(cols["cx_px"], dtype=np.float32),
        "cy_px": np.array(cols["cy_px"], dtype=np.float32),
        "midline": np.array(cols["midline"], dtype=np.float32),
        "body_len_px": np.array(cols["body_len_px"], dtype=np.float32),
        "head_angle_deg": np.array(cols["head_angle_deg"], dtype=np.float32),
        "lost": np.array(cols["lost"], dtype=np.int8),
        "fps": np.float32(fps),
        "mm_per_px": np.float32(mm_per_px),
    }
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

    dom = np.full(len(X), -1, int)              # index into classes; -1 = no call
    conf_top = np.full(len(X), np.nan)
    if ok.any():
        dom[ok] = np.argmax(proba[ok], axis=1)
        conf_top[ok] = np.max(proba[ok], axis=1)

    # clip-level summary for active-learning keep decisions
    n_ok = int(ok.sum())
    summary = {"n_frames": len(X), "n_scored": n_ok}
    if n_ok:
        topc = conf_top[ok]
        domc = dom[ok]
        summary["mean_conf"] = float(np.mean(topc))
        summary["frac_lowconf"] = float(np.mean(topc < 0.6))     # uncertain frames
        # dominant-behavior fractions across the clip
        frac = {classes[i]: float(np.mean(domc == i)) for i in range(len(classes))}
        summary["behavior_frac"] = frac
        summary["dominant"] = max(frac, key=frac.get)
    return {"native_frame": nf, "time_s": ts, "dom": dom, "conf": conf_top,
            "classes": classes, "proba": proba, "ok": ok, "summary": summary}
