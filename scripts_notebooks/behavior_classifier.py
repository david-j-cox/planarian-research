#!/usr/bin/env python3
"""
behavior_classifier.py — Train a multi-label behavior classifier on the 80
blind human-labeled windows, using the track features from behavior_features.py.

Each labeled window (manifest.json) maps to a signal row via (video,
center_frame); features are computed at the manifest's window_s so they match
how the rule baseline (predictions_hidden.json) was produced. Behaviors are
multi-label (e.g. "gliding;turning"), so we train one-vs-rest and evaluate with
leave-one-out CV (only 80 samples). The rule classifier is the baseline to beat.

Usage:
  python behavior_classifier.py \
      --signals ../realtime_runs/S3_signals.npz \
      --blind_dir ../realtime_runs/S3_labels_blind \
      --out ../realtime_runs/behavior_clf.joblib
"""
import argparse
import json
import os
import csv
import numpy as np

from behavior_features import load_signals, compute_features

FEATURES = ["speed_mm_s", "disp_mm", "head_osc_deg", "head_reversals",
            "bodylen_cv", "bodylen_cycles", "bodylen_contract",
            "heading_change_deg", "frac_lost",
            "ang_vel_p90_deg_s", "path_curv_deg_mm", "body_curv_deg"]


def build_dataset(signals, blind_dir):
    man = json.load(open(os.path.join(blind_dir, "manifest.json")))
    window_s = float(man.get("window_s", 3.0))
    windows = {w["window_id"]: w for w in man["windows"]}

    labels = {}
    with open(os.path.join(blind_dir, "human_labels.csv")) as f:
        for r in csv.DictReader(f):
            labels[int(r["window_id"])] = set(
                b for b in r["behavior"].split(";") if b)

    sig = load_signals(signals)
    feats = compute_features(sig, window_s)
    video = np.asarray([str(v) for v in sig["video"]])
    nframe = np.asarray(sig["native_frame"]).astype(int)

    # "no_worm" marks unusable windows (detection failure / dish edge); drop them
    # and never let them become a class.
    SKIP = {"no_worm", "unknown"}
    classes = sorted({b for bs in labels.values() for b in bs} - SKIP)
    X, Y, rule, wids = [], [], [], []
    rule_pred = {p["window_id"]: p.get("rule_pred", p.get("model_pred", "unknown"))
                 for p in json.load(open(os.path.join(blind_dir, "predictions_hidden.json")))}

    for wid, behs in sorted(labels.items()):
        behs = behs - SKIP
        if not behs:                       # unusable / empty window: skip
            continue
        w = windows[wid]
        m = (video == w["video"]) & (nframe == int(w["center_frame"]))
        if not m.any():
            continue
        i = int(np.where(m)[0][0])
        row = [feats[k][i] for k in FEATURES]
        if not np.all(np.isfinite(row)):
            continue
        X.append(row)
        Y.append([1 if c in behs else 0 for c in classes])
        rule.append(rule_pred.get(wid, "unknown"))
        wids.append(wid)
    return np.array(X, float), np.array(Y, int), classes, rule, wids, window_s


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", default=os.path.join(here, "..", "realtime_runs", "S3_signals.npz"))
    ap.add_argument("--blind_dir", default=os.path.join(here, "..", "realtime_runs", "S3_labels_blind"))
    ap.add_argument("--out", default=os.path.join(here, "..", "realtime_runs", "behavior_clf.joblib"))
    a = ap.parse_args()

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import f1_score, precision_score, recall_score
    import joblib

    X, Y, classes, rule, wids, window_s = build_dataset(a.signals, a.blind_dir)
    print(f"dataset: {X.shape[0]} windows, {X.shape[1]} features, {len(classes)} classes")
    print(f"classes: {classes}")
    print(f"label support: " + "  ".join(f"{c}={int(Y[:,j].sum())}" for j, c in enumerate(classes)))

    candidates = {
        "logreg": OneVsRestClassifier(make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"))),
        "rf": OneVsRestClassifier(
            RandomForestClassifier(n_estimators=300, max_depth=6,
                                   class_weight="balanced", random_state=0)),
    }

    loo = LeaveOneOut()
    results = {}
    for name, clf in candidates.items():
        P = np.zeros_like(Y)
        for tr, te in loo.split(X):
            clf.fit(X[tr], Y[tr])
            P[te] = clf.predict(X[te])
        macro_f1 = f1_score(Y, P, average="macro", zero_division=0)
        micro_f1 = f1_score(Y, P, average="micro", zero_division=0)
        subset_acc = float((P == Y).all(axis=1).mean())
        results[name] = (macro_f1, micro_f1, subset_acc, P)
        print(f"\n[{name}] LOO-CV  macro-F1={macro_f1:.3f}  micro-F1={micro_f1:.3f}  subset-acc={subset_acc:.3f}")
        for j, c in enumerate(classes):
            p = precision_score(Y[:, j], P[:, j], zero_division=0)
            r = recall_score(Y[:, j], P[:, j], zero_division=0)
            f = f1_score(Y[:, j], P[:, j], zero_division=0)
            print(f"    {c:14s} P={p:.2f} R={r:.2f} F1={f:.2f}  (support={int(Y[:,j].sum())})")

    # Rule baseline: treat single-label rule_pred as a one-hot prediction.
    Pr = np.zeros_like(Y)
    for k, rp in enumerate(rule):
        if rp in classes:
            Pr[k, classes.index(rp)] = 1
    rule_macro = f1_score(Y, Pr, average="macro", zero_division=0)
    rule_micro = f1_score(Y, Pr, average="micro", zero_division=0)
    print(f"\n[rule baseline] macro-F1={rule_macro:.3f}  micro-F1={rule_micro:.3f}")

    best = max(results, key=lambda k: results[k][0])
    print(f"\nBest model: {best} (macro-F1={results[best][0]:.3f})")
    final = candidates[best]
    final.fit(X, Y)
    joblib.dump({"model": final, "features": FEATURES, "classes": classes,
                 "window_s": window_s}, a.out)
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
