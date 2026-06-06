#!/usr/bin/env python3
"""
behavior_label_tool.py — Blind behavior labeling to validate the rule classifier.

Two subcommands:

  sample  — pick ~N short windows spread across a session, STRATIFIED by the rule
            classifier's predicted behavior (so rare gaits appear in the sample),
            and write a manifest. The predictions are stored in a SEPARATE file
            the GUI never reads, so labeling stays blind/unbiased.

  label   — GUI: for each manifest window, loop the worm video (zoomed crop on
            the worm) and let you assign ONE dominant behavior, blind to the
            rule's guess. Saves human labels. Resumable.

Then behavior_accuracy.py joins the human labels to the rule predictions and
reports per-behavior precision/recall + a confusion matrix.

Usage (run `label` on the machine with the MKVs + a display):
  python behavior_label_tool.py sample \
      --signals ../realtime_runs/S3_signals.npz --clips_dir ../live_capture \
      --n 150 --window_s 3 --out_dir ../realtime_runs/S3_labels_blind
  python behavior_label_tool.py label --manifest_dir ../realtime_runs/S3_labels_blind

Controls (label GUI): number keys pick a behavior, R replay, n/SPACE next
(must label first), b back, u clear, q save+quit.
"""

import os
import sys
import csv
import json
import glob
import argparse

import cv2
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
from open_dish_tracker import open_video        # noqa: E402
from behavior_features import load_signals, compute_features  # noqa: E402
from behavior_rules import classify, BEHAVIORS   # noqa: E402

WIN = "Blind Behavior Labeler"
LABELABLE = [b for b in BEHAVIORS if b != "unknown"]  # human picks a real behavior
DISP_MAX_W = 900


# ── sample ────────────────────────────────────────────────────────────
def cmd_sample(args):
    sig = load_signals(args.signals)
    feats = compute_features(sig, args.window_s)
    pred = classify(feats)
    video = feats["_video"]
    nf = np.asarray(sig["native_frame"])
    cx = np.asarray(sig["cx_px"]); cy = np.asarray(sig["cy_px"])
    fps = float(sig["fps"]) if "fps" in sig else 30.0
    half = int(round(args.window_s * fps / 2))

    # Candidate centers = frames whose surrounding window is mostly detected.
    rng = np.random.default_rng(args.seed)
    by_beh = {}
    for i in range(len(pred)):
        by_beh.setdefault(pred[i], []).append(i)

    # Stratified pick: even share per predicted behavior, capped by availability.
    per = max(1, args.n // max(1, len([b for b in by_beh if b != "unknown"])))
    chosen = []
    for b, idxs in by_beh.items():
        if b == "unknown":
            continue
        idxs = [i for i in idxs if not np.isnan(cx[i])]
        if not idxs:
            continue
        take = min(per, len(idxs))
        chosen.extend(rng.choice(idxs, size=take, replace=False).tolist())
    chosen = sorted(set(chosen))[:args.n]

    os.makedirs(args.out_dir, exist_ok=True)
    manifest, hidden = [], []
    for wid, i in enumerate(chosen):
        v = str(video[i]); f0 = int(nf[i])
        manifest.append({"window_id": wid, "video": v,
                         "center_frame": f0,
                         "start_frame": max(0, f0 - half),
                         "end_frame": f0 + half,
                         "cx": float(cx[i]), "cy": float(cy[i])})
        hidden.append({"window_id": wid, "rule_pred": str(pred[i]),
                       "time_s": float(feats["_time_s"][i])})

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump({"window_s": args.window_s, "clips_dir": args.clips_dir,
                   "windows": manifest}, f, indent=2)
    with open(os.path.join(args.out_dir, "predictions_hidden.json"), "w") as f:
        json.dump(hidden, f, indent=2)

    from collections import Counter
    c = Counter(h["rule_pred"] for h in hidden)
    print(f"Sampled {len(manifest)} windows ({args.window_s}s each) -> {args.out_dir}")
    print("Stratification (rule pred, HIDDEN from labeler):",
          {k: c[k] for k in sorted(c)})
    print("Now run:  python behavior_label_tool.py label "
          f"--manifest_dir {args.out_dir}")


# ── label GUI ─────────────────────────────────────────────────────────
def _crop(frame, cx, cy, size):
    h, w = frame.shape[:2]
    half = size // 2
    x0 = int(np.clip(cx - half, 0, max(0, w - size)))
    y0 = int(np.clip(cy - half, 0, max(0, h - size)))
    return frame[y0:y0 + size, x0:x0 + size]


def cmd_label(args):
    md = args.manifest_dir
    with open(os.path.join(md, "manifest.json")) as f:
        man = json.load(f)
    windows = man["windows"]
    clips_dir = args.clips_dir or man["clips_dir"]
    labels_path = os.path.join(md, "human_labels.csv")

    done = {}
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            for r in csv.DictReader(f):
                done[int(r["window_id"])] = r["behavior"]

    # Preload window frames (crops) so playback is smooth.
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    crop_size = args.crop_px

    def load_window(wm):
        cap = open_video(os.path.join(clips_dir, wm["video"]))
        frames = []
        for fr in range(wm["start_frame"], wm["end_frame"] + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            ok, img = cap.read()
            if not ok:
                break
            c = _crop(img, wm["cx"], wm["cy"], crop_size)
            s = min(1.0, DISP_MAX_W / max(1, c.shape[1]))
            frames.append(cv2.resize(c, (int(c.shape[1] * s), int(c.shape[0] * s))))
        cap.release()
        return frames

    i = 0
    n = len(windows)
    while 0 <= i < n:
        wm = windows[i]
        frames = load_window(wm)
        if not frames:
            i += 1
            continue
        sel = done.get(wm["window_id"], "")
        fi = 0
        while True:
            base = frames[fi % len(frames)].copy()
            fi += 1
            panel = np.full((base.shape[0] + 200, max(base.shape[1], 360), 3),
                            25, np.uint8)
            panel[:base.shape[0], :base.shape[1]] = base
            y0 = base.shape[0]
            _put(panel, f"window {i+1}/{n}   [{len(done)} labeled]", (12, y0 + 24),
                 0.6, (0, 220, 0))
            _put(panel, "Pick the DOMINANT behavior (blind):", (12, y0 + 50),
                 0.55, (255, 255, 255))
            for k, b in enumerate(LABELABLE):
                col = (0, 255, 255) if sel == b else (200, 200, 200)
                _put(panel, f"{k+1}. {b}", (12 + (k % 4) * 170,
                     y0 + 78 + (k // 4) * 26), 0.55, col)
            _put(panel, ("SELECTED: " + sel) if sel else "SELECTED: (none)",
                 (12, y0 + 160), 0.6,
                 (0, 255, 0) if sel else (0, 0, 255))
            _put(panel, "R replay  n/SPACE next  b back  u clear  q save+quit",
                 (12, y0 + 186), 0.5, (180, 180, 180))
            cv2.imshow(WIN, panel)
            k = cv2.waitKey(40) & 0xFF
            if ord('1') <= k <= ord('9'):
                j = k - ord('1')
                if j < len(LABELABLE):
                    sel = LABELABLE[j]
            elif k == ord('u'):
                sel = ""
            elif k == ord('r'):
                fi = 0
            elif k == ord('b'):
                if sel:
                    done[wm["window_id"]] = sel
                i = max(0, i - 1)
                break
            elif k in (ord('n'), ord(' ')):
                if not sel:
                    continue  # must label before advancing
                done[wm["window_id"]] = sel
                i += 1
                break
            elif k in (ord('q'), 27):
                if sel:
                    done[wm["window_id"]] = sel
                _save_labels(labels_path, done)
                cv2.destroyAllWindows()
                print(f"Saved {len(done)}/{n} labels -> {labels_path}")
                return
        _save_labels(labels_path, done)

    _save_labels(labels_path, done)
    cv2.destroyAllWindows()
    print(f"Done. {len(done)}/{n} labeled -> {labels_path}")


def _save_labels(path, done):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window_id", "behavior"])
        for wid in sorted(done):
            w.writerow([wid, done[wid]])


def _put(img, s, org, scale, color):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="Build the blind window manifest.")
    s.add_argument("--signals", required=True)
    s.add_argument("--clips_dir", required=True)
    s.add_argument("--n", type=int, default=150)
    s.add_argument("--window_s", type=float, default=3.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out_dir", required=True)
    s.set_defaults(func=cmd_sample)

    l = sub.add_parser("label", help="Blind labeling GUI.")
    l.add_argument("--manifest_dir", required=True)
    l.add_argument("--clips_dir", default=None)
    l.add_argument("--crop_px", type=int, default=500)
    l.set_defaults(func=cmd_label)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
