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

import threading
import time as _time

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
                   "fps": fps,
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
    fps = float(man.get("fps", 30.0))
    labels_path = os.path.join(md, "human_labels.csv")

    # done[window_id] = set(behaviors). Backward compatible: old single-label
    # rows (one behavior string) load as a one-element set.
    done = {}
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            for r in csv.DictReader(f):
                bs = {b for b in r["behavior"].split(";") if b}
                done[int(r["window_id"])] = bs

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    crop_size = args.crop_px

    # One cached VideoCapture per video, reused across windows (the manifest is
    # sorted, so consecutive windows usually share a clip). Guarded by a lock
    # because the prefetch thread also reads.
    _caps = {}
    _cap_lock = threading.Lock()

    def _get_cap(video):
        cap = _caps.get(video)
        if cap is None:
            cap = open_video(os.path.join(clips_dir, video))
            _caps[video] = cap
        return cap

    def load_window(wm):
        # Seek ONCE to the window start, then read sequentially. Re-seeking every
        # frame (the old way) forced a keyframe decode per frame — the 10-15s lag.
        frames = []
        with _cap_lock:
            cap = _get_cap(wm["video"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, wm["start_frame"])
            for _ in range(wm["start_frame"], wm["end_frame"] + 1):
                ok, img = cap.read()
                if not ok:
                    break
                c = _crop(img, wm["cx"], wm["cy"], crop_size)
                s = min(1.0, DISP_MAX_W / max(1, c.shape[1]))
                frames.append(cv2.resize(
                    c, (int(c.shape[1] * s), int(c.shape[0] * s))))
        return frames

    # Prefetch cache: window_id -> frames, filled by a background thread so the
    # NEXT window is decoded while you label the current one (no wait on `n`).
    _pf = {}
    _pf_lock = threading.Lock()

    def _prefetch(idx):
        if not (0 <= idx < len(windows)):
            return
        wid = windows[idx]["window_id"]
        with _pf_lock:
            if wid in _pf:
                return
        fr = load_window(windows[idx])
        with _pf_lock:
            _pf[wid] = fr

    def get_frames(idx):
        wid = windows[idx]["window_id"]
        with _pf_lock:
            fr = _pf.pop(wid, None)
        if fr is None:
            fr = load_window(windows[idx])
        # kick off prefetch of the next window in the background
        threading.Thread(target=_prefetch, args=(idx + 1,), daemon=True).start()
        return fr

    i = 0
    n = len(windows)
    # Bottom panel sized to fit the title + one-behavior-per-line menu + footer,
    # so nothing is ever clipped regardless of crop size.
    menu_h = 40 + len(LABELABLE) * 30 + 64
    while 0 <= i < n:
        wm = windows[i]
        frames = get_frames(i)
        if not frames:
            i += 1
            continue
        sel = set(done.get(wm["window_id"], set()))
        bh, bw_ = frames[0].shape[:2]
        pw = max(bw_, 420)
        # Pre-render each video frame onto a full-size canvas ONCE (the costly
        # crop/resize is already done in load_window; this just blits). The menu
        # strip is rebuilt only when `sel` changes, not every frame — that
        # per-frame text drawing was what made playback choppy.
        canvases = []
        for f_img in frames:
            cvs = np.full((bh + menu_h, pw, 3), 25, np.uint8)
            cvs[:bh, :f_img.shape[1]] = f_img
            canvases.append(cvs)

        def menu_strip():
            strip = np.full((menu_h, pw, 3), 25, np.uint8)
            y = 26
            _put(strip, f"window {i+1}/{n}   [{len(done)} done]   "
                 f"TOGGLE behaviors (multi):", (12, y), 0.55, (0, 220, 0))
            y += 30
            for k, b in enumerate(LABELABLE):
                on = b in sel
                col = (0, 255, 255) if on else (190, 190, 190)
                _put(strip, f"{k+1}  {'[x]' if on else '[ ]'} {b}", (16, y), 0.6, col)
                y += 30
            y += 8
            cur = " + ".join(sorted(sel)) if sel else "(none)"
            _put(strip, "SELECTED: " + cur, (12, y), 0.6,
                 (0, 255, 0) if sel else (0, 0, 255))
            y += 28
            _put(strip, "1-7 toggle   n/SPACE next   b back   u clear   q save+quit",
                 (12, y), 0.5, (180, 180, 180))
            return strip

        strip = menu_strip()
        t0 = _time.monotonic()
        nframes = len(canvases)
        while True:
            # Wall-clock playback at the real capture fps -> smooth 3s loop.
            fi = int(((_time.monotonic() - t0) * fps)) % nframes
            disp = canvases[fi].copy()
            disp[bh:, :] = strip
            cv2.imshow(WIN, disp)
            k = cv2.waitKey(15) & 0xFF
            if ord('1') <= k <= ord('9'):
                j = k - ord('1')
                if j < len(LABELABLE):
                    b = LABELABLE[j]
                    sel.discard(b) if b in sel else sel.add(b)   # toggle
                    strip = menu_strip()
            elif k == ord('u'):
                sel = set(); strip = menu_strip()
            elif k == ord('r'):
                t0 = _time.monotonic()
            elif k == ord('b'):
                if sel:
                    done[wm["window_id"]] = set(sel)
                i = max(0, i - 1)
                break
            elif k in (ord('n'), ord(' ')):
                if not sel:
                    continue  # must select >=1 before advancing
                done[wm["window_id"]] = set(sel)
                i += 1
                break
            elif k in (ord('q'), 27):
                if sel:
                    done[wm["window_id"]] = set(sel)
                _save_labels(labels_path, done)
                _release_caps(_caps, _cap_lock)
                cv2.destroyAllWindows()
                print(f"Saved {len(done)}/{n} labels -> {labels_path}")
                return
        _save_labels(labels_path, done)

    _save_labels(labels_path, done)
    _release_caps(_caps, _cap_lock)
    cv2.destroyAllWindows()
    print(f"Done. {len(done)}/{n} labeled -> {labels_path}")


def _release_caps(caps, lock):
    with lock:
        for c in caps.values():
            try:
                c.release()
            except Exception:
                pass
        caps.clear()


def _save_labels(path, done):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window_id", "behavior"])      # behavior = ';'-joined set
        for wid in sorted(done):
            w.writerow([wid, ";".join(sorted(done[wid]))])


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
