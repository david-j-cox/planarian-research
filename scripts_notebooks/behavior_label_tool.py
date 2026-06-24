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
NO_WORM = "no_worm"   # window has no/unusable worm (detection failure, dish edge);
                      # recorded so it can be advanced past, dropped downstream


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


# ── active-learning sample ─────────────────────────────────────────────
def cmd_active(args):
    """Pick the NEXT labeling batch with the trained model, not the rule.

    Two acquisition signals, because random/rule-stratified sampling wastes
    labels on easy gliding:
      - UNCERTAINTY: windows near the model's decision boundary (some class
        probability close to 0.5) -- the gliding/turning confusions where a label
        is most informative.
      - RARE-CLASS coverage: windows the model thinks are likely turning/scrunch
        (the starved classes), so the next round actually grows them.
    Already-labeled windows are excluded and a minimum center spacing is enforced
    so we don't relabel near-duplicate windows.
    """
    import joblib
    from collections import defaultdict, Counter

    sig = load_signals(args.signals)
    art = joblib.load(args.model)
    model, FEATURES, classes = art["model"], art["features"], art["classes"]
    window_s = float(art.get("window_s", args.window_s))

    feats = compute_features(sig, window_s)
    video = feats["_video"]
    nf = np.asarray(sig["native_frame"])
    cx = np.asarray(sig["cx_px"]); cy = np.asarray(sig["cy_px"])
    lost = np.asarray(sig["lost"])
    fps = float(sig["fps"]) if "fps" in sig else 30.0
    mmpp = float(sig["mm_per_px"]) if float(sig["mm_per_px"]) > 0 else 1.0
    half = int(round(window_s * fps / 2))
    min_gap = int(round(args.min_gap_s * fps))

    # JITTER GATE: exclude windows containing a SEVERE position jump. The worm
    # maxes ~7 mm/s; a spike of 100s mm/s is the tracker bouncing on an unreliable
    # dish-edge/rim track where the worm curls into the meniscus and is hard to
    # even see (it appears to "disappear"). Uncertainty sampling otherwise walks
    # straight into these. A modest threshold (default 50 mm/s) ignores ordinary
    # box-center wobble but catches the wild rim jitter; the bad window is killed
    # within +-half a window of the spike.
    jbad = np.zeros(len(cx), bool)
    for v in np.unique(video):
        idx = np.where(video == v)[0]
        idx = idx[np.argsort(nf[idx])]
        x, y, L = cx[idx], cy[idx], lost[idx]
        sp = np.zeros(len(idx))
        sp[1:] = np.hypot(np.diff(x), np.diff(y)) * mmpp * fps
        jump = (sp > args.max_jump_mm_s) & (L == 0)
        if jump.any():
            jbad[idx[np.convolve(jump.astype(int), np.ones(2 * half + 1), mode="same") > 0]] = True

    X = np.column_stack([feats[k] for k in FEATURES])
    ok = np.all(np.isfinite(X), axis=1) & (lost == 0) & ~np.isnan(cx) & ~jbad
    print(f"jitter gate (>{args.max_jump_mm_s:.0f} mm/s spike): {int(jbad.sum())} frames "
          f"excluded; {int(ok.sum())} clean candidate frames remain")
    proba = np.zeros((len(X), len(classes)))
    proba[ok] = model.predict_proba(X[ok])   # proba[ok] is NaN-free

    # Uncertainty = closeness of the nearest class probability to 0.5 (small =
    # on the fence). Rare-score = max prob among the starved target classes.
    # Computed only on valid rows (selection draws from `ok` anyway).
    margin = np.full(len(X), np.inf)
    margin[ok] = np.min(np.abs(proba[ok] - 0.5), axis=1)   # small -> uncertain
    rare_idx = [classes.index(c) for c in args.rare if c in classes]
    rare_score = np.zeros(len(X))
    if rare_idx:
        rare_score[ok] = np.max(proba[ok][:, rare_idx], axis=1)
    top = np.full(len(X), -1)
    top[ok] = np.argmax(proba[ok], axis=1)

    # Already-shown windows only need to be NON-OVERLAPPING with new picks (a
    # half-window gap); applying the full min_gap to all of them blankets every
    # clip and leaves nothing. New picks are spaced from each other by min_gap.
    shown = defaultdict(list)
    if args.exclude_manifest_dir:
        ex = json.load(open(os.path.join(args.exclude_manifest_dir, "manifest.json")))
        for w in ex["windows"]:
            shown[w["video"]].append(int(w["center_frame"]))
    taken = defaultdict(list)   # newly picked centers

    def far_enough(v, f):
        return (all(abs(f - c) >= half for c in shown[v]) and
                all(abs(f - c) >= min_gap for c in taken[v]))

    cand = np.where(ok)[0]

    def greedy(order, budget):
        picked = []
        for i in order:
            if len(picked) >= budget:
                break
            v, f = str(video[i]), int(nf[i])
            if not far_enough(v, f):
                continue
            taken[v].append(f); picked.append(i)
        return picked

    # Split the budget: half rare-class coverage, half pure uncertainty.
    n_rare = args.n // 2
    rare_order = sorted(cand, key=lambda i: -rare_score[i])
    unc_order = sorted(cand, key=lambda i: margin[i])
    chosen = greedy(rare_order, n_rare)
    chosen += greedy(unc_order, args.n - len(chosen))
    chosen = sorted(set(chosen))

    os.makedirs(args.out_dir, exist_ok=True)
    manifest, hidden = [], []
    for wid, i in enumerate(chosen):
        v = str(video[i]); f0 = int(nf[i])
        manifest.append({"window_id": wid, "video": v, "center_frame": f0,
                         "start_frame": max(0, f0 - half), "end_frame": f0 + half,
                         "cx": float(cx[i]), "cy": float(cy[i])})
        pr = {classes[j]: round(float(proba[i, j]), 3) for j in range(len(classes))}
        hidden.append({"window_id": wid,
                       "model_pred": classes[int(top[i])] if top[i] >= 0 else "unknown",
                       "model_proba": pr, "min_margin": round(float(margin[i]), 3),
                       "time_s": float(feats["_time_s"][i])})

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump({"window_s": window_s, "clips_dir": args.clips_dir, "fps": fps,
                   "windows": manifest}, f, indent=2)
    with open(os.path.join(args.out_dir, "predictions_hidden.json"), "w") as f:
        json.dump(hidden, f, indent=2)

    c = Counter(h["model_pred"] for h in hidden)
    print(f"Active-sampled {len(manifest)} windows ({window_s}s) -> {args.out_dir}")
    print(f"  model-pred mix (HIDDEN): {dict(sorted(c.items()))}")
    print(f"  median min-margin of picks: {np.median([h['min_margin'] for h in hidden]):.3f} "
          f"(lower = more uncertain)")
    print(f"Now label:  python behavior_label_tool.py label --manifest_dir {args.out_dir}")


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
    # Display width the crop is scaled to. Decoupled from crop_px so a generous
    # crop (worm stays in frame) still renders small enough to play at true fps.
    disp_w = args.disp_w

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
        #
        # DROP exact-duplicate consecutive frames. The source MKVs contain padded
        # duplicate frames (the high-res capture can't sustain a true 30fps, so
        # OBS repeats frames to fill the timeline) -- runs of up to ~6 identical
        # frames make a MOVING worm look frozen-then-jump. We keep only frames
        # whose full image actually changed, so playback shows real motion. A
        # resting worm is unaffected (sensor noise makes its frames non-identical;
        # only byte-identical padded frames are dropped).
        frames = []
        prev_sig = None
        with _cap_lock:
            cap = _get_cap(wm["video"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, wm["start_frame"])
            for _ in range(wm["start_frame"], wm["end_frame"] + 1):
                ok, img = cap.read()
                if not ok:
                    break
                sig = cv2.resize(img, (160, 100)).astype(np.int16)
                if prev_sig is not None and not np.any(sig - prev_sig):
                    continue                       # exact padded duplicate -> skip
                prev_sig = sig
                c = _crop(img, wm["cx"], wm["cy"], crop_size)
                s = min(1.0, disp_w / max(1, c.shape[1]))
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

    n = len(windows)
    # Resume at the first not-yet-labeled window (a relaunch shouldn't re-walk
    # everything already done).
    i = next((k for k, w in enumerate(windows)
              if w["window_id"] not in done), 0)
    # Bottom panel sized to fit the title + one-behavior-per-line menu + footer,
    # so nothing is ever clipped regardless of crop size.
    menu_h = 40 + len(LABELABLE) * 30 + 64
    frame_period = 1.0 / fps
    while 0 <= i < n:
        wm = windows[i]
        frames = get_frames(i)
        if not frames:
            i += 1
            continue
        sel = set(done.get(wm["window_id"], set()))
        bh, bw_ = frames[0].shape[:2]
        pw = max(bw_, 420)
        nframes = len(frames)
        # One PERSISTENT display buffer. Per frame we overwrite only the video
        # region (the part that changes); the menu strip is drawn into the bottom
        # only when `sel` changes. The previous code copied a full canvas AND
        # re-blitted the strip every frame, and picked the frame by wall clock
        # (so a slow render SKIPPED frames unevenly) -- that was the choppiness.
        disp = np.full((bh + menu_h, pw, 3), 25, np.uint8)

        def draw_strip():
            strip = disp[bh:, :]
            strip[:] = 25
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
            _put(strip, "1-7 toggle  n/SPACE next  x no-worm  b back  u clear  r replay  q quit",
                 (12, y), 0.5, (180, 180, 180))

        draw_strip()
        fi = 0
        step = 1            # PING-PONG direction: forward then reverse
        next_t = _time.monotonic()
        while True:
            vf = frames[fi]
            disp[:bh, :vf.shape[1]] = vf
            # Playback heartbeat: a bar that sweeps with playback, so a perfectly
            # STILL worm (resting) is visibly distinguishable from a frozen
            # player. Purely a time indicator -> does not bias the blind call.
            prog = int(pw * fi / max(1, nframes - 1))
            disp[0:5, :, :] = 40
            disp[0:5, :prog, :] = (0, 220, 0)
            cv2.imshow(WIN, disp)
            # Advance one frame per period; waitKey absorbs the remaining time so
            # playback holds true fps and, under load, slows EVENLY (no skips).
            now = _time.monotonic()
            delay = max(1, int((next_t + frame_period - now) * 1000))
            k = cv2.waitKey(delay) & 0xFF
            next_t += frame_period
            # Ping-pong instead of wrap-around: at a loop wrap the worm teleports
            # from its end position back to its start (a ~35x single-frame jump
            # that reads as worm motion). Bouncing forward<->backward removes that
            # discontinuity; motion stays smooth and is shown both directions.
            fi += step
            if fi >= nframes - 1:
                fi = nframes - 1; step = -1
            elif fi <= 0:
                fi = 0; step = 1
            if next_t < now - frame_period:       # fell far behind: resync clock
                next_t = now
            if ord('1') <= k <= ord('9'):
                j = k - ord('1')
                if j < len(LABELABLE):
                    b = LABELABLE[j]
                    sel.discard(b) if b in sel else sel.add(b)   # toggle
                    draw_strip()
            elif k == ord('u'):
                sel = set(); draw_strip()
            elif k == ord('x'):
                # No worm / unusable window (detection failure, dish edge).
                # Recorded as NO_WORM and dropped from training + accuracy.
                done[wm["window_id"]] = {NO_WORM}
                i += 1
                break
            elif k == ord('r'):
                fi = 0; step = 1; next_t = _time.monotonic()
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

    a = sub.add_parser("active", help="Pick the next batch by model uncertainty "
                                      "+ rare-class coverage (active learning).")
    a.add_argument("--signals", required=True)
    a.add_argument("--model", required=True, help="trained behavior_clf joblib")
    a.add_argument("--clips_dir", required=True)
    a.add_argument("--n", type=int, default=120)
    a.add_argument("--window_s", type=float, default=3.0,
                   help="fallback if the model joblib lacks window_s")
    a.add_argument("--rare", nargs="*", default=["turning", "scrunching"],
                   help="starved classes to over-sample")
    a.add_argument("--min_gap_s", type=float, default=4.0,
                   help="min spacing (s) between picked window centers per clip")
    a.add_argument("--max_jump_mm_s", type=float, default=50.0,
                   help="exclude a window if the worm position spikes faster than "
                        "this (mm/s) -- severe rim/edge jitter; worm max ~7 mm/s")
    a.add_argument("--exclude_manifest_dir", default=None,
                   help="manifest dir whose windows are already labeled (skip near them)")
    a.add_argument("--out_dir", required=True)
    a.set_defaults(func=cmd_active)

    l = sub.add_parser("label", help="Blind labeling GUI.")
    l.add_argument("--manifest_dir", required=True)
    l.add_argument("--clips_dir", default=None)
    l.add_argument("--crop_px", type=int, default=600,
                   help="crop window around the worm (px); larger keeps a moving "
                        "worm in frame")
    l.add_argument("--disp_w", type=int, default=560,
                   help="display width the crop is scaled to; smaller plays "
                        "smoother (decoupled from crop_px)")
    l.set_defaults(func=cmd_label)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
