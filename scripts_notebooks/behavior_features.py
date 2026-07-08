#!/usr/bin/env python3
"""
behavior_features.py — Windowed behavior features from per-frame tracker signals.

Reads a <session>_signals.npz (written by watch_folder_tracker.py) and computes,
for every frame, a set of features over a trailing window. The features are the
inputs to both the rule-based classifier and any trained model, and are designed
to be CAUSAL (trailing window only) so the same code runs in real time.

Per-frame signals consumed (from the npz):
  cx_px, cy_px        worm centroid (pixels)
  midline (mp,2)      head->tail body curve
  body_len_px         body length along the midline
  head_angle_deg      direction of the head-end segment
  time_s, lost, mm_per_px, fps

Features computed per frame (trailing window W seconds):
  speed_mm_s          centroid speed, glitch-capped (rest vs glide)
  disp_mm             net displacement over the window (move vs stay-put)
  head_osc_deg        std of detrended head angle (wigwagging amplitude)
  head_reversals      #direction reversals of head angle in window (wigwag rate)
  bodylen_cv          coeff. of variation of body length (peristalsis/scrunch)
  bodylen_cycles      #length oscillation cycles in window
  bodylen_contract    max fractional shortening over window (scrunch indicator)
  heading_change_deg  net change in body heading (turning)
  frac_lost           fraction of window with no detection (confidence guard)

These are deliberately interpretable so the rule classifier can threshold them
directly; the same vector also trains a model later.

Usage:
  from behavior_features import load_signals, compute_features
  sig = load_signals("../realtime_runs/S3_signals.npz")
  feats = compute_features(sig, window_s=1.0)   # dict of (N,) arrays

CLI (summary + sanity stats):
  python behavior_features.py ../realtime_runs/S3_signals.npz
"""

import argparse

import numpy as np

MAX_SPEED_MM_S = 7.0   # cap instantaneous speed (matches filter_jumps; see that file)
# Oscillation gating: a swing must exceed these to count as a real reversal,
# rejecting the sub-threshold jitter a near-flat resting signal produces.
HEAD_MIN_SWING_DEG = 8.0       # head must swing >8 deg to count a wigwag reversal
BODYLEN_MIN_SWING_FRAC = 0.05  # body length must change >5% to count a length cycle
MIN_PATH_SPEED_MM_S = 0.3      # path-curvature ignores steps slower than this
                               # (sub-pixel steps have jitter-dominated direction)


def load_signals(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _angdiff(a, b):
    """Smallest signed difference a-b in degrees, wrapped to [-180,180]."""
    return (a - b + 180.0) % 360.0 - 180.0


def _gated_oscillations(x, min_amp):
    """Count true oscillation half-cycles in 1-D series `x`, ignoring jitter.

    Walks the series tracking the running extremum; a reversal is only counted
    once the signal has swung at least `min_amp` from the last confirmed
    extremum (hysteresis). This rejects the sub-threshold sign-flipping that a
    near-flat noisy signal produces — a RESTING worm's tiny head/length jitter
    no longer reads as oscillation. Returns (n_reversals, peak_to_peak_amp).
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return 0, 0.0
    n_rev = 0
    direction = 0           # +1 rising, -1 falling, 0 unknown
    last_ext = x[0]
    lo = hi = x[0]
    for v in x[1:]:
        lo = min(lo, v); hi = max(hi, v)
        if direction >= 0 and v <= last_ext - min_amp:
            if direction == 1:
                n_rev += 1
            direction = -1; last_ext = v
        elif direction <= 0 and v >= last_ext + min_amp:
            if direction == -1:
                n_rev += 1
            direction = 1; last_ext = v
        elif (direction >= 0 and v > last_ext) or (direction <= 0 and v < last_ext):
            last_ext = v
    return n_rev, float(hi - lo)


def compute_features(sig, window_s=1.0):
    """Per-frame trailing-window features. Returns dict of (N,) float arrays.

    Frames are processed PER CLIP (window never crosses a clip boundary), in
    capture order. The result arrays are aligned 1:1 with the npz frames.
    """
    video = np.asarray([str(v) for v in sig["video"]])
    t = np.asarray(sig["time_s"], dtype=np.float64)
    cx = np.asarray(sig["cx_px"], dtype=np.float64)
    cy = np.asarray(sig["cy_px"], dtype=np.float64)
    blen = np.asarray(sig["body_len_px"], dtype=np.float64)
    head = np.asarray(sig["head_angle_deg"], dtype=np.float64)
    lost = np.asarray(sig["lost"], dtype=np.int8)
    mmpp = float(sig["mm_per_px"]) if float(sig["mm_per_px"]) > 0 else 1.0
    fps = float(sig["fps"]) if "fps" in sig else 30.0
    win = max(1, int(round(window_s * fps)))
    n = len(t)

    feats = {k: np.full(n, np.nan, np.float64) for k in (
        "speed_mm_s", "disp_mm", "head_osc_deg", "head_reversals",
        "bodylen_cv", "bodylen_cycles", "bodylen_contract",
        "heading_change_deg", "frac_lost",
        # motion-dynamics features (target turning / scrunch separation)
        "ang_vel_p90_deg_s", "path_curv_deg_mm", "body_curv_deg")}

    # Per-frame body heading (tail->head vector) and body curvature, from midline.
    ml = np.asarray(sig["midline"], dtype=np.float64)  # (N, mp, 2)
    heading = np.full(n, np.nan)
    bodycurv = np.full(n, np.nan)                       # total body bend (deg)
    for i in range(n):
        if lost[i]:
            continue
        p = ml[i]
        good = ~np.isnan(p[:, 0])
        idx = np.where(good)[0]
        if len(idx) >= 2:
            hv = p[idx[0]] - p[idx[-1]]   # head - tail
            heading[i] = np.degrees(np.arctan2(hv[1], hv[0]))
        if len(idx) >= 3:
            # Sum of absolute turning between consecutive midline segments: a
            # straight worm ~0, a bent/curling (turning, scrunch) worm is large.
            segs = np.diff(p[idx], axis=0)
            sa = np.degrees(np.arctan2(segs[:, 1], segs[:, 0]))
            bodycurv[i] = float(np.sum(np.abs(_angdiff(sa[1:], sa[:-1]))))

    # Process each clip's frames independently (preserve order).
    for vid in np.unique(video):
        idx = np.where(video == vid)[0]
        idx = idx[np.argsort(t[idx])]
        for pos, i in enumerate(idx):
            lo = max(0, pos - win + 1)
            w = idx[lo:pos + 1]                     # trailing window indices
            det = w[lost[w] == 0]
            feats["frac_lost"][i] = 1.0 - len(det) / len(w)
            if len(det) < 2:
                continue
            # finite defaults so a degenerate sub-computation never drops the row
            feats["ang_vel_p90_deg_s"][i] = 0.0
            feats["path_curv_deg_mm"][i] = 0.0
            feats["body_curv_deg"][i] = 0.0

            # --- centroid kinematics ---
            wx, wy, wt = cx[det], cy[det], t[det]
            step = np.hypot(np.diff(wx), np.diff(wy)) * mmpp
            dts = np.diff(wt)
            inst = np.divide(step, dts, out=np.zeros_like(step), where=dts > 0)
            inst = inst[inst <= MAX_SPEED_MM_S]      # drop glitch steps
            feats["speed_mm_s"][i] = float(np.mean(inst)) if len(inst) else 0.0
            feats["disp_mm"][i] = float(np.hypot(wx[-1]-wx[0], wy[-1]-wy[0]) * mmpp)

            # --- head-angle oscillation (wigwagging) ---
            hw = head[det]
            hw = hw[~np.isnan(hw)]
            if len(hw) >= 3:
                # Unwrap the circular angle to a continuous series, then count
                # amplitude-gated reversals so resting jitter doesn't register.
                unw = hw[0] + np.concatenate([[0.0], np.cumsum(_angdiff(hw[1:], hw[:-1]))])
                rev, pp = _gated_oscillations(unw, HEAD_MIN_SWING_DEG)
                feats["head_osc_deg"][i] = pp                 # peak-to-peak swing
                feats["head_reversals"][i] = rev

            # --- body-length dynamics (peristalsis / scrunch) ---
            bw = blen[det]
            bw = bw[~np.isnan(bw) & (bw > 0)]
            if len(bw) >= 3:
                mean_b = np.mean(bw)
                feats["bodylen_cv"][i] = float(np.std(bw) / mean_b) if mean_b else 0.0
                feats["bodylen_contract"][i] = float((np.max(bw)-np.min(bw))/np.max(bw))
                # Gate length cycles on a fraction of mean length (jitter reject).
                rev, _ = _gated_oscillations(bw, BODYLEN_MIN_SWING_FRAC * mean_b)
                feats["bodylen_cycles"][i] = rev / 2.0        # half-cycles -> cycles

            # --- heading change + turning dynamics ---
            hd = heading[det]; htd = t[det]
            gv = ~np.isnan(hd)
            if gv.sum() >= 2:
                hh, tt = hd[gv], htd[gv]
                feats["heading_change_deg"][i] = abs(float(_angdiff(hh[-1], hh[0])))
                # Turning RATE (deg/s): unlike net heading change it doesn't cancel
                # when the worm turns then back. Use the 90th percentile, not the
                # max, so a single jitter/flip frame doesn't define it.
                dts2 = np.diff(tt)
                dh = np.abs(_angdiff(hh[1:], hh[:-1]))
                rate = np.divide(dh, dts2, out=np.zeros_like(dh), where=dts2 > 0)
                if len(rate):
                    feats["ang_vel_p90_deg_s"][i] = float(np.percentile(rate, 90))

            # --- path curvature (curving glide = turning vs straight glide) ---
            # Only over steps where the worm REALLY moved: at sub-pixel speed the
            # step direction is jitter, which otherwise dominates this feature.
            if len(det) >= 3:
                steps = np.column_stack([np.diff(cx[det]), np.diff(cy[det])])
                seglen = np.hypot(steps[:, 0], steps[:, 1])
                min_step_px = (MIN_PATH_SPEED_MM_S / fps) / mmpp
                mv = seglen >= min_step_px
                if mv.sum() >= 2:
                    sa = np.degrees(np.arctan2(steps[mv, 1], steps[mv, 0]))
                    turn = float(np.sum(np.abs(_angdiff(sa[1:], sa[:-1]))))
                    plen_mm = float(np.sum(seglen[mv]) * mmpp)
                    if plen_mm > 1e-6:
                        feats["path_curv_deg_mm"][i] = turn / plen_mm

            # --- body curvature (bent body: turning / scrunch) ---
            bc = bodycurv[det]; bc = bc[~np.isnan(bc)]
            if len(bc):
                feats["body_curv_deg"][i] = float(np.mean(bc))

    feats["_video"] = video
    feats["_time_s"] = t
    feats["_lost"] = lost
    return feats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("signals_npz", help="<session>_signals.npz from the tracker")
    ap.add_argument("--window_s", type=float, default=1.0)
    args = ap.parse_args()

    sig = load_signals(args.signals_npz)
    f = compute_features(sig, args.window_s)
    n = len(f["_time_s"])
    print(f"{n} frames, window {args.window_s}s. Feature summary (median / 90th pct):")
    for k in ("speed_mm_s", "disp_mm", "head_osc_deg", "head_reversals",
              "bodylen_cv", "bodylen_cycles", "bodylen_contract",
              "heading_change_deg", "frac_lost",
              "ang_vel_p90_deg_s", "path_curv_deg_mm", "body_curv_deg"):
        v = f[k][~np.isnan(f[k])]
        if len(v):
            print(f"  {k:20s} {np.median(v):8.3f} / {np.percentile(v,90):8.3f}")
        else:
            print(f"  {k:20s}   (all NaN)")


if __name__ == "__main__":
    main()
