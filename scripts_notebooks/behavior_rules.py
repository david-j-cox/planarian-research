#!/usr/bin/env python3
"""
behavior_rules.py — Rule-based planarian behavior classifier (v1 + pre-labeler).

Turns the windowed features from behavior_features.py into a per-frame behavior
label. It serves two roles at once:
  1. a usable, real-time, interpretable v1 classifier, and
  2. a PRE-LABELER: its output seeds the human labeling/correction tool so a
     person confirms/fixes labels instead of marking every frame from scratch.

IMPORTANT: these rules are a starting HYPOTHESIS, not ground truth. The
thresholds are grounded in the S3 feature distributions and the planarian speed
literature (gliding ~1.5-2, max ~5 mm/s), but their accuracy is unknown until
measured against human labels (that is the next pipeline step). Treat the
proportions this produces as provisional.

Behaviors (priority order — first match wins, most specific first):
  scrunching   strong body-length oscillation with low net travel (escape gait)
  peristalsis  body-length oscillation, slower, lower amplitude than scrunch
  reversing    moving while body heading flips ~180 deg (head/tail lead swap)
  turning      large sustained heading change while moving
  wig_wag      head sweeping (head_sweep) with little net travel
  gliding      steady directed travel above the rest speed
  resting      none of the above; essentially still
  unknown      not enough detection in the window to decide

Usage:
  from behavior_rules import classify
  labels = classify(feats)              # feats from compute_features(); (N,) str

CLI:
  python behavior_rules.py ../realtime_runs/S3_signals.npz          # prints proportions
  python behavior_rules.py ../realtime_runs/S3_signals.npz --out ../realtime_runs/S3_behavior.csv
"""

import argparse
import numpy as np


# Thresholds (grounded in S3 feature percentiles + planarian speed literature).
# All tunable; the labeling/accuracy step will refine them against human truth.
SPEED_REST_MM_S = 0.10     # below this = not meaningfully translating
SPEED_GLIDE_MM_S = 0.10    # at/above this = gliding (range checked vs literature)
HEAD_SWEEP_DEG = 20.0      # peak-to-peak head swing marking a wig-wag
HEAD_SWEEP_MIN_REV = 1     # at least one gated head reversal in the window
BODYLEN_CYCLE_MIN = 1.0    # >=1 length cycle in window = peristaltic/scrunch
SCRUNCH_CONTRACT = 0.20    # >=20% length change in window = scrunch-scale
TURN_DEG = 40.0            # sustained heading change marking a turn
REVERSE_DEG = 140.0        # heading flip near 180 deg = reversal


BEHAVIORS = ["scrunching", "peristalsis", "reversing", "turning",
             "wig_wag", "gliding", "resting", "unknown"]


def classify(feats, max_frac_lost=0.5):
    """Per-frame behavior label (N,) of strings. `feats` is compute_features()."""
    n = len(feats["_time_s"])
    speed = np.nan_to_num(feats["speed_mm_s"], nan=0.0)
    disp = np.nan_to_num(feats["disp_mm"], nan=0.0)
    head_osc = np.nan_to_num(feats["head_osc_deg"], nan=0.0)
    head_rev = np.nan_to_num(feats["head_reversals"], nan=0.0)
    blcyc = np.nan_to_num(feats["bodylen_cycles"], nan=0.0)
    blcon = np.nan_to_num(feats["bodylen_contract"], nan=0.0)
    heading = np.nan_to_num(feats["heading_change_deg"], nan=0.0)
    frac_lost = np.nan_to_num(feats["frac_lost"], nan=1.0)
    lost = feats["_lost"]

    out = np.empty(n, dtype=object)
    for i in range(n):
        if lost[i] or frac_lost[i] > max_frac_lost:
            out[i] = "unknown"
            continue
        moving = speed[i] >= SPEED_REST_MM_S
        # Most specific gaits first.
        if blcyc[i] >= BODYLEN_CYCLE_MIN and blcon[i] >= SCRUNCH_CONTRACT:
            out[i] = "scrunching"
        elif blcyc[i] >= BODYLEN_CYCLE_MIN:
            out[i] = "peristalsis"
        elif moving and heading[i] >= REVERSE_DEG:
            out[i] = "reversing"
        elif moving and heading[i] >= TURN_DEG:
            out[i] = "turning"
        elif (head_osc[i] >= HEAD_SWEEP_DEG and head_rev[i] >= HEAD_SWEEP_MIN_REV
              and not moving):
            out[i] = "wig_wag"
        elif speed[i] >= SPEED_GLIDE_MM_S:
            out[i] = "gliding"
        else:
            out[i] = "resting"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("signals_npz")
    ap.add_argument("--window_s", type=float, default=1.0)
    ap.add_argument("--out", default=None,
                    help="Optional CSV: video,native_frame,time_s,behavior")
    args = ap.parse_args()

    from behavior_features import load_signals, compute_features
    sig = load_signals(args.signals_npz)
    feats = compute_features(sig, args.window_s)
    labels = classify(feats)

    uniq, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    print(f"{total} frames, window {args.window_s}s. Rule-classifier proportions "
          f"(PROVISIONAL — unvalidated vs human):")
    order = {b: i for i, b in enumerate(BEHAVIORS)}
    for b in sorted(uniq, key=lambda x: order.get(x, 99)):
        c = int(counts[list(uniq).index(b)])
        print(f"  {b:12s} {c:6d}  ({100*c/total:.1f}%)")

    if args.out:
        vid = feats["_video"]; t = feats["_time_s"]
        nf = np.asarray(sig["native_frame"])
        with open(args.out, "w") as f:
            f.write("video,native_frame,time_s,behavior\n")
            for i in range(total):
                f.write(f"{vid[i]},{int(nf[i])},{t[i]:.3f},{labels[i]}\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
