#!/usr/bin/env python3
"""
measure_accuracy.py — Measure tracker accuracy against human-clicked ground truth.

Compares the tracker's per-frame worm position (from a *_tracks.csv) to the
worm positions you clicked in label_setup.py (*_labels.json -> worm_truth).
Reports localization error in pixels and mm, and how often the tracker agreed
(detected a worm near your click) vs missed it.

Usage:
  cd scripts_notebooks
  source ../venv/bin/activate
  python measure_accuracy.py \
      --csv ../realtime_runs/pilot_tracks.csv \
      --labels ../realtime_runs/pilot_labels.json

Match is by (video_file, frame). The tracker CSV must have been produced with
the same clips you labeled. Error is the distance between your click and the
tracker centroid for the same frame.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="Tracker *_tracks.csv")
    ap.add_argument("--labels", required=True, help="*_labels.json with worm_truth")
    ap.add_argument("--near_px", type=float, default=40.0,
                    help="Tracker counts as 'agreeing' if within this many px "
                         "of the human click (default: 40).")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, skiprows=2)
    for c in ("frame", "centroid_x_px", "centroid_y_px", "is_lost"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    with open(args.labels) as f:
        lab = json.load(f)
    truth = lab.get("worm_truth", [])
    mmpp = lab.get("mm_per_px")
    if not truth:
        print("No worm_truth in labels — run label_setup.py --stage worm first.")
        return

    # Index tracker rows by (video_file, frame).
    df["video_file"] = df["video_file"].astype(str)
    key = df.set_index(["video_file", "frame"])

    rows = []
    for t in truth:
        k = (str(t["video"]), int(t["frame"]))
        try:
            r = key.loc[k]
        except KeyError:
            rows.append({"matched": False, "lost": None, "err_px": None})
            continue
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        lost = bool(r["is_lost"] == 1) or pd.isna(r["centroid_x_px"])
        if lost:
            rows.append({"matched": True, "lost": True, "err_px": None})
            continue
        err = np.hypot(r["centroid_x_px"] - t["x_px"],
                       r["centroid_y_px"] - t["y_px"])
        rows.append({"matched": True, "lost": False, "err_px": float(err)})

    res = pd.DataFrame(rows)
    n = len(res)
    n_in_csv = int(res["matched"].sum())
    n_lost = int((res["lost"] == True).sum())  # noqa: E712
    errs = res.loc[res["err_px"].notna(), "err_px"].values
    agree = (errs <= args.near_px).sum() if len(errs) else 0

    print("=" * 60)
    print("TRACKER ACCURACY vs HUMAN GROUND TRUTH")
    print("=" * 60)
    print(f"Ground-truth frames clicked: {n}")
    print(f"  found in tracker CSV:      {n_in_csv}")
    print(f"  tracker LOST on these:     {n_lost}  "
          f"({100*n_lost/max(1,n_in_csv):.0f}% of matched)")
    print(f"  tracker detected:          {len(errs)}")
    if len(errs):
        print(f"\nLocalization error (tracker vs your click):")
        print(f"  within {args.near_px:.0f}px ('correct'): {agree}/{len(errs)} "
              f"({100*agree/len(errs):.0f}%)")
        print(f"  median error: {np.median(errs):.1f} px"
              + (f"  ({np.median(errs)*mmpp:.2f} mm)" if mmpp else ""))
        print(f"  mean error:   {np.mean(errs):.1f} px"
              + (f"  ({np.mean(errs)*mmpp:.2f} mm)" if mmpp else ""))
        print(f"  90th pct:     {np.percentile(errs,90):.1f} px"
              + (f"  ({np.percentile(errs,90)*mmpp:.2f} mm)" if mmpp else ""))
        # Overall accuracy = detected AND within near_px, out of all clicks.
        print(f"\nOVERALL: {agree}/{n} clicked frames tracked correctly "
              f"({100*agree/n:.0f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
