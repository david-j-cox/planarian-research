#!/usr/bin/env python3
"""
pilot_report.py — End-to-end accuracy + movement report for a tracking run.

Reads a rolling *_tracks.csv from watch_folder_tracker.py and prints/plots:
  - Detection rate overall and per clip (the accuracy check).
  - Total distance traveled, mean/max speed, % time moving vs stopped.
  - Plots: detection-rate-per-clip bar, position trajectory, speed over time,
    cumulative distance.

Tracker errors (implausible speed > --error_speed mm/s, i.e. blob jumps) are
flagged and excluded from movement metrics — a planarian glides ~0.5-3 mm/s.

Usage:
  cd scripts_notebooks
  source ../venv/bin/activate
  python pilot_report.py --csv ../realtime_runs/pilot_tracks.csv --out_dir ../realtime_runs/pilot_report
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ERROR_SPEED_MM_S = 15.0
STOP_SPEED_MM_S = 0.1


def load(csv_path):
    df = pd.read_csv(csv_path, skiprows=2)
    for c in ("time_s", "centroid_x_px", "centroid_y_px",
              "centroid_x_mm", "centroid_y_mm", "speed_mm_s", "speed_px_s",
              "area_px", "is_lost"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--error_speed", type=float, default=ERROR_SPEED_MM_S)
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(out_dir, exist_ok=True)

    df = load(args.csv)
    n = len(df)
    if n == 0:
        print("Empty CSV.")
        return
    has_mm = df["centroid_x_mm"].notna().any()
    unit = "mm" if has_mm else "px"
    xcol = "centroid_x_mm" if has_mm else "centroid_x_px"
    ycol = "centroid_y_mm" if has_mm else "centroid_y_px"
    scol = "speed_mm_s" if has_mm else "speed_px_s"

    det = df["is_lost"] == 0
    n_det = int(det.sum())
    dur_s = float(df["time_s"].max() - df["time_s"].min())

    # Tracker-error flag (blob jumps).
    err = det & (df[scol] > args.error_speed)
    n_err = int(err.sum())
    good = det & ~err

    print("=" * 64)
    print("PILOT TRACKING ACCURACY REPORT")
    print("=" * 64)
    print(f"CSV:            {args.csv}")
    print(f"Duration:       {dur_s/60:.1f} min ({n} frames)")
    print(f"Clips:          {df['video_file'].nunique()}")
    print(f"Detection rate: {100*n_det/n:.1f}%  ({n_det}/{n} frames)")
    print(f"Tracker errors: {n_err} frames (speed > {args.error_speed} {unit}/s, excluded)")
    print(f"Usable frames:  {int(good.sum())} ({100*good.sum()/n:.1f}%)")

    # Movement metrics on good frames.
    g = df[good]
    if len(g) > 1:
        dx = g[xcol].diff()
        dy = g[ycol].diff()
        step = np.sqrt(dx**2 + dy**2)
        total_dist = float(np.nansum(step))
        sp = g[scol].dropna()
        moving = (g[scol] > STOP_SPEED_MM_S).mean() * 100 if has_mm else float("nan")
        print(f"\nMOVEMENT ({unit}):")
        print(f"  Total distance:  {total_dist:.1f} {unit}"
              + (f"  ({total_dist/10:.1f} cm)" if has_mm else ""))
        print(f"  Mean speed:      {sp.mean():.3f} {unit}/s")
        print(f"  Median speed:    {sp.median():.3f} {unit}/s")
        print(f"  Max speed:       {sp.max():.3f} {unit}/s")
        if has_mm:
            print(f"  % time moving:   {moving:.0f}%  (>{STOP_SPEED_MM_S} mm/s)")
    print("=" * 64)

    # ---- Plots ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1) detection rate per clip
    per = df.groupby("video_file")["is_lost"].apply(
        lambda s: 100 * (s == 0).mean())
    axes[0, 0].bar(range(len(per)), per.values, color="0.3")
    axes[0, 0].axhline(100, color="green", ls="--", lw=0.8)
    axes[0, 0].set_title("Detection rate per clip (%)")
    axes[0, 0].set_xlabel("clip #")
    axes[0, 0].set_ylabel("% detected")
    axes[0, 0].set_ylim(0, 105)

    def _empty(ax, title):
        ax.text(0.5, 0.5, "no detections", ha="center", va="center",
                transform=ax.transAxes, color="0.5")
        ax.set_title(title)

    # 2) trajectory
    if len(g):
        axes[0, 1].plot(g[xcol], g[ycol], lw=0.6, color="0.2")
        axes[0, 1].scatter(g[xcol].iloc[0], g[ycol].iloc[0], c="green", s=40, label="start")
        axes[0, 1].scatter(g[xcol].iloc[-1], g[ycol].iloc[-1], c="red", s=40, label="end")
        axes[0, 1].set_title(f"Worm trajectory ({unit})")
        axes[0, 1].set_aspect("equal", "datalim")
        axes[0, 1].invert_yaxis()
        axes[0, 1].legend(fontsize=8)
    else:
        _empty(axes[0, 1], f"Worm trajectory ({unit})")

    # 3) speed over time
    if len(g):
        axes[1, 0].plot(g["time_s"] / 60, g[scol], lw=0.5, color="0.3")
    else:
        _empty(axes[1, 0], f"Speed over time ({unit}/s)")
    axes[1, 0].set_title(f"Speed over time ({unit}/s)")
    axes[1, 0].set_xlabel("time (min)")
    axes[1, 0].set_ylabel(f"speed ({unit}/s)")

    # 4) cumulative distance
    if len(g):
        cum = np.nancumsum(np.sqrt(g[xcol].diff()**2 + g[ycol].diff()**2))
        axes[1, 1].plot(g["time_s"] / 60, cum, color="0.2")
    else:
        _empty(axes[1, 1], f"Cumulative distance ({unit})")
    axes[1, 1].set_title(f"Cumulative distance ({unit})")
    axes[1, 1].set_xlabel("time (min)")
    axes[1, 1].set_ylabel(f"distance ({unit})")

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = os.path.join(out_dir, "pilot_report.png")
    fig.savefig(path, dpi=140)
    print(f"\nPlots saved: {path}")


if __name__ == "__main__":
    main()
