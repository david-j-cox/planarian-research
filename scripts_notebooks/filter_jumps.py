#!/usr/bin/env python3
"""
filter_jumps.py — Flag physically-impossible instantaneous speeds in a tracks CSV.

The watch-folder tracker localizes the worm well (validated ~0.5 mm vs human
ground truth). The per-frame POSITIONS are generally trustworthy, but some
frames show a physically-impossible instantaneous speed: small fast steps
(~tens of mm/s, a few mm in one frame) and occasional large excursions (1000+
mm/s, the centroid briefly grabbing something far away then returning). Either
way the per-frame VELOCITY at that frame is not trustworthy and corrupts speed
and %-time-moving statistics.

Investigation (S3) showed the common moderate steps are NOT false latches: blob
area (~17000 px) and confidence (1.0) are unchanged across the step, and frame
timing is steady 30 fps — the worm really is at A then at B. (The worm also sits
genuinely still for long stretches — bit-identical centroids during rest are
correct tracking, not a glitch, and are left alone.) Rather than guess which
spikes are real-but-fast vs mis-localizations, we keep every position and only
flag the offending SPEED value, so speed/movement stats can exclude it without
altering the trajectory.

Planarian gliding is ~1.5-2 mm/s (range ~1-5; Rompolas 2010, Talbot & Schotz
2011, Sabry 2022 review). Scrunching escape is faster but unquantified. Default
ceiling 7 mm/s: above any reported gliding, with headroom, so we only flag the
clearly-impossible steps.

What this writes (in place-ish, alongside the CSV):
  - adds/updates a `speed_flag` column: "" for trusted, "impossible_step" for
    speeds above the ceiling.
  - sets the flagged rows' speed_mm_s / speed_px_s to NaN (so naive consumers
    don't average in a 1000 mm/s spike) while leaving the centroid columns and
    is_lost untouched (positions stay; distance/trajectory are unaffected).

Usage:
  cd scripts_notebooks
  python filter_jumps.py ../realtime_runs/S3_tracks.csv
  python filter_jumps.py ../realtime_runs/S3_tracks.csv --max_speed_mm_s 7 --dry_run

Idempotent: previously flagged rows are recomputed each run, so changing the
ceiling re-evaluates from the original positions (speed is recomputed from
centroids, not from the possibly-NaN'd column).
"""

import os
import math
import argparse

import numpy as np
import pandas as pd


MAX_SPEED_MM_S = 7.0   # flag instantaneous speeds above this (see module docstring)


def _read_tracks(csv_path):
    with open(csv_path) as f:
        meta = [ln.rstrip("\n") for ln in f if ln.startswith("#")]
    df = pd.read_csv(csv_path, skiprows=len(meta))
    return meta, df


def _write_tracks(csv_path, meta, df, dry_run):
    if dry_run:
        return
    tmp = csv_path + ".tmp"
    with open(tmp, "w", newline="") as f:
        for m in meta:
            f.write(m + "\n")
        df.to_csv(f, index=False)
    os.replace(tmp, csv_path)


def _mm_per_px(meta):
    for m in meta:
        if "mm_per_px=" in m:
            try:
                return float(m.split("mm_per_px=")[1].split()[0])
            except (ValueError, IndexError):
                return None
    return None


def filter_one(csv_path, max_speed, dry_run):
    meta, df = _read_tracks(csv_path)
    mmpp = _mm_per_px(meta)
    for c in ("native_frame", "time_s", "centroid_x_mm", "centroid_y_mm",
              "speed_mm_s", "speed_px_s", "is_lost"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["speed_flag"] = ""

    # Recompute instantaneous speed from POSITIONS per clip (don't trust the
    # stored speed column — it may already be NaN'd from a prior run), then flag
    # frames whose recomputed speed exceeds the ceiling.
    n_flag = 0
    n_det = 0
    for vid, idx in df.groupby("video_file").groups.items():
        sub = df.loc[idx].sort_values("native_frame")
        rows = sub.index.tolist()
        xm = sub["centroid_x_mm"].to_numpy()
        ym = sub["centroid_y_mm"].to_numpy()
        t = sub["time_s"].to_numpy()
        lost = sub["is_lost"].to_numpy()
        last = None
        for i, r in enumerate(rows):
            if lost[i] == 1 or np.isnan(xm[i]):
                last = None
                continue
            n_det += 1
            if last is None:
                sp = 0.0
            else:
                dt = t[i] - last[2]
                sp = math.hypot(xm[i] - last[0], ym[i] - last[1]) / dt if dt > 0 else 0.0
            if sp > max_speed:
                df.at[r, "speed_flag"] = "impossible_step"
                df.at[r, "speed_mm_s"] = np.nan
                if "speed_px_s" in df.columns:
                    df.at[r, "speed_px_s"] = np.nan
                n_flag += 1
                # Keep `last` at the new position: the worm IS there now, so the
                # NEXT frame's speed is measured from B (the real current spot),
                # not from the stale A. Only this one step's speed is untrusted.
                last = (xm[i], ym[i], t[i])
            else:
                df.at[r, "speed_mm_s"] = round(sp, 4)
                if "speed_px_s" in df.columns and mmpp:
                    df.at[r, "speed_px_s"] = round(sp / mmpp, 3)
                last = (xm[i], ym[i], t[i])

    _write_tracks(csv_path, meta, df, dry_run)
    return {"csv": csv_path, "flagged": n_flag, "detected": n_det,
            "pct": 100 * n_flag / max(1, n_det)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_paths", nargs="+", help="One or more *_tracks.csv")
    ap.add_argument("--max_speed_mm_s", type=float, default=MAX_SPEED_MM_S,
                    help=f"Flag instantaneous speeds above this (default {MAX_SPEED_MM_S}).")
    ap.add_argument("--dry_run", action="store_true",
                    help="Report what would change without writing.")
    args = ap.parse_args()

    for p in args.csv_paths:
        s = filter_one(p, args.max_speed_mm_s, args.dry_run)
        tag = " (dry run)" if args.dry_run else ""
        print(f"{os.path.basename(s['csv'])}{tag}: flagged {s['flagged']} "
              f"impossible-speed frames of {s['detected']} detected "
              f"({s['pct']:.1f}%) at >{args.max_speed_mm_s} mm/s")


if __name__ == "__main__":
    main()
