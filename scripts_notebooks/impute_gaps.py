#!/usr/bin/env python3
"""
impute_gaps.py — Fill LOST tracking gaps with provenance-tagged interpolation.

Reads a *_tracks.csv and its *_gaps.json sidecar, fills short gaps with
linear interpolation when the implied bridging speed is physically plausible
for a planarian, leaves the rest as 'unrecoverable' for the gap-scrubber
GUI to handle. Every imputed row gets a `source` column tag so downstream
analysis can stratify or filter.

Usage:
    python impute_gaps.py PATH/TO/*_tracks.csv [...]
    python impute_gaps.py --data_dir OpenDishWork/tracker_results

Idempotent: re-running does not double-impute. Rows already tagged
'imputed_short' are recomputed from scratch (so threshold changes apply),
but anchored/human-traced rows are left untouched.
"""
from __future__ import annotations
import argparse
import glob
import json
import math
import os
import sys
from typing import List

import pandas as pd

# ─── Imputation policy ────────────────────────────────────────────────
# Short gaps shorter than this duration are eligible for auto-bridge,
# *provided* the implied speed across the gap is physically plausible.
AUTO_BRIDGE_MAX_S = 1.0

# Maximum plausible bridging speed (mm/s). Planarians move ~0.5-2 mm/s under
# normal conditions; we set a generous ceiling of 3 mm/s. Bridges that
# require speeds above this are tagged 'unrecoverable' rather than
# silently filled — almost certainly the tracker re-acquired the worm
# elsewhere and we don't know where it actually went.
MAX_BRIDGE_SPEED_MM_S = 3.0


def _read_tracks(csv_path: str) -> tuple[list[str], list[str], pd.DataFrame]:
    """Read a tracks.csv preserving the two metadata header lines."""
    with open(csv_path, "r") as f:
        lines = f.readlines()
    # First two lines are CSV-quoted comments, third is the column header.
    meta = lines[:2]
    df = pd.read_csv(csv_path, skiprows=2)
    return meta, list(df.columns), df


def _write_tracks(csv_path: str, meta: list[str], df: pd.DataFrame) -> None:
    """Write tracks.csv back with the original metadata header intact."""
    tmp_path = csv_path + ".tmp"
    with open(tmp_path, "w") as f:
        for line in meta:
            f.write(line if line.endswith("\n") else line + "\n")
        df.to_csv(f, index=False)
    os.replace(tmp_path, csv_path)


def _impute_one_session(csv_path: str, gaps_path: str,
                        dry_run: bool = False) -> dict:
    """Impute short, physically-plausible gaps in one session."""
    if not os.path.exists(gaps_path):
        raise FileNotFoundError(
            f"Missing gaps sidecar for {csv_path}: expected {gaps_path}. "
            f"Re-run the tracker to regenerate it.")

    meta, cols, df = _read_tracks(csv_path)
    if "source" not in df.columns:
        # Older CSVs (pre-provenance) — add the column with 'tracked' for
        # all detected rows and empty for LOST rows.
        df["source"] = df["centroid_x_mm"].notna().map(
            {True: "tracked", False: ""})

    # Reset previously-auto-imputed rows so this run is idempotent and
    # picks up any threshold changes. Human-anchored / traced rows stay.
    auto_mask = df["source"] == "imputed_short"
    if auto_mask.any():
        for c in ("centroid_x_px", "centroid_y_px",
                  "centroid_x_mm", "centroid_y_mm",
                  "speed_px_s", "speed_mm_s",
                  "body_angle_deg", "head_angle_deg",
                  "area_px", "confidence",
                  "body_length_px", "body_length_mm"):
            if c in df.columns:
                df.loc[auto_mask, c] = pd.NA
        df.loc[auto_mask, "source"] = ""

    with open(gaps_path) as f:
        gaps = json.load(f)

    stats = {
        "csv": csv_path,
        "n_gaps_total": len(gaps["gaps"]),
        "auto_bridged": 0,
        "skipped_too_long": 0,
        "skipped_unrecoverable_jump": 0,
        "rows_imputed": 0,
    }

    for g in gaps["gaps"]:
        # Only auto-handle short gaps; medium/long need the scrubber GUI.
        if g["tier"] != "short" or (g["duration_s"] or 0) >= AUTO_BRIDGE_MAX_S:
            stats["skipped_too_long"] += 1
            continue

        lk = g["last_known"]
        nk = g["next_known"]
        if not lk or not nk or lk["x_mm"] is None or nk["x_mm"] is None:
            # Gap at the very start or end of the session, no anchor on
            # one side — skip. (Rare; usually one or two frames.)
            stats["skipped_unrecoverable_jump"] += 1
            continue

        dx_mm = nk["x_mm"] - lk["x_mm"]
        dy_mm = nk["y_mm"] - lk["y_mm"]
        dist_mm = math.hypot(dx_mm, dy_mm)
        dt_s = (nk["time_s"] - lk["time_s"]) if (
            nk["time_s"] is not None and lk["time_s"] is not None) else None

        if dt_s is None or dt_s <= 0:
            stats["skipped_unrecoverable_jump"] += 1
            continue

        implied_speed = dist_mm / dt_s
        if implied_speed > MAX_BRIDGE_SPEED_MM_S:
            # Tracker almost certainly re-acquired elsewhere; the actual
            # path the worm took is unknown. Leave it for the scrubber.
            stats["skipped_unrecoverable_jump"] += 1
            continue

        # Linearly interpolate every gap row by time.
        dx_px = nk["x_px"] - lk["x_px"]
        dy_px = nk["y_px"] - lk["y_px"]
        dist_px = math.hypot(dx_px, dy_px)
        implied_speed_px = dist_px / dt_s

        start = g["start_row"]
        end = g["end_row"]
        for r in range(start, end + 1):
            t_r = df.at[r, "time_s"]
            if pd.isna(t_r):
                continue
            frac = (t_r - lk["time_s"]) / dt_s
            df.at[r, "centroid_x_mm"] = lk["x_mm"] + frac * dx_mm
            df.at[r, "centroid_y_mm"] = lk["y_mm"] + frac * dy_mm
            df.at[r, "centroid_x_px"] = lk["x_px"] + frac * dx_px
            df.at[r, "centroid_y_px"] = lk["y_px"] + frac * dy_px
            df.at[r, "speed_mm_s"] = implied_speed
            df.at[r, "speed_px_s"] = implied_speed_px
            df.at[r, "confidence"] = 0.0  # imputed → low confidence
            df.at[r, "source"] = "imputed_short"

        stats["auto_bridged"] += 1
        stats["rows_imputed"] += (end - start + 1)

    if not dry_run:
        _write_tracks(csv_path, meta, df)
    return stats


def main():
    global AUTO_BRIDGE_MAX_S, MAX_BRIDGE_SPEED_MM_S
    ap = argparse.ArgumentParser(
        description="Fill short, physically-plausible LOST gaps in tracks.csv "
                    "files using linear interpolation. Medium/long gaps and "
                    "implausible bridges are left for the gap-scrubber GUI.")
    ap.add_argument("--data_dir", type=str, default=None,
                    help="Directory containing *_tracks.csv + *_gaps.json. "
                         "Processes every session in the directory.")
    ap.add_argument("csv_paths", nargs="*", default=[],
                    help="Specific tracks.csv files to impute "
                         "(alternative to --data_dir).")
    ap.add_argument("--auto_bridge_max_s", type=float,
                    default=AUTO_BRIDGE_MAX_S,
                    help=f"Max gap duration eligible for auto-bridge "
                         f"(default: {AUTO_BRIDGE_MAX_S}s).")
    ap.add_argument("--max_speed_mm_s", type=float,
                    default=MAX_BRIDGE_SPEED_MM_S,
                    help=f"Max plausible bridging speed (default: "
                         f"{MAX_BRIDGE_SPEED_MM_S} mm/s).")
    ap.add_argument("--dry_run", action="store_true",
                    help="Compute stats but don't write changes back.")
    args = ap.parse_args()

    if not args.data_dir and not args.csv_paths:
        ap.error("Either --data_dir or one or more csv_paths must be provided.")

    # Allow per-run overrides without touching the module-level constants.
    AUTO_BRIDGE_MAX_S = args.auto_bridge_max_s
    MAX_BRIDGE_SPEED_MM_S = args.max_speed_mm_s

    if args.data_dir:
        csvs = sorted(glob.glob(
            os.path.join(args.data_dir, "*_tracks.csv")))
    else:
        csvs = args.csv_paths

    if not csvs:
        print("No tracks.csv files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Imputing {len(csvs)} session(s); "
          f"auto_bridge<{AUTO_BRIDGE_MAX_S}s, "
          f"max_speed={MAX_BRIDGE_SPEED_MM_S}mm/s")
    print()

    totals = {"auto_bridged": 0, "skipped_too_long": 0,
              "skipped_unrecoverable_jump": 0, "rows_imputed": 0}
    for csv_path in csvs:
        gaps_path = csv_path.replace("_tracks.csv", "_gaps.json")
        try:
            s = _impute_one_session(csv_path, gaps_path,
                                    dry_run=args.dry_run)
        except FileNotFoundError as e:
            print(f"SKIP {os.path.basename(csv_path)}: {e}", file=sys.stderr)
            continue
        name = os.path.basename(csv_path).replace("_tracks.csv", "")
        print(f"{name:38s}  bridged={s['auto_bridged']:4d}/{s['n_gaps_total']:<4d}"
              f"  rows_imputed={s['rows_imputed']:5d}"
              f"  unrecoverable_jumps={s['skipped_unrecoverable_jump']:3d}")
        for k in totals:
            totals[k] += s[k]

    print()
    print(f"TOTAL: bridged={totals['auto_bridged']} gaps, "
          f"{totals['rows_imputed']} rows imputed, "
          f"{totals['skipped_unrecoverable_jump']} unrecoverable jumps left "
          f"for the scrubber GUI.")


if __name__ == "__main__":
    main()
