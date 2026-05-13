#!/usr/bin/env python3
"""
validate_imputation.py — Empirical error characterization for gap imputation.

Takes a high-quality tracked session (default: Bubba_0001), synthetically
deletes windows of varying duration, runs each imputation tier (linear
interp, two-anchor pchip, simulated human-trace), and measures per-row
Euclidean error in mm against the ground truth.

Produces:
    docs/imputation_validation/error_by_duration.png   (the methods figure)
    docs/imputation_validation/error_summary.csv       (numeric table)

This is what defends the imputer at Q&A: "We don't just trust linear
interp — here's the empirical error budget at each gap length."

Usage:
    python validate_imputation.py
    python validate_imputation.py --tracks OpenDishWork/tracker_results/Bubba_0001_021426_tracks.csv
"""
from __future__ import annotations
import argparse
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


# Gap durations to test (seconds). Match the tier boundaries used elsewhere.
DURATIONS_S = [1.0, 5.0, 30.0, 120.0]

# Number of synthetic gaps to draw per duration.
N_GAPS_PER_DURATION = 30

# Imputation tiers we compare.
TIERS = ["linear (no anchor)", "two-anchor pchip", "human-traced (every 2s)"]


def _load_clean_tracks(csv_path: str) -> pd.DataFrame:
    """Load a tracks.csv and keep only rows with valid positions."""
    df = pd.read_csv(csv_path, skiprows=2)
    for c in ("time_s", "centroid_x_mm", "centroid_y_mm"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["centroid_x_mm", "centroid_y_mm", "time_s"])
    df = df.reset_index(drop=True)
    return df


def _draw_synthetic_gaps(df: pd.DataFrame, duration_s: float,
                        n_gaps: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Pick n_gaps random (start_idx, end_idx) windows of ~duration_s.

    Avoids the first/last 5 seconds so anchors always exist on both sides.
    """
    t = df["time_s"].to_numpy()
    t_min, t_max = t[0] + 5.0, t[-1] - 5.0
    windows = []
    attempts = 0
    while len(windows) < n_gaps and attempts < n_gaps * 20:
        attempts += 1
        t_start = rng.uniform(t_min, t_max - duration_s)
        t_end = t_start + duration_s
        # Find the row indices that fall inside the window.
        start_idx = int(np.searchsorted(t, t_start, side="left"))
        end_idx = int(np.searchsorted(t, t_end, side="right")) - 1
        if end_idx <= start_idx:
            continue
        if start_idx == 0 or end_idx >= len(df) - 1:
            continue
        windows.append((start_idx, end_idx))
    return windows


def _impute_linear(df: pd.DataFrame, start: int, end: int) -> np.ndarray:
    """Linear interp between row start-1 and row end+1 (no human input)."""
    lk = df.iloc[start - 1]
    nk = df.iloc[end + 1]
    t_anchor = np.array([lk["time_s"], nk["time_s"]])
    xs = np.array([lk["centroid_x_mm"], nk["centroid_x_mm"]])
    ys = np.array([lk["centroid_y_mm"], nk["centroid_y_mm"]])
    t_gap = df["time_s"].iloc[start:end + 1].to_numpy()
    x_imp = np.interp(t_gap, t_anchor, xs)
    y_imp = np.interp(t_gap, t_anchor, ys)
    return np.column_stack([x_imp, y_imp])


def _impute_two_anchor(df: pd.DataFrame, start: int, end: int,
                       gt: pd.DataFrame) -> np.ndarray:
    """Simulate the scrubber GUI: two human anchor clicks at gap 50% and 75%.

    Anchors are drawn from ground truth (representing perfect clicks).
    Interpolation uses pchip across {last_known, anchor_50, anchor_75, next_known}.
    """
    lk_row = df.iloc[start - 1]
    nk_row = df.iloc[end + 1]
    mid_idx = start + (end - start) // 2
    q3_idx = start + 3 * (end - start) // 4
    a50 = gt.iloc[mid_idx]
    a75 = gt.iloc[q3_idx]

    t_anchor = np.array([lk_row["time_s"], a50["time_s"],
                         a75["time_s"], nk_row["time_s"]])
    xs = np.array([lk_row["centroid_x_mm"], a50["centroid_x_mm"],
                   a75["centroid_x_mm"], nk_row["centroid_x_mm"]])
    ys = np.array([lk_row["centroid_y_mm"], a50["centroid_y_mm"],
                   a75["centroid_y_mm"], nk_row["centroid_y_mm"]])

    # PCHIP needs strictly increasing x — guarantee that.
    order = np.argsort(t_anchor)
    t_anchor, xs, ys = t_anchor[order], xs[order], ys[order]

    # Deduplicate any equal timestamps (rare but defensive).
    keep = np.concatenate([[True], np.diff(t_anchor) > 1e-9])
    t_anchor, xs, ys = t_anchor[keep], xs[keep], ys[keep]

    t_gap = df["time_s"].iloc[start:end + 1].to_numpy()
    if len(t_anchor) >= 3:
        fx = PchipInterpolator(t_anchor, xs)
        fy = PchipInterpolator(t_anchor, ys)
        x_imp = fx(t_gap)
        y_imp = fy(t_gap)
    else:
        x_imp = np.interp(t_gap, t_anchor, xs)
        y_imp = np.interp(t_gap, t_anchor, ys)
    return np.column_stack([x_imp, y_imp])


def _impute_human_traced(df: pd.DataFrame, start: int, end: int,
                         gt: pd.DataFrame, click_interval_s: float = 2.0) -> np.ndarray:
    """Simulate a user clicking every click_interval_s seconds through the gap.

    Anchors are exact ground-truth positions at those timestamps; the rest
    is linearly interpolated between clicks. This is the 'gold standard'
    for the scrubber GUI's long-gap trace mode.
    """
    lk_row = df.iloc[start - 1]
    nk_row = df.iloc[end + 1]
    t_start = lk_row["time_s"]
    t_end = nk_row["time_s"]
    click_ts = np.arange(t_start, t_end + 1e-9, click_interval_s)
    if click_ts[-1] < t_end - 1e-9:
        click_ts = np.append(click_ts, t_end)

    # For each click timestamp, find the nearest ground-truth row.
    gt_t = gt["time_s"].to_numpy()
    click_xs, click_ys = [], []
    for ct in click_ts:
        idx = int(np.argmin(np.abs(gt_t - ct)))
        click_xs.append(gt.iloc[idx]["centroid_x_mm"])
        click_ys.append(gt.iloc[idx]["centroid_y_mm"])

    t_gap = df["time_s"].iloc[start:end + 1].to_numpy()
    x_imp = np.interp(t_gap, click_ts, click_xs)
    y_imp = np.interp(t_gap, click_ts, click_ys)
    return np.column_stack([x_imp, y_imp])


def _run_validation(csv_path: str, out_dir: str, seed: int = 42) -> pd.DataFrame:
    """Run all imputation tiers across all durations, collect per-row errors."""
    gt = _load_clean_tracks(csv_path)
    print(f"Ground-truth session: {os.path.basename(csv_path)}  "
          f"({len(gt)} clean rows)")

    rng = np.random.default_rng(seed)
    results = []  # rows of (duration_s, tier, gap_idx, row_idx_in_gap, error_mm)

    for dur in DURATIONS_S:
        gaps = _draw_synthetic_gaps(gt, dur, N_GAPS_PER_DURATION, rng)
        print(f"  duration={dur:>5.1f}s  n_gaps={len(gaps)}")
        for gi, (start, end) in enumerate(gaps):
            truth = gt[["centroid_x_mm", "centroid_y_mm"]].iloc[start:end + 1].to_numpy()
            n_rows = end - start + 1

            for tier in TIERS:
                if tier == "linear (no anchor)":
                    pred = _impute_linear(gt, start, end)
                elif tier == "two-anchor pchip":
                    pred = _impute_two_anchor(gt, start, end, gt)
                elif tier == "human-traced (every 2s)":
                    pred = _impute_human_traced(gt, start, end, gt, click_interval_s=2.0)
                else:
                    continue

                err_mm = np.sqrt(np.sum((pred - truth) ** 2, axis=1))
                for r, e in enumerate(err_mm):
                    results.append({
                        "duration_s": dur,
                        "tier": tier,
                        "gap_idx": gi,
                        "row_offset": r,
                        "n_rows": n_rows,
                        "error_mm": float(e),
                    })

    return pd.DataFrame(results)


def _plot_and_save(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # ─── Numeric summary ──────────────────────────────
    summary = (df.groupby(["duration_s", "tier"])["error_mm"]
                 .agg(["mean", "median",
                       ("p95", lambda s: np.percentile(s, 95)),
                       "max", "count"])
                 .reset_index())
    summary_path = os.path.join(out_dir, "error_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary: {summary_path}")
    print(summary.to_string(index=False))

    # ─── Box-plot figure ──────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    tiers = list(df["tier"].unique())
    durations = sorted(df["duration_s"].unique())
    n_tiers = len(tiers)
    n_durs = len(durations)
    width = 0.8 / n_tiers
    colors = ["#E8853A", "#3874A8", "#5BA85B"]

    for ti, tier in enumerate(tiers):
        positions = np.arange(n_durs) + (ti - (n_tiers - 1) / 2) * width
        data = [df[(df["duration_s"] == d) & (df["tier"] == tier)]["error_mm"].to_numpy()
                for d in durations]
        bp = ax.boxplot(data, positions=positions, widths=width * 0.85,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(facecolor=colors[ti % len(colors)],
                                      alpha=0.65, edgecolor="black"),
                        medianprops=dict(color="black", linewidth=1.5),
                        whiskerprops=dict(color="black"),
                        capprops=dict(color="black"))

    ax.set_xticks(np.arange(n_durs))
    ax.set_xticklabels([f"{d:g}s" for d in durations])
    ax.set_xlabel("Synthetic gap duration")
    ax.set_ylabel("Imputation error per row (mm)")
    ax.set_title(f"Imputation error vs gap duration\n"
                 f"({N_GAPS_PER_DURATION} synthetic gaps per duration; "
                 f"box=IQR, line=median, whiskers=1.5×IQR)")
    # Legend handles
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colors[i % len(colors)], alpha=0.65,
                     edgecolor="black", label=tiers[i])
               for i in range(n_tiers)]
    ax.legend(handles=handles, loc="upper left", framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()

    png_path = os.path.join(out_dir, "error_by_duration.png")
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    print(f"Figure:  {png_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", type=str,
                    default="OpenDishWork/tracker_results/Bubba_0001_021426_tracks.csv",
                    help="Ground-truth tracks.csv (use the cleanest session).")
    ap.add_argument("--out_dir", type=str,
                    default="docs/imputation_validation",
                    help="Output directory for figure + CSV.")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for reproducible gap placement.")
    args = ap.parse_args()

    if not os.path.exists(args.tracks):
        print(f"Tracks file not found: {args.tracks}", file=sys.stderr)
        sys.exit(1)

    df = _run_validation(args.tracks, args.out_dir, seed=args.seed)
    _plot_and_save(df, args.out_dir)


if __name__ == "__main__":
    main()
