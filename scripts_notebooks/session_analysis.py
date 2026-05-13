#!/usr/bin/env python3
"""
Session Analysis Graphs for Planarian Tracking Data
====================================================
Produces three conference-ready figures from tracked CSV files:
  1. Total distance traveled per session (grouped bar)
  2. Cumulative distance over time (line plot)
  3. Time to first sustained stop per session (grouped bar)

Usage:
    cd scripts_notebooks
    source ../venv/bin/activate
    python session_analysis.py
"""

import argparse
import glob
import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
STOP_SPEED_THRESH = 0.1   # mm/s — below this counts as "stopped"
STOP_DURATION_THRESH = 60  # seconds of sustained stop to count

WORM_COLORS = {"Bubba": "#3874A8", "Champ": "#E8853A"}  # blue / orange
ANALYZE_WORMS = set(WORM_COLORS.keys())  # only these names are analyzed

# ---------------------------------------------------------------------------
# 1. Load and parse all CSVs
# ---------------------------------------------------------------------------
def _load_truncations(repo_root):
    """Load per-session truncate_at_s overrides from session_truncations.json
    at the repo root. Missing file = no truncations (return empty dict).
    """
    path = os.path.join(repo_root, "session_truncations.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


def load_sessions(data_dir, truncations=None):
    """Load all session CSVs, return list of dicts with metadata + DataFrame.

    `truncations` maps session_base → {"truncate_at_s": float, ...}; matching
    sessions are clipped to time_s < truncate_at_s.
    """
    truncations = truncations or {}
    csv_pattern = os.path.join(data_dir, "*_tracks.csv")
    csv_files = sorted(glob.glob(csv_pattern))
    if not csv_files:
        sys.exit(f"No CSV files found matching {csv_pattern}")

    sessions = []
    for path in csv_files:
        basename = path.rsplit("/", 1)[-1]
        session_base = basename.replace("_tracks.csv", "")
        match = re.match(r"(\w+?)_(\d+)_\d+_tracks\.csv", basename)
        if not match:
            print(f"Skipping unrecognised file: {basename}")
            continue

        worm = match.group(1)
        if worm not in ANALYZE_WORMS:
            continue
        session_num = int(match.group(2))

        df = pd.read_csv(path, skiprows=2)
        for col in ("speed_mm_s", "centroid_x_mm", "centroid_y_mm", "time_s"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Optional truncation: clip to time_s < cutoff. Done at load time so
        # every downstream metric (distance, stop-time, cumulative) sees the
        # trimmed frame.
        trunc = truncations.get(session_base)
        if trunc and "truncate_at_s" in trunc:
            cutoff = float(trunc["truncate_at_s"])
            before = len(df)
            df = df[df["time_s"] < cutoff].reset_index(drop=True)
            print(f"  [truncate] {session_base}: dropped {before - len(df)} "
                  f"rows past t={cutoff:.1f}s "
                  f"({trunc.get('reason', 'no reason given')})")

        sessions.append({
            "worm": worm,
            "session": session_num,
            "path": path,
            "df": df,
        })

    return sessions


# ---------------------------------------------------------------------------
# 2. Compute total distance per session
# ---------------------------------------------------------------------------
def compute_distance(df):
    """Return total distance (mm) from speed × dt integration."""
    dt = df["time_s"].diff()
    incremental = df["speed_mm_s"] * dt  # mm
    return incremental.sum()  # NaN terms drop automatically


# ---------------------------------------------------------------------------
# 3. Compute time to first sustained stop
# ---------------------------------------------------------------------------
def compute_time_to_stop(df, speed_thresh=STOP_SPEED_THRESH,
                         duration_thresh=STOP_DURATION_THRESH):
    """
    Find the first moment the worm is stopped (speed < thresh) for at least
    `duration_thresh` consecutive seconds.  Returns time in seconds, or None.
    """
    stopped = df["speed_mm_s"].fillna(0) < speed_thresh
    times = df["time_s"].values

    run_start = None
    for i, is_stopped in enumerate(stopped):
        if is_stopped:
            if run_start is None:
                run_start = i
            elapsed = times[i] - times[run_start]
            if elapsed >= duration_thresh:
                return times[run_start]
        else:
            run_start = None

    return None  # never stopped for long enough


# ---------------------------------------------------------------------------
# 4. Compute cumulative distance series
# ---------------------------------------------------------------------------
def cumulative_distance_series(df):
    """Return (time_minutes, cumulative_distance_cm) arrays."""
    dt = df["time_s"].diff().fillna(0)
    incremental = (df["speed_mm_s"].fillna(0) * dt)
    cum_mm = incremental.cumsum()
    return df["time_s"].values / 60.0, cum_mm.values / 10.0  # min, cm


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True,
                    help="Directory containing *_tracks.csv files.")
    ap.add_argument("--output_dir", default=None,
                    help="Where to save figures (default: same as data_dir).")
    args = ap.parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir or data_dir

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    truncations = _load_truncations(repo_root)
    if truncations:
        print(f"Truncations from session_truncations.json:")
        for k in truncations:
            print(f"  {k}: truncate_at_s={truncations[k].get('truncate_at_s')}")
    sessions = load_sessions(data_dir, truncations=truncations)
    worms = sorted(set(s["worm"] for s in sessions))
    session_nums = sorted(set(s["session"] for s in sessions))

    # Tracked-vs-imputed breakdown per session (only if `source` column exists).
    print("\nSource breakdown:")
    print(f"  {'session':<25}{'tracked%':>10}{'imputed%':>10}{'lost%':>8}")
    for s in sorted(sessions, key=lambda s: (s["worm"], s["session"])):
        df = s["df"]
        n = len(df)
        if "source" in df.columns and n:
            src = df["source"].fillna("").astype(str)
            tracked = (src == "tracked").sum()
            imputed = src.str.startswith("imputed_").sum() + (src == "human_traced").sum()
            lost = n - tracked - imputed
            tag = f"{s['worm']}_{s['session']:04d}"
            print(f"  {tag:<25}{100*tracked/n:>9.1f}%"
                  f"{100*imputed/n:>9.1f}%{100*lost/n:>7.1f}%")

    # --- Compute metrics ---------------------------------------------------
    dist_table = {}   # (worm, session) → distance_cm
    stop_table = {}   # (worm, session) → stop_time_min or None
    max_time_min = 0

    for s in sessions:
        key = (s["worm"], s["session"])
        dist_mm = compute_distance(s["df"])
        dist_table[key] = dist_mm / 10.0  # → cm

        stop_s = compute_time_to_stop(s["df"])
        stop_table[key] = stop_s / 60.0 if stop_s is not None else None

        t_max = s["df"]["time_s"].max() / 60.0
        if t_max > max_time_min:
            max_time_min = t_max

    # --- Print summary table -----------------------------------------------
    print("\n" + "=" * 72)
    print(f"{'Session':<12} {'Worm':<8} {'Distance (cm)':>14} {'Time to Stop (min)':>20}")
    print("-" * 72)
    for sn in session_nums:
        for w in worms:
            key = (w, sn)
            d = dist_table.get(key, 0)
            ts = stop_table.get(key)
            ts_str = f"{ts:.1f}" if ts is not None else "never"
            print(f"  {sn:<10} {w:<8} {d:>14.1f} {ts_str:>20}")
    print("=" * 72 + "\n")

    # ======================================================================
    # FIGURE 1 — Total distance per session (grouped bar)
    # ======================================================================
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(session_nums))
    bar_w = 0.35

    for i, w in enumerate(worms):
        vals = [dist_table.get((w, sn), 0) for sn in session_nums]
        ax1.bar(x + i * bar_w, vals, bar_w, label=w,
                color=WORM_COLORS[w], edgecolor="white", linewidth=0.5)

    ax1.set_xticks(x + bar_w / 2)
    ax1.set_xticklabels([str(s) for s in session_nums])
    _style_ax(ax1, "Total Distance Traveled per Session",
              "Session", "Total Distance (cm)")
    ax1.legend(fontsize=11, frameon=False)
    fig1.tight_layout()
    fig1.savefig(f"{output_dir}/total_distance_per_session.png", dpi=200)
    print("Saved: total_distance_per_session.png")

    # ======================================================================
    # FIGURE 2 — Cumulative distance over time (line plot)
    # ======================================================================
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, w in zip(axes2, worms):
        w_sessions = [s for s in sessions if s["worm"] == w]
        cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(w_sessions)))
        for s, c in zip(sorted(w_sessions, key=lambda s: s["session"]), cmap):
            t_min, cum_cm = cumulative_distance_series(s["df"])
            ax.plot(t_min, cum_cm, linewidth=1.8, color=c,
                    label=f"Session {s['session']}")
        _style_ax(ax, f"{w} — Cumulative Distance",
                  "Time (min)", "Cumulative Distance (cm)")
        ax.legend(fontsize=9, frameon=False, loc="upper left")

    fig2.tight_layout()
    fig2.savefig(f"{output_dir}/cumulative_distance.png", dpi=200)
    print("Saved: cumulative_distance.png")

    # ======================================================================
    # FIGURE 3 — Time to first sustained stop (grouped bar)
    # ======================================================================
    fig3, ax3 = plt.subplots(figsize=(8, 5))

    for i, w in enumerate(worms):
        vals = []
        hatches = []
        for sn in session_nums:
            ts = stop_table.get((w, sn))
            if ts is None:
                vals.append(max_time_min)
                hatches.append("//")
            else:
                vals.append(ts)
                hatches.append("")

        bars = ax3.bar(x + i * bar_w, vals, bar_w, label=w,
                       color=WORM_COLORS[w], edgecolor="white", linewidth=0.5)
        # Apply per-bar hatching for "never stopped" sessions
        for bar, h in zip(bars, hatches):
            if h:
                bar.set_hatch(h)
                bar.set_edgecolor("white")

    ax3.set_xticks(x + bar_w / 2)
    ax3.set_xticklabels([str(s) for s in session_nums])
    _style_ax(ax3, "Time to First Sustained Stop (≥60 s below 0.1 mm/s)",
              "Session", "Time to Stop (min)")
    ax3.legend(fontsize=11, frameon=False)
    # Add note about hatched bars
    ax3.annotate("Hatched = never stopped for ≥60 s",
                 xy=(0.98, 0.97), xycoords="axes fraction",
                 ha="right", va="top", fontsize=9, fontstyle="italic",
                 color="gray")
    fig3.tight_layout()
    fig3.savefig(f"{output_dir}/time_to_stop.png", dpi=200)
    print("Saved: time_to_stop.png")

    plt.close("all")


if __name__ == "__main__":
    main()
