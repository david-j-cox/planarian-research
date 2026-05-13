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

import glob
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = "/tmp/tracker_output"
CSV_PATTERN = f"{DATA_DIR}/*_tracks.csv"
STOP_SPEED_THRESH = 0.1   # mm/s — below this counts as "stopped"
STOP_DURATION_THRESH = 60  # seconds of sustained stop to count

WORM_COLORS = {"Bubba": "#3874A8", "Champ": "#E8853A"}  # blue / orange

# ---------------------------------------------------------------------------
# 1. Load and parse all CSVs
# ---------------------------------------------------------------------------
def load_sessions():
    """Load all session CSVs, return list of dicts with metadata + DataFrame."""
    csv_files = sorted(glob.glob(CSV_PATTERN))
    if not csv_files:
        sys.exit(f"No CSV files found matching {CSV_PATTERN}")

    sessions = []
    for path in csv_files:
        # Extract worm name and session number from filename
        # e.g. Bubba_0001_021426_tracks.csv → Bubba, 1
        basename = path.rsplit("/", 1)[-1]
        match = re.match(r"(\w+?)_(\d+)_\d+_tracks\.csv", basename)
        if not match:
            print(f"Skipping unrecognised file: {basename}")
            continue

        worm = match.group(1)
        session_num = int(match.group(2))

        df = pd.read_csv(path, skiprows=2)
        # Coerce numeric columns (empty strings → NaN)
        for col in ("speed_mm_s", "centroid_x_mm", "centroid_y_mm", "time_s"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

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
    sessions = load_sessions()
    worms = sorted(set(s["worm"] for s in sessions))
    session_nums = sorted(set(s["session"] for s in sessions))

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
    fig1.savefig(f"{DATA_DIR}/total_distance_per_session.png", dpi=200)
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
    fig2.savefig(f"{DATA_DIR}/cumulative_distance.png", dpi=200)
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
    fig3.savefig(f"{DATA_DIR}/time_to_stop.png", dpi=200)
    print("Saved: time_to_stop.png")

    plt.show()


if __name__ == "__main__":
    main()
