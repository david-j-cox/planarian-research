#!/usr/bin/env python3
"""Habituation analysis for planarian open-dish tracking data.

Analyses:
  (a) Total movement (distance traveled) per session
  (b) Time course of movement — when does the worm stop moving?

Generates plots saved to the output directory.

Usage:
    python habituation_analysis.py --data_dir /tmp/tracker_output_v2
"""

import argparse
import csv
import json
import os
import glob
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from collections import OrderedDict


# Only analyze files whose worm name is in this set (excludes yuja_, etc.)
ANALYZE_WORMS = {"Bubba", "Champ"}


def _load_truncations(repo_root):
    """Per-session truncate_at_s overrides from session_truncations.json."""
    path = os.path.join(repo_root, "session_truncations.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


# ── Data loading ─────────────────────────────────────────────────────

def load_session(csv_path, truncate_at_s=None):
    """Load a tracking CSV and return structured data as a dict of arrays.

    If truncate_at_s is given, rows with time_s >= cutoff are dropped at load.
    """
    with open(csv_path) as f:
        lines = f.readlines()

    # Find header line (skip comment lines starting with #)
    header_idx = next(i for i, l in enumerate(lines) if l.startswith("video_file"))
    reader = csv.DictReader(lines[header_idx:])
    rows = list(reader)

    # Parse into arrays
    time_s = []
    x_mm = []
    y_mm = []
    speed_mm_s = []
    detected = []

    for r in rows:
        t = float(r["time_s"])
        if truncate_at_s is not None and t >= truncate_at_s:
            continue
        has_det = bool(r["centroid_x_mm"].strip())
        time_s.append(t)
        detected.append(has_det)
        if has_det:
            x_mm.append(float(r["centroid_x_mm"]))
            y_mm.append(float(r["centroid_y_mm"]))
            sp = r["speed_mm_s"].strip()
            speed_mm_s.append(float(sp) if sp else 0.0)
        else:
            x_mm.append(np.nan)
            y_mm.append(np.nan)
            speed_mm_s.append(np.nan)

    return {
        "time_s": np.array(time_s),
        "x_mm": np.array(x_mm),
        "y_mm": np.array(y_mm),
        "speed_mm_s": np.array(speed_mm_s),
        "detected": np.array(detected),
    }


def compute_step_distances(data):
    """Compute frame-to-frame distances in mm (NaN where detection gaps)."""
    dx = np.diff(data["x_mm"])
    dy = np.diff(data["y_mm"])
    return np.sqrt(dx**2 + dy**2)


def rolling_mean(arr, window):
    """Compute rolling mean, ignoring NaNs. Returns same-length array."""
    out = np.full_like(arr, np.nan)
    half = window // 2
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        chunk = arr[lo:hi]
        valid = chunk[~np.isnan(chunk)]
        if len(valid) >= window // 4:  # need at least 25% valid
            out[i] = np.mean(valid)
    return out


# ── Analysis (a): Total distance per session ─────────────────────────

def analyze_total_distance(sessions):
    """Compute total distance traveled (mm) for each session."""
    results = OrderedDict()
    for name, data in sessions.items():
        steps = compute_step_distances(data)
        # Filter out unreasonably large jumps (tracking errors)
        # A planarian moves at most ~3 mm/s, at ~0.3s intervals → max ~1 mm/step
        max_step = 2.0  # mm — generous threshold
        valid_steps = steps[~np.isnan(steps)]
        valid_steps = valid_steps[valid_steps <= max_step]
        total_mm = np.sum(valid_steps)
        det_rate = np.mean(data["detected"]) * 100
        duration_min = (data["time_s"][-1] - data["time_s"][0]) / 60.0
        results[name] = {
            "total_mm": total_mm,
            "total_cm": total_mm / 10,
            "detection_rate": det_rate,
            "duration_min": duration_min,
            "n_frames": len(data["time_s"]),
        }
    return results


# ── Analysis (b): Movement over time / cessation ─────────────────────

def net_displacement_speed(x_mm, y_mm, time_s, window_sec=300):
    """Compute net displacement over sliding windows, returned as mm/s.

    Instead of averaging instantaneous (jitter-contaminated) speeds, this
    measures how far the centroid actually moved over each window.  A
    stationary worm with pixel jitter will show ~0 net displacement while
    a moving worm will show real displacement.

    Returns same-length array (NaN-padded at edges).
    """
    dt = np.median(np.diff(time_s[~np.isnan(time_s)][:100]))
    half = max(1, int((window_sec / dt) / 2))
    out = np.full(len(x_mm), np.nan)
    for i in range(half, len(x_mm) - half):
        x0, y0 = x_mm[i - half], y_mm[i - half]
        x1, y1 = x_mm[i + half], y_mm[i + half]
        if np.isnan(x0) or np.isnan(x1):
            continue
        t_span = time_s[i + half] - time_s[i - half]
        if t_span > 0:
            out[i] = np.sqrt((x1 - x0)**2 + (y1 - y0)**2) / t_span
    return out


def analyze_movement_timecourse(data, window_sec=300):
    """Compute smoothed speed over time and find when worm stops.

    Uses net displacement over a sliding window rather than averaging
    instantaneous speed.  This eliminates centroid jitter artifacts:
    at ~0.107 mm/px and ~0.3 s frame intervals, 1-pixel jitter produces
    ~0.36 mm/s of fake instantaneous speed that never drops to zero.
    Net displacement over 5 minutes correctly reads ~0 for a still worm.

    Parameters
    ----------
    data : dict from load_session
    window_sec : int
        Sliding window size in seconds (default: 300 = 5 min).

    Returns
    -------
    dict with time_min, smoothed_speed, cessation_time_min
    """
    time_s = data["time_s"]
    speed = data["speed_mm_s"].copy()
    x_mm = data["x_mm"]
    y_mm = data["y_mm"]

    # Estimate frame interval
    dt = np.median(np.diff(time_s[~np.isnan(time_s)][:100]))
    window_frames = max(1, int(window_sec / dt))

    # Net displacement speed (jitter-robust)
    smoothed = net_displacement_speed(x_mm, y_mm, time_s, window_sec)
    time_min = (time_s - time_s[0]) / 60.0

    # Find cessation: first time the net-displacement speed drops below
    # threshold and stays below for at least another window
    speed_threshold = 0.05  # mm/s — well above 0 but below real movement
    sustain_frames = window_frames

    cessation_min = None
    below = smoothed < speed_threshold
    for i in range(len(below)):
        if below[i] and not np.isnan(smoothed[i]):
            end = min(len(below), i + sustain_frames)
            segment = below[i:end]
            valid_segment = segment[~np.isnan(smoothed[i:end])]
            if len(valid_segment) > 0 and np.all(valid_segment):
                cessation_min = time_min[i]
                break

    return {
        "time_min": time_min,
        "smoothed_speed": smoothed,
        "raw_speed": speed,
        "cessation_min": cessation_min,
        "window_sec": window_sec,
    }


# ── Plotting ─────────────────────────────────────────────────────────

def plot_total_distance(results, output_dir, worm_name):
    """Bar chart of total distance per session for one worm."""
    sessions = [k for k in results if k.startswith(worm_name)]
    if not sessions:
        return

    labels = [s.split("_")[1] for s in sessions]  # "0001", "0002", etc.
    distances = [results[s]["total_cm"] for s in sessions]
    det_rates = [results[s]["detection_rate"] for s in sessions]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    bars = ax1.bar(labels, distances, color="#4C72B0", alpha=0.85, label="Distance (cm)")
    ax1.set_xlabel("Session", fontsize=12)
    ax1.set_ylabel("Total Distance Traveled (cm)", fontsize=12, color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")

    # Annotate bars with detection rate
    for bar, rate in zip(bars, det_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{rate:.0f}%", ha="center", va="bottom", fontsize=9, color="gray")

    ax1.set_title(f"{worm_name} — Total Distance per Session\n(% = detection rate)",
                  fontsize=13, fontweight="bold")
    ax1.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{worm_name}_total_distance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_speed_timecourse(sessions_data, timecourses, output_dir, worm_name):
    """Speed over time for all sessions of one worm, stacked."""
    worm_sessions = [k for k in sessions_data if k.startswith(worm_name)]
    if not worm_sessions:
        return

    n = len(worm_sessions)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n))

    for i, (sess_name, ax) in enumerate(zip(worm_sessions, axes)):
        tc = timecourses[sess_name]
        label = sess_name.split("_")[1]

        # Plot raw speed as faint dots
        valid = ~np.isnan(tc["raw_speed"])
        ax.scatter(tc["time_min"][valid], tc["raw_speed"][valid],
                   s=0.3, alpha=0.15, color="gray", rasterized=True)

        # Plot smoothed speed as line
        ax.plot(tc["time_min"], tc["smoothed_speed"],
                color=colors[i], linewidth=1.5, label=f"Session {label}")

        # Mark cessation point
        if tc["cessation_min"] is not None:
            ax.axvline(tc["cessation_min"], color="red", linestyle="--",
                       alpha=0.7, linewidth=1)
            ax.text(tc["cessation_min"] + 0.5, ax.get_ylim()[1] * 0.85,
                    f"Stops: {tc['cessation_min']:.1f} min",
                    color="red", fontsize=9)

        ax.set_ylabel("Speed\n(mm/s)", fontsize=10)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_ylim(bottom=0)

    axes[-1].set_xlabel("Time into session (minutes)", fontsize=12)
    fig.suptitle(f"{worm_name} — Speed Over Time ({tc['window_sec']//60}-min rolling avg)\n"
                 f"Red dashed = sustained cessation of movement",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{worm_name}_speed_timecourse.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_habituation_summary(timecourses, output_dir):
    """Cross-session comparison: cessation time across repeated exposures."""
    worms = OrderedDict()
    for name, tc in timecourses.items():
        worm = name.split("_")[0]
        if worm not in worms:
            worms[worm] = []
        worms[worm].append({
            "session": name.split("_")[1],
            "cessation_min": tc["cessation_min"],
        })

    fig, ax = plt.subplots(figsize=(8, 5))

    x_offset = 0
    colors = {"Bubba": "#4C72B0", "Champ": "#DD8452"}

    for worm, sess_list in worms.items():
        xs = list(range(1, len(sess_list) + 1))
        ys = []
        for s in sess_list:
            if s["cessation_min"] is not None:
                ys.append(s["cessation_min"])
            else:
                ys.append(np.nan)  # never stopped

        color = colors.get(worm, "gray")
        valid = [i for i, y in enumerate(ys) if not np.isnan(y)]
        invalid = [i for i, y in enumerate(ys) if np.isnan(y)]

        # Plot points
        ax.scatter([xs[i] for i in valid], [ys[i] for i in valid],
                   s=100, color=color, zorder=5, label=worm)
        ax.scatter([xs[i] for i in invalid],
                   [60 for _ in invalid],  # plot at top with different marker
                   s=100, color=color, marker="^", alpha=0.5, zorder=5)

        # Connect with line
        ax.plot(xs, [y if not np.isnan(y) else 60 for y in ys],
                color=color, alpha=0.5, linestyle="--")

    ax.set_xlabel("Session Number (repeated exposure)", fontsize=12)
    ax.set_ylabel("Time to Stop Moving (minutes)", fontsize=12)
    ax.set_title("Habituation: Does the worm stop sooner with repeated exposure?",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xticks(range(1, 6))
    ax.set_ylim(bottom=0)
    ax.text(0.98, 0.95, "▲ = never stopped during session",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="gray")

    plt.tight_layout()
    path = os.path.join(output_dir, "habituation_cessation_summary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Summary table ────────────────────────────────────────────────────

def print_summary(dist_results, timecourses):
    """Print a text summary table."""
    print("\n" + "=" * 80)
    print("HABITUATION ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"{'Session':<25s} {'Dist (cm)':>10s} {'Det Rate':>10s} "
          f"{'Duration':>10s} {'Stop Time':>12s}")
    print("-" * 80)
    for name in dist_results:
        d = dist_results[name]
        tc = timecourses.get(name)
        stop = f"{tc['cessation_min']:.1f} min" if tc and tc["cessation_min"] else "never"
        print(f"{name:<25s} {d['total_cm']:>10.1f} {d['detection_rate']:>9.1f}% "
              f"{d['duration_min']:>9.1f}m {stop:>12s}")
    print("=" * 80)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Habituation analysis for planarian tracking")
    ap.add_argument("--data_dir", required=True,
                    help="Directory containing *_tracks.csv files")
    ap.add_argument("--output_dir", default=None,
                    help="Where to save plots (default: data_dir)")
    ap.add_argument("--window_sec", type=int, default=300,
                    help="Rolling window for speed smoothing in seconds (default: 300 = 5 min)")
    ap.add_argument("--min_detection_rate", type=float, default=50.0,
                    help="Skip sessions with detection rate below this %% (default: 50)")
    args = ap.parse_args()

    output_dir = args.output_dir or args.data_dir
    os.makedirs(output_dir, exist_ok=True)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    truncations = _load_truncations(repo_root)
    if truncations:
        print("Truncations from session_truncations.json:")
        for k, v in truncations.items():
            print(f"  {k}: truncate_at_s={v.get('truncate_at_s')} "
                  f"({v.get('reason', '')})")

    # Find all track CSVs
    csv_files = sorted(glob.glob(os.path.join(args.data_dir, "*_tracks.csv")))
    if not csv_files:
        print(f"No *_tracks.csv files found in {args.data_dir}")
        return

    print(f"Found {len(csv_files)} session files\n")

    # Load all sessions (filter to Bubba/Champ only).
    sessions = OrderedDict()
    for f in csv_files:
        name = os.path.basename(f).replace("_tracks.csv", "")
        worm_match = re.match(r"(\w+?)_", name)
        if not worm_match or worm_match.group(1) not in ANALYZE_WORMS:
            continue
        trunc = truncations.get(name, {}).get("truncate_at_s")
        sessions[name] = load_session(f, truncate_at_s=trunc)
        det_pct = np.mean(sessions[name]["detected"]) * 100
        note = f" [truncated at {trunc:.1f}s]" if trunc else ""
        print(f"  Loaded {name}: {len(sessions[name]['time_s'])} frames, "
              f"{det_pct:.1f}% detected{note}")

    # Analysis (a): Total distance
    print("\n── Analysis (a): Total Distance ──")
    dist_results = analyze_total_distance(sessions)

    # Analysis (b): Speed timecourse and cessation
    print("\n── Analysis (b): Movement Timecourse ──")
    timecourses = OrderedDict()
    for name, data in sessions.items():
        det_rate = np.mean(data["detected"]) * 100
        if det_rate < args.min_detection_rate:
            print(f"  {name}: SKIPPED (detection rate {det_rate:.1f}% < {args.min_detection_rate}%)")
            timecourses[name] = {
                "time_min": (data["time_s"] - data["time_s"][0]) / 60.0,
                "smoothed_speed": np.full_like(data["speed_mm_s"], np.nan),
                "raw_speed": data["speed_mm_s"],
                "cessation_min": None,
                "window_sec": args.window_sec,
            }
            continue
        tc = analyze_movement_timecourse(data, window_sec=args.window_sec)
        timecourses[name] = tc
        stop_str = f"{tc['cessation_min']:.1f} min" if tc["cessation_min"] else "never"
        print(f"  {name}: cessation at {stop_str}")

    # Summary
    print_summary(dist_results, timecourses)

    # Plots
    print("\n── Generating Plots ──")
    for worm in ["Bubba", "Champ"]:
        plot_total_distance(dist_results, output_dir, worm)
        plot_speed_timecourse(sessions, timecourses, output_dir, worm)

    plot_habituation_summary(timecourses, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
