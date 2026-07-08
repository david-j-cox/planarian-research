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

# Implausibly fast frames are almost always tracker blob-jumps. Real planarian
# glide speed tops out near 3 mm/s; >15 mm/s is unambiguous tracker error.
TRACKER_ERROR_SPEED_MM_S = 15.0
ANALYZABLE_SOURCES = {
    "tracked", "imputed_short", "imputed_bisect",
    "imputed_anchored", "human_traced",
}


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

    Tracker errors are re-tagged in-memory: rows currently 'tracked' with
    speed_mm_s > TRACKER_ERROR_SPEED_MM_S become source='tracker_error' and
    are dropped from the analyzable mask.
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
    source = []
    analyzable = []
    n_error = 0

    for r in rows:
        t = float(r["time_s"])
        if truncate_at_s is not None and t >= truncate_at_s:
            continue
        has_det = bool(r["centroid_x_mm"].strip())
        src = (r.get("source") or "").strip()
        time_s.append(t)
        detected.append(has_det)
        if has_det:
            x_mm.append(float(r["centroid_x_mm"]))
            y_mm.append(float(r["centroid_y_mm"]))
            sp = r["speed_mm_s"].strip()
            sp_val = float(sp) if sp else 0.0
            speed_mm_s.append(sp_val)
            if src == "tracked" and sp_val > TRACKER_ERROR_SPEED_MM_S:
                src = "tracker_error"
                n_error += 1
        else:
            x_mm.append(np.nan)
            y_mm.append(np.nan)
            speed_mm_s.append(np.nan)
        source.append(src)
        analyzable.append(src in ANALYZABLE_SOURCES)

    if n_error:
        print(f"  [tracker-error] {os.path.basename(csv_path)}: re-tagged "
              f"{n_error} rows with speed > {TRACKER_ERROR_SPEED_MM_S} mm/s")

    return {
        "time_s": np.array(time_s),
        "x_mm": np.array(x_mm),
        "y_mm": np.array(y_mm),
        "speed_mm_s": np.array(speed_mm_s),
        "detected": np.array(detected),
        "source": np.array(source),
        "analyzable": np.array(analyzable),
    }


def compute_step_distances(data):
    """Frame-to-frame distances in mm. Zeros out steps where either endpoint
    is non-analyzable (tracker error, LOST, etc.) so tracker glitches don't
    inflate distance."""
    dx = np.diff(data["x_mm"])
    dy = np.diff(data["y_mm"])
    steps = np.sqrt(dx**2 + dy**2)
    ana = data["analyzable"]
    edge_ok = ana[:-1] & ana[1:]
    steps = np.where(edge_ok, steps, np.nan)
    return steps


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
    """Compute total distance traveled (mm) for each session.

    Tracker-error and LOST rows are upstream-filtered in compute_step_distances
    (their steps are NaN), so plain nansum is the right aggregator.
    """
    results = OrderedDict()
    for name, data in sessions.items():
        steps = compute_step_distances(data)
        total_mm = np.nansum(steps)
        det_rate = np.mean(data["detected"]) * 100
        ana_rate = np.mean(data["analyzable"]) * 100
        duration_min = (data["time_s"][-1] - data["time_s"][0]) / 60.0
        results[name] = {
            "total_mm": total_mm,
            "total_cm": total_mm / 10,
            "detection_rate": det_rate,
            "analyzable_rate": ana_rate,
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

    Non-analyzable rows (tracker_error, LOST) get NaN x/y locally so a
    blob-jump endpoint can't poison the window's displacement.

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
    ana = data["analyzable"]
    x_mm = np.where(ana, data["x_mm"], np.nan)
    y_mm = np.where(ana, data["y_mm"], np.nan)

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
    """Bar chart of total distance per session for one worm (APA grayscale)."""
    sessions = [k for k in results if k.startswith(worm_name)]
    if not sessions:
        return

    labels = [s.split("_")[1] for s in sessions]  # "0001", "0002", etc.
    distances = [results[s]["total_cm"] for s in sessions]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Bubba = solid black, Champ = white with diagonal hatching.
    if worm_name == "Champ":
        ax1.bar(labels, distances, facecolor="white", hatch="///",
                edgecolor="black", linewidth=1.0)
    else:
        ax1.bar(labels, distances, facecolor="black",
                edgecolor="black", linewidth=1.0)

    ax1.set_xlabel("Session Number", fontsize=24, labelpad=12)
    ax1.set_ylabel("Total Distance\nTraveled (cm)", fontsize=24, labelpad=12)
    ax1.tick_params(labelsize=14)
    ax1.set_ylim(bottom=0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{worm_name}_total_distance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_speed_timecourse(sessions_data, timecourses, output_dir, worm_name):
    """Speed over time for all sessions of one worm, stacked (APA grayscale)."""
    worm_sessions = [k for k in sessions_data if k.startswith(worm_name)]
    if not worm_sessions:
        return

    n = len(worm_sessions)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    # Light-to-dark gray gradient across sessions.
    grays = [str(g) for g in np.linspace(0.65, 0.0, n)]

    for i, (sess_name, ax) in enumerate(zip(worm_sessions, axes)):
        tc = timecourses[sess_name]
        label = sess_name.split("_")[1]

        # Raw speed as faint dots
        valid = ~np.isnan(tc["raw_speed"])
        ax.scatter(tc["time_min"][valid], tc["raw_speed"][valid],
                   s=0.3, alpha=0.15, color="0.7", rasterized=True)

        # Smoothed speed as line
        ax.plot(tc["time_min"], tc["smoothed_speed"],
                color=grays[i], linewidth=1.5, label=f"Session {label}")

        # Mark cessation point with a thin black dashed line
        if tc["cessation_min"] is not None:
            ax.axvline(tc["cessation_min"], color="black", linestyle="--",
                       alpha=0.6, linewidth=1)

        ax.set_ylabel("Speed (mm/s)", fontsize=24, labelpad=12)
        ax.tick_params(labelsize=14)
        ax.legend(loc="upper right", fontsize=12, frameon=False)
        ax.set_ylim(bottom=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time (min)", fontsize=24, labelpad=12)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{worm_name}_speed_timecourse.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_habituation_summary(timecourses, output_dir):
    """Cross-session comparison: cessation time across repeated exposures
    (APA grayscale: Bubba=filled circle/solid line, Champ=hollow circle/dashed)."""
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

    worm_marker_style = {
        "Bubba": {"marker": "o", "facecolor": "black",
                  "edgecolor": "black", "linestyle": "-"},
        "Champ": {"marker": "o", "facecolor": "white",
                  "edgecolor": "black", "linestyle": "--"},
    }

    for worm, sess_list in worms.items():
        style = worm_marker_style.get(worm,
                                      {"marker": "s", "facecolor": "0.5",
                                       "edgecolor": "black", "linestyle": ":"})
        xs = list(range(1, len(sess_list) + 1))
        ys = [s["cessation_min"] if s["cessation_min"] is not None else np.nan
              for s in sess_list]
        valid = [i for i, y in enumerate(ys) if not np.isnan(y)]
        invalid = [i for i, y in enumerate(ys) if np.isnan(y)]

        # Stopped points
        ax.scatter([xs[i] for i in valid], [ys[i] for i in valid],
                   s=100, facecolor=style["facecolor"],
                   edgecolor=style["edgecolor"], linewidth=1.2,
                   marker=style["marker"], zorder=5, label=worm)
        # Never-stopped: triangle at top of plot
        ax.scatter([xs[i] for i in invalid], [60 for _ in invalid],
                   s=100, facecolor=style["facecolor"],
                   edgecolor=style["edgecolor"], linewidth=1.2,
                   marker="^", zorder=5)

        # Connecting line (black, linestyle distinguishes worm)
        ax.plot(xs, [y if not np.isnan(y) else 60 for y in ys],
                color="black", alpha=0.6, linestyle=style["linestyle"],
                linewidth=1.0)

    ax.set_xlabel("Session Number", fontsize=24, labelpad=12)
    ax.set_ylabel("Time to First\nSustained Stop (min)", fontsize=24, labelpad=12)
    ax.set_xticks(range(1, 6))
    ax.tick_params(labelsize=14)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=14, frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "habituation_cessation_summary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Summary table ────────────────────────────────────────────────────

def print_summary(dist_results, timecourses):
    """Print a text summary table."""
    print("\n" + "=" * 92)
    print("HABITUATION ANALYSIS SUMMARY")
    print("=" * 92)
    print(f"{'Session':<25s} {'Dist (cm)':>10s} {'Det':>7s} {'Analyz':>8s} "
          f"{'Duration':>10s} {'Stop Time':>12s}")
    print("-" * 92)
    for name in dist_results:
        d = dist_results[name]
        tc = timecourses.get(name)
        stop = f"{tc['cessation_min']:.1f} min" if tc and tc["cessation_min"] else "never"
        print(f"{name:<25s} {d['total_cm']:>10.1f} {d['detection_rate']:>6.1f}% "
              f"{d['analyzable_rate']:>7.1f}% {d['duration_min']:>9.1f}m "
              f"{stop:>12s}")
    print("=" * 92)


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
        ana_pct = np.mean(sessions[name]["analyzable"]) * 100
        note = f" [truncated at {trunc:.1f}s]" if trunc else ""
        print(f"  Loaded {name}: {len(sessions[name]['time_s'])} frames, "
              f"{det_pct:.1f}% detected, {ana_pct:.1f}% analyzable{note}")

    # Analysis (a): Total distance
    print("\n── Analysis (a): Total Distance ──")
    dist_results = analyze_total_distance(sessions)

    # Analysis (b): Speed timecourse and cessation
    print("\n── Analysis (b): Movement Timecourse ──")
    timecourses = OrderedDict()
    for name, data in sessions.items():
        ana_rate = np.mean(data["analyzable"]) * 100
        if ana_rate < args.min_detection_rate:
            print(f"  {name}: SKIPPED (analyzable rate {ana_rate:.1f}% < {args.min_detection_rate}%)")
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
