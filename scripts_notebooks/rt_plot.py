#!/usr/bin/env python3
"""
rt_plot.py -- render the live worm location CSV to a 2D figure.

Reads realtime_runs/<session>_tracks.csv and writes a PNG with two panels:
  (left)  trajectory in the dish plane, colored by time (old=blue -> new=red),
          with the most recent position marked; unphysical single-frame jumps
          (speed > --max_speed) are broken so they don't draw streaks.
  (right) 2D occupancy heatmap (where the worm spent time).

One-shot:   python rt_plot.py --session worm_run_01 --out ../realtime_runs/worm_run_01_live.png
Live loop:  python rt_plot.py --session worm_run_01 --loop 30   # re-render every 30s
"""
import argparse
import csv
import os
import time
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless: write files, no GUI needed
import matplotlib.pyplot as plt

# Live view shows only the most recent HISTORY_S seconds. The full session is
# ~400k+ points; rendering all of it starves a small CPU box. The complete
# record stays in the CSV; only the plot window is capped.
HISTORY_S = float(os.environ.get("PLOT_HISTORY_S", 3 * 3600))    # default: last 3 h


def _clip_start(name):
    """Parse wall-clock start time from a clip filename like
    '2026-06-25_19-39-15.mkv'. Returns epoch seconds, or None."""
    try:
        return datetime.strptime(os.path.basename(name)[:19],
                                 "%Y-%m-%d_%H-%M-%S").timestamp()
    except (ValueError, IndexError):
        return None


def load(csv_path, max_speed):
    xs, ys, sp, tt = [], [], [], []
    for_idx = 0
    with open(csv_path) as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#") or row[0] == "video_file":
                continue
            try:
                x, y, s = float(row[3]), float(row[4]), float(row[5])
                tsec = float(row[2])
            except (ValueError, IndexError):
                continue
            cs = _clip_start(row[0])
            wall = cs + tsec if cs is not None else for_idx / 5.0   # fallback: ~5 samples/s
            xs.append(x); ys.append(y); sp.append(s); tt.append(wall)
            for_idx += 1
    xs, ys, sp, tt = map(lambda a: np.array(a, float), (xs, ys, sp, tt))
    if len(tt):
        # Rows are in PROCESSING order, not time order: clips reprocessed during
        # the run get appended out of sequence, which makes the trace jump
        # backwards in time. Sort by true wall-clock time so the path is
        # chronological. (stable sort keeps within-clip frame order.)
        order = np.argsort(tt, kind="stable")
        xs, ys, sp, tt = xs[order], ys[order], sp[order], tt[order]
        if HISTORY_S and len(tt):                                  # keep only recent window
            keep = tt >= (tt.max() - HISTORY_S)
            xs, ys, sp, tt = xs[keep], ys[keep], sp[keep], tt[keep]
        tmin = (tt - tt.min()) / 60.0                              # minutes since start
        t0 = float(tt.min())
    else:
        tmin = tt; t0 = 0.0
    return xs, ys, sp, tmin, t0


def load_behavior(path, t0):
    """Load the per-second behavior CSV and align it to the position timeline
    (same t0). Returns (minutes_since_start, behavior_label) arrays. Empty if the
    file is missing/empty -- behavior only populates once it's running live."""
    if not os.path.exists(path) or t0 == 0.0:
        return np.array([]), np.array([])
    tmins, labs = [], []
    for row in csv.reader(open(path)):
        if not row or row[0].startswith("#") or row[0] == "video_file":
            continue
        try:
            cs = _clip_start(row[0]); tsec = float(row[2])
        except (ValueError, IndexError):
            continue
        if cs is None:
            continue
        tmins.append((cs + tsec - t0) / 60.0); labs.append(row[3])
    return np.array(tmins), np.array(labs)


def _rolling_nanmean(a, w):
    """Rolling mean over window w, ignoring NaNs (the blanked tracking jumps)."""
    m = ~np.isnan(a)
    af = np.where(m, a, 0.0)
    k = np.ones(w)
    num = np.convolve(af, k, mode="same")
    den = np.convolve(m.astype(float), k, mode="same")
    out = np.full_like(a, np.nan)
    out[den > 0] = num[den > 0] / den[den > 0]
    return out


# activity-bucket edges (mm/s) -- match rt_watch.py --keep_edges so the speed
# panel's regimes line up with the rest/cruise/active training buckets
ACT_EDGES = (0.7, 1.05)

SMOOTH_WIN = 15        # rolling-average window for time-series panels (~3 s @ 5 Hz)


def turning_rate(xs, ys, tmin, sp, jump, min_speed=0.3, step=5):
    """Rate of change of TRAVEL DIRECTION (deg/s), aligned to tmin.

    'Heading' = the compass direction the centroid is moving. We estimate it from
    displacement over `step` samples (~1 s at 5 Hz) rather than adjacent frames,
    so sub-pixel jitter doesn't masquerade as turning. The output is how many
    degrees that travel direction changes per second: 0 = moving straight, large
    = sharp/erratic turning. NOTE: this is direction of motion, not body/head
    orientation (head/tail isn't in this CSV). NaN when ~stationary (direction is
    undefined at near-zero speed) or across a tracking jump."""
    n = len(xs)
    out = np.full(n, np.nan)
    if n < 2 * step + 1:
        return out
    head = np.arctan2(ys[step:] - ys[:-step], xs[step:] - xs[:-step])   # len n-step
    dh = head[step:] - head[:-step]                                     # len n-2*step
    dh = (dh + np.pi) % (2 * np.pi) - np.pi                             # wrap
    dt = (tmin[2 * step:] - tmin[:-2 * step]) * 60.0                    # seconds spanned
    rate = np.degrees(np.abs(dh)) / np.where(dt > 0, dt, np.nan)
    idx = np.arange(step, n - step)                                    # center indices
    mid = sp[idx]
    good = (mid > min_speed) & (mid <= 40) & ~jump[idx]
    out[idx] = np.where(good, rate, np.nan)
    return out


def estimate_dish(xs, ys, jump):
    """Estimate dish center + radius (mm) from the worm's coverage. Uses the
    1/99 percentile extent of jump-filtered points so tracking-spike outliers
    don't blow up the bounds; converges to the true center as the worm works
    the whole dish."""
    gx, gy = (xs[~jump], ys[~jump]) if (~jump).any() else (xs, ys)
    cx = (np.percentile(gx, 1) + np.percentile(gx, 99)) / 2
    cy = (np.percentile(gy, 1) + np.percentile(gy, 99)) / 2
    # R = farthest the centroid reaches from center = effective wall, so the worm
    # never plots beyond the rim (the old quarter-span estimate understated it).
    R = float(np.nanmax(np.hypot(gx - cx, gy - cy)))
    return cx, cy, R


def render(csv_path, out_path, max_speed, center=True, behavior_csv=None):
    xs, ys, sp, tmin, t0 = load(csv_path, max_speed)
    if len(xs) < 2:
        return None, 0
    jump = sp > max_speed
    cx, cy, R = estimate_dish(xs, ys, jump)
    if center:
        xs, ys = xs - cx, ys - cy            # 0,0 = dish center; rim at radius R
    lab = "from center" if center else "(mm)"
    fig = plt.figure(figsize=(13, 17.5))
    gs = fig.add_gridspec(6, 2, height_ratios=[1.5, 0.8, 0.8, 0.8, 0.8, 0.5],
                          hspace=0.45, wspace=0.22)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    ax4 = fig.add_subplot(gs[2, :], sharex=ax3)
    ax5 = fig.add_subplot(gs[3, :], sharex=ax3)
    ax6 = fig.add_subplot(gs[4, :], sharex=ax3)
    ax7 = fig.add_subplot(gs[5, :], sharex=ax3)   # ethogram

    lim = R * 1.12                                  # symmetric plot extent (rim + margin)
    rim_kw = dict(fill=False, ls="--", lw=1.0, color="0.4")

    # ---- panel 1: trajectory colored by time ----
    # Only break the path on a TRUE teleport (gross mis-detection). The 6 mm/s
    # `jump` mask is for the metric panels; using it here also erased real brisk
    # transits (the worm crosses a band fast -> segment >6 mm/s -> blanked ->
    # false vertical gap even though the path runs through). Draw real moves;
    # break only on >15 mm/s, which is unambiguously a mis-detection.
    teleport = sp > 15.0
    t = tmin / tmin.max() if tmin.max() else tmin
    pts = np.column_stack([xs, ys])
    for i in range(1, len(pts)):
        if teleport[i]:
            continue
        ax1.plot(pts[i-1:i+1, 0], pts[i-1:i+1, 1],
                 color=plt.cm.turbo(t[i]), lw=0.6, alpha=0.7)
    if center:
        ax1.add_patch(plt.Circle((0, 0), R, **rim_kw))      # dish rim
        ax1.plot(0, 0, "+", color="0.4", ms=10)             # center
        ax1.set_xlim(-lim, lim); ax1.set_ylim(-lim, lim)
    ax1.scatter(xs[-1], ys[-1], s=90, c="red", edgecolor="k", zorder=5, label="now")
    ax1.set_title(f"Trajectory  ({len(xs)} pts, blue=start -> red=now)")
    ax1.set_xlabel(f"x {lab} (mm)"); ax1.set_ylabel(f"y {lab} (mm)")
    ax1.set_aspect("equal"); ax1.invert_yaxis()     # image coords: y down
    ax1.legend(loc="upper right")

    # ---- panel 2: occupancy heatmap ----
    h, xe, ye = np.histogram2d(xs, ys, bins=60)
    ax2.imshow(np.log1p(h.T), origin="upper",
               extent=[xe[0], xe[-1], ye[-1], ye[0]],
               cmap="magma", aspect="equal")
    if center:
        ax2.add_patch(plt.Circle((0, 0), R, **rim_kw))
    ax2.set_title("Occupancy (log dwell time)")
    ax2.set_xlabel(f"x {lab} (mm)"); ax2.set_ylabel(f"y {lab} (mm)")

    # ---- panel 3: x(t) and y(t) time series, jumps blanked ----
    xp = np.where(jump, np.nan, xs)
    yp = np.where(jump, np.nan, ys)
    ax3.plot(tmin, xp, lw=0.8, color="#1f77b4", label="x")
    ax3.plot(tmin, yp, lw=0.8, color="#d62728", label="y")
    if center:
        for s in (R, -R):
            ax3.axhline(s, ls="--", lw=0.8, color="0.6")
        ax3.axhline(0, lw=0.6, color="0.8")
        ax3.text(0.2, R, "wall", va="bottom", fontsize=8, color="0.5")
        ax3.text(0.2, -R, "wall", va="top", fontsize=8, color="0.5")
    ax3.set_title("Position vs time  (0 = dish center)")
    ax3.set_ylabel(f"position {lab} (mm)")
    ax3.set_xlim(0, max(tmin.max(), 1)); ax3.grid(alpha=0.25)
    if center:
        ax3.set_ylim(-R * 1.22, R * 1.32)        # headroom so the legend clears the data
    ax3.legend(loc="upper right", ncol=2)
    plt.setp(ax3.get_xticklabels(), visible=False)   # shared x with ax4 below

    # ---- panel 4: speed over time, LOG y (speed is heavily right-skewed) ----
    spd = np.where(jump, np.nan, sp)                 # blank tracking-jump spikes
    spd_s = _rolling_nanmean(spd, SMOOTH_WIN)
    floor = 0.02                                     # log floor (mm/s); rest sits near here
    smax = np.nanmax(spd_s) if np.isfinite(np.nanmax(spd_s)) else 2.0
    ax4.plot(tmin, spd_s, lw=1.3, color="#2ca02c", label="speed (3 s avg)")
    ax4.set_yscale("log"); ax4.set_ylim(floor, smax * 1.3)        # contains all data
    ax4.axhspan(floor, ACT_EDGES[0], color="#3b6", alpha=0.05)
    ax4.axhspan(ACT_EDGES[0], ACT_EDGES[1], color="#fb3", alpha=0.05)
    ax4.axhspan(ACT_EDGES[1], smax * 1.3, color="#f55", alpha=0.05)
    for e in ACT_EDGES:
        ax4.axhline(e, ls="--", lw=0.7, color="0.6")
    ax4.set_title("Speed vs time  (log scale; bands = rest / cruise / active)")
    ax4.set_ylabel("speed (mm/s)")
    ax4.grid(alpha=0.25, which="both"); ax4.legend(loc="upper right")
    plt.setp(ax4.get_xticklabels(), visible=False)

    # ---- panel 5: distance from dish center (thigmotaxis) ----
    rad = np.hypot(xs, ys) if center else np.hypot(xs - cx, ys - cy)
    rad = np.where(jump, np.nan, rad)
    rmax = np.nanmax(rad) if np.isfinite(np.nanmax(rad)) else R   # farthest reach = wall
    ax5.plot(tmin, rad, lw=0.5, color="0.8", alpha=0.45)         # raw (real fast excursions)
    ax5.plot(tmin, _rolling_nanmean(rad, SMOOTH_WIN), lw=1.3, color="#1f77b4",
             label="radius (3 s avg)")
    ax5.axhline(rmax, ls="--", lw=0.8, color="0.6")
    ax5.text(0.2, rmax, "max reach (~wall)", fontsize=7, color="0.5", va="bottom")
    ax5.set_ylim(0, rmax * 1.28)                                  # headroom: label/legend clear data
    ax5.set_title("Distance from center  (0 = center, line = wall; high = edge-following)")
    ax5.set_ylabel("radius (mm)")
    ax5.grid(alpha=0.25); ax5.legend(loc="upper right")
    plt.setp(ax5.get_xticklabels(), visible=False)

    # ---- panel 6: turning rate of travel direction ----
    tr = turning_rate(xs, ys, tmin, sp, jump)
    trs = _rolling_nanmean(tr, SMOOTH_WIN)
    ax6.plot(tmin, trs, lw=1.1, color="#9467bd", label="turn rate (3 s avg)")
    tmax = np.nanmax(trs) if np.isfinite(np.nanmax(trs)) else 90
    ax6.set_ylim(0, tmax * 1.3)                                   # headroom so the legend clears the data
    ax6.set_title("Turning rate of travel direction  (deg/s: 0 = straight, high = sharp turns)")
    ax6.set_ylabel("deg/s")
    ax6.grid(alpha=0.25); ax6.legend(loc="upper right")
    plt.setp(ax6.get_xticklabels(), visible=False)

    # ---- panel 7: ethogram (behavior state over time) ----
    BEH_COLORS = {"resting": "#9e9e9e", "gliding": "#2ca02c", "turning": "#1f77b4",
                  "wig_wag": "#9467bd", "contracted": "#d62728",
                  "peristalsis": "#ff7f0e", "reversing": "#17becf"}
    bt, blab = load_behavior(behavior_csv, t0) if behavior_csv else (np.array([]), np.array([]))
    if len(bt):
        # Proper ethogram: bin time, fill each bin with its DOMINANT behavior color.
        # (Overlapping per-point markers let whichever class drew last paint over
        # the rest at this compressed scale -- that made it look all-contracted.)
        span = max(tmin.max(), 1.0)
        nbins = int(np.clip(span, 60, 700))
        edges = np.linspace(0, span, nbins + 1)
        bidx = np.clip(np.digitize(bt, edges) - 1, 0, nbins - 1)
        seen = []
        for b in range(nbins):
            sel = bidx == b
            if sel.any():
                vals, cnts = np.unique(blab[sel], return_counts=True)
                dom = vals[np.argmax(cnts)]
                ax7.axvspan(edges[b], edges[b + 1], color=BEH_COLORS.get(dom, "0.5"), lw=0)
                seen.append(dom)
        order = ["resting", "gliding", "turning", "wig_wag", "contracted",
                 "peristalsis", "reversing"]
        present = [b for b in order if b in set(seen)]
        ax7.legend([plt.Rectangle((0, 0), 1, 1, color=BEH_COLORS[b]) for b in present],
                   present, loc="upper right", ncol=5, fontsize=7, framealpha=0.95,
                   handlelength=1.0, handletextpad=0.3, columnspacing=0.8)
        ax7.set_title("Ethogram  (dominant behavior per ~1-min bin)")
    else:
        ax7.text(0.5, 0.5, "ethogram populates once behavior tracking is live",
                 ha="center", va="center", transform=ax7.transAxes, color="0.55", fontsize=10)
        ax7.set_title("Ethogram")
    ax7.set_yticks([]); ax7.set_ylim(-1, 1)
    ax7.set_xlabel("time (min since start)"); ax7.set_xlim(0, max(tmin.max(), 1))

    mins = tmin.max()
    fig.suptitle(f"worm_run_01 -- live location  |  ~{mins:.0f} min tracked  |  "
                 f"updated {time.strftime('%H:%M:%S')}", fontsize=12)
    fig.subplots_adjust(top=0.95, bottom=0.04, left=0.06, right=0.97)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path, len(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="worm_run_01")
    ap.add_argument("--runs_dir", default="../realtime_runs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_speed", type=float, default=6.0,
                    help="mm/s above which a step is treated as a tracking artifact "
                         "(planarian gliding tops out ~5 mm/s; box-center wobble reads higher)")
    ap.add_argument("--loop", type=float, default=0,
                    help="re-render every N seconds (0 = one-shot)")
    ap.add_argument("--no_center", action="store_true",
                    help="keep raw image coords instead of dish-centered (0,0=dish center)")
    a = ap.parse_args()
    csv_path = os.path.join(a.runs_dir, f"{a.session}_tracks.csv")
    behavior_csv = os.path.join(a.runs_dir, f"{a.session}_behavior.csv")
    out_path = a.out or os.path.join(a.runs_dir, f"{a.session}_live.png")
    while True:
        p, n = render(csv_path, out_path, a.max_speed, center=not a.no_center,
                      behavior_csv=behavior_csv)
        print(f"{time.strftime('%H:%M:%S')}  rendered {n} points -> {p}" if p
              else "not enough data yet")
        if not a.loop:
            break
        time.sleep(a.loop)


if __name__ == "__main__":
    main()
