#!/usr/bin/env python3
"""
clean_tracks.py -- produce an analysis-ready copy of the live tracks CSV.

The live CSV is an append-log in PROCESSING order: clips reprocessed during the
run land out of sequence, and (defensively) a clip could appear twice. This
writes a SEPARATE cleaned file -- it never modifies the live CSV -- that is:
  - chronological: sorted by (clip start time, native_frame)
  - de-duplicated: one row per (video_file, native_frame), last write wins
  - annotated: adds a leading wall_time ISO column (clip start + time_s)

Usage:
  python clean_tracks.py --session worm_run_01
  -> writes ../realtime_runs/worm_run_01_tracks_clean.csv
"""
import argparse
import csv
import os
from datetime import datetime, timedelta

COLS = ["video_file", "native_frame", "time_s", "x_mm", "y_mm",
        "speed_mm_s", "conf", "state"]


def clip_start(name):
    try:
        return datetime.strptime(os.path.basename(name)[:19], "%Y-%m-%d_%H-%M-%S")
    except (ValueError, IndexError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="worm_run_01")
    ap.add_argument("--runs_dir", default="../realtime_runs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no_trim", action="store_true",
                    help="keep overlapping frames instead of trimming to a strictly "
                         "increasing timeline")
    a = ap.parse_args()
    src = os.path.join(a.runs_dir, f"{a.session}_tracks.csv")
    out = a.out or os.path.join(a.runs_dir, f"{a.session}_tracks_clean.csv")

    rows = {}                       # (video_file, native_frame) -> row, last wins
    raw = 0; bad = 0; meta = None
    with open(src) as f:
        for row in csv.reader(f):
            if not row:
                continue
            if row[0].startswith("#"):
                meta = ",".join(row); continue
            if row[0] == "video_file":
                continue
            if len(row) < 8:
                bad += 1; continue
            try:
                int(row[1]); [float(row[i]) for i in (2, 3, 4, 5, 6)]
            except ValueError:
                bad += 1; continue          # skip a partial last line mid-append
            raw += 1
            rows[(row[0], row[1])] = row

    # attach a true wall-clock time to each row, then sort by it
    timed = []
    for r in rows.values():
        cs = clip_start(r[0])
        wt = cs + timedelta(seconds=float(r[2])) if cs else datetime.max
        timed.append((wt, r))
    timed.sort(key=lambda x: (x[0], x[1][0]))

    # trim clip-to-clip overlap: keep a strictly increasing timeline so the same
    # ~0.2-2s window isn't counted twice where consecutive clips overlap
    kept = []
    last = None
    trimmed = 0
    for wt, r in timed:
        if last is not None and wt <= last and not a.no_trim:
            trimmed += 1; continue
        kept.append((wt, r)); last = wt

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        if meta:
            w.writerow([meta + f"  (cleaned: sorted by wall_time, deduped, "
                               f"{'overlap-trimmed, ' if not a.no_trim else ''}"
                               f"{len(kept)} rows)"])
        w.writerow(["wall_time"] + COLS)
        for wt, r in kept:
            w.writerow([wt.isoformat(timespec="milliseconds") if wt != datetime.max else ""] + r)

    dups = raw - len(rows)
    span = ""
    if kept:
        t0, t1 = kept[0][0], kept[-1][0]
        if t0 != datetime.max and t1 != datetime.max:
            span = f"  span {t0.strftime('%H:%M')}..{t1.strftime('%H:%M')}"
    print(f"read {raw} rows, dropped {bad} malformed, removed {dups} duplicate, "
          f"trimmed {trimmed} overlap")
    print(f"wrote {len(kept)} clean rows -> {out}{span}")


if __name__ == "__main__":
    main()
