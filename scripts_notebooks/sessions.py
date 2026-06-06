#!/usr/bin/env python3
"""
sessions.py — Group a flat folder of clips into recording sessions by time gaps.

The live_capture clips are 1-minute MKVs named "<YYYY-MM-DD_HH-MM-SS>.mkv". Within
one recording session they sit ~60s apart (continuous capture); a real break
between sessions (swap animal, reset condition) shows up as a much larger gap.
We split on any gap larger than GAP_THRESHOLD_S and label the runs S1, S2, ...

This lives in one place so label_setup.py, watch_folder_tracker.py, and
measure_accuracy.py all agree on which clip belongs to which session. The clips
stay flat in live_capture/ — callers pass --session S2 and we filter.

Usage from a script:
    from sessions import list_session_clips, group_sessions
    clips = list_session_clips(args.clips_dir, args.session)   # only S2's clips

CLI (for inspection):
    python sessions.py --clips_dir ../live_capture
"""

import os
import re
import glob
import argparse
from datetime import datetime

VIDEO_EXTS = (".mkv", ".mp4", ".avi", ".mov")
GAP_THRESHOLD_S = 120          # gap > this between consecutive clips => new session
TS_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})[_ ](\d{2})-(\d{2})-(\d{2})")


def list_clips(d):
    out = []
    for ext in VIDEO_EXTS:
        out.extend(glob.glob(os.path.join(d, f"*{ext}")))
    return sorted(out)


def clip_time(path):
    """Parse the capture time from the filename; None if it doesn't match."""
    m = TS_RE.search(os.path.basename(path))
    if not m:
        return None
    return datetime(*(int(g) for g in m.groups()))


def group_sessions(clips, gap_threshold_s=GAP_THRESHOLD_S):
    """Return {session_id: [clip_path, ...]} keyed S1, S2, ... in time order.

    Clips whose names carry no parseable timestamp are dropped (with the count
    returned separately so callers can warn). Sorting is by capture time.
    """
    timed = [(clip_time(c), c) for c in clips]
    n_skipped = sum(1 for t, _ in timed if t is None)
    timed = sorted((t, c) for t, c in timed if t is not None)

    groups = {}
    prev = None
    idx = 0
    for t, c in timed:
        if prev is None or (t - prev).total_seconds() > gap_threshold_s:
            idx += 1
        groups.setdefault(f"S{idx}", []).append(c)
        prev = t
    return groups, n_skipped


def list_session_clips(clips_dir, session, gap_threshold_s=GAP_THRESHOLD_S):
    """Clips belonging to `session` (e.g. "S2"). If session is None/"all",
    return every timestamped clip in time order. Raises on an unknown session."""
    clips = list_clips(clips_dir)
    groups, _ = group_sessions(clips, gap_threshold_s)
    if session in (None, "all", "ALL"):
        return [c for s in sorted(groups, key=_skey) for c in groups[s]]
    key = session if session.upper().startswith("S") else f"S{session}"
    key = key.upper()
    if key not in groups:
        raise SystemExit(
            f"Session {session!r} not found. Available: "
            f"{', '.join(sorted(groups, key=_skey))}")
    return groups[key]


def _skey(s):
    return int(s[1:]) if s[1:].isdigit() else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips_dir", required=True)
    ap.add_argument("--gap", type=float, default=GAP_THRESHOLD_S,
                    help=f"Gap (s) that starts a new session (default {GAP_THRESHOLD_S}).")
    args = ap.parse_args()

    clips = list_clips(args.clips_dir)
    groups, n_skipped = group_sessions(clips, args.gap)
    print(f"{len(clips)} clips in {args.clips_dir}  "
          f"({n_skipped} without a parseable timestamp)\n")
    for s in sorted(groups, key=_skey):
        cs = groups[s]
        t0 = clip_time(cs[0]).strftime("%H:%M:%S")
        t1 = clip_time(cs[-1]).strftime("%H:%M:%S")
        print(f"  {s}: {len(cs):>2} clips   {t0}–{t1}")
    print(f"\n{len(groups)} sessions (gap > {args.gap:.0f}s).")


if __name__ == "__main__":
    main()
