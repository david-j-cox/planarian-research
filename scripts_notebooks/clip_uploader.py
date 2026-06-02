#!/usr/bin/env python3
"""
clip_uploader.py — Capture-node janitor: move finished OBS clips into Google
Drive so a faster compute machine can pull and process them, keeping this Mac's
disk near-empty during a multi-week run.

How it works (no Drive API needed — uses the Google Drive desktop app):
  - Watches --watch_dir (where OBS writes 1-minute clips).
  - When a clip's size has been stable for --settle seconds (OBS finished it),
    MOVES it into --drive_dir (a folder inside your synced Google Drive).
  - The Drive desktop app uploads it to the cloud; the other Mac's Drive app
    pulls it down into that machine's Drive folder, where its tracker watches.
  - Moving (not copying) frees local disk immediately. A manifest logs every
    moved clip so the run is auditable and restart-safe.

Default --drive_dir points at:
  ~/Library/CloudStorage/GoogleDrive-<account>/My Drive/planarian_clips

Usage:
  cd scripts_notebooks
  source ../venv/bin/activate
  python clip_uploader.py \
      --watch_dir ../live_capture \
      --drive_dir "/Users/davidjcox/Library/CloudStorage/GoogleDrive-cox.david.j@gmail.com/My Drive/planarian_clips"

Stop with Ctrl-C. Safe to restart — already-moved clips are gone from
watch_dir, so they won't be reprocessed.

Python 3.9+. Stdlib only.
"""

import os
import sys
import time
import glob
import json
import shutil
import argparse
from datetime import datetime

VIDEO_EXTS = (".mkv", ".mp4", ".avi", ".mov")

DEFAULT_DRIVE = ("/Users/davidjcox/Library/CloudStorage/"
                 "GoogleDrive-cox.david.j@gmail.com/My Drive/planarian_clips")


def list_videos(d):
    out = []
    for ext in VIDEO_EXTS:
        out.extend(glob.glob(os.path.join(d, f"*{ext}")))
    return sorted(out)


def is_settled(path, settle_s):
    """True if the file size is stable for settle_s seconds (OBS done writing)."""
    try:
        s1 = os.path.getsize(path)
    except OSError:
        return False
    time.sleep(settle_s)
    try:
        s2 = os.path.getsize(path)
    except OSError:
        return False
    return s1 == s2 and s2 > 0


def run(args):
    if not os.path.isdir(args.drive_dir):
        # Don't silently create outside Drive — make the user confirm the path.
        os.makedirs(args.drive_dir, exist_ok=True)
        print(f"Created {args.drive_dir}")
    manifest = os.path.join(args.drive_dir, "_upload_manifest.jsonl")

    print(f"Watching   {os.path.abspath(args.watch_dir)}")
    print(f"Moving to  {args.drive_dir}")
    print(f"Poll {args.poll}s, settle {args.settle}s. Ctrl-C to stop.\n")

    moved = 0
    try:
        while True:
            for path in list_videos(args.watch_dir):
                name = os.path.basename(path)
                if not is_settled(path, args.settle):
                    continue
                size = os.path.getsize(path)
                dest = os.path.join(args.drive_dir, name)
                try:
                    shutil.move(path, dest)
                except Exception as e:
                    print(f"[{datetime.now():%H:%M:%S}] FAILED to move {name}: {e}")
                    continue
                rec = {"clip": name, "bytes": size,
                       "moved_at": datetime.now().isoformat(timespec="seconds")}
                with open(manifest, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                moved += 1
                print(f"[{datetime.now():%H:%M:%S}] moved {name} "
                      f"({size/1e6:.0f} MB) -> Drive  [{moved} total]")
            time.sleep(args.poll)
    except KeyboardInterrupt:
        print(f"\nStopped. {moved} clips moved this session.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch_dir", default="../live_capture",
                    help="Folder OBS writes clips into (default: ../live_capture).")
    ap.add_argument("--drive_dir", default=DEFAULT_DRIVE,
                    help="Google Drive synced folder to move clips into.")
    ap.add_argument("--poll", type=float, default=5.0,
                    help="Seconds between scans (default: 5).")
    ap.add_argument("--settle", type=float, default=3.0,
                    help="Seconds a clip's size must be stable before moving "
                         "(default: 3) so we never move a half-written file.")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
