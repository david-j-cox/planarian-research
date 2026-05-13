#!/usr/bin/env python3
"""
yuja_pipeline.py — Enumerate, download, and track a YuJa folder's videos
using the documented YuJa REST API.

Workflow:
    1. List the folder's videos via /services/media/retrievefolderassets
    2. Group videos into sessions by parsing the title's timestamp (gap >
       SESSION_BREAK_S seconds = new session boundary).
    3. For each session, download every video, run the tracker on the
       session as a whole, then delete the raw mp4s.
    4. Resumable: a manifest JSON tracks each video's status.

Auth: API token from YuJa Admin Panel → API. Pass via:
    --token_file ~/.yuja_token        (preferred — file containing the token)
    --token <hex_token>               (less safe — appears in shell history)
    YUJA_TOKEN env var                (third fallback)

Usage:
    python yuja_pipeline.py \\
        --folder_id 62258391 \\
        --manifest OpenDishWork/yuja_manifest.json \\
        --output_dir OpenDishWork/tracker_results \\
        --scratch_dir OpenDishWork/yuja_scratch \\
        --token_file ~/.yuja_token \\
        --max_videos 5     # smoke-test first

Notes on volume: the Infrared folder has 18,171 videos at ~1 min each. At
~24s per video on a MacBook, a full run is ~5 days CPU. Use --max_videos
or --subsample to bound a run.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path


SESSION_BREAK_S = 180   # 3-minute gap → new session
API_BASE = "https://endicott.yuja.com/services"

TITLE_RE = re.compile(r'(\d{4})-(\d{2})-(\d{2})\s+(\d{2})-(\d{2})-(\d{2})')


# ─── Auth & API helpers ──────────────────────────────────────────────────

def _read_token(args) -> str:
    if args.token:
        return args.token.strip()
    if args.token_file:
        return Path(os.path.expanduser(args.token_file)).read_text().strip()
    env = os.environ.get("YUJA_TOKEN")
    if env:
        return env.strip()
    raise SystemExit(
        "No API token provided. Set --token, --token_file, or YUJA_TOKEN.")


def _api_get(path: str, token: str) -> dict | list:
    """GET an API path, return parsed JSON. Path starts with /."""
    url = API_BASE + path
    req = urllib.request.Request(url, headers={"authToken": token})
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read()
    if not body:
        return None
    return json.loads(body)


# ─── Manifest helpers ────────────────────────────────────────────────────

def _load_manifest(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"folder_id": None, "videos": {}, "scraped_at": None}


def _save_manifest(path: str, m: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(m, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


# ─── Folder enumeration ─────────────────────────────────────────────────

def enumerate_folder(folder_id: int, token: str) -> list[dict]:
    """Return every video asset directly under folder_id.

    The API returns folders mixed with video files; for now we only walk
    video files at this level. (Recursion into subfolders is one extra
    line if ever needed.)
    """
    data = _api_get(f"/media/retrievefolderassets/{folder_id}", token)
    if not isinstance(data, list):
        raise RuntimeError(
            f"Unexpected response from retrievefolderassets/{folder_id}: "
            f"{data!r}")
    return [e for e in data if e.get("Asset_type") == "Video File"]


def get_mp4_url(video_id: int, token: str) -> str | None:
    """Return the signed CloudFront MP4 URL for a video, or None if
    unavailable.
    """
    data = _api_get(f"/media/videos/{video_id}/streams/hls", token)
    if not data or not data.get("success") or not data.get("streams"):
        return None
    streams = data["streams"]
    if not streams:
        return None
    mp4 = streams[0].get("typeAndVideoSourceMap", {}).get("MP4") or {}
    url = mp4.get("fileURL")
    if not url:
        # Fall back to HLS cloudFrontURL if MP4 missing
        hls = streams[0].get("typeAndVideoSourceMap", {}).get("HLS") or {}
        url = hls.get("cloudFrontURL") or hls.get("fileURL")
    return url


# ─── Session grouping ───────────────────────────────────────────────────

def _parse_title_timestamp(title: str):
    m = TITLE_RE.match((title or "").strip())
    if not m:
        return None
    return datetime(*[int(g) for g in m.groups()])


def group_into_sessions(videos: list[dict]) -> list[list[dict]]:
    """Sort videos by parsed timestamp and split into sessions on long gaps."""
    parsed = []
    unknown = []
    for v in videos:
        ts = _parse_title_timestamp(v.get("title"))
        if ts is None:
            unknown.append(v)
        else:
            parsed.append((ts, v))
    parsed.sort(key=lambda t: t[0])

    sessions: list[list[dict]] = []
    cur: list[dict] = []
    prev_ts = None
    for ts, v in parsed:
        if prev_ts is not None and (ts - prev_ts).total_seconds() > SESSION_BREAK_S:
            sessions.append(cur)
            cur = []
        cur.append({**v, "_ts": ts.isoformat()})
        prev_ts = ts
    if cur:
        sessions.append(cur)
    if unknown:
        sessions.append(unknown)
    return sessions


def _session_name(session: list[dict]) -> str:
    first = session[0]
    ts = _parse_title_timestamp(first.get("title"))
    if ts is None:
        return f"yuja_unknown_{first.get('video_id', 'x')}"
    return ts.strftime("yuja_%Y%m%d_%H%M%S")


# ─── Download & stage ───────────────────────────────────────────────────

def download_to(url: str, out_path: str, max_retries: int = 3) -> None:
    """Download with retry. Fail loudly on a too-small payload."""
    last_err = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=600) as resp, \
                    open(out_path, "wb") as f:
                shutil.copyfileobj(resp, f)
            if os.path.getsize(out_path) < 4096:
                raise RuntimeError(
                    f"Downloaded file is suspiciously small: "
                    f"{os.path.getsize(out_path)} bytes")
            return
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Download failed after {max_retries} tries: {last_err}")


def stage_for_tracker(mp4_path: str, target_dir: str, title: str) -> str:
    """Hardlink/copy the mp4 into target_dir with a tracker-compatible name.

    Tracker globs *.mkv and parses 'YYYY-MM-DD HH-MM-SS' filenames; OpenCV
    sniffs container by content, so a .mkv-named MP4 opens fine.
    """
    os.makedirs(target_dir, exist_ok=True)
    target = os.path.join(target_dir, f"{title}.mkv")
    if os.path.exists(target):
        os.remove(target)
    try:
        os.link(mp4_path, target)
    except OSError:
        shutil.copy2(mp4_path, target)
    return target


# ─── Tracker integration ────────────────────────────────────────────────

def run_tracker_on_session(session_dir: str, output_dir: str) -> bool:
    cmd = [sys.executable, "scripts_notebooks/open_dish_tracker.py",
           "--data_root", os.path.dirname(session_dir),
           "--output_dir", output_dir,
           "--sessions", os.path.basename(session_dir)]
    print(f"    tracker: {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd).returncode == 0


# ─── Main loop ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder_id", type=int, required=True,
                    help="YuJa folder_id (e.g., 62258391 for Infrared).")
    ap.add_argument("--manifest", required=True,
                    help="Where to read/write the manifest JSON.")
    ap.add_argument("--output_dir", required=True,
                    help="Where tracker outputs go.")
    ap.add_argument("--scratch_dir", required=True,
                    help="Scratch dir for downloaded MP4s "
                         "(cleaned between sessions).")
    ap.add_argument("--token", default=None)
    ap.add_argument("--token_file", default=None)
    ap.add_argument("--max_videos", type=int, default=None,
                    help="Cap on number of videos to process this run.")
    ap.add_argument("--order", choices=["chronological", "subsample"],
                    default="chronological",
                    help="Processing order. 'chronological' (default): "
                         "process pending videos in chronological order so "
                         "we accumulate the longest unbroken continuous "
                         "stretch possible — matches the project's "
                         "continuous-data thesis. 'subsample' (legacy): "
                         "every Nth video uniformly across time.")
    ap.add_argument("--subsample", type=int, default=1,
                    help="With --order=subsample, process every Nth video. "
                         "Ignored when --order=chronological.")
    ap.add_argument("--refresh_manifest", action="store_true",
                    help="Re-fetch the folder listing from YuJa even if a "
                         "manifest already exists.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print planned sessions; do not download or track.")
    args = ap.parse_args()

    token = _read_token(args)

    # 1. Load or rebuild manifest
    manifest = _load_manifest(args.manifest)
    if (not manifest["videos"]) or args.refresh_manifest \
            or manifest.get("folder_id") != args.folder_id:
        print(f"Enumerating folder {args.folder_id} via API...", flush=True)
        videos = enumerate_folder(args.folder_id, token)
        print(f"  found {len(videos)} videos", flush=True)
        manifest["folder_id"] = args.folder_id
        manifest["scraped_at"] = datetime.utcnow().isoformat() + "Z"
        for v in videos:
            key = str(v["video_id"])
            if key not in manifest["videos"]:
                manifest["videos"][key] = {
                    "video_id": v["video_id"],
                    "node_id": v.get("Node_id"),
                    "title": v.get("title", ""),
                    "status": "pending",
                }
        _save_manifest(args.manifest, manifest)
        print(f"  manifest: {args.manifest}", flush=True)
    else:
        print(f"Using existing manifest with {len(manifest['videos'])} "
              f"videos.", flush=True)

    # 2. Filter to pending + order
    pending_all = [v for v in manifest["videos"].values()
                   if v.get("status") != "tracked"]
    # Always sort chronologically — both order modes need this baseline.
    pending_all.sort(key=lambda v: v.get("title", ""))
    if args.order == "subsample" and args.subsample > 1:
        pending = pending_all[::args.subsample]
        print(f"  subsample 1-in-{args.subsample}: "
              f"{len(pending_all)} → {len(pending)} videos", flush=True)
    else:
        pending = pending_all
        print(f"  chronological order: "
              f"{len(pending)} pending videos (starting from "
              f"{pending[0].get('title', '?') if pending else 'n/a'})",
              flush=True)
    if args.max_videos:
        pending = pending[:args.max_videos]
        print(f"  capped to --max_videos={args.max_videos}", flush=True)

    # 3. Group into sessions
    sessions = group_into_sessions(pending)
    print(f"  grouped into {len(sessions)} session(s) "
          f"(session_break={SESSION_BREAK_S}s)", flush=True)
    for i, s in enumerate(sessions[:5]):
        first_ts = _parse_title_timestamp(s[0].get("title", ""))
        last_ts = _parse_title_timestamp(s[-1].get("title", ""))
        print(f"    [{i+1}] {_session_name(s)}  n={len(s)}  "
              f"{first_ts} .. {last_ts}", flush=True)
    if len(sessions) > 5:
        print(f"    ... + {len(sessions) - 5} more session(s)", flush=True)

    if args.dry_run:
        print("\n(dry-run) exiting.", flush=True)
        return

    if not pending:
        print("Nothing to do.", flush=True)
        return

    os.makedirs(args.scratch_dir, exist_ok=True)

    # 4. Process each session sequentially
    processed = 0
    for s_idx, session in enumerate(sessions):
        sess_name = _session_name(session)
        sess_dir = os.path.join(args.scratch_dir, sess_name)
        os.makedirs(sess_dir, exist_ok=True)
        print(f"\n[session {s_idx+1}/{len(sessions)}] {sess_name} "
              f"(n={len(session)})", flush=True)

        for v in session:
            vid = v["video_id"]
            mkey = str(vid)
            print(f"  video {vid}: {v.get('title')!r}", flush=True)
            try:
                url = get_mp4_url(vid, token)
                if not url:
                    raise RuntimeError("no MP4 URL returned")
            except Exception as e:
                print(f"    SKIP: get_mp4_url failed: {e}", flush=True)
                manifest["videos"][mkey]["status"] = "error_url"
                manifest["videos"][mkey]["error"] = str(e)[:200]
                _save_manifest(args.manifest, manifest)
                continue

            tmp_mp4 = os.path.join(args.scratch_dir, f"yuja_{vid}.mp4")
            try:
                t0 = time.monotonic()
                download_to(url, tmp_mp4)
                size_mb = os.path.getsize(tmp_mp4) / 1e6
                dt = time.monotonic() - t0
                print(f"    downloaded {size_mb:.1f} MB in {dt:.1f}s", flush=True)
            except Exception as e:
                print(f"    SKIP: download failed: {e}", flush=True)
                manifest["videos"][mkey]["status"] = "error_download"
                manifest["videos"][mkey]["error"] = str(e)[:200]
                _save_manifest(args.manifest, manifest)
                continue

            try:
                stage_for_tracker(tmp_mp4, sess_dir, v["title"])
                manifest["videos"][mkey]["status"] = "downloaded"
                _save_manifest(args.manifest, manifest)
            finally:
                try:
                    os.remove(tmp_mp4)
                except OSError:
                    pass

            processed += 1
            if args.max_videos and processed >= args.max_videos:
                break

        # Track the session
        if not list(Path(sess_dir).glob("*.mkv")):
            print(f"  no videos staged; skipping tracker", flush=True)
            continue

        ok = run_tracker_on_session(sess_dir, args.output_dir)
        if ok:
            for v in session:
                key = str(v["video_id"])
                if manifest["videos"].get(key, {}).get("status") == "downloaded":
                    manifest["videos"][key]["status"] = "tracked"
                    manifest["videos"][key]["tracked_at"] = \
                        datetime.utcnow().isoformat() + "Z"
            _save_manifest(args.manifest, manifest)
            shutil.rmtree(sess_dir, ignore_errors=True)
            print(f"  session {sess_name}: tracked & cleaned.", flush=True)
        else:
            print(f"  session {sess_name}: TRACKER FAILED; scratch left "
                  f"for inspection.", flush=True)
            for v in session:
                manifest["videos"][str(v["video_id"])]["status"] = "error_tracker"
            _save_manifest(args.manifest, manifest)

        if args.max_videos and processed >= args.max_videos:
            print(f"\nReached --max_videos={args.max_videos}; stopping.",
                  flush=True)
            break

    print(f"\nDone. Processed {processed} videos this run.", flush=True)


if __name__ == "__main__":
    main()
