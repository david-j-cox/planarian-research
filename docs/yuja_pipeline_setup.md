# YuJa pipeline setup — Mac & Windows

This document describes how to set up `yuja_pipeline.py` on a fresh
machine (mac or Windows) so it can run unattended for days against a
YuJa media folder.

## What it does

1. Calls the documented YuJa REST API to enumerate every video in a
   folder (e.g., the "Infrared" folder).
2. Groups videos into "sessions" by parsing the timestamp in each video's
   title. A gap longer than 3 minutes between consecutive videos starts
   a new session.
3. For each session, downloads every MP4, runs the planarian tracker
   over the session, then deletes the raw MP4s.
4. Tracks progress in a manifest JSON so runs are resumable.

## One-time setup

### 1. Get a YuJa API token

YuJa Admin Panel → API → generate token. Copy the 32-character hex
string somewhere safe (you cannot retrieve it later, only regenerate).

### 2. Save the token to a file

**Mac / Linux:**

```bash
echo 'your-32-char-token-here' > ~/.yuja_token
chmod 600 ~/.yuja_token
```

**Windows (PowerShell):**

```powershell
Set-Content -Path $env:USERPROFILE\.yuja_token -Value 'your-32-char-token-here'
# Then restrict to your user account:
icacls $env:USERPROFILE\.yuja_token /inheritance:r /grant:r "$env:USERNAME:F"
```

The pipeline reads this file via `--token_file ~/.yuja_token`. Or set
`YUJA_TOKEN` environment variable instead.

### 3. Clone the repo & set up Python

```bash
git clone <repo-url> "Planarian Research"
cd "Planarian Research"
python -m venv venv
# Mac/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

pip install opencv-python numpy pandas scipy scikit-image tqdm matplotlib
```

### 4. Find the folder ID

```bash
python scripts_notebooks/yuja_pipeline.py --folder_id 0 --manifest /tmp/_ \
    --output_dir /tmp/_ --scratch_dir /tmp/_ --token_file ~/.yuja_token \
    --dry_run 2>&1 | head
```

This will error — but in the process, you can also probe the API:

```bash
curl -s 'https://endicott.yuja.com/services/media/retrieveuserassets/<your_login_id>' \
    -H "authToken: $(cat ~/.yuja_token)"
```

Returns the list of top-level folders with `folder_id` integers. Use
the one whose `title` matches your collection.

## Running

### Smoke test (5 videos)

```bash
python scripts_notebooks/yuja_pipeline.py \
    --folder_id 62258391 \
    --manifest OpenDishWork/yuja_manifest.json \
    --output_dir OpenDishWork/tracker_results \
    --scratch_dir OpenDishWork/yuja_scratch \
    --token_file ~/.yuja_token \
    --max_videos 5
```

Confirms downloading + tracking work end-to-end. Should take ~3 minutes.

### Production run

```bash
python scripts_notebooks/yuja_pipeline.py \
    --folder_id 62258391 \
    --manifest OpenDishWork/yuja_manifest.json \
    --output_dir OpenDishWork/tracker_results \
    --scratch_dir OpenDishWork/yuja_scratch \
    --token_file ~/.yuja_token
```

Default order is `chronological`: videos processed in order of recording
timestamp so you accumulate the longest unbroken continuous stretch
possible. Matches the project's continuous-data goal.

The full Infrared folder (18,171 videos × ~24s/video) takes roughly
**5 CPU-days** on a single machine. Plan accordingly.

### Running unattended for days

The pipeline is resumable — interrupt and restart any time, it picks up
where it left off via the manifest.

**Mac/Linux (`tmux` or `screen`):**

```bash
tmux new -s yuja
# Inside the tmux session:
python scripts_notebooks/yuja_pipeline.py [args...]
# Detach: Ctrl+B then D
# Reattach later: tmux attach -t yuja
```

**Windows (PowerShell, background job):**

```powershell
Start-Job -ScriptBlock {
    cd "C:\path\to\Planarian Research"
    .\venv\Scripts\python.exe scripts_notebooks\yuja_pipeline.py `
        --folder_id 62258391 `
        --manifest OpenDishWork\yuja_manifest.json `
        --output_dir OpenDishWork\tracker_results `
        --scratch_dir OpenDishWork\yuja_scratch `
        --token_file $env:USERPROFILE\.yuja_token `
        *> yuja_run.log
}
```

Or simpler: run inside Windows Terminal in a tab, log it to a file:

```powershell
.\venv\Scripts\python.exe scripts_notebooks\yuja_pipeline.py [args...] *> yuja_run.log
```

### Monitoring progress

The manifest at `OpenDishWork/yuja_manifest.json` is updated after every
video. To see overall progress:

```bash
python -c "import json,collections; m=json.load(open('OpenDishWork/yuja_manifest.json')); print(collections.Counter(v['status'] for v in m['videos'].values()))"
```

## Caveats

- The token authenticates as **you** (an IT Manager). The pipeline can
  see every video your account can see. Keep `~/.yuja_token` secure.
- CloudFront URLs are signed and expire — the pipeline mints a fresh
  one for every download via `streams/hls`, so expiration during a long
  run isn't a problem.
- Tracker calibration: the YuJa videos use a different dish size
  (~62 mm) than the Bubba/Champ benchtop setup (~114 mm). Calibration
  auto-adapts per session.
- Disk usage during a run: at most ~25 MB × N videos per session
  staged at once. After tracking, only the CSV + midlines.npz +
  gaps.json remain (~few MB per session).
