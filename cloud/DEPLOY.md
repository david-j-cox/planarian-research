# Cloud deploy runbook (RETIRED 2026-06-26 -- reference only)

> The cloud box was decommissioned in favor of **batch catch-up on the Mac**
> (`catch_up.sh`): MPS is ~10x faster and higher quality than a 4-core CPU box, for
> $0. See SESSION_HANDOFF.md. This runbook is kept in case cloud is revisited.

# Cloud deploy runbook + live-state (planarian compute node)

## LIVE (as of 2026-06-26 migration)
- **Box:** Hetzner **CX33** (4 vCPU / 8 GB, x86, Ubuntu 24.04 LTS), Nuremberg.
  Public IP **78.47.76.201**. SSH: `ssh root@78.47.76.201` (key: ~/.ssh/id_ed25519).
- Runs the FULL pipeline on CPU: unified single YOLO pass (location + behavior),
  imgsz **1024**, **stride 4**, ~44s/clip (16s margin under the 60s/clip budget).
- Reads clips from Google Drive via rclone (endicott account), processes, deletes.
- Publishes the live plot to the planarian-live GitHub Pages site (deploy key).
- 4 systemd services, auto-start on boot: planarian-{drivepull,watch,plot,publish}.

## Architecture (why the box can run alone)
Separate **recording Mac** (OBS) records 1-min clips -> Google Drive (continuous,
independent). The **box** pulls + processes them. The old **processing Mac** is
retired -- nothing in the live path depends on it. Turning it off is safe.

## Manage the box
    systemctl status planarian-watch
    journalctl -u planarian-watch -f          # per-clip times (watch for OVER 60s)
    systemctl restart planarian-plot          # etc.
    ls ~/clips_in | wc -l                      # queue depth (backlog if growing)

## Rebuild from scratch (if the box is lost)
1. Provision CX33 (Ubuntu 24.04, x86), add the planarian-deploy SSH key.
2. `git clone https://github.com/david-j-cox/planarian-research.git ~/planarian-research`
   IMPORTANT: core modules (yolo_tracker, fusion_tracker, behavior_features,
   yolo_to_signals) historically lived on **dev**, not main -- if missing after
   clone, scp the Mac's scripts_notebooks/*.py over.
3. scp the gitignored weights from a Mac:
   - `scripts_notebooks/runs/pose/worm_white7mp_n/weights/best.pt`
   - `realtime_runs/behavior_clf.joblib`
4. rclone Drive auth (endicott): on box `rclone config` (headless) OR run
   `rclone authorize "drive"` on a machine WITH a browser, sign in with endicott,
   paste the token into ~/.config/rclone/rclone.conf as `[drive] type=drive
   scope=drive token=<json>`. Verify: `rclone lsf "drive:Planarian Research/planarian_clips"`.
5. Site push: `ssh-keygen` on box, add the .pub as a **write** deploy key on
   david-j-cox/planarian-live (`gh repo deploy-key add ... --allow-write`), then
   `git clone git@github.com:david-j-cox/planarian-live.git ~/planarian-live`.
6. `bash ~/planarian-research/cloud/setup.sh`  (installs deps + registers services)
7. (continuity) scp the Mac's worm_run_01_{tracks,behavior}.csv + _processed.txt
   into ~/planarian-research/realtime_runs so the session continues seamlessly.

## Gotchas that bit us (do not re-learn)
- **imgsz must be 1024.** At 640 the worm (tiny in a 7MP frame) is mostly lost
  (~73% vs ~30% at 1024); behavior macro-F1 collapsed 0.84 -> 0.50. No tuning down.
- **2 cores is too slow** (~123s/clip). 4 cores @ stride 3 = ~50s. RAM matters too:
  4 GB OOMs the background builder (np.median of 50x 7MP frames). 8 GB is fine.
- **torch + torchvision must come from the SAME cpu index**, else
  `operator torchvision::nms does not exist` at inference.
- **systemd sets no HOME** -> rclone/git/ultralytics fail and drive_pull.sh (set -u)
  errors on $HOME. Every unit needs `Environment=HOME=/root`.
- **The plot must be cheap + low priority.** Rendering the full ~430k-point session
  (2.2 GB, ~70s CPU) starves the watcher. Fix: cap to last 3h (PLOT_HISTORY_S) and
  run the plot service at Nice=19. Render ~5 min.
- **Drive upload race:** the recording Mac is still UPLOADING the newest clip when
  rclone pulls -> 0-byte / half-written files ("corrupted on transfer: sizes
  differ") that jam the watcher. Fix: `rclone move --min-age 2m` (only pull clips
  settled >2 min, i.e. fully uploaded). Costs ~2 min latency. A clip that gets
  permanently corrupted in Drive must be deleted or rclone retries it every cycle.
- **A growing backlog is self-defeating** on a small box: the watcher re-checks
  every pending clip each loop, so a big queue adds overhead and never drains. Keep
  it caught up; if a backlog builds, archive the old clips (~/clips_archive) and
  let the box run on fresh clips only.
- The macOS File-Provider gates (materialized/prefetch) in rt_watch are harmless on
  Linux IF clips arrive complete (the --min-age fix ensures that); a 0-byte file
  makes the watcher loop on prefetch forever.
- Behavior is served at **stride 4** but the model was trained on all-frame signals
  -> rougher predictions. Follow-up: retrain on stride-4 signals for train/serve match.
