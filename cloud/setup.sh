#!/bin/bash
# One-shot setup for the planarian compute node (Hetzner CX33: Ubuntu 24.04, x86,
# 4 vCPU / 8 GB). Run as root after: git clone + scp'ing the model weights +
# rclone config. Installs deps, builds the venv, and registers systemd services
# that auto-start on boot. See DEPLOY.md for the full runbook and gotchas.
#
# Sizing note (measured): the full pipeline (location+behavior, unified single
# YOLO pass) at imgsz 1024 / stride 4 is ~44s/clip on 4 cores (16s margin under
# the 60s budget; stride 3 was ~50s but too tight once the plot + drive_pull
# contend). A 2-core box (~123s) and imgsz 640 (detection collapses) do NOT work.
#
# Assumes:
#   ~/planarian-research            <- git clone (code; note core modules live on dev)
#   .../runs/pose/worm_white7mp_n/weights/best.pt   <- scp'd (gitignored)
#   ~/planarian-research/realtime_runs/behavior_clf.joblib  <- scp'd (gitignored)
#   rclone remote "drive" configured (see DEPLOY.md step 3)
set -euo pipefail
REPO="$HOME/planarian-research"; SN="$REPO/scripts_notebooks"; PY="$REPO/venv/bin/python"

echo "== system deps =="
apt-get update -y
apt-get install -y python3-venv python3-pip git rclone ffmpeg libgl1 libglib2.0-0

echo "== python venv (CPU torch, x86) =="
python3 -m venv "$REPO/venv"
"$REPO/venv/bin/pip" install --upgrade pip wheel
# Install torch AND torchvision from the SAME cpu index so the torchvision::nms
# operator links (Ultralytics pulls a PyPI torchvision otherwise -> NMS error).
"$REPO/venv/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cpu
"$REPO/venv/bin/pip" install ultralytics opencv-python-headless av numpy scipy \
    scikit-image scikit-learn joblib matplotlib tqdm pandas

mkdir -p "$HOME/clips_in" "$REPO/realtime_runs/label_queue"

echo "== systemd services =="
SVC=/etc/systemd/system
# NOTE: systemd does not set HOME -> rclone/git/ultralytics break (and drive_pull.sh
# uses set -u). Every unit must set Environment=HOME=/root.

cat > $SVC/planarian-drivepull.service <<EOF
[Unit]
Description=Planarian: pull clips from Drive (rclone)
After=network-online.target
Wants=network-online.target
[Service]
Environment=HOME=/root
Environment=WATCH_DIR=$HOME/clips_in
ExecStart=/bin/bash $REPO/cloud/drive_pull.sh
Restart=always
RestartSec=10
[Install]
WantedBy=multi-user.target
EOF

cat > $SVC/planarian-watch.service <<EOF
[Unit]
Description=Planarian: track + behavior (rt_watch, unified single pass)
After=planarian-drivepull.service
[Service]
Environment=HOME=/root
WorkingDirectory=$SN
ExecStart=$PY -u rt_watch.py --watch_dir $HOME/clips_in --session_id worm_run_01 --out_dir $REPO/realtime_runs --device cpu --imgsz 1024 --stride 4 --stable_s 2 --min_frames 560 --delete_after --behavior_model $REPO/realtime_runs/behavior_clf.joblib --label_queue $REPO/realtime_runs/label_queue --keep_per_day 150
Restart=always
RestartSec=10
[Install]
WantedBy=multi-user.target
EOF

# Plot runs at low priority (Nice 19) so it can NEVER starve the watcher on a
# small box; it renders only the last 3h (PLOT_HISTORY_S) every 5 min.
cat > $SVC/planarian-plot.service <<EOF
[Unit]
Description=Planarian: live plot loop (low priority, 3h window)
After=planarian-watch.service
[Service]
Environment=HOME=/root
WorkingDirectory=$SN
Nice=19
CPUWeight=10
ExecStart=$PY rt_plot.py --session worm_run_01 --loop 300
Restart=always
RestartSec=15
[Install]
WantedBy=multi-user.target
EOF

cat > $SVC/planarian-publish.service <<EOF
[Unit]
Description=Planarian: publish plot to GitHub Pages
After=planarian-plot.service
[Service]
Environment=HOME=/root
WorkingDirectory=$REPO/cloud
ExecStart=/bin/bash $REPO/cloud/publish_live.sh
Restart=always
RestartSec=30
[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now planarian-drivepull planarian-watch planarian-plot planarian-publish
echo "== done. tail logs: journalctl -u planarian-watch -f =="
