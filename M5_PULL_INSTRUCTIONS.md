# Pulling the planarian data onto the M5 Max

All data is verified on Google Drive (account: cox.david.j@gmail.com) under `PlanarianVideos/`.
Code lives on GitHub (`dev` branch).

## 1. Get the code
```
git clone https://github.com/david-j-cox/planarian-research.git
cd planarian-research && git checkout dev
```

## 2. Connect rclone to the same Drive (one-time, opens browser)
```
brew install rclone
rclone config create gdrive drive scope drive
# Log in as cox.david.j@gmail.com and approve.
```

## 3. Pull the data

### Tier 1 — labeled training set (~625 MB) — pull this first
```
rclone copy "gdrive:PlanarianVideos/tier1_training" ./tier1_training \
  --transfers 8 --drive-chunk-size 64M --progress
```
Contains: 15 labeled videos (videos/), label+calibration files (labels/), MIGRATION_MANIFEST.md.

### Tier 2 — full corpus (~39 GB)
```
rclone copy "gdrive:PlanarianVideos/tier2_corpus" ./tier2_corpus \
  --transfers 8 --drive-chunk-size 128M --progress
```
Contains: OpenDishWork/ (692 files), additional_videos/ (297), live_capture/ (24 unlabeled),
planarian_social_interactions/ (9).

## 4. Verify the pull (optional but recommended)
```
rclone check ./tier1_training "gdrive:PlanarianVideos/tier1_training"
rclone check ./tier2_corpus  "gdrive:PlanarianVideos/tier2_corpus"
```
Both should report "0 differences found".

## Notes
- Tier 1 alone is enough to start finetuning; Tier 2 is unlabeled corpus for expanding the dataset.
- On the M5: install PyTorch with MPS backend + Ultralytics (YOLO). Avoid Detectron2 (poor Apple Silicon support).
- First training step: extract labeled frames from the 15 tier1 videos and convert the point
  labels in S1/S3_labels.json to YOLO boxes/keypoints. See MIGRATION_MANIFEST.md.
