#!/usr/bin/env python3
"""
build_yolo_dataset.py

Build a YOLO pose dataset (worm box + head/tail keypoints) from the classical
tracker's dense per-frame output (realtime_runs/S3_signals.npz) and the source
videos. Pseudo-labels were validated against the 73 hand-marked GT points
(median ~0.5 mm error, 88% within 1 mm).

Label geometry per frame:
  - box  : bounding box of the 20-point midline, padded for body thickness
  - kpt0 : head  = midline[0]
  - kpt1 : tail  = midline[-1]

Frames are subsampled by centroid displacement to avoid near-duplicate poses,
and the train/val split is by whole video so val is on unseen footage.

Usage:
  python build_yolo_dataset.py \
      --signals ../realtime_runs/S3_signals.npz \
      --videos_dir ../live_capture \
      --out ../realtime_runs/yolo_dataset \
      --val_videos 2026-06-02_15-05-45 2026-06-02_15-07-26
"""
import argparse
import os
import shutil
import numpy as np
import cv2
from tqdm import tqdm


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--signals", default=os.path.join(here, "..", "realtime_runs", "S3_signals.npz"))
    ap.add_argument("--videos_dir", default=os.path.join(here, "..", "live_capture"))
    ap.add_argument("--out", default=os.path.join(here, "..", "realtime_runs", "yolo_dataset"))
    ap.add_argument("--val_videos", nargs="*", default=["2026-06-02_15-05-45", "2026-06-02_15-07-26"],
                    help="Video stems (without .mkv) held out for validation.")
    ap.add_argument("--min_disp_px", type=float, default=10.0,
                    help="Keep a frame only if centroid moved >= this since the last kept frame.")
    ap.add_argument("--max_gap_s", type=float, default=1.5,
                    help="Force-keep a frame if this many seconds passed since the last kept one.")
    ap.add_argument("--pad_frac", type=float, default=0.14,
                    help="Box padding on each side as a fraction of body_len_px (covers body thickness).")
    ap.add_argument("--pad_min_px", type=float, default=12.0, help="Minimum box padding per side in px.")
    ap.add_argument("--jpg_quality", type=int, default=92)
    ap.add_argument("--max_per_video", type=int, default=600, help="Safety cap on kept frames per video.")
    return ap.parse_args()


def main():
    a = parse_args()
    z = np.load(a.signals, allow_pickle=True)
    video = z["video"]
    nframe = z["native_frame"].astype(int)
    cx, cy = z["cx_px"], z["cy_px"]
    mid = z["midline"]            # (N, 20, 2)
    blen = z["body_len_px"]
    lost = z["lost"]
    fps = float(z["fps"])
    mmpp = float(z["mm_per_px"])

    finite = np.isfinite(cx) & np.isfinite(cy) & np.isfinite(blen) & np.isfinite(mid).all(axis=(1, 2))
    usable = (lost == 0) & finite
    print(f"usable frames: {int(usable.sum())} / {len(cx)}  (fps={fps}, mm_per_px={mmpp:.5f})")

    # Output layout
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)

    val_set = set(a.val_videos)
    counts = {"train": 0, "val": 0}
    skipped_oob = 0

    vids = [v for v in sorted(set(video.tolist()))]
    for vid in vids:
        stem = os.path.splitext(vid)[0]
        split = "val" if stem in val_set else "train"
        vpath = os.path.join(a.videos_dir, vid)
        if not os.path.exists(vpath):
            print(f"  WARN missing video, skipping: {vpath}")
            continue

        # Frames to keep for this video: subsample by displacement / time gap.
        sel = np.where(usable & (video == vid))[0]
        sel = sel[np.argsort(nframe[sel])]
        keep_idx = []
        last_xy = None
        last_t = None
        for i in sel:
            t = nframe[i] / fps
            move = np.inf if last_xy is None else float(np.hypot(cx[i] - last_xy[0], cy[i] - last_xy[1]))
            gap = np.inf if last_t is None else (t - last_t)
            if move >= a.min_disp_px or gap >= a.max_gap_s:
                keep_idx.append(i)
                last_xy = (cx[i], cy[i])
                last_t = t
            if len(keep_idx) >= a.max_per_video:
                break
        want = {int(nframe[i]): i for i in keep_idx}
        if not want:
            continue

        cap = cv2.VideoCapture(vpath)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fno = 0
        got = 0
        pbar = tqdm(total=len(want), desc=f"{stem}[{split}]", leave=False)
        while got < len(want):
            ok, frame = cap.read()
            if not ok:
                break
            if fno in want:
                i = want[fno]
                m = mid[i]
                x0, y0 = m[:, 0].min(), m[:, 1].min()
                x1, y1 = m[:, 0].max(), m[:, 1].max()
                pad = max(a.pad_frac * float(blen[i]), a.pad_min_px)
                bx0, by0 = max(0.0, x0 - pad), max(0.0, y0 - pad)
                bx1, by1 = min(W - 1.0, x1 + pad), min(H - 1.0, y1 + pad)
                bw, bh = bx1 - bx0, by1 - by0
                if bw <= 1 or bh <= 1:
                    skipped_oob += 1
                    fno += 1
                    continue
                # normalized YOLO pose line: cls xc yc w h  hx hy hv  tx ty tv
                xc = (bx0 + bx1) / 2.0 / W
                yc = (by0 + by1) / 2.0 / H
                wn, hn = bw / W, bh / H
                hx, hy = float(m[0, 0]) / W, float(m[0, 1]) / H
                tx, ty = float(m[-1, 0]) / W, float(m[-1, 1]) / H
                label = f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f} {hx:.6f} {hy:.6f} 2 {tx:.6f} {ty:.6f} 2\n"

                base = f"{stem}_f{fno:06d}"
                cv2.imwrite(os.path.join(a.out, f"images/{split}", base + ".jpg"),
                            frame, [cv2.IMWRITE_JPEG_QUALITY, a.jpg_quality])
                with open(os.path.join(a.out, f"labels/{split}", base + ".txt"), "w") as f:
                    f.write(label)
                counts[split] += 1
                got += 1
                pbar.update(1)
            fno += 1
        pbar.close()
        cap.release()
        print(f"  {stem}[{split}]: kept {got}/{len(want)} frames")

    # data.yaml
    yaml_path = os.path.join(a.out, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(
            "# Auto-generated by build_yolo_dataset.py\n"
            f"path: {os.path.abspath(a.out)}\n"
            "train: images/train\n"
            "val: images/val\n"
            "kpt_shape: [2, 3]   # head, tail; (x, y, visibility)\n"
            "flip_idx: [0, 1]    # head/tail not mirror-paired; identity\n"
            "names:\n"
            "  0: worm\n"
        )
    print(f"\nDONE  train={counts['train']}  val={counts['val']}  skipped_oob={skipped_oob}")
    print(f"data.yaml -> {yaml_path}")


if __name__ == "__main__":
    main()
