# batch_worm_tracker_chain.py
# Python 3.9+. Requires: opencv-python, numpy, tqdm
# Optional (for Parquet): pyarrow or fastparquet
#
# 1) Seed once (interactive on a representative first video):
#    python batch_worm_tracker_chain.py --seed --video "../planarian_social_interactions/2025-10-14 03-00-15.mkv" --calibration ./calibration.json
# 2) Batch (headless), chained across sorted files:
#    python batch_worm_tracker_chain.py --videos_dir "../planarian_social_interactions" --calibration ./calibration.json --no_overlay --export_parquet_wide
#
# Options: --glob "*.mkv"  --roi_px 120  --delta_gate 0.05  --darkness_thresh 0.15
# New:     --export_parquet_wide  (writes <video>_pixelvec.parquet, wide)
#          --save_mask_npz        (writes <video>_wormmask.npz, packed bits)

import os, json, math, argparse, glob
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import csv

# Optional: for Parquet export (only needed if you pass --export_parquet_wide)
try:
    import pandas as pd
    _PANDAS_OK = True
except Exception:
    _PANDAS_OK = False

# ---------- Utils ----------
def to_gray01(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

def relative_darkness(gray):
    lo, hi = np.percentile(gray, 2.0), np.percentile(gray, 98.0)
    if hi <= lo:
        hi, lo = float(gray.max()), float(gray.min())
    norm = np.clip((gray - lo) / max(1e-6, (hi - lo)), 0, 1)
    return 1.0 - norm

def circle_mask(shape_hw, center_xy, radius_px):
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    if center_xy and radius_px and radius_px > 0:
        cv2.circle(mask, (int(center_xy[0]), int(center_xy[1])), int(radius_px), 255, -1)
    return mask

def largest_component(mask):
    num, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1: return None, None, None
    comp = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    area = int(stats[comp, cv2.CC_STAT_AREA])
    cx, cy = cents[comp]
    cmask = (labels == comp).astype(np.uint8) * 255
    return (float(cx), float(cy)), area, cmask

def template_from(gray_like, center, half=32):
    h, w = gray_like.shape
    cx, cy = int(round(center[0])), int(round(center[1]))
    x1, x2 = max(0, cx-half), min(w, cx+half)
    y1, y2 = max(0, cy-half), min(h, cy+half)
    patch = gray_like[y1:y2, x1:x2].copy()
    if patch.size == 0: return None
    m, s = patch.mean(), patch.std() + 1e-6
    return (patch - m) / s  # Z-scored for NCC

def ncc_match(map_img, templ):
    """TM_CCOEFF_NORMED, safe: require ROI >= template in both dims."""
    if templ is None or templ.size == 0:
        return None, None, None
    th, tw = templ.shape[:2]
    mh, mw = map_img.shape[:2]
    if mh < th or mw < tw:
        return None, None, None
    t = templ.astype(np.float32)
    m = cv2.matchTemplate(map_img.astype(np.float32), t, cv2.TM_CCOEFF_NORMED)
    _, maxv, _, maxloc = cv2.minMaxLoc(m)
    cx = maxloc[0] + tw/2.0
    cy = maxloc[1] + th/2.0
    return (cx, cy), maxv, m

def build_static_background_darkness(cap, sample_frames=300, stride=3, dish_mask_bool=None):
    frames = []
    pos0 = cap.get(cv2.CAP_PROP_POS_FRAMES)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    n_to_sample = min(sample_frames, total) if total > 0 else sample_frames
    for i in range(n_to_sample):
        ok, frame = cap.read()
        if not ok: break
        if i % stride == 0:
            g = to_gray01(frame)
            d = relative_darkness(g)
            if dish_mask_bool is not None:
                d = d * dish_mask_bool
            frames.append(d)
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos0)
    if not frames:
        raise RuntimeError("No frames to build background.")
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.float32)

# ---------- Interactive helpers (seeding only) ----------
class ClickCollector:
    def __init__(self, win, img):
        self.win, self.img, self.base = win, img.copy(), img.copy()
        self.pts = []
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win, self.on_mouse)
    def on_mouse(self, ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            self.pts.append((x, y))
            cv2.circle(self.img, (x, y), 4, (0,255,0), -1)
            cv2.imshow(self.win, self.img)
        elif ev == cv2.EVENT_RBUTTONDOWN and self.pts:
            self.pts.pop()
            self.img = self.base.copy()
            for (px, py) in self.pts:
                cv2.circle(self.img, (px, py), 4, (0,255,0), -1)
            cv2.imshow(self.win, self.img)
    def get(self, prompt, min_pts=1):
        overlay = self.img.copy()
        cv2.putText(overlay, prompt, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30,220,30), 2)
        self.img = overlay
        cv2.imshow(self.win, self.img)
        while True:
            k = cv2.waitKey(33) & 0xFF
            if k in (13, 32):
                if len(self.pts) >= min_pts: break
            elif k == 27:
                self.pts = []; break
        cv2.destroyWindow(self.win)
        return self.pts

def seed_first_video(video_path, save_json):
    cap = cv2.VideoCapture(str(video_path)); ok, frame0 = cap.read(); cap.release()
    if not ok: raise RuntimeError("Cannot read first frame.")
    H, W = frame0.shape[:2]

    # Dish
    cc = ClickCollector("Seed: Dish", frame0)
    pts = cc.get("Dish: click CENTER then RIM. Enter to accept.", 2)
    if len(pts) < 2: raise RuntimeError("Dish not tagged.")
    (cx, cy), (rx, ry) = pts[0], pts[1]; radius = float(math.dist((cx, cy), (rx, ry)))
    dish_center = [float(cx), float(cy)]; dish_radius = float(radius)

    # 1 cm
    cc2 = ClickCollector("Seed: 1 cm", frame0)
    spts = cc2.get("Scale: click START and END of 1 cm line. Enter to accept.", 2)
    if len(spts) < 2: raise RuntimeError("Scale not tagged.")
    (sx, sy), (ex, ey) = spts[0], spts[1]
    mm_per_px = 10.0 / float(math.dist((sx, sy), (ex, ey)))

    # Worm polygon → centroid + template
    cc3 = ClickCollector("Seed: Worm", frame0)
    wpoly = cc3.get("Worm: outline with left-clicks. Enter to accept.", 3)
    if len(wpoly) < 3: raise RuntimeError("Worm not tagged.")
    mask = np.zeros((H, W), np.uint8); cv2.fillPoly(mask, [np.array(wpoly, np.int32)], 255)
    _, _, comp_mask = largest_component(mask)
    if comp_mask is None: comp_mask = mask
    num, labels, stats, cents = cv2.connectedComponentsWithStats(comp_mask, 8)
    if num <= 1: raise RuntimeError("Could not get worm component.")
    centroid = [float(cents[1][0]), float(cents[1][1])]
    g0 = to_gray01(frame0); dark0 = relative_darkness(g0)
    templ = template_from(dark0, centroid, half=32)
    if templ is None: raise RuntimeError("Failed to build template.")

    calib = dict(
        dish_center=dish_center,
        dish_radius_px=dish_radius,
        mm_per_px=float(mm_per_px),
        worm_seed_centroid=centroid,
        worm_template=templ.tolist()
    )
    with open(save_json, "w") as f: json.dump(calib, f, indent=2)
    print(f"Saved seed+calibration → {save_json}")
    return calib

# ---------- Per-video chained processing ----------
def process_video_chained(
    video_path, calib, state,
    darkness_thresh=0.15, delta_gate=0.05,
    min_area_px=50, morph_open=3, morph_close=5,
    max_jump_px=200, roi_px=120, write_overlay=False, fps_out=None,
    save_mask_npz=False, export_parquet_wide=False
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise RuntimeError(f"Could not open {video_path}")
    ok, frame0 = cap.read()
    if not ok: raise RuntimeError("Empty video.")
    H, W = frame0.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps_out is None: fps_out = fps

    # Dish + scale
    dish_center = tuple(calib["dish_center"]); dish_radius = float(calib["dish_radius_px"])
    mm_per_px = float(calib["mm_per_px"])
    dishMask = circle_mask((H, W), dish_center, dish_radius); dishMask_bool = (dishMask > 0)

    # Static background
    bg_dark = build_static_background_darkness(cap, sample_frames=300, stride=3, dish_mask_bool=dishMask_bool)

    # Overlay writer next to the video file
    out_prefix = str(Path(video_path).with_suffix('').as_posix())
    overlay_path = f"{out_prefix}_overlay.mp4"
    writer = None
    if write_overlay:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(overlay_path, fourcc, fps_out, (W, H), True)

    # --------- Buffer CSV rows ----------
    rows = []
    rows.append([f"# mm_per_px={mm_per_px:.8f}"])
    rows.append([
        "frame","time_s","centroid_x","centroid_y","centroid_x_mm","centroid_y_mm",
        "area_px","mean_darkness","confidence","speed_px_s","speed_mm_s","heading_deg"
    ])

    # Collect per-frame masks to export later (optional)
    collect_masks = save_mask_npz or export_parquet_wide
    masks = [] if collect_masks else None

    # State
    last_centroid = state.get("centroid", calib.get("worm_seed_centroid"))
    templ = state.get("template", np.array(calib.get("worm_template"), dtype=np.float32))
    if isinstance(templ, list): templ = np.array(templ, dtype=np.float32)

    # Rewind and iterate
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc=f"Processing {Path(video_path).name}", unit="frame")

    frame_idx = 0
    last_time = 0.0
    while True:
        ok, frame = cap.read()
        if not ok: break
        t = frame_idx / fps

        gray = to_gray01(frame)
        dark = relative_darkness(gray)

        dish_dark = dark[dishMask_bool]
        dish_med = np.median(dish_dark) if dish_dark.size else 0.0

        base_dark = (dark >= (dish_med + darkness_thresh)).astype(np.uint8)
        delta = (dark - bg_dark)
        moving = (delta >= delta_gate).astype(np.uint8)
        cand_global = (base_dark & moving & (dishMask > 0)).astype(np.uint8) * 255

        # ROI-first search
        use_roi = last_centroid is not None
        centroid = None; area = 0; comp_mask = None
        if use_roi:
            cx, cy = last_centroid
            x1 = int(max(0, cx - roi_px)); x2 = int(min(W, cx + roi_px))
            y1 = int(max(0, cy - roi_px)); y2 = int(min(H, cy + roi_px))
            cand_roi = np.zeros_like(cand_global)
            cand_roi[y1:y2, x1:x2] = cand_global[y1:y2, x1:x2]
            centroid, area, comp_mask = largest_component(cand_roi)

            # NCC on positive delta map
            if templ is not None and templ.size > 0:
                delta_pos = np.clip(delta, 0, 1)
                roi_map = delta_pos[y1:y2, x1:x2]
                ncc_c, ncc_v, _ = ncc_match(roi_map, templ)
                if ncc_c is not None:
                    ncc_c = (ncc_c[0] + x1, ncc_c[1] + y1)
                    if centroid is None or area < min_area_px or ncc_v > 0.4:
                        centroid = ncc_c
                        area = max(area, min_area_px)
                        comp_mask = None

        # Fallback: global
        if centroid is None:
            centroid, area, comp_mask = largest_component(cand_global)

        # Build a full-frame binary mask for this frame
        worm_mask_full = np.zeros((H, W), dtype=np.uint8)
        if centroid is not None and area >= min_area_px:
            if comp_mask is not None:
                worm_mask_full = (comp_mask > 0).astype(np.uint8)
            else:
                cx, cy = int(round(centroid[0])), int(round(centroid[1]))
                cv2.circle(worm_mask_full, (cx, cy), 6, 1, -1)

        # Accumulate outputs (no immediate writes)
        if centroid is not None and area >= min_area_px:
            cx, cy = centroid
            plausible = True
            if last_centroid is not None and max_jump_px is not None:
                if math.dist((cx, cy), last_centroid) > max_jump_px:
                    plausible = False

            if plausible:
                comp_bool = worm_mask_full.astype(bool)
                if comp_bool.any():
                    comp_dark_mean = float(dark[comp_bool].mean())
                    comp_delta_mean = float(delta[comp_bool].mean())
                else:
                    comp_dark_mean = float(dark[int(cy), int(cx)])
                    comp_delta_mean = float(delta[int(cy), int(cx)])
                confidence = float(np.clip(0.5 * (comp_dark_mean - dish_med) + 0.5 * comp_delta_mean, 0, 1))

                if last_centroid is not None:
                    dt = max(1e-6, t - last_time)
                    dx, dy = (cx - last_centroid[0]), (cy - last_centroid[1])
                    step_px = float(np.hypot(dx, dy))
                    speed_px_s = step_px / dt
                    speed_mm_s = speed_px_s * mm_per_px
                    heading_deg = (np.degrees(np.arctan2(-dy, dx)) + 360.0) % 360.0
                else:
                    speed_px_s = speed_mm_s = heading_deg = ""

                rows.append([
                    frame_idx, round(t,4), cx, cy, cx*mm_per_px, cy*mm_per_px,
                    int(worm_mask_full.sum()), round(comp_dark_mean,4), round(confidence,4),
                    (round(speed_px_s,4) if speed_px_s != "" else ""),
                    (round(speed_mm_s,4) if speed_mm_s != "" else ""),
                    (round(heading_deg,2) if heading_deg != "" else "")
                ])

                if collect_masks:
                    masks.append(worm_mask_full)

                # Update state
                last_centroid = (cx, cy)
                last_time = t

                # Update template occasionally
                if frame_idx % 60 == 0:
                    new_templ = template_from(np.clip(delta, 0, 1), last_centroid, half=32)
                    if new_templ is not None:
                        templ = new_templ

                if writer:
                    vis = frame.copy()
                    cv2.circle(vis, (int(dish_center[0]), int(dish_center[1])), int(dish_radius), (255,200,0), 2)

                    # ---- Red "mesh" (contours) + soft fill ----
                    if worm_mask_full.any():
                        contours, _ = cv2.findContours(worm_mask_full*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        cv2.drawContours(vis, contours, -1, (0,0,255), 1)
                        overlay = vis.copy()
                        cv2.drawContours(overlay, contours, -1, (0,0,255), thickness=cv2.FILLED)
                        vis = cv2.addWeighted(overlay, 0.15, vis, 0.85, 0)

                    cv2.circle(vis, (int(cx), int(cy)), 4, (0,0,255), -1)
                    lab = f"t={t:.2f}s"
                    if speed_px_s != "":
                        lab += f" v={speed_px_s:.1f}px/s ({speed_mm_s:.2f} mm/s)"
                    cv2.putText(vis, lab, (15,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    writer.write(vis)
            else:
                rows.append([frame_idx, round(t,4), "", "", "", "", 0, 0.0, 0.0, "", "", ""])
                if collect_masks: masks.append(np.zeros((H,W), np.uint8))
                if writer:
                    vis = frame.copy()
                    cv2.putText(vis, f"t={t:.2f}s (skip: jump)", (15,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40,40,255), 2)
                    writer.write(vis)
        else:
            rows.append([frame_idx, round(t,4), "", "", "", "", 0, 0.0, 0.0, "", "", ""])
            if collect_masks: masks.append(np.zeros((H,W), np.uint8))
            if writer:
                vis = frame.copy()
                cv2.putText(vis, f"t={t:.2f}s (no component)", (15,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40,40,255), 2)
                writer.write(vis)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if writer: writer.release()

    # --------- Write CSV once to CURRENT WORKING DIR ----------
    base = Path(video_path).with_suffix('').name
    csv_path = Path.cwd() / f"{base}_tracks.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)
    print(f"Wrote {csv_path}")

    # --------- Save masks (NPZ and/or Parquet wide) ----------
    if collect_masks and len(masks) > 0:
        masks_bool = np.stack(masks, axis=0).astype(bool)  # [T, H, W]
        T = masks_bool.shape[0]

        if save_mask_npz:
            packed = np.packbits(masks_bool.reshape(T, -1), axis=1)   # [T, ceil(H*W/8)]
            npz_path = Path.cwd() / f"{base}_wormmask.npz"
            np.savez_compressed(npz_path, packed=packed, meta=np.array([H, W, T], dtype=np.int32))
            print(f"Wrote {npz_path} (packed bits; H={H}, W={W}, T={T})")

        if export_parquet_wide:
            if not _PANDAS_OK:
                raise RuntimeError("pandas is required for --export_parquet_wide. Install pandas + (pyarrow|fastparquet).")
            flat = masks_bool.reshape(T, -1).astype(np.uint8)   # [T, H*W]
            # Build DataFrame: VERY WIDE (one column per pixel)
            df = pd.DataFrame(flat)
            df.insert(0, "frame", range(T))
            # Optionally label pixel columns p0..pN (skip if extremely large to save time)
            # df.columns = ["frame"] + [f"p{j}" for j in range(H*W)]
            pq_path = Path.cwd() / f"{base}_pixelvec.parquet"
            # Requires: pip install pyarrow  (or)  pip install fastparquet
            df.to_parquet(pq_path, index=False)
            print(f"Wrote {pq_path} (wide; shape={df.shape})")

    # return updated chaining state
    return {"centroid": last_centroid, "template": templ}

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Chained planarian tracker: seed once, then process all videos headlessly.")
    ap.add_argument("--seed", action="store_true", help="Run interactive seeding on --video and save calibration+seed JSON.")
    ap.add_argument("--video", type=str, help="Video used for seeding (first frame).")
    ap.add_argument("--calibration", type=str, default="./calibration.json", help="Path to calibration JSON.")
    ap.add_argument("--videos_dir", type=str, default="../planarian_social_interactions", help="Folder of videos to process.")
    ap.add_argument("--glob", type=str, default="*.mkv", help="Filename glob (e.g., *.mp4).")
    ap.add_argument("--no_overlay", action="store_true")
    ap.add_argument("--roi_px", type=int, default=120, help="ROI radius around previous centroid.")
    ap.add_argument("--darkness_thresh", type=float, default=0.15)
    ap.add_argument("--delta_gate", type=float, default=0.05)
    ap.add_argument("--min_area_px", type=int, default=50)
    ap.add_argument("--morph_open", type=int, default=3)
    ap.add_argument("--morph_close", type=int, default=5)
    ap.add_argument("--max_jump_px", type=float, default=200)
    ap.add_argument("--save_mask_npz", action="store_true", help="Save per-frame worm masks as packed NPZ (compact).")
    ap.add_argument("--export_parquet_wide", action="store_true", help="Export per-frame worm mask as wide Parquet.")
    args = ap.parse_args()

    if args.seed:
        if not args.video:
            raise SystemExit("Provide --video with --seed.")
        seed_first_video(args.video, args.calibration)
        return

    if not os.path.exists(args.calibration):
        raise SystemExit(f"Missing {args.calibration}. Run with --seed first.")
    with open(args.calibration, "r") as f:
        calib = json.load(f)

    files = sorted(glob.glob(os.path.join(args.videos_dir, args.glob)))
    if not files:
        raise SystemExit(f"No videos found in {args.videos_dir} with pattern {args.glob}")

    # Initialize state from calibration
    state = {
        "centroid": calib.get("worm_seed_centroid"),
        "template": np.array(calib.get("worm_template"), dtype=np.float32)
    }

    for vp in files:
        print(f"\n=== {vp} ===")
        state = process_video_chained(
            video_path=vp,
            calib=calib,
            state=state,
            darkness_thresh=args.darkness_thresh,
            delta_gate=args.delta_gate,
            min_area_px=args.min_area_px,
            morph_open=args.morph_open,
            morph_close=args.morph_close,
            max_jump_px=args.max_jump_px,
            roi_px=args.roi_px,
            write_overlay=not args.no_overlay,
            fps_out=None,
            save_mask_npz=args.save_mask_npz,
            export_parquet_wide=args.export_parquet_wide
        )

if __name__ == "__main__":
    main()