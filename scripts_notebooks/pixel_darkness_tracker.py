# pixel_darkness_tracker.py
# Requires: Python 3.9+, opencv-python, numpy, tqdm
# Usage (single file):
#   python pixel_darkness_tracker.py --video "../planarian_social_interactions/your_video.mkv"
# Batch (all .mkv in default folder):
#   python pixel_darkness_tracker.py
# Optional explicit scale if you skip clicks:
#   --mm_per_px 0.05  OR  --px_per_mm 20

import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import csv
import math
import argparse

# ----------------------------
# Utility: normalization, masks
# ----------------------------
def to_gray_float01(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return g

def relative_darkness(gray):
    # 0 (bright) .. 1 (dark); percentile stretch for robustness
    lo = np.percentile(gray, 2.0)
    hi = np.percentile(gray, 98.0)
    if hi <= lo:
        hi, lo = float(gray.max()), float(gray.min())
    norm = np.clip((gray - lo) / max(1e-6, (hi - lo)), 0, 1)
    return 1.0 - norm

def polygon_mask(shape, pts):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if len(pts) >= 3:
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    return mask

def circle_mask(shape, center, radius):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if center is not None and radius is not None and radius > 0:
        cv2.circle(mask, tuple(map(int, center)), int(radius), 255, thickness=-1)
    return mask

def largest_component(mask):
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return None, None, None
    comp = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])   # skip bg
    area = int(stats[comp, cv2.CC_STAT_AREA])
    cx, cy = centroids[comp]
    cc_mask = (labels == comp).astype(np.uint8) * 255
    return (float(cx), float(cy)), area, cc_mask

# ----------------------------
# Interactive tagging
# ----------------------------
class ClickCollector:
    def __init__(self, win, img):
        self.win = win
        self.img = img.copy()
        self.base = img.copy()
        self.points = []
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            cv2.circle(self.img, (x, y), 4, (0, 255, 0), -1)
            cv2.imshow(self.win, self.img)
        elif event == cv2.EVENT_RBUTTONDOWN and self.points:
            self.points.pop()
            self.img = self.base.copy()
            for (px, py) in self.points:
                cv2.circle(self.img, (px, py), 4, (0,255,0), -1)
            cv2.imshow(self.win, self.img)

    def get_points(self, prompt, min_points=1):
        overlay = self.img.copy()
        cv2.putText(overlay, prompt, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 220, 30), 2)
        self.img = overlay
        cv2.imshow(self.win, self.img)
        while True:
            k = cv2.waitKey(33) & 0xFF
            if k in (13, 32):  # Enter/Space
                if len(self.points) >= min_points:
                    break
            elif k == 27:      # ESC = cancel/skip
                self.points = []
                break
        cv2.destroyWindow(self.win)
        return self.points

def estimate_circle_from_two_points(center_pt, rim_pt):
    cx, cy = center_pt
    rx, ry = rim_pt
    r = math.dist((cx, cy), (rx, ry))
    return (cx, cy), r

# ----------------------------
# Core tracking
# ----------------------------
def run_tracker(
    video_path,
    out_prefix=None,
    ema_alpha=0.02,           # background EMA speed
    darkness_thresh=0.15,     # base darkness gate (0..1)
    delta_gate=0.05,          # min delta change to treat as "movement"
    k_profile=0.5,            # tightness of worm darkness profile gate
    min_area_px=50,           # ignore tiny specks
    morph_open=3,             # kernel for opening
    morph_close=5,            # kernel for closing
    max_jump_px=200,          # reject implausible teleports (set None to disable)
    write_overlay=True,
    fps_out=None,
    mm_per_px_cli=None,       # optional fallback scale from CLI
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # First frame
    ret, frame0 = cap.read()
    if not ret:
        raise RuntimeError("Empty video.")
    H, W = frame0.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps_out is None:
        fps_out = fps

    # --- Tag dish (optional) ---
    dish_clicks = ClickCollector("Tag Dish", frame0).get_points(
        "Dish: Left-click CENTER, then a RIM point. Enter to accept. ESC to skip.",
        min_points=2
    )
    dish_center, dish_radius = (None, None)
    if len(dish_clicks) >= 2:
        dish_center, dish_radius = estimate_circle_from_two_points(dish_clicks[0], dish_clicks[1])
    dishMask = circle_mask(frame0.shape, dish_center, dish_radius) if dish_center else np.ones((H, W), np.uint8)*255
    dishMask_bool = (dishMask > 0)

    # --- Tag 1 cm scale (NEW) ---
    scale_clicks = ClickCollector("Tag Scale", frame0).get_points(
        "Scale: Left-click START and END of the 1 cm line. Enter to accept. ESC to skip.",
        min_points=2
    )
    mm_per_px = None
    scale_seg = None
    if len(scale_clicks) >= 2:
        (x1, y1), (x2, y2) = scale_clicks[0], scale_clicks[1]
        px_dist = math.dist((x1, y1), (x2, y2))
        if px_dist > 0:
            mm_per_px = 10.0 / px_dist  # 10 mm in 1 cm
            scale_seg = ((int(x1), int(y1)), (int(x2), int(y2)))
    # Fallback to CLI if clicks were skipped
    if mm_per_px is None:
        mm_per_px = mm_per_px_cli

    # --- Tag worm polygon ---
    worm_poly = ClickCollector("Tag Worm", frame0).get_points(
        "Worm: Outline with left-clicks. Enter to accept. Right-click to undo.",
        min_points=3
    )
    if len(worm_poly) < 3:
        raise RuntimeError("Worm polygon not provided.")
    wormMask0 = polygon_mask(frame0.shape, worm_poly)

    # --- Build initial profiles ---
    g0 = to_gray_float01(frame0)
    dark0 = relative_darkness(g0)
    wormMask_bool = (wormMask0 > 0) & dishMask_bool
    worm_vals = dark0[wormMask_bool]
    bg_vals   = dark0[dishMask_bool & (~wormMask_bool)]
    worm_mean, worm_std = float(np.median(worm_vals)), float(np.std(worm_vals) + 1e-6)

    # Background EMA
    bg_ema = dark0.copy()

    # Outputs
    if out_prefix is None:
        out_prefix = str(Path(video_path).with_suffix('').as_posix())
    overlay_path = f"{out_prefix}_overlay.mp4"
    csv_path     = f"{out_prefix}_tracks.csv"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None
    if write_overlay:
        writer = cv2.VideoWriter(overlay_path, fourcc, fps_out, (W, H), True)

    # CSV
    csv_file = open(csv_path, 'w', newline='')
    # Write a comment line noting calibration (if present)
    if mm_per_px is not None:
        csv_file.write(f"# mm_per_px={mm_per_px:.8f}\n")
    csvw = csv.writer(csv_file)
    header = ["frame", "time_s", "centroid_x", "centroid_y",
              "area_px", "mean_darkness", "confidence",
              "speed_px_s", "heading_deg"]
    if mm_per_px is not None:
        header += ["speed_mm_s", "centroid_x_mm", "centroid_y_mm"]
    csvw.writerow(header)

    # Tracking state
    last_centroid = None
    last_time = 0.0

    # First row from initial polygon
    comp_centroid, comp_area, comp_mask = largest_component(wormMask0)
    if comp_centroid is not None:
        cx, cy = comp_centroid
        mean_dark = float(dark0[wormMask_bool].mean())
        row = [0, 0.0, cx, cy, comp_area, mean_dark, 1.0, "", ""]
        if mm_per_px is not None:
            row += ["", cx * mm_per_px, cy * mm_per_px]
        csvw.writerow(row)
        last_centroid = (cx, cy)
        last_time = 0.0
    else:
        row = [0, 0.0, "", "", 0, float(worm_vals.mean()), 0.0, "", ""]
        if mm_per_px is not None:
            row += ["", "", ""]
        csvw.writerow(row)

    # Visualize first frame
    vis0 = frame0.copy()
    if dish_center:
        cv2.circle(vis0, tuple(map(int, dish_center)), int(dish_radius), (255, 200, 0), 2)
    cv2.polylines(vis0, [np.array(worm_poly, dtype=np.int32)], True, (0, 255, 0), 2)
    if scale_seg:
        cv2.line(vis0, scale_seg[0], scale_seg[1], (255, 0, 255), 2)
        cv2.putText(vis0, f"1 cm  |  {mm_per_px:.4f} mm/px", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    if write_overlay:
        writer.write(vis0)

    # Main loop
    frame_idx = 1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar_total = (total_frames - 1) if total_frames > 0 else 0
    pbar = tqdm(total=pbar_total, desc="Tracking", unit="frame")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = to_gray_float01(frame)
        dark = relative_darkness(gray)

        # Update background EMA (only within dish)
        bg_ema[dishMask_bool] = (1 - ema_alpha) * bg_ema[dishMask_bool] + ema_alpha * dark[dishMask_bool]

        # Darkness gates
        dish_dark = dark[dishMask_bool]
        dish_med = np.median(dish_dark)
        base_dark = (dark >= (dish_med + darkness_thresh)).astype(np.uint8)
        prof_dark = (dark >= (worm_mean - k_profile * worm_std)).astype(np.uint8)
        delta = (dark - bg_ema)
        moving = (delta >= delta_gate).astype(np.uint8)

        # Combine and constrain to dish
        cand = (base_dark & prof_dark & moving & (dishMask > 0)).astype(np.uint8) * 255

        # Morphology
        if morph_open and morph_open > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))
            cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, k)
        if morph_close and morph_close > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
            cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k)

        centroid, area, comp_mask = largest_component(cand)
        t = frame_idx / fps

        vis = frame.copy()
        if dish_center:
            cv2.circle(vis, tuple(map(int, dish_center)), int(dish_radius), (255, 200, 0), 2)
        if scale_seg:
            cv2.line(vis, scale_seg[0], scale_seg[1], (255, 0, 255), 2)
            cv2.putText(vis, f"{mm_per_px:.4f} mm/px", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        if centroid is not None and area >= min_area_px:
            cx, cy = centroid
            plausible = True
            if last_centroid and max_jump_px is not None:
                if math.dist((cx, cy), last_centroid) > max_jump_px:
                    plausible = False

            if plausible:
                comp_bool = (comp_mask > 0)
                comp_dark_mean = float(dark[comp_bool].mean())
                comp_delta_mean = float(delta[comp_bool].mean())
                confidence = float(np.clip(0.5 * (comp_dark_mean - dish_med) + 0.5 * comp_delta_mean, 0, 1))

                # Kinematics
                if last_centroid is not None:
                    dt = max(1e-6, t - last_time)
                    dx = cx - last_centroid[0]
                    dy = cy - last_centroid[1]
                    step_px = float(np.hypot(dx, dy))
                    speed_px_s = step_px / dt
                    heading_deg = (np.degrees(np.arctan2(-dy, dx)) + 360.0) % 360.0
                    speed_mm_s = (speed_px_s * mm_per_px) if (mm_per_px is not None) else None
                else:
                    speed_px_s, heading_deg, speed_mm_s = "", "", None

                row = [frame_idx, round(t, 4), cx, cy, int(area),
                       round(comp_dark_mean, 4), round(confidence, 4),
                       (round(speed_px_s, 4) if speed_px_s != "" else ""),
                       (round(heading_deg, 2) if heading_deg != "" else "")]
                if mm_per_px is not None:
                    row += [(round(speed_mm_s, 4) if speed_mm_s is not None else ""),
                            cx * mm_per_px, cy * mm_per_px]
                csvw.writerow(row)

                # Overlay
                cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cnts, -1, (0, 0, 255), 2)
                cv2.circle(vis, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                label = f"t={t:.2f}s  area={area}  conf={confidence:.2f}"
                if isinstance(speed_px_s, float):
                    label += f"  v={speed_px_s:.1f}px/s"
                    if mm_per_px is not None:
                        label += f" ({speed_mm_s:.2f} mm/s)"
                    label += f"  θ={heading_deg:.0f}°"
                cv2.putText(vis, label, (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Update state
                last_centroid = (cx, cy)
                last_time = t
            else:
                row = [frame_idx, round(t, 4), "", "", 0, 0.0, 0.0, "", ""]
                if mm_per_px is not None:
                    row += ["", "", ""]
                csvw.writerow(row)
                cv2.putText(vis, f"t={t:.2f}s  (skip: jump)", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 255), 2)
        else:
            row = [frame_idx, round(t, 4), "", "", 0, 0.0, 0.0, "", ""]
            if mm_per_px is not None:
                row += ["", "", ""]
            csvw.writerow(row)
            cv2.putText(vis, f"t={t:.2f}s  (no component)", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 255), 2)

        if write_overlay:
            writer.write(vis)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    csv_file.close()
    if write_overlay:
        writer.release()

    print("Wrote CSV to:", csv_path)
    if write_overlay:
        print("Wrote overlay MP4 to:", overlay_path)
    if mm_per_px is not None:
        print(f"Calibration: {mm_per_px:.6f} mm/px (from clicks{'' if scale_seg else ' or CLI'})")

    return csv_path, (overlay_path if write_overlay else None)

# ----------------------------
# CLI
# ----------------------------
def main():
    import glob
    parser = argparse.ArgumentParser(description="Pixel-intensity worm tracker with initial tag + 1 cm scale calibration.")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to a single video (.mkv or .mp4). If omitted, batch over ../planarian_social_interactions/*.mkv")
    parser.add_argument("--out_prefix", type=str, default=None, help="Output prefix for CSV/MP4.")
    parser.add_argument("--no_overlay", action="store_true", help="Disable overlay MP4 writing.")
    parser.add_argument("--fps_out", type=float, default=None, help="FPS for overlay output (defaults to source FPS).")
    parser.add_argument("--mm_per_px", type=float, default=None, help="Millimeters per pixel (fallback if skipping clicks).")
    parser.add_argument("--px_per_mm", type=float, default=None, help="Pixels per millimeter (alternative fallback).")
    args = parser.parse_args()

    mm_per_px = args.mm_per_px
    if mm_per_px is None and args.px_per_mm is not None and args.px_per_mm > 0:
        mm_per_px = 1.0 / args.px_per_mm

    if args.video is not None:
        run_tracker(
            video_path=args.video,
            out_prefix=args.out_prefix,
            write_overlay=not args.no_overlay,
            fps_out=args.fps_out,
            mm_per_px_cli=mm_per_px
        )
        return

    # Batch over default folder
    folder = "../planarian_social_interactions"
    patterns = [os.path.join(folder, "*.mkv")]
    files = [p for pat in patterns for p in glob.glob(pat)]
    if not files:
        raise RuntimeError(f"No videos found in {folder}")

    for vp in files:
        base = os.path.splitext(vp)[0]
        out_prefix = args.out_prefix or base
        print(f"\n=== Processing: {vp} ===")
        run_tracker(
            video_path=vp,
            out_prefix=out_prefix,
            write_overlay=not args.no_overlay,
            fps_out=args.fps_out,
            mm_per_px_cli=mm_per_px
        )

if __name__ == "__main__":
    main()