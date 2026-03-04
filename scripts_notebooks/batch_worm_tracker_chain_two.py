# batch_worm_tracker_chain_two.py
# Two-worm, chained tracker
# 1) Seed once (interactive): dish + 1 cm + Worm A polygon + Worm B polygon
# 2) Batch over the folder headlessly; each video starts from last known A/B centroids + templates
#
# Quick start:
#   pip install opencv-python numpy tqdm
#   python batch_worm_tracker_chain_two.py --seed --video "../planarian_social_interactions/2025-10-14 03-00-15.mkv" --calibration ./calibration_2worms.json
#   python batch_worm_tracker_chain_two.py --videos_dir "../planarian_social_interactions" --glob "*.mkv" --calibration ./calibration_2worms.json --no_overlay
#
# CSV per frame (one row): A_* then B_* blocks with px/mm positions, speed, heading, etc.

import os, json, math, argparse, glob
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import csv
from itertools import permutations

# ---------- Basic utils ----------
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

def template_from(map_img, center, half=24):
    """Return a z-scored template patch centered at `center` (half-size = `half`)."""
    h, w = map_img.shape
    cx, cy = int(round(center[0])), int(round(center[1]))
    x1, x2 = max(0, cx-half), min(w, cx+half)
    y1, y2 = max(0, cy-half), min(h, cy+half)
    patch = map_img[y1:y2, x1:x2].copy()
    if patch.size == 0:
        return None
    m, s = patch.mean(), patch.std() + 1e-6
    return (patch - m) / s

def ncc_score(map_img, templ):
    """Safe NCC: only run if ROI >= template in both dimensions."""
    if templ is None or templ.size == 0:
        return (None, None)
    th, tw = templ.shape[:2]
    mh, mw = map_img.shape[:2]
    if mh < th or mw < tw:
        return (None, None)
    res = cv2.matchTemplate(map_img.astype(np.float32),
                            templ.astype(np.float32),
                            cv2.TM_CCOEFF_NORMED)
    _, maxv, _, maxloc = cv2.minMaxLoc(res)
    cy = maxloc[1] + th / 2.0
    cx = maxloc[0] + tw / 2.0
    return (cx, cy), float(maxv)

def build_static_background_darkness(cap, sample_frames=300, stride=3, dish_mask_bool=None):
    frames = []
    pos0 = cap.get(cv2.CAP_PROP_POS_FRAMES)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    n_to_sample = min(sample_frames, total) if total > 0 else sample_frames
    for i in range(n_to_sample):
        ok, frame = cap.read()
        if not ok:
            break
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

# ---------- Components ----------
def extract_components(bin_mask, max_k=6, min_area=1):
    num, labels, stats, cents = cv2.connectedComponentsWithStats(bin_mask, 8)
    comps = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x, y, w, h, _ = stats[i]
        cx, cy = cents[i]
        cmask = (labels == i).astype(np.uint8) * 255
        comps.append(dict(
            centroid=(float(cx), float(cy)),
            area=area,
            bbox=(x, y, w, h),
            mask=cmask
        ))
    comps.sort(key=lambda c: c["area"], reverse=True)
    return comps[:max_k]

# ---------- Interactive (seeding) ----------
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
                cv2.circle(self.img,(px,py),4,(0,255,0),-1)
            cv2.imshow(self.win, self.img)
    def get(self, prompt, min_pts=1):
        overlay = self.img.copy()
        cv2.putText(overlay, prompt, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30,220,30), 2)
        self.img = overlay
        cv2.imshow(self.win, self.img)
        while True:
            k = cv2.waitKey(33) & 0xFF
            if k in (13, 32):  # Enter/Space
                if len(self.pts) >= min_pts:
                    break
            elif k == 27:      # ESC
                self.pts = []
                break
        cv2.destroyWindow(self.win)
        return self.pts

def polygon_mask(shape, pts):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if len(pts) >= 3:
        cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)
    return mask

def seed_two(video_path, save_json):
    cap = cv2.VideoCapture(str(video_path)); ok, frame0 = cap.read(); cap.release()
    if not ok:
        raise RuntimeError("Cannot read first frame for seeding.")
    H, W = frame0.shape[:2]

    # Dish
    cc = ClickCollector("Seed: Dish", frame0)
    pts = cc.get("Dish: click CENTER then RIM. Enter to accept.", 2)
    if len(pts) < 2:
        raise RuntimeError("Dish not tagged.")
    (cx, cy), (rx, ry) = pts[0], pts[1]
    dish_center = [float(cx), float(cy)]
    dish_radius = float(math.dist((cx, cy), (rx, ry)))

    # 1 cm
    cc2 = ClickCollector("Seed: 1 cm", frame0)
    spts = cc2.get("Scale: click START and END of 1 cm line. Enter to accept.", 2)
    if len(spts) < 2:
        raise RuntimeError("Scale not tagged.")
    (sx, sy), (ex, ey) = spts[0], spts[1]
    mm_per_px = 10.0 / float(math.dist((sx, sy), (ex, ey)))

    # Worm A & B polygons
    cc3 = ClickCollector("Seed: Worm A", frame0)
    wA = cc3.get("Worm A: outline polygon. Enter to accept.", 3)
    cc4 = ClickCollector("Seed: Worm B", frame0)
    wB = cc4.get("Worm B: outline polygon. Enter to accept.", 3)
    if len(wA) < 3 or len(wB) < 3:
        raise RuntimeError("Both worms must be tagged.")

    # Initial centroids & templates
    g0 = to_gray01(frame0); dark0 = relative_darkness(g0)
    mA = polygon_mask(frame0.shape, wA); mB = polygon_mask(frame0.shape, wB)
    compsA = extract_components(mA, max_k=1)
    compsB = extract_components(mB, max_k=1)
    if not compsA or not compsB:
        raise RuntimeError("Could not extract both worm components.")
    cA = compsA[0]["centroid"]; cB = compsB[0]["centroid"]
    tA = template_from(dark0, cA, half=24)
    tB = template_from(dark0, cB, half=24)
    if tA is None or tB is None:
        raise RuntimeError("Failed to build templates for both worms.")

    calib = dict(
        dish_center=dish_center,
        dish_radius_px=dish_radius,
        mm_per_px=float(mm_per_px),
        wormA_seed_centroid=[float(cA[0]), float(cA[1])],
        wormB_seed_centroid=[float(cB[0]), float(cB[1])],
        wormA_template=tA.tolist(),
        wormB_template=tB.tolist()
    )
    with open(save_json, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"Saved seed+calibration → {save_json}")
    return calib

# ---------- Assignment helper for two worms ----------
def assign_to_tracks(comps, delta_pos, tracks, alpha_ncc=0.6, max_jump=None):
    """
    comps: list of dicts with 'centroid','area','bbox','mask'
    tracks: [{'last': (x,y) or None, 'templ': np.array or None}, ...]
    Returns chosen list for each track and used indices.
    """
    K, T = len(comps), len(tracks)
    if K == 0:
        return [None]*T, set()

    dist = np.full((T, K), 1e9, dtype=np.float32)
    nccs = np.zeros((T, K), dtype=np.float32)

    for ti, tr in enumerate(tracks):
        last = tr["last"]
        templ = tr["templ"]
        for ki, c in enumerate(comps):
            cx, cy = c["centroid"]
            # distance cost
            if last is not None:
                d = math.dist((cx, cy), last)
                if (max_jump is not None) and (d > max_jump * 2.0):
                    d += 1e4
            else:
                d = 500.0  # no prior
            # NCC score in padded ROI around component
            ncc = 0.0
            if templ is not None and templ.size > 0:
                x, y, w, h = c["bbox"]
                pad = 32
                H, W = delta_pos.shape
                x1 = max(0, x - pad); y1 = max(0, y - pad)
                x2 = min(W, x + w + pad); y2 = min(H, y + h + pad)
                roi = delta_pos[y1:y2, x1:x2]
                if roi.size > 0:
                    th, tw = templ.shape[:2]
                    mh, mw = roi.shape[:2]
                    if mh >= th and mw >= tw:
                        loc, ncc_val = ncc_score(roi, templ)
                        if loc is not None:
                            ncc = float(max(0.0, ncc_val))
            dist[ti, ki] = d
            nccs[ti, ki] = ncc

    best = None
    best_pair = None
    comp_indices = list(range(K))
    for perm in permutations(comp_indices, min(T, K)):
        total_cost = 0.0
        ok = True
        for ti in range(T):
            if ti >= len(perm):
                ok = False; break
            ki = perm[ti]
            cost = dist[ti, ki] - alpha_ncc * (100.0 * nccs[ti, ki])
            total_cost += cost
        if ok and (best is None or total_cost < best):
            best = total_cost
            best_pair = perm

    chosen = [None]*T
    used = set()
    if best_pair is not None:
        for ti, ki in enumerate(best_pair):
            chosen[ti] = {**comps[ki], "ncc": float(nccs[ti, ki])}
            used.add(ki)

    return chosen, used

# ---------- Per-video (two worms) ----------
def process_video_two(
    video_path, calib, state,
    darkness_thresh=0.15, delta_gate=0.05,
    min_area_px=50, morph_open=3, morph_close=5,
    max_jump_px=200, roi_px=140, write_overlay=False, fps_out=None, 
    out_tag="2worms"
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("Empty video.")
    H, W = frame0.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps_out is None:
        fps_out = fps

    # Dish + scale
    dish_center = tuple(calib["dish_center"])
    dish_radius = float(calib["dish_radius_px"])
    mm_per_px = float(calib["mm_per_px"])
    dishMask = circle_mask((H, W), dish_center, dish_radius)
    dishMask_bool = (dishMask > 0)

    # Static background from this video
    bg_dark = build_static_background_darkness(cap, sample_frames=300, stride=3, dish_mask_bool=dishMask_bool)

    # Outputs
    out_prefix = str(Path(video_path).with_suffix('').as_posix())
    csv_path = f"{out_prefix}_tracks_{out_tag}.csv"
    overlay_path = f"{out_prefix}_overlay_{out_tag}.mp4"
    writer = None
    if write_overlay:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(overlay_path, fourcc, fps_out, (W, H), True)

    # CSV
    csv_file = open(csv_path, 'w', newline='')
    csv_file.write(f"# mm_per_px={mm_per_px:.8f}\n")
    csvw = csv.writer(csv_file)
    csvw.writerow([
        "frame","time_s",
        "A_x","A_y","A_x_mm","A_y_mm","A_area_px","A_mean_dark","A_conf","A_speed_px_s","A_speed_mm_s","A_heading_deg",
        "B_x","B_y","B_x_mm","B_y_mm","B_area_px","B_mean_dark","B_conf","B_speed_px_s","B_speed_mm_s","B_heading_deg",
    ])

    # Chain state in
    A_last = state.get("A_centroid", calib.get("wormA_seed_centroid"))
    B_last = state.get("B_centroid", calib.get("wormB_seed_centroid"))
    A_templ = state.get("A_template", np.array(calib.get("wormA_template"), dtype=np.float32))
    B_templ = state.get("B_template", np.array(calib.get("wormB_template"), dtype=np.float32))
    if isinstance(A_templ, list): A_templ = np.array(A_templ, dtype=np.float32)
    if isinstance(B_templ, list): B_templ = np.array(B_templ, dtype=np.float32)

    # Rewind to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc=f"Processing {Path(video_path).name}", unit="frame")

    frame_idx = 0
    A_last_time = 0.0
    B_last_time = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps

        gray = to_gray01(frame)
        dark = relative_darkness(gray)
        dish_dark = dark[dishMask_bool]
        dish_med = np.median(dish_dark) if dish_dark.size else 0.0

        base_dark = (dark >= (dish_med + darkness_thresh)).astype(np.uint8)
        delta = (dark - bg_dark)
        delta_pos = np.clip(delta, 0, 1)
        moving = (delta_pos >= delta_gate).astype(np.uint8)
        cand = (base_dark & moving & (dishMask > 0)).astype(np.uint8) * 255

        # Morphology cleanup
        if morph_open and morph_open > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))
            cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, k)
        if morph_close and morph_close > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
            cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k)

        comps = extract_components(cand, max_k=6, min_area=min_area_px)

        # Assign components to A/B using distance + NCC
        chosen, used = assign_to_tracks(
            comps,
            delta_pos,
            tracks=[{"last": A_last, "templ": A_templ},
                    {"last": B_last, "templ": B_templ}],
            alpha_ncc=0.6,
            max_jump=max_jump_px
        )

        # For any missing track, try NCC in a ROI around last position
        for ti, (last, templ) in enumerate([(A_last, A_templ), (B_last, B_templ)]):
            if chosen[ti] is not None:
                continue
            if last is None or templ is None or templ.size == 0:
                continue
            cx, cy = last
            Hh, Ww = delta_pos.shape
            x1 = int(max(0, cx - roi_px)); x2 = int(min(Ww, cx + roi_px))
            y1 = int(max(0, cy - roi_px)); y2 = int(min(Hh, cy + roi_px))
            roi = delta_pos[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            th, tw = templ.shape[:2]
            mh, mw = roi.shape[:2]
            if mh >= th and mw >= tw:
                loc, ncc = ncc_score(roi, templ)
                if loc is not None:
                    loc = (loc[0] + x1, loc[1] + y1)
                    chosen[ti] = dict(centroid=loc, area=min_area_px, mask=None, ncc=float(ncc))

        # Helpers
        def kinematics(prev, now_center, now_time, prev_time):
            if prev is None or now_center is None:
                return "", "", ""
            dt = max(1e-6, now_time - prev_time)
            dx, dy = (now_center[0] - prev[0]), (now_center[1] - prev[1])
            step_px = float(np.hypot(dx, dy))
            vpx = step_px / dt
            vmm = vpx * mm_per_px
            heading = (np.degrees(np.arctan2(-dy, dx)) + 360.0) % 360.0
            return vpx, vmm, heading

        def comp_stats(center, mask):
            if center is None:
                return 0.0, 0.0
            if mask is not None and mask.any():
                m = (mask > 0)
                c_dark = float(dark[m].mean())
                c_delta = float(delta[m].mean())
            else:
                x, y = int(round(center[0])), int(round(center[1]))
                y = np.clip(y, 0, dark.shape[0]-1); x = np.clip(x, 0, dark.shape[1]-1)
                c_dark = float(dark[y, x]); c_delta = float(delta[y, x])
            conf = float(np.clip(0.5*(c_dark - dish_med) + 0.5*c_delta, 0, 1))
            return c_dark, conf

        # A stats
        A_center = chosen[0]["centroid"] if chosen[0] is not None else None
        A_area = (chosen[0]["area"] if chosen[0] is not None else 0)
        A_dark, A_conf = comp_stats(A_center, chosen[0]["mask"] if chosen[0] is not None else None)
        A_vpx, A_vmm, A_head = kinematics(A_last, A_center, t, A_last_time)

        # B stats
        B_center = chosen[1]["centroid"] if chosen[1] is not None else None
        B_area = (chosen[1]["area"] if chosen[1] is not None else 0)
        B_dark, B_conf = comp_stats(B_center, chosen[1]["mask"] if chosen[1] is not None else None)
        B_vpx, B_vmm, B_head = kinematics(B_last, B_center, t, B_last_time)

        # Write CSV
        row = [frame_idx, round(t, 4)]
        if A_center is not None:
            row += [A_center[0], A_center[1], A_center[0]*mm_per_px, A_center[1]*mm_per_px,
                    int(A_area), round(A_dark,4), round(A_conf,4),
                    (round(A_vpx,4) if A_vpx != "" else ""), (round(A_vmm,4) if A_vmm != "" else ""), (round(A_head,2) if A_head != "" else "")]
        else:
            row += ["","","","",0,0.0,0.0,"","",""]
        if B_center is not None:
            row += [B_center[0], B_center[1], B_center[0]*mm_per_px, B_center[1]*mm_per_px,
                    int(B_area), round(B_dark,4), round(B_conf,4),
                    (round(B_vpx,4) if B_vpx != "" else ""), (round(B_vmm,4) if B_vmm != "" else ""), (round(B_head,2) if B_head != "" else "")]
        else:
            row += ["","","","",0,0.0,0.0,"","",""]
        csvw.writerow(row)

        # Update states & templates
        if A_center is not None:
            A_last = A_center; A_last_time = t
            if frame_idx % 60 == 0:
                nt = template_from(delta_pos, A_center, half=24)
                if nt is not None: A_templ = nt
        if B_center is not None:
            B_last = B_center; B_last_time = t
            if frame_idx % 60 == 0:
                nt = template_from(delta_pos, B_center, half=24)
                if nt is not None: B_templ = nt

        if writer:
            vis = frame.copy()
            cv2.circle(vis, (int(dish_center[0]), int(dish_center[1])), int(dish_radius), (255,200,0), 2)
            if A_center is not None:
                cv2.circle(vis, (int(A_center[0]), int(A_center[1])), 5, (0,0,255), -1)
                cv2.putText(vis, "A", (int(A_center[0])+6, int(A_center[1])-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            if B_center is not None:
                cv2.circle(vis, (int(B_center[0]), int(B_center[1])), 5, (0,255,0), -1)
                cv2.putText(vis, "B", (int(B_center[0])+6, int(B_center[1])-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            writer.write(vis)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    csv_file.close()
    if writer:
        writer.release()

    # Return updated chain state
    return dict(A_centroid=A_last, B_centroid=B_last, A_template=A_templ, B_template=B_templ)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Chained two-worm tracker: seed once, process all videos headlessly.")
    ap.add_argument("--seed", action="store_true", help="Run interactive seeding (dish + scale + worm A + worm B).")
    ap.add_argument("--video", type=str, help="Video for seeding.")
    ap.add_argument("--calibration", type=str, default="./calibration_2worms.json", help="Calibration JSON path.")
    ap.add_argument("--videos_dir", type=str, default="../planarian_social_interactions", help="Folder to process.")
    ap.add_argument("--glob", type=str, default="*.mkv", help="Pattern (e.g., *.mp4).")
    ap.add_argument("--no_overlay", action="store_true")
    ap.add_argument("--roi_px", type=int, default=140, help="ROI radius (px) around previous centroid.")
    ap.add_argument("--darkness_thresh", type=float, default=0.15)
    ap.add_argument("--delta_gate", type=float, default=0.05)
    ap.add_argument("--min_area_px", type=int, default=50)
    ap.add_argument("--morph_open", type=int, default=3)
    ap.add_argument("--morph_close", type=int, default=5)
    ap.add_argument("--max_jump_px", type=float, default=200)
    ap.add_argument("--out_tag", type=str, default="2worms",
                help="Tag appended to output filenames (e.g., _tracks_<tag>.csv). Default: 2worms")
    args = ap.parse_args()

    if args.seed:
        if not args.video:
            raise SystemExit("Provide --video with --seed.")
        seed_two(args.video, args.calibration)
        return

    if not os.path.exists(args.calibration):
        raise SystemExit(f"Missing {args.calibration}. Run with --seed first.")
    with open(args.calibration, "r") as f:
        calib = json.load(f)

    files = sorted(glob.glob(os.path.join(args.videos_dir, args.glob)))
    if not files:
        raise SystemExit(f"No videos found in {args.videos_dir} with pattern {args.glob}")

    # Initialize chain state
    state = dict(
        A_centroid=calib.get("wormA_seed_centroid"),
        B_centroid=calib.get("wormB_seed_centroid"),
        A_template=np.array(calib.get("wormA_template"), dtype=np.float32),
        B_template=np.array(calib.get("wormB_template"), dtype=np.float32),
    )

    for vp in files:
        print(f"\n=== {vp} ===")
        state = process_video_two(
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
            out_tag=args.out_tag
        )

if __name__ == "__main__":
    main()