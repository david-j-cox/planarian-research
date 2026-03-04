#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, glob, json, math
from dataclasses import dataclass
import numpy as np
import pandas as pd
import cv2
from scipy.optimize import linear_sum_assignment

# =========================
# Utilities
# =========================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def angle_wrap_deg(a):
    return (a + 180) % 360 - 180

def vec_angle_deg(dx, dy):
    # screen coords (y down): atan2(dy, dx)
    return (math.degrees(math.atan2(dy, dx)) + 360) % 360

def poly_to_mask(poly_pts, shape):
    mask = np.zeros(shape[:2], np.uint8)
    if len(poly_pts) >= 3:
        cv2.fillPoly(mask, [np.array(poly_pts, np.int32)], 255)
    return mask

def circle_from_cross(top, bottom, left, right):
    pts = np.array([top, bottom, left, right], dtype=float)
    c = pts.mean(axis=0)
    r = np.mean(np.linalg.norm(pts - c, axis=1))
    return (int(c[0]), int(c[1]), int(r))

def hsv_hist_template(frame, mask, h_bins=32, s_bins=32):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1], mask, [h_bins, s_bins], [0,180, 0,256])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return hist

def backproject_prob(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    bp = cv2.calcBackProject([hsv], [0,1], hist, [0,180, 0,256], scale=1)
    return cv2.GaussianBlur(bp, (9,9), 0)

# =========================
# Interactive initializer
# =========================
class InitTagger:
    """
    Step 1: click TWO endpoints of 1 cm line
    Step 2: click dish Top, Bottom, Left, Right (in that order)
    Step 3: press '1' to outline Worm A (poly) then ENTER;
            press '2' to outline Worm B (poly) then ENTER
    'Z' undo, 'S' save and proceed, 'Q' cancel
    """
    def __init__(self, first_frame, cache_path):
        self.base = first_frame.copy()
        self.im = first_frame.copy()
        self.h, self.w = self.im.shape[:2]
        self.cache_path = cache_path
        self.win = "init_tagging"
        self.step = 0
        self.points = []   # [cm1, cm2, top, bottom, left, right]
        self.poly1, self.poly2 = [], []
        self.active_poly = 0  # 0 none, 1 worm1, 2 worm2

    def _draw_ui(self):
        self.im = self.base.copy()
        lines = [
            "Step 1/3: Click TWO endpoints of the 1 cm mark (Undo=Z)",
            "Step 2/3: Click dish Top, Bottom, Left, Right (in that order; Undo=Z)",
            "Step 3/3: Press '1' to outline Worm A (poly), ENTER to close; then '2' for Worm B",
            "Press 'S' to save & start. 'Q' to cancel."
        ]
        y = 24
        for i, t in enumerate(lines):
            color = (0,255,255) if i == self.step else (120,180,180)
            cv2.putText(self.im, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 22

        # draw scale points/line
        if len(self.points) >= 1:
            for p in self.points[:2]:
                cv2.circle(self.im, tuple(map(int,p)), 5, (0,255,255), -1)
            if len(self.points) >= 2:
                cv2.line(self.im, tuple(map(int,self.points[0])),
                         tuple(map(int,self.points[1])), (0,255,255), 2)

        # draw dish circle if available
        if len(self.points) >= 6:
            top,bottom,left,right = self.points[2], self.points[3], self.points[4], self.points[5]
            cx,cy,r = circle_from_cross(top,bottom,left,right)
            cv2.circle(self.im, (cx,cy), int(r), (0,200,0), 2)

        # draw polygons
        for poly, col in [(self.poly1,(255,180,0)), (self.poly2,(180,255,0))]:
            for i in range(1, len(poly)):
                cv2.line(self.im, poly[i-1], poly[i], col, 2)

        # active poly hint
        if self.active_poly in (1,2):
            cv2.putText(self.im, f"Outlining Worm {'A' if self.active_poly==1 else 'B'} (click points, ENTER to close)",
                        (12, self.h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    def _on_mouse(self, ev,x,y,flags,ud):
        if ev != cv2.EVENT_LBUTTONDOWN: return
        if self.active_poly == 1:
            self.poly1.append((x,y))
            return
        if self.active_poly == 2:
            self.poly2.append((x,y))
            return
        if self.step == 0 and len(self.points) < 2:
            self.points.append((x,y))
        elif self.step == 1 and len(self.points) < 6:
            self.points.append((x,y))

    def run(self):
        cv2.namedWindow(self.win)
        cv2.setMouseCallback(self.win, self._on_mouse)

        while True:
            self._draw_ui()
            cv2.imshow(self.win, self.im)
            k = cv2.waitKey(20) & 0xFF

            if k == ord('q'):
                cv2.destroyWindow(self.win)
                return None

            if k == ord('z'):
                if self.active_poly == 1 and self.poly1:
                    self.poly1.pop()
                elif self.active_poly == 2 and self.poly2:
                    self.poly2.pop()
                elif self.points:
                    self.points.pop()

            if k == ord('1'): self.active_poly = 1
            if k == ord('2'): self.active_poly = 2
            if k == 13:       self.active_poly = 0  # ENTER

            # advance steps
            if self.step == 0 and len(self.points) >= 2: self.step = 1
            if self.step == 1 and len(self.points) >= 6: self.step = 2
            if self.step == 2 and self.poly1:            self.step = 3
            if self.step == 3 and self.poly2:            self.step = 4

            if k == ord('s'):
                if len(self.points) >= 6 and self.poly1 and self.poly2:
                    break

        cv2.destroyWindow(self.win)

        # outputs
        (x1,y1),(x2,y2) = self.points[0], self.points[1]
        px_per_cm = math.hypot(x2-x1, y2-y1) / 1.0
        top,bottom,left,right = self.points[2], self.points[3], self.points[4], self.points[5]
        cx,cy,r = circle_from_cross(top,bottom,left,right)
        petri_mask = np.zeros(self.base.shape[:2], np.uint8)
        cv2.circle(petri_mask, (cx,cy), int(r)-2, 255, -1)

        mask1 = poly_to_mask(self.poly1, self.base.shape)
        mask2 = poly_to_mask(self.poly2, self.base.shape)
        hist1 = hsv_hist_template(self.base, mask1)
        hist2 = hsv_hist_template(self.base, mask2)

        cache = {
            "px_per_cm": float(px_per_cm),
            "dish_circle": [int(cx), int(cy), int(r)],
            "poly1": self.poly1,
            "poly2": self.poly2
        }
        with open(self.cache_path, "w") as f:
            json.dump(cache, f)

        return px_per_cm, petri_mask, hist1, hist2

def load_cached_init(cache_path, frame):
    if not os.path.exists(cache_path): return None
    with open(cache_path,"r") as f:
        c = json.load(f)
    px_per_cm = float(c["px_per_cm"])
    cx,cy,r = c["dish_circle"]
    petri_mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.circle(petri_mask, (cx,cy), int(r)-2, 255, -1)
    mask1 = poly_to_mask([tuple(p) for p in c["poly1"]], frame.shape)
    mask2 = poly_to_mask([tuple(p) for p in c["poly2"]], frame.shape)
    hist1 = hsv_hist_template(frame, mask1)
    hist2 = hsv_hist_template(frame, mask2)
    return px_per_cm, petri_mask, hist1, hist2

# =========================
# Detection
# =========================
@dataclass
class Detection:
    centroid: np.ndarray
    angle_deg: float
    area: float
    ellipse: tuple

def detect_with_templates(frame, petri_mask, hists, min_area=20, max_area=60000):
    dets = []
    for hist in hists:
        bp = backproject_prob(frame, hist)
        if petri_mask is not None:
            bp = cv2.bitwise_and(bp, bp, mask=petri_mask)
        # Otsu on backprojection
        _, th = cv2.threshold(bp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), 1)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8), 1)
        cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            dets.append(None); continue
        best, best_score = None, -1
        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area or area > max_area: continue
            if len(c) < 5: continue
            ellipse = cv2.fitEllipse(c)
            (cx,cy),(MA,ma),angle = ellipse
            short,long = sorted([MA,ma])
            if short < 3: continue
            elong = long / (short + 1e-6)
            score = area * elong
            if score > best_score:
                best_score = score
                best = Detection(np.array([cx,cy], dtype=float), float(angle), float(area), ellipse)
        dets.append(best)
    return dets  # [d1, d2] (can include None)

def detect_generic(frame, petri_mask=None):
    if petri_mask is not None:
        frame = cv2.bitwise_and(frame, frame, mask=petri_mask)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    # try both polarities
    def th_fn(inv=False):
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV if not inv else cv2.THRESH_BINARY,
                                   31, 7)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), 1)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), 1)
        return th
    ths = [th_fn(False), th_fn(True)]
    best_cands = []
    for th in ths:
        cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 25 or area > 60000: continue
            if len(c) < 5: continue
            ellipse = cv2.fitEllipse(c)
            (cx,cy),(MA,ma),angle = ellipse
            short,long = sorted([MA,ma])
            if short < 3: continue
            aspect = long/(short+1e-6)
            if aspect < 1.3: continue
            best_cands.append(Detection(np.array([cx,cy],float), float(angle), float(area), ellipse))
    # keep top 2 by area*aspect
    best_cands = sorted(best_cands, key=lambda d: d.area, reverse=True)[:2]
    return best_cands

# =========================
# Tracking
# =========================
@dataclass
class Kalman1D:
    x: float
    v: float = 0.0
    p: float = 10.0
    q: float = 0.05
    r: float = 2.0
    def predict(self, dt):
        self.x = self.x + self.v*dt
        self.p = self.p + self.q
        return self.x
    def update(self, z):
        k = self.p / (self.p + self.r)
        self.x = self.x + k*(z - self.x)
        self.p = (1 - k)*self.p
        return self.x

class Kalman2D:
    def __init__(self, x, y):
        self.kx, self.ky = Kalman1D(x), Kalman1D(y)
    def predict(self, dt):
        return np.array([self.kx.predict(dt), self.ky.predict(dt)])
    def update(self, z):
        return np.array([self.kx.update(z[0]), self.ky.update(z[1])])

class WormTracker:
    def __init__(self, fps, px_per_cm):
        self.fps = fps
        self.dt = 1.0 / max(fps, 1e-6)
        self.px_per_cm = px_per_cm
        self.filters = {}
        self.prev_pos = {}
        self.next_id = 0

    def _assign(self, dets):
        ids = list(self.filters.keys())
        det_positions = [d["centroid"] for d in dets]
        if not ids:
            return {}, set(range(len(dets))), []
        preds = np.array([self.filters[i].predict(self.dt) for i in ids])
        dets_arr = np.array(det_positions)
        cost = np.linalg.norm(preds[:,None,:] - dets_arr[None,:,:], axis=2)
        row_ind, col_ind = linear_sum_assignment(cost)
        matches, unmatched = {}, set(range(len(dets)))
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < 100:  # pixels
                matches[ids[r]] = c
                unmatched.discard(c)
        for i in ids:
            if i not in matches:
                self.filters[i].predict(self.dt)
        return matches, unmatched, ids

    def _init(self, pos):
        return Kalman2D(pos[0], pos[1])

    def step(self, dets):
        # boot if empty
        if not self.filters and dets:
            for d in dets[:2]:
                self.filters[self.next_id] = self._init(d["centroid"])
                self.prev_pos[self.next_id] = d["centroid"].copy()
                self.next_id += 1

        matches, unmatched, ids = self._assign(dets) if dets else ({}, set(), list(self.filters.keys()))
        outputs = []

        for wid, di in matches.items():
            z = dets[di]["centroid"]
            pos = self.filters[wid].update(z)
            outputs.append((wid, pos, dets[di]))

        for di in unmatched:
            wid = self.next_id
            self.filters[wid] = self._init(dets[di]["centroid"])
            self.prev_pos[wid] = dets[di]["centroid"].copy()
            self.next_id += 1
            outputs.append((wid, dets[di]["centroid"], dets[di]))

        rows = []
        for wid, pos, det in outputs:
            prev = self.prev_pos.get(wid, pos)
            dx, dy = pos[0]-prev[0], pos[1]-prev[1]
            self.prev_pos[wid] = pos.copy()

            heading = vec_angle_deg(dx, dy)
            dist_px = math.hypot(dx, dy)
            speed_cm_s = (dist_px / self.px_per_cm) * self.fps
            moving = 1 if speed_cm_s > 0.02 else 0

            rows.append({
                "worm_id": wid,
                "x_px": pos[0], "y_px": pos[1],
                "x_cm": pos[0]/self.px_per_cm, "y_cm": pos[1]/self.px_per_cm,
                "orientation_deg": angle_wrap_deg(det["angle_deg"]),
                "heading_deg": heading,
                "speed_cm_s": speed_cm_s,
                "moving": moving,
                "ellipse": det["ellipse"],
                "area": det["area"]
            })
        return rows

# =========================
# Per-video processing
# =========================
def process_video(path, px_per_cm, petri_mask, hists, write_viz=True, out_dir=None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[WARN] Could not open {path}")
        return
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ret, frame0 = cap.read()
    if not ret:
        print(f"[WARN] Empty video: {path}")
        return
    h, w = frame0.shape[:2]
    base = os.path.splitext(os.path.basename(path))[0]
    out_dir = ensure_dir(out_dir or os.path.dirname(path))
    csv_path = os.path.join(out_dir, f"{base}_tracks.csv")

    writer = None
    if write_viz:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(os.path.join(out_dir, f"{base}_annotated.mp4"), fourcc, fps, (w, h))

    tracker = WormTracker(fps=fps, px_per_cm=px_per_cm)
    rows = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        # primary: template-based (two slots)
        dets_tmpl = detect_with_templates(frame, petri_mask, hists)
        dets = []
        for d in dets_tmpl:
            if d is not None:
                dets.append({
                    "centroid": d.centroid,
                    "angle_deg": d.angle_deg,
                    "ellipse": d.ellipse,
                    "area": d.area
                })

        # fallback: generic if fewer than 2 detections
        if len(dets) < 2:
            generic = detect_generic(frame, petri_mask)
            # add non-overlapping generics
            for g in generic:
                if all(np.linalg.norm(g.centroid - dd["centroid"]) > 20 for dd in dets):
                    dets.append({
                        "centroid": g.centroid,
                        "angle_deg": g.angle_deg,
                        "ellipse": g.ellipse,
                        "area": g.area
                    })
                if len(dets) == 2: break

        step_rows = tracker.step(dets)
        for r in step_rows:
            r["frame"] = frame_idx
            r["time_s"] = frame_idx / fps
            rows.append(r)

        if writer is not None:
            disp = frame.copy()
            # draw mask rim
            if petri_mask is not None:
                edges = cv2.Canny(petri_mask, 50, 150)
                disp[edges>0] = (0.6*disp[edges>0] + 0.4*np.array([0,255,0])).astype(np.uint8)
            # draw current detections
            for r in step_rows:
                cx, cy = int(r["x_px"]), int(r["y_px"])
                wid = r["worm_id"]
                speed = r["speed_cm_s"]
                cv2.circle(disp, (cx,cy), 5, (0,200,255), -1)
                cv2.putText(disp, f"ID {wid}  {speed:.2f} cm/s",
                            (cx+8, cy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,255,255), 1)
                # draw orientation axis
                L = 25
                theta = math.radians(r["orientation_deg"])
                x2 = int(cx + L*math.cos(theta)); y2 = int(cy + L*math.sin(theta))
                x1 = int(cx - L*math.cos(theta)); y1 = int(cy - L*math.sin(theta))
                cv2.line(disp, (x1,y1), (x2,y2), (255,180,0), 2)

            cv2.putText(disp, f"{base} | {frame_idx+1}/{total} | {fps:.1f} fps",
                        (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            writer.write(disp)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    # write CSV (even if empty, so issues are obvious)
    df = pd.DataFrame(rows, columns=[
        "frame","time_s","worm_id","x_px","y_px","x_cm","y_cm",
        "orientation_deg","heading_deg","speed_cm_s","moving","area"
    ])
    df.to_csv(csv_path, index=False)
    print(f"[OK] Saved {csv_path} ({len(df)} rows)")

# =========================
# Driver
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", required=True, help="Folder with .mkv videos")
    ap.add_argument("--glob", default="*.mkv", help="Filename pattern")
    ap.add_argument("--write_viz", type=int, default=1, help="Write annotated MP4s (1/0)")
    ap.add_argument("--reinit", type=int, default=0, help="Force new tagging even if cache exists")
    args = ap.parse_args()

    vids = sorted(glob.glob(os.path.join(args.video_dir, args.glob)))
    if not vids:
        print("[ERR] No videos found.")
        return

    # grab first frame for init/cached loading
    cap0 = cv2.VideoCapture(vids[0])
    ret, frame0 = cap0.read()
    cap0.release()
    if not ret:
        print("[ERR] Could not read first frame for init.")
        return

    cache_path = os.path.join(args.video_dir, "init_cache.json")
    if args.reinit or not os.path.exists(cache_path):
        init = InitTagger(frame0, cache_path).run()
        if init is None:
            print("[ERR] Initialization cancelled.")
            return
        px_per_cm, petri_mask, hist1, hist2 = init
    else:
        cached = load_cached_init(cache_path, frame0)
        if cached is None:
            print("[WARN] Cache missing/corrupt; reinitializing.")
            init = InitTagger(frame0, cache_path).run()
            if init is None: return
            px_per_cm, petri_mask, hist1, hist2 = init
        else:
            px_per_cm, petri_mask, hist1, hist2 = cached

    print(f"[INFO] Scale: {px_per_cm:.3f} px/cm")

    for vp in vids:
        process_video(vp, px_per_cm, petri_mask, [hist1, hist2],
                      write_viz=bool(args.write_viz), out_dir=args.video_dir)

if __name__ == "__main__":
    main()
