#!/usr/bin/env python3
"""
behavior_labeler.py — Interactive behavior labeling tool for planarian tracking data.

Surfaces 5-second windows of worm movement (ranked by activity), renders a composite
display with midline animation, curvature kymograph, speed/angle/body-length traces,
and lets the user label behaviors via keyboard toggles or mouse clicks.

Usage:
  cd scripts_notebooks
  source ../venv/bin/activate
  python behavior_labeler.py                        # default: /tmp/tracker_output
  python behavior_labeler.py --data_dir /path/to/data
  python behavior_labeler.py --sort sequential      # sequential | most_active | unlabeled
"""

import os, sys, json, glob, re, argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
WINDOW_SEC = 5.0
STRIDE_SEC = 4.0           # 1-second overlap
GAP_THRESH_SEC = 7.0       # skip windows that straddle video boundaries
ANIM_FPS = 15
FRAME_DELAY_MS = int(1000 / ANIM_FPS)

# Display geometry
DISP_W, DISP_H = 1400, 800
MID_W, MID_H = 400, 500     # midline animation panel
RIGHT_X = MID_W              # right column starts here
RIGHT_W = DISP_W - MID_W     # = 1000
KYMO_H = 220
SPEED_H = 130
BLEN_H = 100
HUD_H = DISP_H - MID_H      # bottom HUD bar spans full width
# Right-column y offsets
KYMO_Y = 0
SPEED_Y = KYMO_H
BLEN_Y = KYMO_H + SPEED_H
# Verify layout fits
assert MID_H + HUD_H == DISP_H, "Layout height mismatch"
assert KYMO_H + SPEED_H + BLEN_H <= MID_H, "Right column exceeds midline panel height"

# Color palette
COL_BG       = (30, 30, 30)
COL_PANEL_BG = (40, 40, 40)
COL_TEXT      = (220, 220, 220)
COL_DIM       = (130, 130, 130)
COL_HEAD     = (60, 60, 255)      # red in BGR
COL_TAIL     = (255, 120, 60)     # blue-ish in BGR
COL_GHOST    = (80, 80, 80)
COL_MARKER   = (0, 230, 230)      # yellow-ish line marker
COL_GRID     = (60, 60, 60)
COL_ACCENT   = (0, 200, 120)      # green accent
COL_BTN      = (70, 70, 70)
COL_BTN_HOV  = (90, 90, 90)

LABEL_COLORS = [
    (100, 200, 100),   # scrunching   - green
    (100, 180, 255),   # head_sweep   - orange (BGR)
    (255, 200, 100),   # gliding      - light blue
    (180, 180, 180),   # resting      - gray
    (80, 220, 255),    # turning      - yellow
    (200, 100, 255),   # reversing    - magenta
    (255, 150, 150),   # peristalsis  - light cyan
]

DEFAULT_BEHAVIORS = [
    "scrunching", "head_sweep", "gliding", "resting",
    "turning", "reversing", "peristalsis",
]

# ──────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────

@dataclass
class WindowData:
    """Data for a single labeling window."""
    midlines: np.ndarray       # (F, 20, 2)
    curvatures: np.ndarray     # (F, 16)
    speeds: np.ndarray         # (F,)
    angles: np.ndarray         # (F,)
    body_lengths: np.ndarray   # (F,)
    centroids: np.ndarray      # (F, 2) — pixel coords
    times: np.ndarray          # (F,) — seconds


@dataclass
class WindowInfo:
    """Manifest entry for one window."""
    session: str
    start_idx: int        # index into NPZ arrays
    end_idx: int
    start_time: float
    end_time: float
    activity_score: float
    window_id: str        # "session:start_idx"


@dataclass
class ClickRect:
    """Hit-test rectangle for mouse interaction."""
    x1: int
    y1: int
    x2: int
    y2: int
    action: str           # e.g. "toggle_0", "next", "prev", "pause", "custom"


# ──────────────────────────────────────────────────────────────────────
# 1. Session Data Loader
# ──────────────────────────────────────────────────────────────────────

class SessionData:
    """Loads and caches NPZ + CSV data for one session."""

    def __init__(self, session_name: str, data_dir: str):
        self.session = session_name
        self.data_dir = data_dir

        npz_path = os.path.join(data_dir, f"{session_name}_midlines.npz")
        csv_path = os.path.join(data_dir, f"{session_name}_tracks.csv")

        # --- Load NPZ ---
        npz = np.load(npz_path)
        self.midlines = npz["midlines"]            # (N, 20, 2)
        self.curvatures = npz["curvatures"]        # (N, 16)
        self.body_lengths_npz = npz["body_lengths"] # (N,)
        self.frame_indices = npz["frame_indices"]   # (N,)
        self.times_s = npz["times_s"]               # (N,)
        self.video_names = npz["video_names"]       # (N,)
        self.mm_per_px = float(npz["mm_per_px"])
        self.n_frames = len(self.midlines)

        # --- Load CSV ---
        self.csv_speeds = np.full(self.n_frames, np.nan, dtype=np.float32)
        self.csv_angles = np.full(self.n_frames, np.nan, dtype=np.float32)
        self.csv_centroids = np.full((self.n_frames, 2), np.nan, dtype=np.float32)
        self.csv_body_lengths = np.full(self.n_frames, np.nan, dtype=np.float32)

        # Build (video_name, frame)→npz_index lookup
        # Frame indices repeat per video, so must key by both
        key_to_npz = {}
        for i in range(self.n_frames):
            key = (str(self.video_names[i]), int(self.frame_indices[i]))
            key_to_npz[key] = i

        with open(csv_path, "r") as f:
            for line in f:
                if line.startswith("#") or line.startswith('"#'):
                    continue
                if line.strip().startswith("video_file"):
                    continue
                parts = line.strip().split(",")
                if len(parts) < 15:
                    continue
                try:
                    video_name = parts[0].strip()
                    frame_num = int(parts[1].strip())
                except (ValueError, IndexError):
                    continue
                npz_idx = key_to_npz.get((video_name, frame_num))
                if npz_idx is None:
                    continue
                # Parse fields
                try:
                    self.csv_centroids[npz_idx, 0] = float(parts[3].strip())  # centroid_x_px
                    self.csv_centroids[npz_idx, 1] = float(parts[4].strip())  # centroid_y_px
                    self.csv_angles[npz_idx] = float(parts[7].strip())        # body_angle_deg
                    spd = parts[10].strip()                                    # speed_px_s
                    if spd:
                        self.csv_speeds[npz_idx] = float(spd)
                    bl = parts[13].strip()                                     # body_length_px
                    if bl:
                        self.csv_body_lengths[npz_idx] = float(bl)
                except (ValueError, IndexError):
                    pass

        # Fill NaN centroids from midline means where possible
        nan_mask = np.isnan(self.csv_centroids[:, 0])
        if nan_mask.any():
            mid_means = self.midlines[nan_mask].mean(axis=1)  # (K, 2)
            self.csv_centroids[nan_mask] = mid_means

        # Fill NaN body lengths from NPZ
        nan_bl = np.isnan(self.csv_body_lengths)
        if nan_bl.any():
            self.csv_body_lengths[nan_bl] = self.body_lengths_npz[nan_bl]

    def get_window(self, start_idx: int, end_idx: int) -> WindowData:
        """Extract a WindowData slice."""
        sl = slice(start_idx, end_idx)
        speeds = self.csv_speeds[sl].copy()
        # Replace NaN speeds with 0
        speeds[np.isnan(speeds)] = 0.0
        angles = self.csv_angles[sl].copy()
        angles[np.isnan(angles)] = 0.0
        bl = self.csv_body_lengths[sl].copy()
        bl[np.isnan(bl)] = 0.0

        return WindowData(
            midlines=self.midlines[sl].copy(),
            curvatures=self.curvatures[sl].copy(),
            speeds=speeds,
            angles=angles,
            body_lengths=bl,
            centroids=self.csv_centroids[sl].copy(),
            times=self.times_s[sl].copy(),
        )


# ──────────────────────────────────────────────────────────────────────
# 2. Window Manifest Generator
# ──────────────────────────────────────────────────────────────────────

def discover_sessions(data_dir: str) -> List[str]:
    """Find all sessions that have both _tracks.csv and _midlines.npz."""
    npz_files = glob.glob(os.path.join(data_dir, "*_midlines.npz"))
    sessions = []
    for npz_path in sorted(npz_files):
        name = os.path.basename(npz_path).replace("_midlines.npz", "")
        csv_path = os.path.join(data_dir, f"{name}_tracks.csv")
        if os.path.exists(csv_path):
            sessions.append(name)
    return sessions


def build_manifest(sessions_data: Dict[str, SessionData]) -> List[WindowInfo]:
    """Generate 5-second windows with 1-second overlap, skip video gaps."""
    windows = []
    for session_name, sd in sorted(sessions_data.items()):
        n = sd.n_frames
        if n < 4:
            continue
        times = sd.times_s
        videos = sd.video_names

        # Determine stride in index space: find typical dt
        dt = np.median(np.diff(times[:min(100, n)]))
        if dt <= 0:
            dt = 0.3  # fallback
        win_len = max(4, int(round(WINDOW_SEC / dt)))
        stride = max(1, int(round(STRIDE_SEC / dt)))

        start = 0
        while start + win_len <= n:
            end = start + win_len
            t_start = float(times[start])
            t_end = float(times[end - 1])

            # Check for video boundary gap
            dts = np.diff(times[start:end])
            if np.any(dts > GAP_THRESH_SEC):
                start += stride
                continue

            # Activity score: curvature var + speed var + body_length var
            curv = sd.curvatures[start:end]
            spd = sd.csv_speeds[start:end]
            bl = sd.csv_body_lengths[start:end]
            curv_var = float(np.nanvar(curv)) if curv.size and not np.all(np.isnan(curv)) else 0.0
            spd_var = float(np.nanvar(spd)) if spd.size and not np.all(np.isnan(spd)) else 0.0
            bl_var = float(np.nanvar(bl)) if bl.size and not np.all(np.isnan(bl)) else 0.0
            # Normalize: curvature is ~0-1 rad, speed ~0-50 px/s, bl ~15-25 px
            activity = curv_var * 100 + spd_var * 0.1 + bl_var * 0.5

            wid = f"{session_name}:{start}"
            windows.append(WindowInfo(
                session=session_name,
                start_idx=start,
                end_idx=end,
                start_time=t_start,
                end_time=t_end,
                activity_score=activity,
                window_id=wid,
            ))
            start += stride

    return windows


def save_manifest(windows: List[WindowInfo], path: str):
    data = []
    for w in windows:
        data.append({
            "session": w.session,
            "start_idx": w.start_idx,
            "end_idx": w.end_idx,
            "start_time": w.start_time,
            "end_time": w.end_time,
            "activity_score": w.activity_score,
            "window_id": w.window_id,
        })
    with open(path, "w") as f:
        json.dump(data, f, indent=1)


def load_manifest(path: str) -> List[WindowInfo]:
    with open(path, "r") as f:
        data = json.load(f)
    return [WindowInfo(**d) for d in data]


# ──────────────────────────────────────────────────────────────────────
# 3. Label Manager
# ──────────────────────────────────────────────────────────────────────

class LabelManager:
    """Persists behavior labels and custom behaviors to JSON."""

    def __init__(self, path: str):
        self.path = path
        self.behaviors: List[str] = list(DEFAULT_BEHAVIORS)
        self.labels: Dict[str, List[str]] = {}   # window_id → [behavior, ...]
        self.last_position: int = 0
        self.sort_mode: str = "most_active"
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        with open(self.path, "r") as f:
            data = json.load(f)
        self.behaviors = data.get("behaviors", list(DEFAULT_BEHAVIORS))
        self.labels = data.get("labels", {})
        self.last_position = data.get("last_position", 0)
        self.sort_mode = data.get("sort_mode", "most_active")

    def save(self):
        data = {
            "behaviors": self.behaviors,
            "labels": self.labels,
            "last_position": self.last_position,
            "sort_mode": self.sort_mode,
            "n_labeled": sum(1 for v in self.labels.values() if v),
            "n_total": len(self.labels),
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=1)

    def get_labels(self, window_id: str) -> List[str]:
        return list(self.labels.get(window_id, []))

    def set_labels(self, window_id: str, labels: List[str]):
        self.labels[window_id] = labels

    def toggle_label(self, window_id: str, behavior: str):
        current = self.get_labels(window_id)
        if behavior in current:
            current.remove(behavior)
        else:
            current.append(behavior)
        self.set_labels(window_id, current)

    def add_custom_behavior(self, name: str):
        name = name.strip().lower().replace(" ", "_")
        if name and name not in self.behaviors:
            self.behaviors.append(name)

    def is_labeled(self, window_id: str) -> bool:
        return bool(self.labels.get(window_id))

    def label_color(self, idx: int) -> Tuple[int, int, int]:
        if idx < len(LABEL_COLORS):
            return LABEL_COLORS[idx]
        # Generate deterministic color for custom behaviors
        np.random.seed(idx * 31 + 7)
        return tuple(int(x) for x in np.random.randint(80, 240, 3))


# ──────────────────────────────────────────────────────────────────────
# 4. Composite Renderer
# ──────────────────────────────────────────────────────────────────────

class Renderer:
    """Draws the 1400×800 composite display."""

    def __init__(self):
        self.click_rects: List[ClickRect] = []
        # Build blue-white-red LUT for kymograph
        self.kymo_lut = self._build_bwr_lut()

    @staticmethod
    def _build_bwr_lut() -> np.ndarray:
        """Blue-white-red colormap, 256 entries, BGR."""
        lut = np.zeros((256, 3), dtype=np.uint8)
        for i in range(128):
            t = i / 127.0
            # blue → white
            lut[i] = [255, int(255 * t), int(255 * t)]  # BGR: blue channel full
        for i in range(128, 256):
            t = (i - 128) / 127.0
            # white → red
            lut[i] = [int(255 * (1 - t)), int(255 * (1 - t)), 255]  # BGR
        return lut

    def render(self, wd: WindowData, anim_frame: int, labels: List[str],
               behaviors: List[str], info: dict, paused: bool) -> np.ndarray:
        """Render full composite frame. Returns BGR image."""
        self.click_rects.clear()
        canvas = np.full((DISP_H, DISP_W, 3), COL_BG, dtype=np.uint8)
        n_frames = len(wd.midlines)
        anim_frame = anim_frame % max(1, n_frames)

        # --- Midline animation panel (left) ---
        self._draw_midline_panel(canvas, wd, anim_frame)

        # --- Right column panels ---
        self._draw_kymograph(canvas, wd, anim_frame)
        self._draw_speed_angle(canvas, wd, anim_frame)
        self._draw_body_length(canvas, wd, anim_frame)

        # --- HUD bar (bottom) ---
        self._draw_hud(canvas, labels, behaviors, info, paused)

        return canvas

    def _draw_midline_panel(self, canvas, wd: WindowData, anim_frame: int):
        """Animate 20-point stick figure centered on window mean centroid."""
        x0, y0 = 0, 0
        panel = canvas[y0:y0 + MID_H, x0:x0 + MID_W]
        panel[:] = COL_PANEL_BG

        n_frames = len(wd.midlines)
        if n_frames == 0:
            return

        # Center on mean centroid of the window
        mean_centroid = np.nanmean(wd.centroids, axis=0)  # (2,)
        if np.any(np.isnan(mean_centroid)):
            mean_centroid = wd.midlines.reshape(-1, 2).mean(axis=0)

        cx, cy = mean_centroid
        scale = 3.0   # zoom factor
        ox, oy = MID_W // 2, MID_H // 2  # panel center

        def to_panel(pts):
            """Transform pixel coords to panel coords."""
            return ((pts - [cx, cy]) * scale + [ox, oy]).astype(np.int32)

        # Ghost trail (3 prior frames)
        for gi in range(3, 0, -1):
            gf = (anim_frame - gi) % n_frames
            ghost_pts = to_panel(wd.midlines[gf])
            alpha = 0.15 + 0.1 * (3 - gi)
            ghost_col = tuple(int(c * alpha) for c in (200, 200, 200))
            for j in range(len(ghost_pts) - 1):
                cv2.line(panel, tuple(ghost_pts[j]), tuple(ghost_pts[j + 1]),
                         ghost_col, 1, cv2.LINE_AA)

        # Current frame midline
        midline = wd.midlines[anim_frame]
        pts = to_panel(midline)
        n_seg = len(pts) - 1
        curvs = wd.curvatures[anim_frame] if len(wd.curvatures) > anim_frame else np.zeros(16)

        # Draw segments colored by curvature
        for j in range(n_seg):
            # Map curvature to color intensity
            if j < len(curvs):
                c = curvs[j]
                # Normalize: negative = blue, positive = red, zero = white
                c_clamp = np.clip(c * 5.0, -1, 1)
                if c_clamp >= 0:
                    col = (int(255 * (1 - c_clamp)), int(255 * (1 - c_clamp)), 255)
                else:
                    col = (255, int(255 * (1 + c_clamp)), int(255 * (1 + c_clamp)))
            else:
                col = COL_TEXT
            cv2.line(panel, tuple(pts[j]), tuple(pts[j + 1]), col, 2, cv2.LINE_AA)

        # Head = red dot, tail = blue dot
        cv2.circle(panel, tuple(pts[0]), 5, COL_HEAD, -1, cv2.LINE_AA)
        cv2.circle(panel, tuple(pts[-1]), 4, COL_TAIL, -1, cv2.LINE_AA)

        # Frame counter
        cv2.putText(panel, f"Frame {anim_frame + 1}/{n_frames}",
                    (10, MID_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_DIM, 1)

        # Time display
        if anim_frame < len(wd.times):
            t = wd.times[anim_frame] - wd.times[0]
            cv2.putText(panel, f"t={t:.1f}s",
                        (10, MID_H - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_DIM, 1)

        # Scale bar (1mm)
        if len(wd.midlines) > 0:
            # Approximate mm_per_px from body length (typical worm ~5-8mm, ~50-80px)
            bar_px = int(1.0 / 0.107 * scale)  # 1mm in panel pixels
            bx = MID_W - 20 - bar_px
            by = MID_H - 20
            cv2.line(panel, (bx, by), (bx + bar_px, by), COL_TEXT, 2)
            cv2.putText(panel, "1mm", (bx, by - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, COL_DIM, 1)

    def _draw_kymograph(self, canvas, wd: WindowData, anim_frame: int):
        """Curvature kymograph: x=time, y=head→tail."""
        x0, y0 = RIGHT_X, KYMO_Y
        panel = canvas[y0:y0 + KYMO_H, x0:x0 + RIGHT_W]
        panel[:] = COL_PANEL_BG

        n_frames = len(wd.curvatures)
        n_seg = wd.curvatures.shape[1] if n_frames > 0 else 16
        if n_frames < 2:
            return

        margin_l, margin_r, margin_t, margin_b = 50, 20, 25, 25
        plot_w = RIGHT_W - margin_l - margin_r
        plot_h = KYMO_H - margin_t - margin_b

        # Build heatmap (n_seg × n_frames), then resize
        heatmap = np.nan_to_num(wd.curvatures.T, nan=0.0)  # (16, n_frames)
        # Normalize to 0-255
        vmin, vmax = -0.3, 0.3
        heatmap_norm = np.clip((heatmap - vmin) / (vmax - vmin), 0, 1)
        heatmap_u8 = (heatmap_norm * 255).astype(np.uint8)

        # Apply LUT
        heatmap_bgr = np.zeros((*heatmap_u8.shape, 3), dtype=np.uint8)
        for c in range(3):
            heatmap_bgr[:, :, c] = self.kymo_lut[heatmap_u8, c]

        # Resize to fill plot area
        heatmap_resized = cv2.resize(heatmap_bgr, (plot_w, plot_h),
                                      interpolation=cv2.INTER_NEAREST)
        panel[margin_t:margin_t + plot_h, margin_l:margin_l + plot_w] = heatmap_resized

        # Current frame marker (vertical yellow line)
        fx = margin_l + int(anim_frame / max(1, n_frames - 1) * (plot_w - 1))
        cv2.line(panel, (fx, margin_t), (fx, margin_t + plot_h), COL_MARKER, 1)

        # Labels
        cv2.putText(panel, "Curvature Kymograph", (margin_l, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_TEXT, 1)
        cv2.putText(panel, "Head", (5, margin_t + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, COL_HEAD, 1)
        cv2.putText(panel, "Tail", (5, margin_t + plot_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, COL_TAIL, 1)

    def _draw_speed_angle(self, canvas, wd: WindowData, anim_frame: int):
        """Speed + angle line plots."""
        x0, y0 = RIGHT_X, SPEED_Y
        panel = canvas[y0:y0 + SPEED_H, x0:x0 + RIGHT_W]
        panel[:] = COL_PANEL_BG

        n = len(wd.speeds)
        if n < 2:
            return

        margin_l, margin_r, margin_t, margin_b = 50, 20, 20, 15
        plot_w = RIGHT_W - margin_l - margin_r
        plot_h = SPEED_H - margin_t - margin_b

        # Draw border
        cv2.rectangle(panel, (margin_l, margin_t),
                      (margin_l + plot_w, margin_t + plot_h), COL_GRID, 1)

        # Speed trace (green)
        speeds = wd.speeds
        s_min, s_max = 0, max(float(np.nanmax(speeds)), 1.0)
        xs = np.linspace(0, plot_w - 1, n).astype(np.int32) + margin_l
        ys_spd = margin_t + plot_h - ((speeds - s_min) / (s_max - s_min) * (plot_h - 4) + 2).astype(np.int32)
        ys_spd = np.clip(ys_spd, margin_t, margin_t + plot_h)
        pts_spd = np.stack([xs, ys_spd], axis=1)
        cv2.polylines(panel, [pts_spd], False, (100, 220, 100), 1, cv2.LINE_AA)

        # Angle trace (blue) — normalize to 0-360
        angles = wd.angles % 360
        a_min, a_max = 0, 360
        ys_ang = margin_t + plot_h - ((angles - a_min) / (a_max - a_min) * (plot_h - 4) + 2).astype(np.int32)
        ys_ang = np.clip(ys_ang, margin_t, margin_t + plot_h)
        pts_ang = np.stack([xs, ys_ang], axis=1)
        cv2.polylines(panel, [pts_ang], False, (255, 160, 80), 1, cv2.LINE_AA)

        # Current frame marker
        fx = margin_l + int(anim_frame / max(1, n - 1) * (plot_w - 1))
        cv2.line(panel, (fx, margin_t), (fx, margin_t + plot_h), COL_MARKER, 1)

        # Labels
        cv2.putText(panel, "Speed", (margin_l, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 220, 100), 1)
        cv2.putText(panel, "Angle", (margin_l + 55, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 160, 80), 1)

        # Y-axis labels
        cv2.putText(panel, f"{s_max:.0f}", (5, margin_t + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 220, 100), 1)
        cv2.putText(panel, "0", (5, margin_t + plot_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, COL_DIM, 1)

    def _draw_body_length(self, canvas, wd: WindowData, anim_frame: int):
        """Body length trace — key scrunching indicator."""
        x0, y0 = RIGHT_X, BLEN_Y
        panel = canvas[y0:y0 + BLEN_H, x0:x0 + RIGHT_W]
        panel[:] = COL_PANEL_BG

        n = len(wd.body_lengths)
        if n < 2:
            return

        margin_l, margin_r, margin_t, margin_b = 50, 20, 18, 12
        plot_w = RIGHT_W - margin_l - margin_r
        plot_h = BLEN_H - margin_t - margin_b

        cv2.rectangle(panel, (margin_l, margin_t),
                      (margin_l + plot_w, margin_t + plot_h), COL_GRID, 1)

        bl = wd.body_lengths
        bl_min = max(float(np.nanmin(bl)) - 2, 0)
        bl_max = float(np.nanmax(bl)) + 2
        bl_range = bl_max - bl_min if bl_max > bl_min else 1.0

        xs = np.linspace(0, plot_w - 1, n).astype(np.int32) + margin_l
        ys = margin_t + plot_h - ((bl - bl_min) / bl_range * (plot_h - 4) + 2).astype(np.int32)
        ys = np.clip(ys, margin_t, margin_t + plot_h)
        pts = np.stack([xs, ys], axis=1)
        cv2.polylines(panel, [pts], False, (200, 180, 100), 1, cv2.LINE_AA)

        # Current frame marker
        fx = margin_l + int(anim_frame / max(1, n - 1) * (plot_w - 1))
        cv2.line(panel, (fx, margin_t), (fx, margin_t + plot_h), COL_MARKER, 1)

        cv2.putText(panel, "Body Length (px)", (margin_l, 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 180, 100), 1)
        cv2.putText(panel, f"{bl_max:.0f}", (5, margin_t + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, COL_DIM, 1)
        cv2.putText(panel, f"{bl_min:.0f}", (5, margin_t + plot_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, COL_DIM, 1)

    def _draw_hud(self, canvas, labels: List[str], behaviors: List[str],
                  info: dict, paused: bool):
        """Bottom HUD: session info, behavior label tags, progress, shortcuts."""
        y0 = MID_H
        hud = canvas[y0:DISP_H, 0:DISP_W]
        hud[:] = (25, 25, 25)

        # Divider line
        cv2.line(canvas, (0, y0), (DISP_W, y0), COL_GRID, 1)

        # --- Row 1: Session info + progress ---
        row1_y = y0 + 25
        session = info.get("session", "")
        pos = info.get("position", 0)
        total = info.get("total", 0)
        n_labeled = info.get("n_labeled", 0)
        sort_mode = info.get("sort_mode", "most_active")
        t_start = info.get("t_start", 0)
        t_end = info.get("t_end", 0)
        activity = info.get("activity", 0)

        cv2.putText(canvas, f"{session}", (10, row1_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_TEXT, 1)
        cv2.putText(canvas, f"t={t_start:.1f}-{t_end:.1f}s  activity={activity:.2f}",
                    (10, row1_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_DIM, 1)

        # Progress
        prog_text = f"Clip {pos + 1}/{total}  |  Labeled: {n_labeled}/{total}  |  Sort: {sort_mode}"
        cv2.putText(canvas, prog_text, (420, row1_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_DIM, 1)

        # Progress bar
        bar_x, bar_y, bar_w, bar_h = 420, row1_y + 8, 400, 12
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), COL_GRID, 1)
        if total > 0:
            fill_w = int(bar_w * n_labeled / total)
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                          COL_ACCENT, -1)

        # Pause indicator
        if paused:
            cv2.putText(canvas, "PAUSED", (DISP_W - 120, row1_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # --- Row 2: Behavior label tags (clickable) ---
        row2_y = row1_y + 50
        tag_x = 10
        for i, beh in enumerate(behaviors):
            key = str(i + 1) if i < 9 else str((i + 1) % 10)
            if i >= 9:
                key = str(0) if i == 9 else "+"
            tag_text = f"[{key}] {beh}"
            text_size = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            tag_w = text_size[0] + 16
            tag_h = 28
            ty = row2_y

            # Check if this behavior is currently selected
            selected = beh in labels
            col = self._label_color_for(i)

            if selected:
                cv2.rectangle(canvas, (tag_x, ty), (tag_x + tag_w, ty + tag_h), col, -1)
                text_col = (0, 0, 0)
            else:
                cv2.rectangle(canvas, (tag_x, ty), (tag_x + tag_w, ty + tag_h), col, 1)
                text_col = col

            cv2.putText(canvas, tag_text, (tag_x + 8, ty + 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_col, 1)

            # Register click rect
            self.click_rects.append(ClickRect(tag_x, ty, tag_x + tag_w, ty + tag_h,
                                              f"toggle_{i}"))

            tag_x += tag_w + 8
            if tag_x > DISP_W - 150:
                tag_x = 10
                row2_y += tag_h + 5

        # [+ Custom] button
        cust_x = tag_x + 10
        cust_y = row2_y if tag_x < DISP_W - 150 else row2_y
        cust_w, cust_h = 100, 28
        cv2.rectangle(canvas, (cust_x, cust_y), (cust_x + cust_w, cust_y + cust_h),
                      COL_ACCENT, 1)
        cv2.putText(canvas, "[c] + Custom", (cust_x + 5, cust_y + 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_ACCENT, 1)
        self.click_rects.append(ClickRect(cust_x, cust_y, cust_x + cust_w,
                                          cust_y + cust_h, "custom"))

        # --- Row 3: Nav buttons + keyboard shortcuts ---
        row3_y = DISP_H - 80

        # Nav buttons
        btn_specs = [
            ("[< Prev]", "prev", 10),
            ("[Play/Pause]", "pause", 100),
            ("[Next >]", "next", 230),
        ]
        for text, action, bx in btn_specs:
            tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0] + 16
            bh = 28
            cv2.rectangle(canvas, (bx, row3_y), (bx + tw, row3_y + bh), COL_BTN, -1)
            cv2.rectangle(canvas, (bx, row3_y), (bx + tw, row3_y + bh), COL_DIM, 1)
            cv2.putText(canvas, text, (bx + 8, row3_y + 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_TEXT, 1)
            self.click_rects.append(ClickRect(bx, row3_y, bx + tw, row3_y + bh, action))

        # Shortcuts reference
        shortcuts = ("Enter/d:accept  a:back  f/b:+/-10  u:unlabeled  "
                     "s:sort  r:reset  Space:pause  q:quit")
        cv2.putText(canvas, shortcuts, (10, DISP_H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COL_DIM, 1)

        # Arrow keys hint when paused
        if paused:
            cv2.putText(canvas, "Arrow keys: step frames", (10, DISP_H - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, COL_DIM, 1)

    def _label_color_for(self, idx: int) -> Tuple[int, int, int]:
        if idx < len(LABEL_COLORS):
            return LABEL_COLORS[idx]
        np.random.seed(idx * 31 + 7)
        return tuple(int(x) for x in np.random.randint(80, 240, 3))

    def hit_test(self, mx: int, my: int) -> Optional[str]:
        """Return action string if (mx, my) hits a clickable rect, else None."""
        for r in self.click_rects:
            if r.x1 <= mx <= r.x2 and r.y1 <= my <= r.y2:
                return r.action
        return None


# ──────────────────────────────────────────────────────────────────────
# 5. Interactive Labeling Loop
# ──────────────────────────────────────────────────────────────────────

def _custom_behavior_dialog() -> Optional[str]:
    """Pop up a tkinter text entry for custom behavior name."""
    try:
        import tkinter as tk
        from tkinter import simpledialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        name = simpledialog.askstring("Custom Behavior",
                                       "Enter behavior name:",
                                       parent=root)
        root.destroy()
        return name
    except Exception:
        return None


def sort_windows(windows: List[WindowInfo], mode: str,
                 label_mgr: LabelManager) -> List[int]:
    """Return ordered indices into windows list based on sort mode."""
    if mode == "most_active":
        return sorted(range(len(windows)),
                      key=lambda i: windows[i].activity_score, reverse=True)
    elif mode == "sequential":
        return list(range(len(windows)))
    elif mode == "unlabeled":
        unlabeled = [i for i in range(len(windows))
                     if not label_mgr.is_labeled(windows[i].window_id)]
        labeled = [i for i in range(len(windows))
                   if label_mgr.is_labeled(windows[i].window_id)]
        # Within each group, sort by activity
        unlabeled.sort(key=lambda i: windows[i].activity_score, reverse=True)
        labeled.sort(key=lambda i: windows[i].activity_score, reverse=True)
        return unlabeled + labeled
    return list(range(len(windows)))


SORT_MODES = ["most_active", "sequential", "unlabeled"]


def run_labeler(data_dir: str, sort_mode: Optional[str] = None):
    """Main entry point: build manifest, load data, run interactive loop."""
    manifest_path = os.path.join(data_dir, "behavior_manifest.json")
    labels_path = os.path.join(data_dir, "behavior_labels.json")

    # --- Discover sessions and load data ---
    print("Discovering sessions...")
    session_names = discover_sessions(data_dir)
    if not session_names:
        print(f"No sessions found in {data_dir}")
        sys.exit(1)
    print(f"Found {len(session_names)} sessions: {', '.join(session_names)}")

    print("Loading session data...")
    sessions_data: Dict[str, SessionData] = {}
    for sn in session_names:
        print(f"  Loading {sn}...")
        sessions_data[sn] = SessionData(sn, data_dir)

    # --- Build or load manifest ---
    if os.path.exists(manifest_path):
        print(f"Loading cached manifest from {manifest_path}")
        windows = load_manifest(manifest_path)
        print(f"  {len(windows)} windows loaded")
    else:
        print("Building window manifest...")
        windows = build_manifest(sessions_data)
        save_manifest(windows, manifest_path)
        print(f"  {len(windows)} windows generated, saved to {manifest_path}")

    if not windows:
        print("No windows generated. Check data.")
        sys.exit(1)

    # --- Label manager ---
    label_mgr = LabelManager(labels_path)
    if sort_mode:
        label_mgr.sort_mode = sort_mode

    # --- Sort windows ---
    order = sort_windows(windows, label_mgr.sort_mode, label_mgr)

    # --- Resume position ---
    pos = min(label_mgr.last_position, len(order) - 1)

    # --- Renderer ---
    renderer = Renderer()

    # --- OpenCV window ---
    win_name = "Planarian Behavior Labeler"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    mouse_click = [None]  # mutable container for callback

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_click[0] = (x, y)

    cv2.setMouseCallback(win_name, on_mouse)

    # --- Animation state ---
    anim_frame = 0
    paused = False
    running = True

    print(f"\nLabeler ready. {len(windows)} clips, {label_mgr.sort_mode} sort.")
    print(f"Resuming at position {pos + 1}.")
    print("Press 'q' or Esc to save and quit.\n")

    while running:
        # Get current window
        win_idx = order[pos]
        w = windows[win_idx]
        sd = sessions_data.get(w.session)
        if sd is None:
            pos = (pos + 1) % len(order)
            continue

        wd = sd.get_window(w.start_idx, w.end_idx)
        n_data = len(wd.midlines)
        current_labels = label_mgr.get_labels(w.window_id)

        # Count labeled
        n_labeled = sum(1 for wi in windows if label_mgr.is_labeled(wi.window_id))

        info = {
            "session": w.session,
            "position": pos,
            "total": len(windows),
            "n_labeled": n_labeled,
            "sort_mode": label_mgr.sort_mode,
            "t_start": w.start_time,
            "t_end": w.end_time,
            "activity": w.activity_score,
        }

        # Inner animation loop for this clip
        clip_done = False
        while not clip_done and running:
            # Render
            frame = renderer.render(wd, anim_frame, current_labels,
                                    label_mgr.behaviors, info, paused)
            cv2.imshow(win_name, frame)

            # Handle input (non-blocking wait)
            key = cv2.waitKey(FRAME_DELAY_MS) & 0xFF

            # Process mouse click
            if mouse_click[0] is not None:
                mx, my = mouse_click[0]
                mouse_click[0] = None
                action = renderer.hit_test(mx, my)
                if action:
                    if action.startswith("toggle_"):
                        idx = int(action.split("_")[1])
                        if idx < len(label_mgr.behaviors):
                            beh = label_mgr.behaviors[idx]
                            label_mgr.toggle_label(w.window_id, beh)
                            current_labels = label_mgr.get_labels(w.window_id)
                    elif action == "next":
                        label_mgr.set_labels(w.window_id, current_labels)
                        label_mgr.last_position = pos
                        label_mgr.save()
                        pos = min(pos + 1, len(order) - 1)
                        anim_frame = 0
                        clip_done = True
                    elif action == "prev":
                        label_mgr.set_labels(w.window_id, current_labels)
                        label_mgr.last_position = pos
                        label_mgr.save()
                        pos = max(pos - 1, 0)
                        anim_frame = 0
                        clip_done = True
                    elif action == "pause":
                        paused = not paused
                    elif action == "custom":
                        name = _custom_behavior_dialog()
                        if name:
                            label_mgr.add_custom_behavior(name)

            # Process keyboard
            if key == 255:
                # No key pressed — advance animation
                if not paused:
                    anim_frame = (anim_frame + 1) % max(1, n_data)
                continue

            if key == ord('q') or key == 27:  # q or Esc
                label_mgr.set_labels(w.window_id, current_labels)
                label_mgr.last_position = pos
                label_mgr.save()
                running = False
                break

            elif key in (13, ord('d')):  # Enter or d — accept and advance
                label_mgr.set_labels(w.window_id, current_labels)
                label_mgr.last_position = pos
                label_mgr.save()
                pos = min(pos + 1, len(order) - 1)
                anim_frame = 0
                clip_done = True

            elif key == ord('a'):  # back
                label_mgr.set_labels(w.window_id, current_labels)
                label_mgr.last_position = pos
                label_mgr.save()
                pos = max(pos - 1, 0)
                anim_frame = 0
                clip_done = True

            elif key == ord('f'):  # skip forward 10
                label_mgr.set_labels(w.window_id, current_labels)
                label_mgr.last_position = pos
                label_mgr.save()
                pos = min(pos + 10, len(order) - 1)
                anim_frame = 0
                clip_done = True

            elif key == ord('b'):  # skip back 10
                label_mgr.set_labels(w.window_id, current_labels)
                label_mgr.last_position = pos
                label_mgr.save()
                pos = max(pos - 10, 0)
                anim_frame = 0
                clip_done = True

            elif key == ord('u'):  # jump to next unlabeled
                label_mgr.set_labels(w.window_id, current_labels)
                label_mgr.save()
                found = False
                for search_pos in range(pos + 1, len(order)):
                    if not label_mgr.is_labeled(windows[order[search_pos]].window_id):
                        pos = search_pos
                        found = True
                        break
                if not found:
                    # Wrap around
                    for search_pos in range(0, pos):
                        if not label_mgr.is_labeled(windows[order[search_pos]].window_id):
                            pos = search_pos
                            found = True
                            break
                if not found:
                    print("All clips labeled!")
                label_mgr.last_position = pos
                label_mgr.save()
                anim_frame = 0
                clip_done = True

            elif key == ord('s'):  # cycle sort mode
                label_mgr.set_labels(w.window_id, current_labels)
                cur_idx = SORT_MODES.index(label_mgr.sort_mode) if label_mgr.sort_mode in SORT_MODES else 0
                label_mgr.sort_mode = SORT_MODES[(cur_idx + 1) % len(SORT_MODES)]
                order = sort_windows(windows, label_mgr.sort_mode, label_mgr)
                # Find current window in new order
                try:
                    pos = order.index(win_idx)
                except ValueError:
                    pos = 0
                label_mgr.last_position = pos
                label_mgr.save()
                anim_frame = 0
                clip_done = True

            elif key == ord('r'):  # reset labels for current clip
                label_mgr.set_labels(w.window_id, [])
                current_labels = []

            elif key == ord('c'):  # custom behavior
                name = _custom_behavior_dialog()
                if name:
                    label_mgr.add_custom_behavior(name)

            elif key == ord(' '):  # space — pause/resume
                paused = not paused

            elif key == 81 and paused:  # left arrow (macOS)
                anim_frame = (anim_frame - 1) % max(1, n_data)
            elif key == 83 and paused:  # right arrow (macOS)
                anim_frame = (anim_frame + 1) % max(1, n_data)
            elif key == 2 and paused:   # left arrow (Linux)
                anim_frame = (anim_frame - 1) % max(1, n_data)
            elif key == 3 and paused:   # right arrow (Linux)
                anim_frame = (anim_frame + 1) % max(1, n_data)

            # Number keys 1-9, 0 for behavior toggles
            elif ord('1') <= key <= ord('9'):
                idx = key - ord('1')  # 0-indexed
                if idx < len(label_mgr.behaviors):
                    beh = label_mgr.behaviors[idx]
                    label_mgr.toggle_label(w.window_id, beh)
                    current_labels = label_mgr.get_labels(w.window_id)
            elif key == ord('0'):
                idx = 9  # 10th behavior
                if idx < len(label_mgr.behaviors):
                    beh = label_mgr.behaviors[idx]
                    label_mgr.toggle_label(w.window_id, beh)
                    current_labels = label_mgr.get_labels(w.window_id)

            # Advance animation if not paused
            if not paused:
                anim_frame = (anim_frame + 1) % max(1, n_data)

        # Check if window was closed
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            label_mgr.set_labels(w.window_id, current_labels)
            label_mgr.last_position = pos
            label_mgr.save()
            running = False

    cv2.destroyAllWindows()
    print(f"\nLabels saved to {labels_path}")
    n_labeled = sum(1 for wi in windows if label_mgr.is_labeled(wi.window_id))
    print(f"Total labeled: {n_labeled}/{len(windows)} clips")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Interactive behavior labeling tool for planarian tracking data.")
    parser.add_argument("--data_dir", default="/tmp/tracker_output",
                        help="Directory containing *_tracks.csv, *_midlines.npz, *_calibration.json")
    parser.add_argument("--sort", choices=SORT_MODES, default=None,
                        help="Initial sort mode (default: resume previous or most_active)")
    parser.add_argument("--rebuild_manifest", action="store_true",
                        help="Force rebuild of window manifest")
    args = parser.parse_args()

    if args.rebuild_manifest:
        manifest_path = os.path.join(args.data_dir, "behavior_manifest.json")
        if os.path.exists(manifest_path):
            os.remove(manifest_path)
            print(f"Removed {manifest_path}, will rebuild.")

    run_labeler(args.data_dir, sort_mode=args.sort)


if __name__ == "__main__":
    main()
