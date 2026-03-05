#!/usr/bin/env python3
"""
open_dish_tracker.py — Fully automated planarian worm tracker for OpenDishWork.

Headless, batch-capable pipeline that auto-detects the petri dish, calibrates
from the 1 cm grid overlay, and tracks the worm across chained one-minute MKV
recordings.  No interactive seeding required.

Usage:
  # Single session:
  python open_dish_tracker.py --data_root ../OpenDishWork --sessions Bubba_0001_021426

  # All sessions (default):
  python open_dish_tracker.py --data_root ../OpenDishWork

  # Custom parameters:
  python open_dish_tracker.py --data_root ../OpenDishWork --frame_skip 1 --min_area 100

Python 3.9+.  Requires: opencv-python, numpy, tqdm
"""

import os, json, math, argparse, glob, re, sys, time, warnings
from collections import deque
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm
import csv

# ──────────────────────────────────────────────────────────────────────
# Optional midline dependencies (scikit-image, scipy)
# ──────────────────────────────────────────────────────────────────────

_HAS_MIDLINE = False
try:
    from skimage.morphology import skeletonize as ski_skeletonize
    from scipy.ndimage import gaussian_filter1d
    _HAS_MIDLINE = True
except ImportError:
    pass

# ──────────────────────────────────────────────────────────────────────
# PyAV-backed VideoCapture (fallback when cv2 can't read MKV)
# ──────────────────────────────────────────────────────────────────────

_HAS_AV = False
try:
    import av
    _HAS_AV = True
except ImportError:
    pass


class AVVideoCapture:
    """Drop-in replacement for cv2.VideoCapture using PyAV (ffmpeg).

    Supports the subset of the cv2.VideoCapture API used by this tracker:
    isOpened, read, get, set (POS_FRAMES), release.
    """

    def __init__(self, path):
        self._path = path
        self._container = None
        self._stream = None
        self._fps = 10.0
        self._frame_count = 0
        self._width = 0
        self._height = 0
        self._current_frame = 0
        self._opened = False

        try:
            self._container = av.open(path)
            self._stream = self._container.streams.video[0]
            self._fps = float(self._stream.average_rate or 10)
            self._width = self._stream.width
            self._height = self._stream.height
            self._frame_count = self._stream.frames or int(
                float(self._container.duration or 0) / 1e6 * self._fps)
            self._stream.thread_type = "AUTO"
            self._opened = True
        except Exception:
            self._opened = False

    def isOpened(self):
        return self._opened

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self._frame_count
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self._width
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self._height
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return self._current_frame
        return 0

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            target = int(value)
            if target == self._current_frame:
                return True
            # Seek to the nearest keyframe at or before target
            target_sec = target / self._fps
            # Use time-based seek (microseconds) for reliability
            self._container.seek(int(target_sec * av.time_base), any_frame=False)
            # Decode forward until we reach the target frame index
            self._current_frame = 0
            for frame in self._container.decode(video=0):
                frame_idx = int(round(float(frame.pts * self._stream.time_base) * self._fps))
                self._current_frame = frame_idx
                if frame_idx >= target:
                    break
            self._current_frame = target
            return True
        return False

    def read(self):
        try:
            for frame in self._container.decode(video=0):
                bgr = frame.to_ndarray(format='bgr24')
                self._current_frame = int(round(
                    float(frame.pts * self._stream.time_base) * self._fps)) + 1
                return True, bgr
        except (av.EOFError, StopIteration, av.error.InvalidDataError):
            pass
        except Exception:
            pass
        return False, None

    def release(self):
        if self._container is not None:
            self._container.close()
            self._container = None
        self._opened = False


def open_video(path):
    """Open a video file, falling back to PyAV if cv2 can't read it."""
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        # Verify it can actually read a frame
        ok, _ = cap.read()
        if ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return cap
        cap.release()
    # Fallback to PyAV
    if _HAS_AV:
        avcap = AVVideoCapture(path)
        if avcap.isOpened():
            return avcap
    raise RuntimeError(f"Cannot open video: {path}")


# ──────────────────────────────────────────────────────────────────────
# Utility functions (adapted from batch_worm_tracker_chain.py)
# ──────────────────────────────────────────────────────────────────────

def to_gray01(bgr):
    """BGR image → float32 grayscale in [0, 1]."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def relative_darkness(gray):
    """Grayscale [0,1] → darkness metric (0 = bright, 1 = dark).

    Uses 2nd/98th percentile normalisation for robustness to outliers.
    """
    lo, hi = np.percentile(gray, 2.0), np.percentile(gray, 98.0)
    if hi <= lo:
        hi, lo = float(gray.max()), float(gray.min())
    norm = np.clip((gray - lo) / max(1e-6, (hi - lo)), 0, 1)
    return 1.0 - norm


def circle_mask(shape_hw, center_xy, radius_px):
    """Return uint8 mask (0/255) with a filled circle."""
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    if center_xy is not None and radius_px and radius_px > 0:
        cv2.circle(mask, (int(center_xy[0]), int(center_xy[1])),
                   int(radius_px), 255, -1)
    return mask


def largest_component(mask):
    """Largest connected component in a uint8 binary mask.

    Returns (centroid_xy, area_px, component_mask) or (None, None, None).
    """
    num, labels, stats, cents = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    if num <= 1:
        return None, None, None
    comp = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    area = int(stats[comp, cv2.CC_STAT_AREA])
    cx, cy = cents[comp]
    cmask = (labels == comp).astype(np.uint8) * 255
    return (float(cx), float(cy)), area, cmask


# ──────────────────────────────────────────────────────────────────────
# Stage 1 — Auto dish detection
# ──────────────────────────────────────────────────────────────────────

def auto_detect_dish(gray, shrink=0.95):
    """Detect the circular petri dish via Otsu threshold + circle fit.

    Parameters
    ----------
    gray : ndarray, float32 [0,1]
        Grayscale image (median background preferred).
    shrink : float
        Fraction of detected radius to use (avoids dish-rim artifacts).

    Returns
    -------
    center_xy : tuple (cx, cy)
    radius_px : float
    """
    # Otsu on 8-bit
    g8 = (gray * 255).astype(np.uint8)
    _, bw = cv2.threshold(g8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # The dish interior is bright → we want the large bright region
    # Invert so dish becomes foreground (255)
    # Otsu may give us either polarity — choose the one with larger bright blob
    for candidate in [bw, 255 - bw]:
        # Morphological cleanup
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        cleaned = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kern, iterations=3)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kern, iterations=2)

        cent, area, cmask = largest_component(cleaned)
        if cent is None or area < 0.1 * gray.size:
            continue

        # Fit a least-squares circle to the contour of the largest component
        contours, _ = cv2.findContours(cmask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        pts = np.vstack(contours).squeeze()
        if pts.ndim != 2 or len(pts) < 5:
            continue

        # Least-squares circle fit: minimise Σ(√((x-a)²+(y-b)²) - r)²
        # Algebraic fit via Kåsa method
        x = pts[:, 0].astype(np.float64)
        y = pts[:, 1].astype(np.float64)
        A = np.column_stack([x, y, np.ones_like(x)])
        b_vec = x**2 + y**2
        result, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
        cx = result[0] / 2.0
        cy = result[1] / 2.0
        r = math.sqrt(result[2] + cx**2 + cy**2)

        # Sanity: radius should be a significant fraction of frame half-width
        h, w = gray.shape
        if r < 0.15 * min(h, w) or r > 0.8 * max(h, w):
            continue

        return (float(cx), float(cy)), float(r * shrink)

    raise RuntimeError("Auto dish detection failed — could not find circular dish.")


# ──────────────────────────────────────────────────────────────────────
# Stage 2 — Auto grid calibration
# ──────────────────────────────────────────────────────────────────────

def auto_grid_calibration(gray, dish_center, dish_radius):
    """Detect grid lines with morphological blackhat + HoughLines.

    Returns
    -------
    mm_per_px : float
        Millimetres per pixel (grid squares are 10 mm).
    spacing_px : float
        Grid spacing in pixels.
    """
    h, w = gray.shape

    # Mask to dish interior
    dmask = circle_mask((h, w), dish_center, dish_radius * 0.90)
    dmask_bool = dmask > 0

    # Morphological blackhat isolates thin dark features (grid lines)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    g8 = (gray * 255).astype(np.uint8)
    blackhat = cv2.morphologyEx(g8, cv2.MORPH_BLACKHAT, kern)
    blackhat[~dmask_bool] = 0

    # Threshold the blackhat response
    _, bh_bw = cv2.threshold(blackhat, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Hough line detection
    lines = cv2.HoughLinesP(bh_bw, rho=1, theta=np.pi / 180, threshold=80,
                            minLineLength=int(dish_radius * 0.3),
                            maxLineGap=15)

    if lines is None or len(lines) < 4:
        # Retry with lower threshold
        lines = cv2.HoughLinesP(bh_bw, rho=1, theta=np.pi / 180, threshold=40,
                                minLineLength=int(dish_radius * 0.2),
                                maxLineGap=20)

    if lines is None or len(lines) < 4:
        warnings.warn("Grid calibration: few lines detected, using fallback.")
        return _fallback_grid_calibration(gray, dish_center, dish_radius)

    # Classify lines as horizontal or vertical based on angle
    horiz_intercepts = []  # y-intercepts of ~horizontal lines
    vert_intercepts = []   # x-intercepts of ~vertical lines

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))

        if angle < 20 or angle > 160:  # ~horizontal
            # y-intercept at x = dish_center_x
            if abs(x2 - x1) > 1:
                slope = (y2 - y1) / (x2 - x1)
                y_at_center = y1 + slope * (dish_center[0] - x1)
            else:
                y_at_center = (y1 + y2) / 2.0
            horiz_intercepts.append(y_at_center)
        elif 70 < angle < 110:  # ~vertical
            # x-intercept at y = dish_center_y
            if abs(y2 - y1) > 1:
                slope = (x2 - x1) / (y2 - y1)
                x_at_center = x1 + slope * (dish_center[1] - y1)
            else:
                x_at_center = (x1 + x2) / 2.0
            vert_intercepts.append(x_at_center)

    # Cluster nearby intercepts and compute spacings
    spacing_estimates = []
    for intercepts in [horiz_intercepts, vert_intercepts]:
        if len(intercepts) < 2:
            continue
        intercepts = sorted(intercepts)
        # Cluster lines within 10px of each other
        clusters = []
        cluster = [intercepts[0]]
        for val in intercepts[1:]:
            if val - cluster[-1] < 10:
                cluster.append(val)
            else:
                clusters.append(np.mean(cluster))
                cluster = [val]
        clusters.append(np.mean(cluster))

        if len(clusters) >= 2:
            diffs = np.diff(clusters)
            # Filter out spacings that are too small (sub-grid noise)
            min_spacing = dish_radius * 0.03
            diffs = diffs[diffs > min_spacing]
            if len(diffs) > 0:
                spacing_estimates.extend(diffs.tolist())

    if not spacing_estimates:
        warnings.warn("Grid calibration: could not compute spacing, using fallback.")
        return _fallback_grid_calibration(gray, dish_center, dish_radius)

    # Use median spacing — this is robust to merged or split lines
    spacing_px = float(np.median(spacing_estimates))

    # Grid squares are 10 mm
    mm_per_px = 10.0 / spacing_px

    # Sanity check: dish diameter in mm should be ~60-100mm for a standard petri dish
    dish_diam_mm = 2 * dish_radius * mm_per_px
    if dish_diam_mm < 20 or dish_diam_mm > 200:
        warnings.warn(f"Grid calibration gave dish diameter {dish_diam_mm:.1f} mm "
                      f"(spacing={spacing_px:.1f}px). Trying fallback.")
        return _fallback_grid_calibration(gray, dish_center, dish_radius)

    return mm_per_px, spacing_px


def _fallback_grid_calibration(gray, dish_center, dish_radius):
    """FFT-based fallback for grid spacing estimation."""
    h, w = gray.shape

    # Extract a horizontal strip through dish centre
    cy = int(dish_center[1])
    strip_h = max(1, int(dish_radius * 0.1))
    y1 = max(0, cy - strip_h)
    y2 = min(h, cy + strip_h)
    cx = int(dish_center[0])
    x1 = max(0, int(cx - dish_radius * 0.8))
    x2 = min(w, int(cx + dish_radius * 0.8))

    strip = gray[y1:y2, x1:x2].mean(axis=0)
    if len(strip) < 20:
        raise RuntimeError("Cannot calibrate grid: dish too small or not detected.")

    # FFT to find dominant frequency (grid spacing)
    strip_detrended = strip - np.mean(strip)
    fft = np.abs(np.fft.rfft(strip_detrended))
    freqs = np.fft.rfftfreq(len(strip_detrended), d=1.0)

    # Ignore DC and very low frequencies
    min_idx = max(2, int(len(fft) * 0.01))
    peak_idx = min_idx + np.argmax(fft[min_idx:])
    if freqs[peak_idx] <= 0:
        raise RuntimeError("FFT grid calibration failed.")

    spacing_px = 1.0 / freqs[peak_idx]
    mm_per_px = 10.0 / spacing_px

    # Sanity check
    dish_diam_mm = 2 * dish_radius * mm_per_px
    if dish_diam_mm < 20 or dish_diam_mm > 200:
        raise RuntimeError(
            f"Grid calibration failed: dish diameter = {dish_diam_mm:.1f} mm "
            f"(expected 60–100 mm for a standard petri dish).")

    return mm_per_px, spacing_px


# ──────────────────────────────────────────────────────────────────────
# Stage 3 — Background model (per video)
# ──────────────────────────────────────────────────────────────────────

def build_background(cap, dish_mask_bool, n_samples=60):
    """Build a per-video background model from evenly sampled frames.

    Returns
    -------
    bg_max : ndarray float32
        Per-pixel maximum brightness (worm-free background estimate).
        Since the worm is dark, the maximum brightness at each pixel
        represents what that pixel looks like without the worm.
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 600
    pos0 = cap.get(cv2.CAP_PROP_POS_FRAMES)

    # Evenly spaced sample indices
    indices = np.linspace(0, total - 1, min(n_samples, total), dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if not ok:
            continue
        g = to_gray01(bgr)
        frames.append(g)

    cap.set(cv2.CAP_PROP_POS_FRAMES, pos0)
    if not frames:
        raise RuntimeError("No frames sampled for background model.")

    stacked = np.stack(frames, axis=0)
    bg_max = np.max(stacked, axis=0).astype(np.float32)

    # Zero outside dish
    if dish_mask_bool is not None:
        bg_max[~dish_mask_bool] = 0

    return bg_max


def build_grid_baseline(video_path, seed_frame, dish_mask_bool, n_samples=20):
    """Build a grid baseline from worm-free frames before the seed frame.

    Averages frames from before the worm was placed in the dish.  This
    captures the static grid pattern which can then be subtracted from
    tracking frames so grid lines cancel out and only the worm remains.

    Parameters
    ----------
    video_path : str
        Path to the first video.
    seed_frame : int
        Frame index where the worm first appears.
    dish_mask_bool : bool array
        True inside the dish.
    n_samples : int
        Number of frames to average from the pre-worm period.

    Returns
    -------
    baseline : ndarray float32
        Mean grayscale of pre-worm frames (the grid pattern).
    """
    cap = open_video(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path} for grid baseline.")

    # Sample evenly from frames 0..seed_frame-1
    end = max(1, seed_frame)
    indices = np.linspace(0, end - 1, min(n_samples, end), dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if not ok:
            continue
        frames.append(to_gray01(bgr))

    cap.release()

    if not frames:
        raise RuntimeError("No pre-worm frames available for grid baseline.")

    baseline = np.mean(np.stack(frames, axis=0), axis=0).astype(np.float32)
    return baseline


def build_grid_baseline_median(video_path, dish_mask_bool, n_samples=30):
    """Build a grid baseline from a video that already contains the worm.

    Uses the *median* of sampled frames spread across the video.  Because
    the worm is small and moves around, at any given pixel the worm is only
    present in a few of the sampled frames.  The median naturally excludes
    it, yielding a clean grid-only background.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    dish_mask_bool : bool array
        True inside the dish.
    n_samples : int
        Number of frames to sample (more = cleaner, but slower).

    Returns
    -------
    baseline : ndarray float32
        Median grayscale of sampled frames (grid pattern, worm removed).
    """
    cap = open_video(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path} for median baseline.")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total < 10:
        cap.release()
        raise RuntimeError(f"Video too short ({total} frames) for median baseline.")

    indices = np.linspace(0, total - 1, min(n_samples, total), dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if not ok:
            continue
        frames.append(to_gray01(bgr))

    cap.release()

    if len(frames) < 5:
        raise RuntimeError("Too few readable frames for median baseline.")

    baseline = np.median(np.stack(frames, axis=0), axis=0).astype(np.float32)
    return baseline


# ──────────────────────────────────────────────────────────────────────
# Stage 4 — Worm detection (per frame)
# ──────────────────────────────────────────────────────────────────────

def detect_worm(gray, bg_max, dish_mask_bool, last_centroid,
                min_area, max_area, roi_px, max_jump_px,
                lost_count, prev_area=None, lost_threshold=30,
                motion_map=None, grid_baseline=None):
    """Detect the worm in a single frame.

    Pipeline:
    1. Subtract grid baseline from current frame (grid lines cancel out)
    2. Threshold the difference to find blobs that are new (the worm)
    3. Score candidates by difference strength and proximity
    4. Strict ROI-based selection (no cross-dish jumps)

    Parameters
    ----------
    gray : float32 array [0,1]
        Current frame grayscale.
    bg_max : float32 array [0,1]
        Per-pixel maximum across sampled frames (worm-free background).
    dish_mask_bool : bool array
        True inside the dish.
    last_centroid : tuple or None
        Previous worm position.
    prev_area : float or None
        Running average of worm area for consistency check.
    motion_map : float32 array or None
        Accumulated absolute frame differences (motion energy).
    grid_baseline : float32 array or None
        Mean of pre-worm frames.  When available, the grid pattern is
        subtracted so only the worm remains as a dark blob.

    Returns
    -------
    centroid : tuple or None
    area : int
    contour : ndarray or None
    confidence : float
    """
    h, w = gray.shape

    if grid_baseline is not None:
        # Grid subtraction: baseline is worm-free, so (baseline - gray)
        # is positive where the worm makes the frame darker than the baseline.
        # Grid lines appear in both → they cancel out.
        #
        # Brightness normalization: scale gray so its mean (inside dish)
        # matches the baseline mean.  This compensates for lighting drift
        # (e.g., sunrise during an hour-long session) so the subtraction
        # captures only pattern differences, not overall illumination.
        bl_mean = grid_baseline[dish_mask_bool].mean()
        gr_mean = gray[dish_mask_bool].mean()
        if gr_mean > 1e-6:
            gray_norm = gray * (bl_mean / gr_mean)
        else:
            gray_norm = gray
        diff = grid_baseline - gray_norm
        diff = np.clip(diff, 0, 1)
        diff[~dish_mask_bool] = 0

        # Threshold: pixels where current frame is notably darker than baseline
        thresh = 0.05  # ~13 grey levels out of 255
        bw = (diff >= thresh).astype(np.uint8) * 255
    else:
        # Fallback: old darkness-based method (no grid baseline available)
        g8 = (gray * 255).astype(np.uint8)
        kern_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        blackhat = cv2.morphologyEx(g8, cv2.MORPH_BLACKHAT, kern_bh)
        g8_clean = cv2.add(g8, blackhat)
        gray_clean = g8_clean.astype(np.float32) / 255.0

        dark = relative_darkness(gray_clean)
        dark[~dish_mask_bool] = 0
        dish_med = np.median(dark[dish_mask_bool])
        bw = (dark >= (dish_med + 0.15)).astype(np.uint8) * 255
        bw[~dish_mask_bool] = 0

    # Morphological cleanup
    kern_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kern_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kern_open)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kern_close)

    # Connected components
    num, labels, stats, cents = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num <= 1:
        return None, 0, None, 0.0

    # Layer 3: Filter and score candidates using motion + darkness
    candidates = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (min_area <= area <= max_area):
            continue

        bw_w = stats[i, cv2.CC_STAT_WIDTH]
        bw_h = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(bw_w, bw_h) / max(min(bw_w, bw_h), 1)
        if aspect > 10:
            continue

        cx, cy = float(cents[i][0]), float(cents[i][1])
        comp_bool = (labels == i)

        # Difference score: how strongly does this blob differ from baseline?
        # With grid baseline: this is the worm signal (grid cancelled out).
        # Without: falls back to bg_max novelty.
        if grid_baseline is not None:
            diff_score = float(np.mean(
                np.clip(grid_baseline[comp_bool] - gray_norm[comp_bool], 0, 1)))
        elif motion_map is not None:
            diff_score = float(np.mean(motion_map[comp_bool]))
        else:
            diff_score = float(np.mean(bg_max[comp_bool] - gray[comp_bool]))

        # Area consistency bonus
        area_bonus = 1.0
        if prev_area is not None and prev_area > 0:
            area_ratio = area / prev_area
            if 0.3 < area_ratio < 3.0:
                area_bonus = 1.2
            elif area_ratio < 0.1 or area_ratio > 10.0:
                area_bonus = 0.5

        score = diff_score * math.sqrt(area) * area_bonus
        candidates.append((i, cx, cy, area, diff_score, score))

    if not candidates:
        return None, 0, None, 0.0

    # Layer 4: Strict proximity-based selection.
    # Worms move slowly — never allow cross-dish jumps.
    # Search radius grows gradually when lost, but is always bounded.
    best = None
    if last_centroid is not None:
        lx, ly = last_centroid

        # Expand search radius gradually when lost: roi_px + 20px per lost frame,
        # but never more than 3× the base ROI.
        search_radius = min(roi_px + lost_count * 20, roi_px * 3)

        roi_cands = []
        for cand in candidates:
            i, cx, cy, area, diff_score, score = cand
            dist = math.dist((cx, cy), (lx, ly))
            if dist <= search_radius:
                # Strong exponential distance decay: worms don't jump.
                # At dist=0: 1.0, dist=30: 0.37, dist=60: 0.14, dist=120: 0.02
                proximity_weight = math.exp(-dist / 30.0)
                weighted_score = score * proximity_weight
                roi_cands.append((*cand, dist, weighted_score))

        if roi_cands:
            roi_cands.sort(key=lambda c: c[7], reverse=True)
            best = roi_cands[0]

    # No last_centroid at all (very first frame): pick best global candidate
    if best is None and last_centroid is None:
        candidates.sort(key=lambda c: c[5], reverse=True)
        best_cand = candidates[0]
        best = (*best_cand, 0.0, best_cand[5])

    # If ROI search found nothing but we've been lost a long time,
    # do a full-dish search to relocate the worm.
    if best is None and last_centroid is not None and lost_count >= 50:
        # Accept best global candidate if it scores well enough
        candidates.sort(key=lambda c: c[5], reverse=True)
        if candidates and candidates[0][5] >= 0.5:
            best_cand = candidates[0]
            best = (*best_cand, 0.0, best_cand[5])

    # If we have a last_centroid but found nothing: report LOST.
    if best is None:
        return None, 0, None, 0.0

    comp_idx = best[0]
    cx, cy, area, diff_score = best[1], best[2], best[3], best[4]

    # Extract contour for orientation
    comp_mask = (labels == comp_idx).astype(np.uint8) * 255
    contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    contour = contours[0] if contours else None

    # Confidence: blend motion score and darkness
    confidence = float(np.clip(diff_score * 10.0, 0, 1))
    if area < min_area * 1.5:
        confidence *= 0.7

    return (cx, cy), area, contour, confidence


# ──────────────────────────────────────────────────────────────────────
# Stage 5 — Orientation via fitEllipse
# ──────────────────────────────────────────────────────────────────────

class OrientationTracker:
    """Track worm body axis and disambiguate head/tail."""

    def __init__(self, smoothing_alpha=0.3):
        self.smoothing_alpha = smoothing_alpha
        self.head_angle = None       # 0–360° direction the head is pointing
        self.body_angle = None       # 0–180° body axis from fitEllipse
        self._prev_centroid = None
        self._head_end = None        # which end of the ellipse is the head

    def update(self, centroid, contour, area):
        """Compute body angle and head angle from contour.

        Returns (body_angle_deg, head_angle_deg) or (None, None).
        """
        if contour is None or len(contour) < 5:
            return self.body_angle, self.head_angle

        try:
            ellipse = cv2.fitEllipse(contour)
        except cv2.error:
            return self.body_angle, self.head_angle

        (ecx, ecy), (ma, MA), angle = ellipse

        # fitEllipse gives angle of the major axis (0–180°)
        # We want this as body_angle
        raw_body_angle = angle % 180.0

        # Smooth body angle (handle wraparound at 0/180)
        if self.body_angle is not None:
            diff = raw_body_angle - self.body_angle
            if diff > 90:
                diff -= 180
            elif diff < -90:
                diff += 180
            smoothed = self.body_angle + self.smoothing_alpha * diff
            self.body_angle = smoothed % 180.0
        else:
            self.body_angle = raw_body_angle

        # Head/tail disambiguation
        head_angle = self._disambiguate_head(
            centroid, contour, self.body_angle, MA, ma)

        if head_angle is not None:
            if self.head_angle is not None:
                # Smooth head angle (handle 0/360 wraparound)
                diff = head_angle - self.head_angle
                if diff > 180:
                    diff -= 360
                elif diff < -180:
                    diff += 360
                self.head_angle = (self.head_angle +
                                   self.smoothing_alpha * diff) % 360.0
            else:
                self.head_angle = head_angle

        self._prev_centroid = centroid
        return self.body_angle, self.head_angle

    def _disambiguate_head(self, centroid, contour, body_angle_deg, MA, ma):
        """Determine which end of the body axis is the head.

        Strategy:
        (a) Motion direction when moving — head leads
        (b) Morphological asymmetry when stationary — planarian heads are wider
        (c) Temporal persistence as fallback
        """
        cx, cy = centroid

        # Two candidate head directions along body axis
        angle_rad = math.radians(body_angle_deg)
        half_len = MA / 2.0 if MA > 0 else 20.0

        end_a = (cx + half_len * math.cos(angle_rad),
                 cy + half_len * math.sin(angle_rad))
        end_b = (cx - half_len * math.cos(angle_rad),
                 cy - half_len * math.sin(angle_rad))

        # (a) Motion-based: if moving, head is in direction of motion
        if self._prev_centroid is not None:
            dx = cx - self._prev_centroid[0]
            dy = cy - self._prev_centroid[1]
            speed = math.hypot(dx, dy)

            if speed > 2.0:  # significant motion
                motion_angle = math.degrees(math.atan2(dy, dx))
                # Which end is closer to the motion direction?
                angle_a = math.degrees(math.atan2(
                    end_a[1] - cy, end_a[0] - cx))
                angle_b = math.degrees(math.atan2(
                    end_b[1] - cy, end_b[0] - cx))

                diff_a = abs(((motion_angle - angle_a) + 180) % 360 - 180)
                diff_b = abs(((motion_angle - angle_b) + 180) % 360 - 180)

                if diff_a < diff_b:
                    self._head_end = 'a'
                    return math.degrees(math.atan2(
                        end_a[1] - cy, end_a[0] - cx)) % 360
                else:
                    self._head_end = 'b'
                    return math.degrees(math.atan2(
                        end_b[1] - cy, end_b[0] - cx)) % 360

        # (b) Morphological asymmetry: measure width at each end
        contour_pts = contour.squeeze()
        if contour_pts.ndim == 2 and len(contour_pts) >= 10:
            # Project contour points onto body axis
            axis = np.array([math.cos(angle_rad), math.sin(angle_rad)])
            projections = (contour_pts - np.array([cx, cy])) @ axis

            # Measure width at the two ends
            q25 = np.percentile(projections, 25)
            q75 = np.percentile(projections, 75)

            near_end_a = contour_pts[projections > q75]
            near_end_b = contour_pts[projections < q25]

            if len(near_end_a) >= 3 and len(near_end_b) >= 3:
                # Width perpendicular to body axis
                perp = np.array([-math.sin(angle_rad), math.cos(angle_rad)])
                width_a = np.ptp((near_end_a - np.array([cx, cy])) @ perp)
                width_b = np.ptp((near_end_b - np.array([cx, cy])) @ perp)

                # Planarian head is wider
                if abs(width_a - width_b) > 1.0:
                    if width_a > width_b:
                        self._head_end = 'a'
                        return math.degrees(math.atan2(
                            end_a[1] - cy, end_a[0] - cx)) % 360
                    else:
                        self._head_end = 'b'
                        return math.degrees(math.atan2(
                            end_b[1] - cy, end_b[0] - cx)) % 360

        # (c) Temporal persistence
        if self._head_end == 'a':
            return math.degrees(math.atan2(
                end_a[1] - cy, end_a[0] - cx)) % 360
        elif self._head_end == 'b':
            return math.degrees(math.atan2(
                end_b[1] - cy, end_b[0] - cx)) % 360

        # Default: pick end_a
        self._head_end = 'a'
        return math.degrees(math.atan2(
            end_a[1] - cy, end_a[0] - cx)) % 360


# ──────────────────────────────────────────────────────────────────────
# Midline / skeleton extraction  (requires scikit-image + scipy)
# ──────────────────────────────────────────────────────────────────────

def extract_skeleton(worm_mask):
    """Skeletonize binary worm mask using Zhang-Suen thinning.

    Returns a clean, 1-pixel-wide skeleton (uint8, 0/255).
    """
    binary = (worm_mask > 0).astype(bool)
    skel = ski_skeletonize(binary)
    return skel.astype(np.uint8) * 255


def prune_skeleton_branches(skel, min_branch_len=8):
    """Remove short branches, keeping only the main spine."""
    skel_bool = skel > 0
    h, w = skel.shape

    def neighbors(x, y):
        nbs = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and skel_bool[ny, nx]:
                    nbs.append((nx, ny))
        return nbs

    changed = True
    iterations = 0
    while changed and iterations < 20:
        changed = False
        iterations += 1
        ys, xs = np.where(skel_bool)

        for x, y in zip(xs, ys):
            nbs = neighbors(x, y)
            if len(nbs) == 1:
                branch = [(x, y)]
                cur = (x, y)
                prev = None
                while True:
                    nbs_cur = [n for n in neighbors(cur[0], cur[1]) if n != prev]
                    if len(nbs_cur) != 1:
                        break
                    prev = cur
                    cur = nbs_cur[0]
                    branch.append(cur)

                if len(branch) < min_branch_len:
                    for bx, by in branch[:-1]:
                        skel_bool[by, bx] = False
                    changed = True

    return skel_bool.astype(np.uint8) * 255


def order_skeleton_points(skel):
    """Order skeleton pixels into a continuous head-to-tail path via double-BFS."""
    ys, xs = np.where(skel > 0)
    if len(xs) < 3:
        return np.column_stack([xs, ys]).astype(np.float64) if len(xs) > 0 else np.empty((0, 2))

    pts = list(zip(xs.tolist(), ys.tolist()))
    pt_to_idx = {p: i for i, p in enumerate(pts)}
    n = len(pts)

    adj = [[] for _ in range(n)]
    for i, (x, y) in enumerate(pts):
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nb = (x + dx, y + dy)
                if nb in pt_to_idx:
                    adj[i].append(pt_to_idx[nb])

    def bfs_furthest(start):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        furthest = start
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                    if dist[v] > dist[furthest]:
                        furthest = v
        return furthest, dist

    end1, _ = bfs_furthest(0)
    end2, _ = bfs_furthest(end1)

    parent = [-1] * n
    visited = [False] * n
    visited[end1] = True
    queue = deque([end1])
    while queue:
        u = queue.popleft()
        if u == end2:
            break
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                queue.append(v)

    path = []
    cur = end2
    while cur != -1:
        path.append(pts[cur])
        cur = parent[cur]
    path.reverse()

    return np.array(path, dtype=np.float64)


def resample_midline(points, n_points=20):
    """Resample an ordered point sequence to N equally-spaced points."""
    if len(points) < 2:
        return points

    diffs = np.diff(points, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cum_len = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_len = cum_len[-1]

    if total_len < 1e-6:
        return points

    target_s = np.linspace(0, total_len, n_points)
    resampled = np.zeros((n_points, 2))
    for i, s in enumerate(target_s):
        idx = np.searchsorted(cum_len, s, side='right') - 1
        idx = max(0, min(idx, len(points) - 2))
        frac = (s - cum_len[idx]) / max(1e-9, cum_len[idx + 1] - cum_len[idx])
        resampled[i] = points[idx] + frac * (points[idx + 1] - points[idx])

    return resampled


def smooth_midline(points, sigma=1.5):
    """Gaussian-smooth the midline coordinates."""
    if len(points) < 5:
        return points
    smoothed = np.copy(points)
    smoothed[:, 0] = gaussian_filter1d(points[:, 0], sigma=sigma, mode='nearest')
    smoothed[:, 1] = gaussian_filter1d(points[:, 1], sigma=sigma, mode='nearest')
    return smoothed


def compute_curvature(points):
    """Compute signed curvature at each interior point.

    Returns array of length N-4 for N input points (central-difference tangents
    → angle differences → curvature).
    """
    if len(points) < 5:
        return np.array([])

    tangents = points[2:] - points[:-2]
    angles = np.arctan2(tangents[:, 1], tangents[:, 0])

    diffs = np.diff(points, axis=0)
    ds = np.sqrt((diffs ** 2).sum(axis=1))

    dtheta = np.diff(angles)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi

    ds_mid = ds[1:-1] + ds[2:]
    ds_mid[ds_mid < 1e-9] = 1e-9

    curvature = dtheta / ds_mid
    return curvature


def extract_midline(contour, prev_midline, n_points, frame_shape):
    """Full midline extraction pipeline from a worm contour.

    Reconstructs binary mask from contour on a tight local patch,
    skeletonizes, orders, resamples, smooths, then applies temporal
    consistency with prev_midline to keep pt0 at the same physical end.

    Returns (midline, curvature, body_length_px) or (None, None, 0.0).
    """
    if contour is None or len(contour) < 5:
        return None, None, 0.0

    # Build tight binary mask from contour (avoids full-frame allocation)
    x, y, bw, bh = cv2.boundingRect(contour)
    pad = 4
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(frame_shape[1], x + bw + pad)
    y1 = min(frame_shape[0], y + bh + pad)
    patch = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    shifted = contour - np.array([x0, y0])
    cv2.drawContours(patch, [shifted], -1, 255, cv2.FILLED)

    # Skeletonize
    skel = extract_skeleton(patch)
    skel = prune_skeleton_branches(skel, min_branch_len=6)

    if cv2.countNonZero(skel) < 5:
        return None, None, 0.0

    # Order → resample → smooth
    ordered = order_skeleton_points(skel)
    if len(ordered) < 5:
        return None, None, 0.0

    # Shift back to full-frame coordinates
    ordered = ordered + np.array([x0, y0], dtype=np.float64)

    # Adaptive point count
    raw_len = np.sum(np.sqrt(np.sum(np.diff(ordered, axis=0)**2, axis=1)))
    adaptive_n = max(10, min(n_points, int(raw_len / 3)))

    midline = resample_midline(ordered, n_points=adaptive_n)
    midline = smooth_midline(midline, sigma=1.5)

    # Temporal consistency: align with previous midline
    if prev_midline is not None and len(prev_midline) >= 3 and len(midline) >= 3:
        def third_centroids(ml):
            k = max(1, len(ml) // 3)
            return ml[:k].mean(axis=0), ml[-k:].mean(axis=0)

        prev_head, prev_tail = third_centroids(prev_midline)
        cur_head, cur_tail = third_centroids(midline)

        d_keep = (np.linalg.norm(cur_head - prev_head) +
                  np.linalg.norm(cur_tail - prev_tail))
        d_flip = (np.linalg.norm(cur_head - prev_tail) +
                  np.linalg.norm(cur_tail - prev_head))

        if d_flip < d_keep:
            midline = midline[::-1].copy()

    body_length_px = np.sum(np.sqrt(np.sum(np.diff(midline, axis=0)**2, axis=1)))
    curvature = compute_curvature(midline)

    return midline, curvature, body_length_px


def _accumulate_ht_vote(midline, centroid, prev_centroid,
                        keep_votes, flip_votes):
    """Accumulate head/tail vote based on motion direction.

    Uses motion vector + third-centroid projection.
    Only votes when displacement > 5px. Mutates keep_votes[0] / flip_votes[0].
    """
    if (centroid is None or prev_centroid is None or
            midline is None or len(midline) < 3):
        return

    motion = np.array(centroid) - np.array(prev_centroid)
    speed = np.linalg.norm(motion)
    if speed < 5.0:
        return
    motion_dir = motion / speed

    k = max(1, len(midline) // 3)
    head_region = midline[:k].mean(axis=0)
    tail_region = midline[-k:].mean(axis=0)
    center = np.array(centroid)

    head_proj = np.dot(head_region - center, motion_dir)
    tail_proj = np.dot(tail_region - center, motion_dir)

    if head_proj > tail_proj:
        keep_votes[0] += 1
    else:
        flip_votes[0] += 1


# ──────────────────────────────────────────────────────────────────────
# Midline overlay rendering (deferred — called after H/T vote is known)
# ──────────────────────────────────────────────────────────────────────

def render_midline_overlay(bgr, contour, centroid, midline, curvature,
                           body_length_px, speed_mm_s, confidence, area,
                           time_s, trail, body_length_history,
                           head_is_pt0, H, W):
    """Render a single midline overlay frame with correct H/T labels.

    Parameters
    ----------
    bgr : ndarray (H, W, 3) — raw source frame
    contour, centroid, midline, curvature — tracking data (may be None if LOST)
    body_length_px, speed_mm_s, confidence, area — scalar metrics
    time_s : float — timestamp for HUD
    trail : deque — recent centroids (mutated: appends centroid)
    body_length_history : deque — recent body lengths (mutated: appends)
    head_is_pt0 : bool — True if midline[0] is head (from session vote)
    H, W : int — frame dimensions

    Returns
    -------
    vis : ndarray (H, W, 3) uint8
    """
    if centroid is not None and contour is not None:
        # Composite: worm pixels from frame, rest white
        worm_mask_ol = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(worm_mask_ol, [contour], -1, 255, cv2.FILLED)
        mask_f = (worm_mask_ol > 0).astype(np.float32)
        mask3 = np.stack([mask_f]*3, axis=-1)
        white_bg = np.full_like(bgr, 240, dtype=np.uint8)
        vis = (mask3 * bgr.astype(np.float32) +
               (1 - mask3) * white_bg.astype(np.float32)
               ).astype(np.uint8)

        trail.append(centroid)

        # Trailing path — fading green
        if len(trail) >= 2:
            pts_trail = list(trail)
            for k in range(1, len(pts_trail)):
                alpha = k / len(pts_trail)
                green = int(80 + 175 * alpha)
                cv2.line(vis,
                         (int(pts_trail[k-1][0]), int(pts_trail[k-1][1])),
                         (int(pts_trail[k][0]), int(pts_trail[k][1])),
                         (0, green, 0), 1)

        # Worm contour outline — red
        cv2.drawContours(vis, [contour], -1, (0, 0, 255), 1)

        # Midline overlay
        if midline is not None and len(midline) >= 2:
            ml_draw = midline.copy()
            curv_draw = curvature.copy() if curvature is not None else None

            # Flip so pt0 = head for rendering
            if not head_is_pt0:
                ml_draw = ml_draw[::-1]
                if curv_draw is not None and len(curv_draw) > 0:
                    curv_draw = -curv_draw[::-1]

            pts_draw = ml_draw.astype(np.int32)

            # Curvature-colored midline
            if curv_draw is not None and len(curv_draw) > 0:
                max_curv = max(abs(curv_draw.max()),
                               abs(curv_draw.min()), 0.005)
                curv_padded = np.zeros(len(ml_draw) - 1)
                off = (len(curv_padded) - len(curv_draw)) // 2
                if 0 <= off and off + len(curv_draw) <= len(curv_padded):
                    curv_padded[off:off+len(curv_draw)] = curv_draw
                for ki in range(len(pts_draw) - 1):
                    val = np.clip(curv_padded[ki] / max_curv, -1, 1)
                    if val > 0:
                        color = (int(100*(1-val)),
                                 int(100*(1-val)), 255)
                    else:
                        color = (255, int(100*(1+val)),
                                 int(100*(1+val)))
                    cv2.line(vis, tuple(pts_draw[ki]),
                             tuple(pts_draw[ki+1]), color, 3)
            else:
                for ki in range(len(pts_draw) - 1):
                    cv2.line(vis, tuple(pts_draw[ki]),
                             tuple(pts_draw[ki+1]),
                             (0, 255, 0), 3)

            # Endpoint markers — pt0 is always Head after flip
            cv2.circle(vis, tuple(pts_draw[0]), 6,
                       (255, 80, 80), -1)
            cv2.putText(vis, "H",
                        (pts_draw[0][0]+8, pts_draw[0][1]-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 120, 120), 2)
            cv2.circle(vis, tuple(pts_draw[-1]), 5,
                       (80, 80, 255), -1)
            cv2.putText(vis, "T",
                        (pts_draw[-1][0]+8, pts_draw[-1][1]-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (120, 120, 255), 2)

        # Track body length for plot
        body_length_history.append(body_length_px)

        # ── Mini curvature plot (bottom-right) ──
        if curvature is not None and len(curvature) > 2:
            plot_w, plot_h = 180, 60
            px0 = W - plot_w - 15
            py0 = H - plot_h - 80

            overlay_bg = vis.copy()
            cv2.rectangle(overlay_bg, (px0-5, py0-5),
                          (px0+plot_w+5, py0+plot_h+5),
                          (0, 0, 0), -1)
            cv2.addWeighted(overlay_bg, 0.6, vis, 0.4, 0, vis)

            cv2.line(vis, (px0, py0+plot_h//2),
                     (px0+plot_w, py0+plot_h//2),
                     (100, 100, 100), 1)

            mc = max(abs(curvature.max()),
                     abs(curvature.min()), 0.005)
            xs_p = np.linspace(px0, px0+plot_w,
                               len(curvature)).astype(int)
            ys_p = (py0 + plot_h//2 -
                    curvature/mc*(plot_h//2-3)).astype(int)
            ys_p = np.clip(ys_p, py0, py0+plot_h)
            for ki in range(len(xs_p)-1):
                c = (80,80,255) if curvature[ki]>0 else (255,80,80)
                cv2.line(vis, (xs_p[ki], ys_p[ki]),
                         (xs_p[ki+1], ys_p[ki+1]), c, 2)
            cv2.putText(vis, "Curvature", (px0, py0-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (200, 200, 200), 1)

        # ── Mini body length plot (bottom-right, below curvature) ──
        if len(body_length_history) >= 2:
            plot_w2, plot_h2 = 180, 50
            px0b = W - plot_w2 - 15
            py0b = H - plot_h2 - 15

            overlay_bg = vis.copy()
            cv2.rectangle(overlay_bg, (px0b-5, py0b-5),
                          (px0b+plot_w2+5, py0b+plot_h2+5),
                          (0, 0, 0), -1)
            cv2.addWeighted(overlay_bg, 0.6, vis, 0.4, 0, vis)

            vals = list(body_length_history)
            vmin = min(vals) * 0.9
            vmax = max(vals) * 1.1
            vrange = max(vmax - vmin, 1.0)
            xs_b = np.linspace(px0b, px0b+plot_w2,
                               len(vals)).astype(int)
            ys_b = (py0b + plot_h2 -
                    (np.array(vals) - vmin) / vrange *
                    plot_h2).astype(int)
            ys_b = np.clip(ys_b, py0b, py0b+plot_h2)
            for ki in range(len(xs_b)-1):
                cv2.line(vis, (xs_b[ki], ys_b[ki]),
                         (xs_b[ki+1], ys_b[ki+1]),
                         (0, 200, 200), 2)
            cv2.putText(vis, "Body length", (px0b, py0b-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (200, 200, 200), 1)

        # ── HUD text (top-left) ──
        spd_txt = (f"{speed_mm_s:.2f}"
                   if speed_mm_s != "" else "---")
        bl_txt = (f"{body_length_px:.0f}"
                  if body_length_px > 0 else "---")
        hud_lines = [
            f"t={time_s:.1f}s  spd={spd_txt} mm/s",
            f"conf={confidence:.2f}  area={area}",
            f"body_len={bl_txt}px",
        ]
        overlay_bg = vis.copy()
        cv2.rectangle(overlay_bg, (5, 5), (320, 85),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay_bg, 0.5, vis, 0.5, 0, vis)
        for li, line_txt in enumerate(hud_lines):
            cv2.putText(vis, line_txt,
                        (10, 22 + li * 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1)
    else:
        # No detection — white frame with LOST
        vis = np.full((H, W, 3), 240, dtype=np.uint8)
        cv2.putText(vis, "LOST", (W//2 - 40, H//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 255), 3)

    return vis


# ──────────────────────────────────────────────────────────────────────
# Video file parsing
# ──────────────────────────────────────────────────────────────────────

def parse_video_timestamp(filename):
    """Extract datetime from filename like '2026-02-14 10-21-55.mkv'."""
    stem = Path(filename).stem
    m = re.match(r'(\d{4})-(\d{2})-(\d{2})\s+(\d{2})-(\d{2})-(\d{2})', stem)
    if not m:
        return None
    return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                    int(m.group(4)), int(m.group(5)), int(m.group(6)))


def get_session_videos(session_dir):
    """Return sorted list of MKV files in a session directory."""
    videos = sorted(glob.glob(os.path.join(session_dir, "*.mkv")))
    if not videos:
        videos = sorted(glob.glob(os.path.join(session_dir, "*.MKV")))
    return videos


# ──────────────────────────────────────────────────────────────────────
# Session processing
# ──────────────────────────────────────────────────────────────────────

def calibrate_session(session_dir, videos):
    """Run auto dish detection + grid calibration on the first video.

    Returns calibration dict and saves JSON alongside output CSV.
    """
    cap = open_video(videos[0])
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {videos[0]}")

    # Read a frame from the middle of the video for a representative image
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 600
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Cannot read frames from {videos[0]}")

    gray = to_gray01(frame)

    # Also build a multi-frame median for more stable calibration
    n_calib = min(30, total)
    indices = np.linspace(0, total - 1, n_calib, dtype=int)
    calib_frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok2, f2 = cap.read()
        if ok2:
            calib_frames.append(to_gray01(f2))
    cap.release()

    if calib_frames:
        median_gray = np.median(np.stack(calib_frames, axis=0),
                                axis=0).astype(np.float32)
    else:
        median_gray = gray

    # Auto dish detection on the median image
    dish_center, dish_radius = auto_detect_dish(median_gray)

    # Auto grid calibration
    mm_per_px, spacing_px = auto_grid_calibration(median_gray, dish_center,
                                                  dish_radius)

    H, W = gray.shape
    calib = {
        "session": os.path.basename(session_dir),
        "dish_center": list(dish_center),
        "dish_radius_px": dish_radius,
        "mm_per_px": mm_per_px,
        "grid_spacing_px": spacing_px,
        "frame_size": [W, H],
        "calibration_video": os.path.basename(videos[0]),
        "n_videos": len(videos),
    }

    return calib


# ──────────────────────────────────────────────────────────────────────
# Manual worm seeding
# ──────────────────────────────────────────────────────────────────────

def manual_seed_worm(video_path, dish_center=None, dish_radius=None):
    """Two-phase interactive seeding for worm tracking.

    Phase 1 — "No Worm": Browse frames of the empty dish.  Advance until
    the worm appears, then press **w** to move to Phase 2.

    Phase 2 — "Mark Worm": Click on the worm inside the cyan dish circle,
    then press **Enter** to confirm.

    Navigation (both phases)
    ------------------------
    d / Space  — +1 frame       a — -1 frame
    f          — +10 frames     b — -10 frames
    g          — +50 frames     v — -50 frames

    Phase transitions
    -----------------
    w          — "I see the worm" → switch to Phase 2
    n          — (Phase 2 only) go back to Phase 1 (no worm yet)
    Enter      — (Phase 2 only) confirm clicked position
    q / Esc    — skip (no seed)

    Returns
    -------
    seed_frame : int or None
    seed_pos   : (x, y) or None
    """
    cap = open_video(video_path)
    if not cap.isOpened():
        print(f"  Cannot open {video_path} for seeding.")
        return None, None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 600
    # phase: 1 = "no worm, scanning", 2 = "worm visible, click to mark"
    state = {"frame": 0, "pos": None, "confirmed": False, "phase": 1}

    def _inside_dish(x, y):
        if dish_center is None or dish_radius is None:
            return True
        dx = x - dish_center[0]
        dy = y - dish_center[1]
        return (dx * dx + dy * dy) <= dish_radius * dish_radius

    win = "Seed Worm Position"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and state["phase"] == 2:
            state["pos"] = (x, y)
            show(state["frame"])

    cv2.setMouseCallback(win, on_mouse)

    def show(frame_num):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ok, bgr = cap.read()
        if not ok:
            return
        vis = bgr.copy()

        # Draw dish boundary
        if dish_center is not None and dish_radius is not None:
            cx, cy = int(dish_center[0]), int(dish_center[1])
            r = int(dish_radius)
            cv2.circle(vis, (cx, cy), r, (255, 255, 0), 2)  # cyan ring

        # Build HUD text based on phase
        if state["phase"] == 1:
            phase_label = "NO WORM — scanning"
            lines = [
                f"Frame {frame_num}/{total}   [{phase_label}]",
                "d/Space=+1  f=+10  g=+50  |  a=-1  b=-10  v=-50",
                "Press W when you see the worm   |   q=skip",
            ]
            hud_color = (0, 200, 255)  # orange-ish
        else:
            phase_label = "WORM VISIBLE — click on it"
            lines = [
                f"Frame {frame_num}/{total}   [{phase_label}]",
                "d/Space=+1  f=+10  g=+50  |  a=-1  b=-10  v=-50",
                "Click worm inside cyan circle, then Enter",
                "n=go back (no worm)   |   q=skip",
            ]
            hud_color = (0, 255, 0)  # green

        inside = True
        if state["pos"] is not None:
            inside = _inside_dish(state["pos"][0], state["pos"][1])
            if not inside:
                lines.append("!! OUTSIDE DISH — click inside the cyan circle !!")
            else:
                lines.append(
                    f"Marked ({state['pos'][0]:.0f}, {state['pos'][1]:.0f})"
                    f" — press Enter to confirm")

        for i, txt in enumerate(lines):
            y_pos = 25 + i * 22
            is_warning = (state["pos"] is not None and not inside
                          and i == len(lines) - 1)
            color = (0, 0, 255) if is_warning else hud_color
            cv2.putText(vis, txt, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 3)
            cv2.putText(vis, txt, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color, 1)

        # Draw seed marker (Phase 2 only)
        if state["phase"] == 2 and state["pos"] is not None:
            px, py = int(state["pos"][0]), int(state["pos"][1])
            marker_color = (0, 255, 0) if inside else (0, 0, 255)
            cv2.circle(vis, (px, py), 15, marker_color, 2)
            cv2.drawMarker(vis, (px, py), marker_color,
                           cv2.MARKER_CROSS, 20, 2)

        cv2.imshow(win, vis)

    show(0)

    while True:
        key = cv2.waitKey(0) & 0xFF

        # --- Navigation (both phases) ---
        if key == ord("q") or key == 27:
            break
        elif key == ord("d") or key == ord(" "):
            state["frame"] = min(state["frame"] + 1, total - 1)
            show(state["frame"])
        elif key == ord("f"):
            state["frame"] = min(state["frame"] + 10, total - 1)
            show(state["frame"])
        elif key == ord("g"):
            state["frame"] = min(state["frame"] + 50, total - 1)
            show(state["frame"])
        elif key == ord("a"):
            state["frame"] = max(state["frame"] - 1, 0)
            show(state["frame"])
        elif key == ord("b"):
            state["frame"] = max(state["frame"] - 10, 0)
            show(state["frame"])
        elif key == ord("v"):
            state["frame"] = max(state["frame"] - 50, 0)
            show(state["frame"])

        # --- Phase transitions ---
        elif key == ord("w"):
            # "I see the worm" — enter Phase 2
            state["phase"] = 2
            state["pos"] = None
            show(state["frame"])
        elif key == ord("n") and state["phase"] == 2:
            # Go back to Phase 1
            state["phase"] = 1
            state["pos"] = None
            show(state["frame"])
        elif key in (13, 10) and state["phase"] == 2:
            if state["pos"] is not None:
                if _inside_dish(state["pos"][0], state["pos"][1]):
                    state["confirmed"] = True
                    break
                else:
                    print("    Click is outside the dish! "
                          "Please click inside the cyan circle.")
                    show(state["frame"])

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    if state["confirmed"] and state["pos"] is not None:
        return state["frame"], state["pos"]
    return None, None


def process_session(session_dir, calib, args, output_dir, seed=None):
    """Process all videos in a session, producing one consolidated CSV.

    Parameters
    ----------
    session_dir : str
        Path to session folder.
    calib : dict
        Calibration data (dish, grid, scale).
    args : argparse.Namespace
        CLI arguments.
    output_dir : str
        Directory for output files.
    seed : tuple or None
        (frame_idx, (x, y)) from manual seeding.  Applied to the first video.
    """
    session_name = os.path.basename(session_dir)
    videos = get_session_videos(session_dir)
    if not videos:
        print(f"  No videos found in {session_dir}, skipping.")
        return

    H, W = calib["frame_size"][1], calib["frame_size"][0]
    dish_center = tuple(calib["dish_center"])
    dish_radius = calib["dish_radius_px"]
    mm_per_px = calib["mm_per_px"]

    dish_mask = circle_mask((H, W), dish_center, dish_radius)
    dish_mask_bool = dish_mask > 0

    # Parse the timestamp of the first video to compute cumulative session time
    t0 = parse_video_timestamp(videos[0])

    # CSV output file
    csv_path = os.path.join(output_dir, f"{session_name}_tracks.csv")
    calib_path = os.path.join(output_dir, f"{session_name}_calibration.json")

    # Save calibration
    with open(calib_path, 'w') as f:
        json.dump(calib, f, indent=2)

    # CSV header
    header = [
        "video_file", "frame", "time_s",
        "centroid_x_px", "centroid_y_px", "centroid_x_mm", "centroid_y_mm",
        "body_angle_deg", "head_angle_deg", "area_px",
        "speed_px_s", "speed_mm_s", "confidence",
        "body_length_px", "body_length_mm"
    ]

    # Chained state across videos
    last_centroid = None
    last_time = None
    prev_area = None  # Running average of worm area for consistency scoring
    orientation_tracker = OrientationTracker(smoothing_alpha=0.3)
    lost_count = 0

    # Midline chained state (persists across videos like last_centroid)
    prev_midline = None
    session_midlines = []         # Accumulator for NPZ output
    ht_keep_votes = [0]           # Mutable counter: pt0 leads
    ht_flip_votes = [0]           # Mutable counter: ptN leads
    overlay_videos_deferred = []  # Deferred overlay data for post-processing

    # Apply manual seed and build grid baseline
    seed_frame = 0
    grid_baseline = None
    if seed is not None:
        seed_frame, seed_pos = seed
        last_centroid = seed_pos

        # Build grid baseline from pre-worm frames (before the seed)
        if seed_frame > 5:
            try:
                grid_baseline = build_grid_baseline(
                    videos[0], seed_frame, dish_mask_bool)
                print(f"  Grid baseline: built from {min(20, seed_frame)} "
                      f"pre-worm frames (0..{seed_frame - 1})")
            except RuntimeError as e:
                print(f"  WARNING: Grid baseline failed: {e}")

    # Overlay state
    trail = deque(maxlen=100)             # recent centroids for trailing path
    body_length_history = deque(maxlen=100)  # recent body lengths for plot

    total_frames_processed = 0
    total_detections = 0

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f"# mm_per_px={mm_per_px:.8f}"])
        writer.writerow([f"# dish_center={dish_center[0]:.1f},{dish_center[1]:.1f}  "
                         f"dish_radius={dish_radius:.1f}px"])
        writer.writerow(header)

        for vid_idx, video_path in enumerate(videos):
            video_name = os.path.basename(video_path)
            vid_ts = parse_video_timestamp(video_path)

            # Cumulative time offset from session start
            if t0 is not None and vid_ts is not None:
                time_offset = (vid_ts - t0).total_seconds()
            else:
                # Approximate: ~60s per video
                time_offset = vid_idx * 60.0

            cap = open_video(video_path)
            if not cap.isOpened():
                print(f"  WARNING: Cannot open {video_name}, skipping.")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

            # Build per-video background
            try:
                bg_max = build_background(cap, dish_mask_bool, n_samples=60)
            except RuntimeError as e:
                print(f"  WARNING: Background build failed for {video_name}: {e}")
                cap.release()
                continue

            # Rebuild grid baseline for each video via temporal median.
            # Video 0 uses the pre-worm mean baseline (built above);
            # all subsequent videos use the median of sampled frames
            # which naturally excludes the moving worm.
            #
            # Caveat: if the worm is stationary throughout the video,
            # the median will absorb it into the baseline, making it
            # invisible to subtraction.  We detect this by checking
            # whether the worm is still visible at its last known
            # position after building the new baseline.  If not, we
            # keep the previous baseline (where the worm was elsewhere).
            if vid_idx > 0 and grid_baseline is not None:
                prev_baseline = grid_baseline
                try:
                    candidate = build_grid_baseline_median(
                        video_path, dish_mask_bool, n_samples=30)
                    # Validate: can we still see the worm?
                    use_candidate = True
                    if last_centroid is not None:
                        lx, ly = int(last_centroid[0]), int(last_centroid[1])
                        roi_r = args.roi_px
                        yy, xx = np.ogrid[:candidate.shape[0], :candidate.shape[1]]
                        roi_mask = ((xx - lx)**2 + (yy - ly)**2) <= roi_r**2
                        roi_mask &= dish_mask_bool
                        # Read a test frame from the new video
                        cap.set(cv2.CAP_PROP_POS_FRAMES, min(50, total_frames - 1))
                        ok_t, bgr_t = cap.read()
                        if ok_t:
                            gray_t = to_gray01(bgr_t)
                            bl_m = candidate[dish_mask_bool].mean()
                            gr_m = gray_t[dish_mask_bool].mean()
                            if gr_m > 1e-6:
                                gray_t = gray_t * (bl_m / gr_m)
                            diff_t = np.clip(candidate - gray_t, 0, 1)
                            diff_t[~dish_mask_bool] = 0
                            worm_pixels = np.sum((diff_t >= 0.05) & roi_mask)
                            if worm_pixels < 30:
                                use_candidate = False
                    if use_candidate:
                        grid_baseline = candidate
                    else:
                        print(f"    Baseline: keeping previous (worm stationary)")
                except RuntimeError as e:
                    pass  # keep previous baseline if build fails

            # Overlay video writer for this video
            overlay_writer = None
            overlay_frame_buf = None      # Deferred midline overlay buffer
            overlay_path = None
            if args.write_overlay and vid_idx < args.overlay_videos:
                eff_fps = fps / max(1, args.frame_skip)
                vid_stem = Path(video_path).stem
                overlay_path = os.path.join(
                    output_dir,
                    f"{session_name}_{vid_stem}_overlay.mp4")
                if args.extract_midline:
                    # Defer rendering until H/T vote is known
                    overlay_frame_buf = []
                else:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    overlay_writer = cv2.VideoWriter(
                        overlay_path, fourcc, eff_fps, (W, H))

            # Process frames — skip to seed frame on first video
            start_frame = seed_frame if vid_idx == 0 else 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_idx = start_frame
            vid_detections = 0

            # Ring buffer of recent grayscale frames for motion detection
            recent_grays = deque(maxlen=4)

            desc = (f"  [{vid_idx + 1}/{len(videos)}] {video_name}")
            pbar = tqdm(total=total_frames, desc=desc, unit="fr",
                        leave=False, ncols=100)

            while True:
                ok, bgr = cap.read()
                if not ok:
                    break

                if frame_idx % args.frame_skip != 0:
                    frame_idx += 1
                    pbar.update(1)
                    continue

                gray = to_gray01(bgr)
                t = time_offset + frame_idx / fps

                # Build motion map from consecutive frame differences
                recent_grays.append(gray)
                motion_map = None
                if len(recent_grays) >= 2:
                    diffs = []
                    for k in range(1, len(recent_grays)):
                        diffs.append(np.abs(recent_grays[k] - recent_grays[k - 1]))
                    motion_map = np.mean(diffs, axis=0).astype(np.float32)

                centroid, area, contour, confidence = detect_worm(
                    gray, bg_max, dish_mask_bool, last_centroid,
                    args.min_area, args.max_area, args.roi_px,
                    args.max_jump_px, lost_count, prev_area=prev_area,
                    motion_map=motion_map, grid_baseline=grid_baseline)

                if centroid is not None:
                    cx, cy = centroid
                    lost_count = 0
                    vid_detections += 1
                    total_detections += 1

                    # Orientation
                    body_angle, head_angle = orientation_tracker.update(
                        centroid, contour, area)

                    # Midline extraction
                    body_length_px = 0.0
                    midline = None
                    curvature = None
                    if args.extract_midline and contour is not None:
                        midline, curvature, body_length_px = extract_midline(
                            contour, prev_midline=prev_midline,
                            n_points=args.midline_points,
                            frame_shape=(H, W))
                        if midline is not None:
                            prev_midline = midline
                            _accumulate_ht_vote(midline, centroid,
                                                last_centroid,
                                                ht_keep_votes, ht_flip_votes)
                            session_midlines.append({
                                'midline': midline,
                                'curvature': curvature,
                                'body_length_px': body_length_px,
                                'frame_idx': frame_idx,
                                'time_s': t,
                                'video_name': video_name,
                                'n_points': len(midline),
                            })

                    # Speed
                    if last_centroid is not None and last_time is not None:
                        dt = max(1e-6, t - last_time)
                        step_px = math.dist(centroid, last_centroid)
                        speed_px_s = step_px / dt
                        speed_mm_s = speed_px_s * mm_per_px
                    else:
                        speed_px_s = ""
                        speed_mm_s = ""

                    # Reduce confidence near dish edge
                    dist_to_dish_center = math.dist(centroid, dish_center)
                    if dist_to_dish_center > dish_radius * 0.9:
                        confidence *= 0.5

                    writer.writerow([
                        video_name, frame_idx, round(t, 4),
                        round(cx, 2), round(cy, 2),
                        round(cx * mm_per_px, 4),
                        round(cy * mm_per_px, 4),
                        round(body_angle, 2) if body_angle is not None else "",
                        round(head_angle, 2) if head_angle is not None else "",
                        area,
                        round(speed_px_s, 4) if speed_px_s != "" else "",
                        round(speed_mm_s, 4) if speed_mm_s != "" else "",
                        round(confidence, 4),
                        round(body_length_px, 2) if body_length_px > 0 else "",
                        round(body_length_px * mm_per_px, 4) if body_length_px > 0 else "",
                    ])

                    last_centroid = centroid
                    last_time = t
                    # Update running area average (exponential smoothing)
                    if prev_area is None:
                        prev_area = float(area)
                    else:
                        prev_area = 0.8 * prev_area + 0.2 * float(area)
                else:
                    lost_count += 1

                    # Lost worm recovery: expand search
                    if lost_count > 30:
                        # Will trigger full-dish fallback in detect_worm
                        pass

                    writer.writerow([
                        video_name, frame_idx, round(t, 4),
                        "", "", "", "",
                        "", "", 0,
                        "", "", 0.0,
                        "", ""
                    ])

                # ── Overlay rendering ──────────────────────────────
                if overlay_frame_buf is not None:
                    # Deferred midline overlay: buffer data for post-processing
                    if centroid is not None:
                        overlay_frame_buf.append({
                            'frame_idx': frame_idx,
                            'centroid': centroid,
                            'contour': contour.copy() if contour is not None else None,
                            'midline': midline.copy() if midline is not None else None,
                            'curvature': curvature.copy() if curvature is not None else None,
                            'body_length_px': body_length_px,
                            'speed_mm_s': speed_mm_s,
                            'confidence': confidence,
                            'area': area,
                            'time_s': t,
                        })
                    else:
                        overlay_frame_buf.append({
                            'frame_idx': frame_idx,
                            'centroid': None,
                            'contour': None,
                            'midline': None,
                            'curvature': None,
                            'body_length_px': 0.0,
                            'speed_mm_s': "",
                            'confidence': 0.0,
                            'area': 0,
                            'time_s': t,
                        })
                elif overlay_writer is not None:
                    # ── Original overlay (no midline) ──
                    vis = bgr.copy()

                    # 1. Dish circle — thin cyan ring
                    cv2.circle(vis,
                               (int(dish_center[0]), int(dish_center[1])),
                               int(dish_radius), (255, 255, 0), 1)

                    # 2. ROI circle around last known position
                    if last_centroid is not None:
                        roi_color = (0, 165, 255) if centroid is None else (0, 255, 255)
                        cv2.circle(vis,
                                   (int(last_centroid[0]), int(last_centroid[1])),
                                   args.roi_px, roi_color, 1)

                    if centroid is not None:
                        trail.append(centroid)

                        # 3. Trailing path — fading green polyline
                        if len(trail) >= 2:
                            pts = list(trail)
                            for k in range(1, len(pts)):
                                alpha = k / len(pts)
                                green = int(80 + 175 * alpha)
                                cv2.line(vis,
                                         (int(pts[k-1][0]), int(pts[k-1][1])),
                                         (int(pts[k][0]), int(pts[k][1])),
                                         (0, green, 0), 1)

                        # 4. Worm contour outline — red
                        if contour is not None:
                            cv2.drawContours(vis, [contour], -1, (0, 0, 255), 1)

                        # 5. Centroid dot — solid red
                        cv2.circle(vis,
                                   (int(centroid[0]), int(centroid[1])),
                                   4, (0, 0, 255), -1)

                        # 6. Head direction arrow — magenta
                        if head_angle is not None:
                            arr_len = 30
                            ha_rad = math.radians(head_angle)
                            tip = (int(centroid[0] + arr_len * math.cos(ha_rad)),
                                   int(centroid[1] + arr_len * math.sin(ha_rad)))
                            cv2.arrowedLine(vis,
                                            (int(centroid[0]), int(centroid[1])),
                                            tip, (255, 0, 255), 2,
                                            tipLength=0.3)

                        # 7. Info HUD — top-left text
                        spd_txt = (f"{speed_mm_s:.2f}"
                                   if speed_mm_s != "" else "---")
                        conf_txt = f"{confidence:.2f}"
                        ba_txt = (f"{body_angle:.1f}"
                                  if body_angle is not None else "---")
                        ha_txt = (f"{head_angle:.1f}"
                                  if head_angle is not None else "---")
                        hud_lines = [
                            f"t={t:.1f}s  spd={spd_txt} mm/s",
                            f"conf={conf_txt}  area={area}",
                            f"body={ba_txt}  head={ha_txt}",
                        ]
                        for li, line_txt in enumerate(hud_lines):
                            cv2.putText(vis, line_txt,
                                        (10, 25 + li * 22),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.55, (255, 255, 255), 2)
                            cv2.putText(vis, line_txt,
                                        (10, 25 + li * 22),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.55, (0, 200, 0), 1)
                    else:
                        # Detection failed — LOST label
                        cv2.putText(vis, "LOST",
                                    (W // 2 - 40, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0, (0, 255, 255), 3)
                        cv2.putText(vis, "LOST",
                                    (W // 2 - 40, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0, (0, 0, 255), 2)

                    overlay_writer.write(vis)

                total_frames_processed += 1
                frame_idx += 1
                pbar.update(1)

            pbar.close()
            cap.release()

            if overlay_frame_buf is not None:
                # Stash deferred overlay for post-processing
                overlay_videos_deferred.append({
                    'source_path': video_path,
                    'output_path': overlay_path,
                    'fps': fps,
                    'frames': overlay_frame_buf,
                })
            elif overlay_writer is not None:
                overlay_writer.release()
                tqdm.write(f"    Overlay written: {overlay_path}")

            detect_pct = (vid_detections / max(1, total_frames // args.frame_skip)
                          * 100)
            tqdm.write(f"    {video_name}: {vid_detections} detections "
                       f"({detect_pct:.0f}%)")

    total_pct = (total_detections / max(1, total_frames_processed) * 100)
    print(f"  Session complete: {total_frames_processed} frames processed, "
          f"{total_detections} detections ({total_pct:.1f}%)")
    print(f"  Output: {csv_path}")
    print(f"  Calibration: {calib_path}")

    # ── Write midline NPZ ──
    if args.extract_midline and session_midlines:
        npz_path = os.path.join(output_dir, f"{session_name}_midlines.npz")
        n_ml = len(session_midlines)
        max_pts = args.midline_points

        # Determine head/tail from accumulated votes
        need_flip = ht_flip_votes[0] > ht_keep_votes[0]
        head_is_pt0 = not need_flip
        total_votes = ht_keep_votes[0] + ht_flip_votes[0]
        if total_votes > 0:
            pct = max(ht_keep_votes[0], ht_flip_votes[0]) / total_votes * 100
            print(f"  Head/tail: votes={ht_keep_votes[0]} keep/"
                  f"{ht_flip_votes[0]} flip → "
                  f"{'pt0' if head_is_pt0 else 'ptN'}=head ({pct:.0f}%)")
        else:
            print(f"  Head/tail: no motion evidence, defaulting to pt0=head")

        # Build NaN-padded arrays
        # Curvature length = N-4 for N points; max curvature length
        max_curv = max(max_pts - 4, 1)
        midlines_arr = np.full((n_ml, max_pts, 2), np.nan, dtype=np.float32)
        curvatures_arr = np.full((n_ml, max_curv), np.nan, dtype=np.float32)
        body_lengths = np.zeros(n_ml, dtype=np.float32)
        n_points_arr = np.zeros(n_ml, dtype=np.int16)
        frame_indices = np.zeros(n_ml, dtype=np.int32)
        times_s = np.zeros(n_ml, dtype=np.float32)
        video_names = []

        for i, md in enumerate(session_midlines):
            ml = md['midline']
            cv = md['curvature']
            np_i = md['n_points']

            # Retroactive flip if needed
            if need_flip:
                ml = ml[::-1].copy()
                if cv is not None and len(cv) > 0:
                    cv = -cv[::-1].copy()

            midlines_arr[i, :np_i, :] = ml.astype(np.float32)
            if cv is not None and len(cv) > 0:
                cl = min(len(cv), max_curv)
                curvatures_arr[i, :cl] = cv[:cl].astype(np.float32)
            body_lengths[i] = md['body_length_px']
            n_points_arr[i] = np_i
            frame_indices[i] = md['frame_idx']
            times_s[i] = md['time_s']
            video_names.append(md['video_name'])

        np.savez_compressed(
            npz_path,
            midlines=midlines_arr,
            curvatures=curvatures_arr,
            body_lengths=body_lengths,
            n_points=n_points_arr,
            frame_indices=frame_indices,
            times_s=times_s,
            video_names=np.array(video_names),
            head_is_pt0=head_is_pt0,
            ht_keep_votes=ht_keep_votes[0],
            ht_flip_votes=ht_flip_votes[0],
            mm_per_px=mm_per_px,
        )
        print(f"  Midlines: {npz_path} ({n_ml} frames)")

    # ── Deferred overlay rendering (correct H/T labels) ──
    if args.extract_midline and overlay_videos_deferred:
        head_is_pt0 = not (ht_flip_votes[0] > ht_keep_votes[0])
        print(f"  Rendering {len(overlay_videos_deferred)} overlay video(s) "
              f"with {'pt0' if head_is_pt0 else 'ptN'}=head...")

        ol_trail = deque(maxlen=100)
        ol_body_length_history = deque(maxlen=100)

        for ov in overlay_videos_deferred:
            cap = open_video(ov['source_path'])
            eff_fps = ov['fps'] / max(1, args.frame_skip)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            ov_writer = cv2.VideoWriter(
                ov['output_path'], fourcc, eff_fps, (W, H))

            for fd in tqdm(ov['frames'],
                           desc=f"    Overlay {Path(ov['source_path']).stem}",
                           unit="fr", leave=False):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fd['frame_idx'])
                ok, bgr = cap.read()
                if not ok:
                    continue

                vis = render_midline_overlay(
                    bgr, fd['contour'], fd['centroid'],
                    fd['midline'], fd['curvature'],
                    fd['body_length_px'], fd['speed_mm_s'],
                    fd['confidence'], fd['area'], fd['time_s'],
                    ol_trail, ol_body_length_history,
                    head_is_pt0, H, W)
                ov_writer.write(vis)

            ov_writer.release()
            cap.release()
            tqdm.write(f"    Overlay written: {ov['output_path']}")

    return csv_path


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Automated planarian worm tracker for OpenDishWork sessions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single session:
  python open_dish_tracker.py --data_root ../OpenDishWork --sessions Bubba_0001_021426

  # All sessions:
  python open_dish_tracker.py --data_root ../OpenDishWork

  # Faster processing (skip more frames):
  python open_dish_tracker.py --data_root ../OpenDishWork --frame_skip 5
        """)

    ap.add_argument("--data_root", type=str, default="../OpenDishWork",
                    help="Root directory containing session folders.")
    ap.add_argument("--output_dir", type=str, default=None,
                    help="Output directory for CSVs and calibrations. "
                         "Defaults to data_root/tracking_output.")
    ap.add_argument("--sessions", type=str, nargs="+", default=None,
                    help="Specific session folder names to process. "
                         "Default: all sessions found.")
    ap.add_argument("--frame_skip", type=int, default=3,
                    help="Process every Nth frame (default: 3 → ~3.3 Hz at 10fps).")
    ap.add_argument("--min_area", type=int, default=80,
                    help="Minimum worm blob area in pixels.")
    ap.add_argument("--max_area", type=int, default=3000,
                    help="Maximum worm blob area in pixels.")
    ap.add_argument("--roi_px", type=int, default=120,
                    help="ROI search radius around last known position (pixels).")
    ap.add_argument("--max_jump_px", type=float, default=100,
                    help="Maximum plausible centroid jump between detections (pixels).")
    ap.add_argument("--manual_seed", action="store_true",
                    help="Interactively click the worm's position in the first "
                         "video of each session before tracking begins.")
    ap.add_argument("--write_overlay", action="store_true",
                    help="Generate diagnostic overlay videos showing tracking.")
    ap.add_argument("--overlay_videos", type=int, default=2,
                    help="Number of videos per session to render overlays for "
                         "(first N). Default: 2.")
    ap.add_argument("--extract_midline", action="store_true",
                    help="Enable midline/skeleton extraction for body shape analysis.")
    ap.add_argument("--midline_points", type=int, default=20,
                    help="Maximum midline sample points (default: 20).")

    args = ap.parse_args()

    # Validate midline dependencies
    if args.extract_midline and not _HAS_MIDLINE:
        raise SystemExit(
            "ERROR: --extract_midline requires scikit-image and scipy.\n"
            "  Install with:  pip install scikit-image scipy")

    # Resolve paths
    data_root = os.path.abspath(args.data_root)
    if not os.path.isdir(data_root):
        raise SystemExit(f"Data root not found: {data_root}")

    if args.output_dir is None:
        output_dir = os.path.join(data_root, "tracking_output")
    else:
        output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Discover sessions
    if args.sessions:
        session_dirs = []
        for s in args.sessions:
            d = os.path.join(data_root, s)
            if os.path.isdir(d):
                session_dirs.append(d)
            else:
                print(f"WARNING: Session directory not found: {d}")
    else:
        session_dirs = sorted([
            os.path.join(data_root, d)
            for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d))
            and not d.startswith('.')
            and d != "tracking_output"
        ])

    if not session_dirs:
        raise SystemExit("No session directories found.")

    print(f"Open Dish Tracker")
    print(f"  Data root:  {data_root}")
    print(f"  Output dir: {output_dir}")
    print(f"  Sessions:   {len(session_dirs)}")
    print(f"  Frame skip: {args.frame_skip}")
    print(f"  Area range: {args.min_area}–{args.max_area} px")
    print(f"  ROI radius: {args.roi_px} px")
    print(f"  Max jump:   {args.max_jump_px} px")
    if args.write_overlay:
        print(f"  Overlay:    enabled (first {args.overlay_videos} videos/session)")
    if args.extract_midline:
        print(f"  Midline:    enabled ({args.midline_points} points)")
    print()

    t_start = time.time()

    for sess_idx, session_dir in enumerate(session_dirs):
        session_name = os.path.basename(session_dir)
        videos = get_session_videos(session_dir)
        n_vids = len(videos)

        print(f"{'='*60}")
        print(f"Session {sess_idx + 1}/{len(session_dirs)}: "
              f"{session_name} ({n_vids} videos)")
        print(f"{'='*60}")

        if n_vids == 0:
            print("  No videos found, skipping.\n")
            continue

        # Calibrate from first video
        try:
            calib = calibrate_session(session_dir, videos)
            print(f"  Dish: center=({calib['dish_center'][0]:.0f}, "
                  f"{calib['dish_center'][1]:.0f}), "
                  f"radius={calib['dish_radius_px']:.0f}px")
            print(f"  Grid: spacing={calib['grid_spacing_px']:.1f}px, "
                  f"scale={calib['mm_per_px']:.5f} mm/px")
            diam_mm = 2 * calib['dish_radius_px'] * calib['mm_per_px']
            print(f"  Dish diameter: {diam_mm:.1f} mm")
        except RuntimeError as e:
            print(f"  ERROR: Calibration failed: {e}")
            print(f"  Skipping session.\n")
            continue

        # Load seed: check for saved seed file first, then manual seed
        seed = None
        seed_file = os.path.join(session_dir, f"{session_name}_seed.json")
        if os.path.exists(seed_file):
            try:
                with open(seed_file) as sf:
                    sd = json.load(sf)
                seed = (sd["seed_frame"], tuple(sd["seed_position"]))
                print(f"  Seed (from file): frame {seed[0]}, "
                      f"pos ({seed[1][0]:.0f}, {seed[1][1]:.0f})")
            except Exception as e:
                print(f"  WARNING: Could not load seed file: {e}")
        elif args.manual_seed:
            print(f"  Manual seeding: opening {os.path.basename(videos[0])}")
            print(f"    Phase 1: Advance frames (d/f/g) until you see the worm.")
            print(f"             Press W when the worm appears.")
            print(f"    Phase 2: Click on the worm inside the cyan circle,")
            print(f"             then press Enter to confirm.")
            dc = tuple(calib['dish_center'])
            dr = calib['dish_radius_px']
            sf, sp = manual_seed_worm(videos[0], dish_center=dc,
                                      dish_radius=dr)
            if sp is not None:
                seed = (sf, sp)
                # Save seed for future runs
                seed_data = {
                    "seed_frame": sf,
                    "seed_position": [sp[0], sp[1]],
                    "source_video": os.path.basename(videos[0]),
                    "source": "manual_seed"
                }
                with open(seed_file, 'w') as f:
                    json.dump(seed_data, f, indent=2)
                print(f"    Seed: frame {sf}, pos ({sp[0]:.0f}, {sp[1]:.0f})")
                print(f"    Saved to {seed_file}")
            else:
                print(f"    No seed provided, using auto-detection.")

        # Process all videos
        try:
            process_session(session_dir, calib, args, output_dir, seed=seed)
        except Exception as e:
            print(f"  ERROR processing session: {e}")
            import traceback
            traceback.print_exc()

        print()

    elapsed = time.time() - t_start
    print(f"{'='*60}")
    print(f"All sessions complete. Total time: {elapsed:.1f}s "
          f"({elapsed/60:.1f} min)")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
