#!/usr/bin/env python3
"""
label_setup.py — Human labeling for the fixed planarian rig.

Since the camera + dish + lighting are locked in, you label the geometry ONCE
by clicking, and tag ground-truth worm positions so tracker accuracy can be
measured (not guessed). Output is a JSON the tracker/report read.

Four stages (run all, or pick with --stage):
  1. dish    — click dish CENTER, then a point on the dish EDGE -> center+radius
  2. scale   — click two grid intersections exactly 1 cm apart -> mm/px
  3. start   — scrub to the frame where the run begins, press 's' to mark it
  4. worm    — for N sampled frames, click the worm -> ground-truth positions

Controls (all stages):
  left-click = place point     n / SPACE = next     b = back a frame
  , / .      = step -10 / +10 frames (scrub stages)
  u          = undo last click
  s          = save current stage / mark start frame
  q / ESC    = quit (saves what's done)

Output JSON (default ../realtime_runs/<session>_labels.json):
  {
    "dish_center":[cx,cy], "dish_radius_px":r,
    "mm_per_px":m, "grid_cm":1.0,
    "run_start_frame":f, "run_start_video":"...",
    "worm_truth":[{"video":..., "frame":..., "x_px":..., "y_px":...}, ...],
    "frame_size":[w,h]
  }

Usage:
  cd scripts_notebooks
  source ../venv/bin/activate
  python label_setup.py --clips_dir ../live_capture --session pilot
  python label_setup.py --clips_dir ../live_capture --session pilot --stage worm --n_worm 40

Python 3.9+. Requires opencv-python with GUI (it opens windows on your Mac).
"""

import os
import sys
import csv
import glob
import json
import math
import argparse

import cv2
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
from open_dish_tracker import open_video  # noqa: E402
from sessions import list_session_clips  # noqa: E402

VIDEO_EXTS = (".mkv", ".mp4", ".avi", ".mov")
WIN = "Planarian Labeler"
DISP_MAX_W = 1500   # window is scaled down to fit; clicks map back to full res


def load_existing(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_labels(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  saved -> {path}")


class Frame:
    """Holds a frame + display scaling so clicks map to full-res pixels."""
    def __init__(self, bgr):
        self.full = bgr
        h, w = bgr.shape[:2]
        self.scale = min(1.0, DISP_MAX_W / w)
        self.disp = cv2.resize(bgr, (int(w * self.scale), int(h * self.scale)))

    def to_full(self, dx, dy):
        return (dx / self.scale, dy / self.scale)

    def click_to_full(self, x, y):
        """Map a window click to full-res pixels, accounting for the
        instruction bar that sits ABOVE the frame (height PANEL_H)."""
        return self.to_full(x, y - PANEL_H)


def read_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, f = cap.read()
    return f if ok else None


# ── On-screen instructions ────────────────────────────────────────────
# Everything a labeler needs is drawn on the image so the tool is
# self-contained for people other than the author. Each stage passes its
# title, the action to do right now, and the key legend.
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, s, org, scale, color, thick=2, shadow=True):
    """Text with a dark outline so it reads on any background."""
    if shadow:
        cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


# Height of the instruction bar drawn ABOVE the video (so nothing covers the
# dish/worm). Mouse callbacks subtract this from the click y to map back onto
# the frame. Kept as a module constant so stages and callbacks agree.
PANEL_H = 120


def with_panel(disp, title, step, keys, sub=None, status=None):
    """Return a new image: a solid instruction bar stacked ABOVE `disp`.

    The video frame is left untouched and sits below the bar, so the worm and
    dish are never covered. Returns the composited image; the caller shows it.
    The bar is PANEL_H px tall — callbacks offset clicks by that amount.

    title  - stage name, e.g. "STAGE 1 of 4 - DISH"
    step   - the single action to do right now (big, yellow)
    keys   - list of "key = meaning" strings shown as a legend
    sub    - optional smaller line under `step` (context/why)
    status - optional readout shown top-right (e.g. measured value)
    """
    h, w = disp.shape[:2]
    bar = np.full((PANEL_H, w, 3), 25, np.uint8)
    cv2.line(bar, (0, PANEL_H - 1), (w, PANEL_H - 1), (0, 220, 0), 2)
    _text(bar, title, (16, 28), 0.7, (0, 220, 0))
    _text(bar, step, (16, 62), 0.8, (0, 255, 255))
    if sub:
        _text(bar, sub, (16, 88), 0.55, (200, 200, 200), thick=1)
    legend = "   ".join(keys)
    _text(bar, legend, (16, PANEL_H - 12), 0.52, (255, 255, 255), thick=1)
    if status:
        (tw, _), _ = cv2.getTextSize(status, FONT, 0.7, 2)
        _text(bar, status, (w - tw - 16, 34), 0.7, (0, 255, 0))
    return np.vstack([bar, disp])


def splash(size, title, lines, footer="Press SPACE or ENTER to continue  (Q = quit)"):
    """Full-screen instruction card shown between stages. Returns False if quit."""
    w, h = size
    while True:
        img = np.full((h, w, 3), 30, np.uint8)
        _text(img, title, (50, 90), 1.1, (0, 255, 255), thick=2)
        cv2.line(img, (50, 110), (w - 50, 110), (0, 220, 0), 2)
        y = 170
        for ln in lines:
            big = ln.startswith("* ")
            _text(img, ln[2:] if big else ln, (60 if big else 80, y),
                  0.75 if big else 0.6,
                  (255, 255, 255) if big else (200, 200, 200),
                  thick=2 if big else 1)
            y += 46 if big else 34
        _text(img, footer, (50, h - 40), 0.65, (0, 255, 0))
        cv2.imshow(WIN, img)
        k = cv2.waitKey(20) & 0xFF
        if k in (ord(' '), 13, 10):
            return True
        if k in (ord('q'), 27):
            return False


# ── Stage 1: dish ─────────────────────────────────────────────────────
def stage_dish(frame, labels):
    clicks = []

    def on_mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
            clicks.append(frame.click_to_full(x, y))

    cv2.setMouseCallback(WIN, on_mouse)
    print("\n[DISH] Click 1) the dish CENTER, then 2) a point on the dish EDGE."
          "  u=undo  s=save  q=quit")
    steps = ["Click the CENTER of the dish",
             "Now click a point on the dish EDGE (rim)",
             "Looks right? Press S to save. Press U to redo."]
    while True:
        disp = frame.disp.copy()
        for i, (px, py) in enumerate(clicks):
            dx, dy = int(px * frame.scale), int(py * frame.scale)
            cv2.circle(disp, (dx, dy), 6, (0, 0, 255), -1)
            _text(disp, ["center", "edge"][i], (dx + 10, dy + 5), 0.6,
                  (0, 0, 255))
        status = None
        if len(clicks) == 2:
            c = clicks[0]
            r = math.dist(clicks[0], clicks[1])
            cv2.circle(disp, (int(c[0] * frame.scale), int(c[1] * frame.scale)),
                       int(r * frame.scale), (0, 255, 0), 2)
            status = f"radius = {r:.0f}px"
        cv2.imshow(WIN, with_panel(
            disp, "STAGE 1 of 4  -  DISH", steps[min(len(clicks), 2)],
            ["L-click = place point", "U = undo", "S = save & next", "Q = quit"],
            sub="Defines the arena. The tracker only looks inside this circle.",
            status=status))
        k = cv2.waitKey(20) & 0xFF
        if k == ord('u') and clicks:
            clicks.pop()
        elif k == ord('s') and len(clicks) == 2:
            labels["dish_center"] = list(clicks[0])
            labels["dish_radius_px"] = math.dist(clicks[0], clicks[1])
            print(f"  dish center={labels['dish_center']} "
                  f"r={labels['dish_radius_px']:.1f}px")
            return True
        elif k in (ord('q'), 27):
            return False


# ── Stage 2: scale ────────────────────────────────────────────────────
def stage_scale(frame, labels, grid_cm):
    clicks = []

    def on_mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
            clicks.append(frame.click_to_full(x, y))

    cv2.setMouseCallback(WIN, on_mouse)
    print(f"\n[SCALE] Click two grid intersections exactly {grid_cm} cm apart "
          f"(adjacent grid corners).  u=undo  s=save  q=quit")
    steps = [f"Click ONE grid-line corner",
             f"Click a SECOND corner exactly {grid_cm} cm away (next one over)",
             "Looks right? Press S to save. Press U to redo."]
    while True:
        disp = frame.disp.copy()
        for px, py in clicks:
            cv2.circle(disp, (int(px * frame.scale), int(py * frame.scale)),
                       6, (255, 0, 255), -1)
        status = None
        if len(clicks) == 2:
            cv2.line(disp,
                     tuple(int(v * frame.scale) for v in clicks[0]),
                     tuple(int(v * frame.scale) for v in clicks[1]),
                     (255, 0, 255), 2)
            dpx = math.dist(clicks[0], clicks[1])
            mmpp = (grid_cm * 10.0) / dpx if dpx else 0
            status = f"{dpx:.0f}px = {grid_cm}cm -> {mmpp:.4f} mm/px"
        cv2.imshow(WIN, with_panel(
            disp, "STAGE 2 of 4  -  SCALE", steps[min(len(clicks), 2)],
            ["L-click = place point", "U = undo", "S = save & next", "Q = quit"],
            sub=f"Two intersections {grid_cm} cm apart set the mm-per-pixel scale.",
            status=status))
        k = cv2.waitKey(20) & 0xFF
        if k == ord('u') and clicks:
            clicks.pop()
        elif k == ord('s') and len(clicks) == 2:
            dpx = math.dist(clicks[0], clicks[1])
            labels["mm_per_px"] = (grid_cm * 10.0) / dpx
            labels["grid_cm"] = grid_cm
            print(f"  mm_per_px={labels['mm_per_px']:.6f}")
            return True
        elif k in (ord('q'), 27):
            return False


# ── Stage 3: run start ────────────────────────────────────────────────
def stage_start(clips, labels):
    print("\n[START] Scrub to the frame where the run begins. "
          ", / . step -10/+10,  b/n step -1/+1,  s=mark start,  q=quit")
    ci, fi = 0, 0
    cap = open_video(clips[ci])
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1800
    cv2.setMouseCallback(WIN, lambda *a: None)
    while True:
        f = read_frame(cap, fi)
        if f is None:
            fi = max(0, min(fi, total - 1))
            f = read_frame(cap, fi) or np.zeros((400, 600, 3), np.uint8)
        fr = Frame(f)
        disp = fr.disp.copy()
        cv2.imshow(WIN, with_panel(
            disp, "STAGE 3 of 4  -  RUN START",
            "Scrub to the frame where the run begins, then press S",
            [". / , = +10/-10 frames", "n / b = +1/-1 frame",
             "S = mark start", "Q = quit"],
            sub="The frame the experiment/trial actually starts on.",
            status=f"frame {fi}/{total}"))
        k = cv2.waitKey(20) & 0xFF
        if k == ord('.'):
            fi = min(total - 1, fi + 10)
        elif k == ord(','):
            fi = max(0, fi - 10)
        elif k == ord('n'):
            fi = min(total - 1, fi + 1)
        elif k == ord('b'):
            fi = max(0, fi - 1)
        elif k == ord('s'):
            labels["run_start_frame"] = int(fi)
            labels["run_start_video"] = os.path.basename(clips[ci])
            print(f"  run start: {clips[ci]} frame {fi}")
            cap.release()
            return True
        elif k in (ord('q'), 27):
            cap.release()
            return False


# ── Stage 4: worm ground truth ────────────────────────────────────────
def stage_worm(clips, labels, n_worm):
    print(f"\n[WORM] Click the worm on each sampled frame ({n_worm} total). "
          "click=place, u=undo, n/SPACE=next (skip if no worm), s=save, q=quit")
    # Build a list of (clip, frame) samples spread across all clips.
    samples = []
    per = max(1, n_worm // max(1, len(clips)))
    for c in clips:
        cap = open_video(c)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1800
        cap.release()
        for fr in np.linspace(total * 0.1, total * 0.9, per, dtype=int):
            samples.append((c, int(fr)))
    samples = samples[:n_worm]

    truth = labels.get("worm_truth", [])
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    i = 0
    cur_cap = None
    cur_path = None
    while i < len(samples):
        c, fr = samples[i]
        if cur_path != c:
            if cur_cap:
                cur_cap.release()
            cur_cap = open_video(c)
            cur_path = c
        f = read_frame(cur_cap, fr)
        if f is None:
            i += 1
            continue
        frame = Frame(f)
        click = {"pt": None}

        def on_mouse(ev, x, y, flags, _):
            if ev == cv2.EVENT_LBUTTONDOWN:
                click["pt"] = frame.click_to_full(x, y)

        cv2.setMouseCallback(WIN, on_mouse)
        while True:
            disp = frame.disp.copy()
            step = ("Click the WORM, then press N for the next frame"
                    if not click["pt"] else
                    "Got it. Press N for next frame (or U to redo)")
            if click["pt"]:
                px, py = click["pt"]
                cv2.circle(disp, (int(px * frame.scale), int(py * frame.scale)),
                           7, (0, 0, 255), 2)
            cv2.imshow(WIN, with_panel(
                disp, f"STAGE 4 of 4  -  WORM   ({i+1} of {len(samples)})", step,
                ["L-click = mark worm", "N / Space = next (skip if none)",
                 "U = undo", "S = save & finish", "Q = quit"],
                sub="If you can't find the worm in a frame, just press N to skip it.",
                status=f"{len(truth)} marked"))
            k = cv2.waitKey(20) & 0xFF
            if k == ord('u'):
                click["pt"] = None
            elif k in (ord('n'), ord(' ')):
                if click["pt"]:
                    truth.append({"video": os.path.basename(c), "frame": fr,
                                  "x_px": round(click["pt"][0], 1),
                                  "y_px": round(click["pt"][1], 1)})
                i += 1
                break
            elif k == ord('s'):
                if click["pt"]:
                    truth.append({"video": os.path.basename(c), "frame": fr,
                                  "x_px": round(click["pt"][0], 1),
                                  "y_px": round(click["pt"][1], 1)})
                labels["worm_truth"] = truth
                if cur_cap:
                    cur_cap.release()
                print(f"  saved {len(truth)} worm points")
                return True
            elif k in (ord('q'), 27):
                labels["worm_truth"] = truth
                if cur_cap:
                    cur_cap.release()
                return True
    labels["worm_truth"] = truth
    if cur_cap:
        cur_cap.release()
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips_dir", required=True, help="Folder of clips.")
    ap.add_argument("--session", default="S1",
                    help="Recording session to label (S1, S2, ... or 'all'). "
                         "Clips are grouped from --clips_dir by capture-time "
                         "gaps; only this session's clips are shown.")
    ap.add_argument("--output_dir", default="../realtime_runs")
    ap.add_argument("--stage", choices=["all", "dish", "scale", "start", "worm"],
                    default="all")
    ap.add_argument("--grid_cm", type=float, default=1.0,
                    help="Real distance between the two grid points you click (cm).")
    ap.add_argument("--n_worm", type=int, default=30,
                    help="How many frames to click the worm on (default: 30).")
    args = ap.parse_args()

    clips = list_session_clips(args.clips_dir, args.session)
    if not clips:
        sys.exit(f"No clips for session {args.session} in {args.clips_dir}")
    print(f"Session {args.session}: {len(clips)} clips "
          f"({os.path.basename(clips[0])} … {os.path.basename(clips[-1])})")
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f"{args.session}_labels.json")
    labels = load_existing(out)

    # A representative frame (middle of the first clip) for dish/scale stages.
    cap = open_video(clips[0])
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 600
    rep = read_frame(cap, total // 2)
    cap.release()
    if rep is None:
        sys.exit("Could not read a frame from the first clip.")
    labels["frame_size"] = [rep.shape[1], rep.shape[0]]
    frame = Frame(rep)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    # Stage views are the frame plus the instruction bar stacked on top, so the
    # window (and the splash cards) use that combined height.
    dw, dh = frame.disp.shape[1], frame.disp.shape[0] + PANEL_H
    cv2.resizeWindow(WIN, dw, dh)
    cv2.moveWindow(WIN, 60, 60)

    # Per-stage intro cards so a first-time labeler knows what each step is.
    INTRO = {
        "dish": ("STAGE 1 of 4 - DISH", [
            "* Tell the tracker where the arena is.",
            "On the picture of the dish you will:",
            "  1. Click the CENTER of the dish.",
            "  2. Click a point on the EDGE (rim).",
            "A green circle shows what you defined. Press S when it fits.",
            "Use U to undo a click, Q to quit (your progress is saved)."]),
        "scale": (f"STAGE 2 of 4 - SCALE", [
            "* Tell the tracker how big a pixel is in real life.",
            f"Click two grid-line corners that are {args.grid_cm} cm apart",
            "(one square on the grid). The tool computes mm-per-pixel.",
            "Press S to save, U to undo."]),
        "start": ("STAGE 3 of 4 - RUN START", [
            "* Mark the frame where the run begins.",
            "Scrub through the video:  . and , jump 10 frames,",
            "n and b step 1 frame at a time.",
            "Press S on the frame where the trial starts."]),
        "worm": (f"STAGE 4 of 4 - WORM ({args.n_worm} frames)", [
            "* Show the tracker where the worm really is.",
            "You'll see sampled frames one at a time. On each:",
            "  - click the worm, then press N for the next frame.",
            "  - can't find it? just press N to skip that frame.",
            "These clicks are the ground truth accuracy is measured against.",
            "Press S any time to save and finish early."]),
    }

    order = ["dish", "scale", "start", "worm"] if args.stage == "all" \
        else [args.stage]
    for st in order:
        title, lines = INTRO[st]
        if not splash((dw, dh), title, lines):
            break
        if st == "dish":
            stage_dish(frame, labels); save_labels(out, labels)
        elif st == "scale":
            stage_scale(frame, labels, args.grid_cm); save_labels(out, labels)
        elif st == "start":
            stage_start(clips, labels); save_labels(out, labels)
        elif st == "worm":
            stage_worm(clips, labels, args.n_worm); save_labels(out, labels)

    have = [k for k in labels if k not in ("frame_size",)]
    splash((dw, dh), "DONE - labels saved", [
        "* This session is labeled.",
        f"Saved to: {os.path.basename(out)}",
        f"Captured: {', '.join(have) if have else '(nothing)'}",
        "",
        "You can close this window. Tell Claude this session is labeled",
        "and it will run the tracker + accuracy report."],
        footer="Press any key to close.")
    cv2.destroyAllWindows()
    print(f"\nLabels written to {out}")
    print("Keys present:", ", ".join(k for k in labels if k != "frame_size"))


if __name__ == "__main__":
    main()
