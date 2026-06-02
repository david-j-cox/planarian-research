#!/usr/bin/env python3
"""
probe_cameras.py — Find which camera index is your USB camera.

OpenCV camera indices on macOS don't match any predictable order — the
built-in FaceTime cam, an iPhone Continuity Camera, and a USB camera can land
on 0/1/2 in any arrangement. This opens indices 0..N, reports which ones work
and at what resolution, and saves one snapshot per working index so you can
eyeball which is the USB camera pointed at your dish.

Usage:
  cd scripts_notebooks
  source ../venv/bin/activate
  python probe_cameras.py                 # probes indices 0..5
  python probe_cameras.py --max_index 8   # probe further

Look in ./camera_probe/ for cam0.jpg, cam1.jpg, ... — whichever shows your
dish is the index to pass to realtime_tracker.py as --camera N.
"""

import os
import argparse
import cv2


def preview(idx):
    """Live viewfinder for one camera index. Use this to aim the camera and
    frame the dish. QuickTime can't see some USB cameras that OpenCV can, so
    this is the reliable way to preview cam 1. Press q to quit, s to snapshot."""
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        print(f"Could not open camera index {idx}.")
        return
    win = f"Camera {idx} preview  (q quit, s snapshot)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print(f"Live preview of camera {idx}. Aim the camera / center the dish. "
          f"q = quit, s = save snapshot.")
    shot = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("No frame.")
            break
        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('s'):
            path = f"preview_cam{idx}_{shot}.jpg"
            cv2.imwrite(path, frame)
            print(f"  saved {path}")
            shot += 1
    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max_index", type=int, default=5,
                    help="Highest camera index to try (default: 5).")
    ap.add_argument("--out_dir", default="camera_probe",
                    help="Where to save snapshots (default: ./camera_probe).")
    ap.add_argument("--preview", type=int, default=None, metavar="INDEX",
                    help="Open a live viewfinder window for this camera index "
                         "instead of probing (e.g. --preview 1).")
    args = ap.parse_args()

    if args.preview is not None:
        preview(args.preview)
        return

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Probing camera indices 0..{args.max_index}\n")
    working = []

    for idx in range(args.max_index + 1):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"  index {idx}: not opened")
            cap.release()
            continue
        # Some cams need a couple of reads before delivering a real frame.
        frame = None
        for _ in range(5):
            ok, f = cap.read()
            if ok and f is not None:
                frame = f
                break
        if frame is None:
            print(f"  index {idx}: opened but no frame")
            cap.release()
            continue

        h, w = frame.shape[:2]
        path = os.path.join(args.out_dir, f"cam{idx}.jpg")
        cv2.imwrite(path, frame)
        print(f"  index {idx}: WORKS  {w}x{h}  -> {path}")
        working.append(idx)
        cap.release()

    print()
    if not working:
        print("No working cameras found. The USB camera may not be enumerating "
              "(check cable is data-capable, try another port, replug).")
    else:
        print(f"Working indices: {working}")
        print(f"Open the snapshots in {args.out_dir}/ and find the one showing "
              f"your dish. Then run:")
        print(f"  python realtime_tracker.py --camera <that_index>")


if __name__ == "__main__":
    main()
