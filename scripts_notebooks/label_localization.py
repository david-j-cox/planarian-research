#!/usr/bin/env python3
"""
label_localization.py — Model-in-the-loop ground-truth labeling for worm
localization. Three subcommands:

  prep    Run the current model over sampled frames from the target videos and
          build a labeling queue. Two pools:
            - EVAL  : random/unbiased frames -> honest accuracy measurement.
            - ACTIVE: frames the model is least sure about (0 boxes, >=2 boxes,
                      or low confidence) -> most informative for training.
          Caches the selected frame images to disk and writes manifest.json +
          proposals.json (the model's proposed boxes).

  label   GUI to review each frame: the model's proposed box is shown; you
          accept it, redraw it, or mark the frame as having no worm. Optional
          head click for orientation. Resumable; saves after every frame.

  export  Convert accepted labels into (a) a YOLO detection dataset (images +
          labels, negatives included as backgrounds) and (b) an eval GT json
          compatible with eval_localizer.py.

GUI keys (label):
  a            accept the model's proposed box as truth
  left-drag    draw/replace the box (marks worm present)
  n            mark NO worm in this frame (negative/background)
  h then click set the head point (orientation); c clears it
  u            undo current frame's decision
  ENTER/SPACE  next  (must decide first)   b  back
  s / q        save and quit

Usage:
  python label_localization.py prep \
      --videos "/path/PlanarianVideos/tier2_corpus/live_capture/2026-06-02_16-*.mkv" \
      --model runs/pose/worm_s3_n/weights/best.pt \
      --out_dir ../realtime_runs/label_white7mp --n_eval 200 --n_active 400
  python label_localization.py label  --out_dir ../realtime_runs/label_white7mp
  python label_localization.py export --out_dir ../realtime_runs/label_white7mp \
      --mm_per_px 0.02657 --dataset ../realtime_runs/yolo_white7mp
"""
import argparse
import glob
import json
import os
import numpy as np
import cv2


# ── helpers ──────────────────────────────────────────────────────────
def resolve_videos(spec):
    out = []
    for s in spec:
        if os.path.isdir(s):
            out += sorted(glob.glob(os.path.join(s, "*.mkv")) + glob.glob(os.path.join(s, "*.mp4")))
        else:
            out += sorted(glob.glob(s))
    return [v for v in out if not v.endswith("_overlay.mp4")]


def informativeness(rec):
    """Higher = more useful to label for training."""
    if rec["ndet"] == 0:
        return 1.0          # possible miss
    if rec["ndet"] >= 2:
        return 0.85         # possible false positive / duplicate
    return 1.0 - rec["conf"]  # low confidence = informative


# ── prep ─────────────────────────────────────────────────────────────
def cmd_prep(a):
    from ultralytics import YOLO
    model = YOLO(a.model)
    rng = np.random.default_rng(a.seed)
    videos = resolve_videos(a.videos)
    if not videos:
        print("no videos matched"); return
    print(f"scanning {len(videos)} videos, ~{a.scan_per_video} frames each")

    records = []
    for v in videos:
        cap = cv2.VideoCapture(v)
        nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if nf <= 0:
            cap.release(); continue
        take = min(a.scan_per_video, nf)
        frames = sorted(rng.choice(nf, size=take, replace=False).tolist())
        for f in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                continue
            r = model.predict(img, imgsz=a.imgsz, conf=a.conf, iou=0.5,
                              agnostic_nms=True, device=a.device, verbose=False)[0]
            boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
            confs = r.boxes.conf.cpu().numpy() if len(boxes) else np.zeros(0)
            if len(boxes):
                bi = int(np.argmax(confs))
                box = [float(x) for x in boxes[bi]]; conf = float(confs[bi])
            else:
                box, conf = None, 0.0
            records.append({"video": os.path.basename(v), "vpath": v, "frame": int(f),
                            "box": box, "conf": conf, "ndet": int(len(boxes))})
        cap.release()
        print(f"  {os.path.basename(v)}: scanned")

    n = len(records)
    print(f"scanned {n} frames total")
    idx = list(range(n)); rng.shuffle(idx)
    eval_ids = set(idx[:a.n_eval])
    rest = [i for i in idx if i not in eval_ids]
    rest.sort(key=lambda i: -informativeness(records[i]))
    active_ids = set(rest[:a.n_active])
    selected = sorted(eval_ids | active_ids)

    os.makedirs(a.out_dir, exist_ok=True)
    cache = os.path.join(a.out_dir, "frames"); os.makedirs(cache, exist_ok=True)

    # Extract selected frames (grouped per video, sequential for efficiency).
    by_vid = {}
    for i in selected:
        by_vid.setdefault(records[i]["vpath"], []).append(i)
    manifest = []
    for vpath, ids in by_vid.items():
        ids.sort(key=lambda i: records[i]["frame"])
        cap = cv2.VideoCapture(vpath)
        for i in ids:
            f = records[i]["frame"]
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                continue
            stem = os.path.splitext(records[i]["video"])[0]
            name = f"{stem}_f{f:06d}.jpg"
            cv2.imwrite(os.path.join(cache, name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            manifest.append({"id": len(manifest), "image": name,
                             "video": records[i]["video"], "frame": f,
                             "pool": "eval" if i in eval_ids else "active",
                             "proposal": records[i]["box"], "conf": records[i]["conf"],
                             "ndet": records[i]["ndet"]})
        cap.release()
    json.dump({"frame_dir": "frames", "items": manifest},
              open(os.path.join(a.out_dir, "manifest.json"), "w"), indent=2)
    n_eval = sum(1 for m in manifest if m["pool"] == "eval")
    print(f"\nprepared {len(manifest)} frames to label: eval={n_eval} active={len(manifest)-n_eval}")
    print(f"  -> {a.out_dir}/manifest.json   (now run: label --out_dir {a.out_dir})")


# ── label (GUI) ──────────────────────────────────────────────────────
def cmd_label(a):
    man = json.load(open(os.path.join(a.out_dir, "manifest.json")))
    items = man["items"]
    cache = os.path.join(a.out_dir, man["frame_dir"])
    labels_path = os.path.join(a.out_dir, "labels.json")
    labels = {}
    if os.path.exists(labels_path):
        labels = {int(k): v for k, v in json.load(open(labels_path)).items()}
    print(f"{len(items)} frames; {len(labels)} already labeled")

    WIN = "Localization Labeler"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    max_w = a.max_w
    state = {"idx": next((i for i in range(len(items)) if items[i]["id"] not in labels), 0),
             "drawing": False, "p0": None, "cur_box": None, "head_mode": False,
             "head": None, "scale": 1.0}

    def on_mouse(ev, x, y, flags, _):
        s = state["scale"]
        if state["head_mode"] and ev == cv2.EVENT_LBUTTONDOWN:
            state["head"] = [x / s, y / s]; state["head_mode"] = False; return
        if ev == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True; state["p0"] = (x, y); state["cur_box"] = None
        elif ev == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["cur_box"] = (state["p0"][0], state["p0"][1], x, y)
        elif ev == cv2.EVENT_LBUTTONUP and state["drawing"]:
            state["drawing"] = False
            x0, y0 = state["p0"];
            bx0, by0, bx1, by1 = min(x0, x)/s, min(y0, y)/s, max(x0, x)/s, max(y0, y)/s
            state["cur_box"] = None
            state["committed_box"] = [bx0, by0, bx1, by1]
    cv2.setMouseCallback(WIN, on_mouse)

    def save():
        json.dump({str(k): v for k, v in labels.items()}, open(labels_path, "w"), indent=2)

    while True:
        i = state["idx"]
        if i < 0 or i >= len(items):
            break
        it = items[i]
        img = cv2.imread(os.path.join(cache, it["image"]))
        H, W = img.shape[:2]
        scale = min(1.0, max_w / W); state["scale"] = scale
        disp = cv2.resize(img, (int(W*scale), int(H*scale)))
        cur = labels.get(it["id"])
        # box to show: committed > existing label > proposal
        box = state.pop("committed_box", None)
        if box is not None:
            labels[it["id"]] = {"present": True, "box": box,
                                "head": state["head"], "pool": it["pool"]}
            cur = labels[it["id"]]
        show_box = cur["box"] if cur and cur.get("box") else it["proposal"]
        color = (0, 255, 0) if (cur and cur.get("present")) else (0, 200, 255)
        if cur and cur.get("present") is False:
            color = (0, 0, 255)
        if show_box and not (cur and cur.get("present") is False):
            x0, y0, x1, y1 = [int(c*scale) for c in show_box]
            cv2.rectangle(disp, (x0, y0), (x1, y1), color, 2)
        hd = (cur or {}).get("head") or state["head"]
        if hd:
            cv2.circle(disp, (int(hd[0]*scale), int(hd[1]*scale)), 6, (255, 0, 255), -1)
        if state["cur_box"]:
            cv2.rectangle(disp, state["cur_box"][:2], state["cur_box"][2:], (255, 255, 0), 1)
        done = sum(1 for it2 in items if it2["id"] in labels)
        status = (f"[{i+1}/{len(items)}] {it['video']} f{it['frame']} pool={it['pool']} "
                  f"conf={it['conf']:.2f} ndet={it['ndet']} | labeled={done} | "
                  f"{'NO-WORM' if cur and cur.get('present') is False else ('SET' if cur else 'undecided')}"
                  f"{'  [HEAD-CLICK]' if state['head_mode'] else ''}")
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(disp, status, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Always-visible controls legend along the bottom.
        Hd = disp.shape[0]
        legend = ("a=ACCEPT box   drag=DRAW box   n=NO worm   ENTER/SPACE=next   "
                  "b=back   u=undo   h+click=head   s/q/ESC or red-X=SAVE+QUIT")
        cv2.rectangle(disp, (0, Hd - 28), (disp.shape[1], Hd), (0, 0, 0), -1)
        cv2.putText(disp, legend, (6, Hd - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)
        cv2.imshow(WIN, disp)
        # Quit if the user clicks the window's red close button.
        try:
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                save(); break
        except cv2.error:
            save(); break
        k = cv2.waitKey(20) & 0xFF
        if k in (ord('s'), ord('q'), 27):  # s / q / ESC
            save(); break
        elif k == ord('a'):  # accept proposal
            if it["proposal"]:
                labels[it["id"]] = {"present": True, "box": it["proposal"],
                                    "head": state["head"], "pool": it["pool"]}
        elif k == ord('n'):  # no worm
            labels[it["id"]] = {"present": False, "box": None, "head": None, "pool": it["pool"]}
        elif k == ord('h'):
            state["head_mode"] = True
        elif k == ord('c'):
            state["head"] = None
            if it["id"] in labels: labels[it["id"]]["head"] = None
        elif k == ord('u'):
            labels.pop(it["id"], None); state["head"] = None
        elif k in (ord('b'),):
            save(); state["idx"] = max(0, i-1); state["head"] = None
        elif k in (13, 32):  # enter/space -> next
            if it["id"] in labels:
                save(); state["idx"] = i+1; state["head"] = None
    cv2.destroyAllWindows()
    save()
    print(f"saved {sum(1 for it in items if it['id'] in labels)} labels -> {labels_path}")


# ── export ───────────────────────────────────────────────────────────
def cmd_export(a):
    man = json.load(open(os.path.join(a.out_dir, "manifest.json")))
    items = {it["id"]: it for it in man["items"]}
    cache = os.path.join(a.out_dir, man["frame_dir"])
    labels = {int(k): v for k, v in json.load(open(os.path.join(a.out_dir, "labels.json"))).items()}

    ds = a.dataset
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(ds, sub), exist_ok=True)
    eval_gt = []
    n = {"train": 0, "val": 0, "neg": 0}
    for cid, lab in labels.items():
        it = items[cid]
        split = "val" if it["pool"] == "eval" else "train"
        src = os.path.join(cache, it["image"])
        img = cv2.imread(src)
        if img is None:
            continue
        H, W = img.shape[:2]
        base = os.path.splitext(it["image"])[0]
        cv2.imwrite(os.path.join(ds, f"images/{split}", base + ".jpg"), img)
        lpath = os.path.join(ds, f"labels/{split}", base + ".txt")
        if lab.get("present") and lab.get("box"):
            x0, y0, x1, y1 = lab["box"]
            xc, yc = (x0+x1)/2/W, (y0+y1)/2/H
            bw, bh = abs(x1-x0)/W, abs(y1-y0)/H
            open(lpath, "w").write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
            n[split] += 1
            if it["pool"] == "eval":
                eval_gt.append({"video": it["video"], "frame": it["frame"],
                                "x_px": (x0+x1)/2, "y_px": (y0+y1)/2})
        else:
            open(lpath, "w").write("")  # negative / background
            n["neg"] += 1
    with open(os.path.join(ds, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(ds)}\ntrain: images/train\nval: images/val\n"
                "names:\n  0: worm\n")
    json.dump({"mm_per_px": a.mm_per_px, "worm_truth": eval_gt},
              open(os.path.join(a.out_dir, "eval_gt.json"), "w"), indent=2)
    print(f"exported dataset -> {ds}  (train={n['train']} val={n['val']} negatives={n['neg']})")
    print(f"eval GT ({len(eval_gt)} positives) -> {a.out_dir}/eval_gt.json")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prep")
    p.add_argument("--videos", nargs="+", required=True)
    p.add_argument("--model", default=os.path.join(here, "runs", "pose", "worm_s3_n", "weights", "best.pt"))
    p.add_argument("--out_dir", required=True)
    p.add_argument("--scan_per_video", type=int, default=120)
    p.add_argument("--n_eval", type=int, default=200)
    p.add_argument("--n_active", type=int, default=400)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_prep)

    p = sub.add_parser("label")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_w", type=int, default=1500)
    p.set_defaults(func=cmd_label)

    p = sub.add_parser("export")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--mm_per_px", type=float, required=True)
    p.set_defaults(func=cmd_export)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
