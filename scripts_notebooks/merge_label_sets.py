#!/usr/bin/env python3
"""
merge_label_sets.py — Merge several blind-label dirs into one training set.

The behavior labels are collected in rounds (an initial stratified `sample`, then
`active`-learning batches). Each round is its own dir (manifest + human_labels +
predictions_hidden) keyed by window_id from 0. This concatenates them into one
combined dir with re-indexed window_ids so behavior_classifier can train on all
rounds at once. Only windows that have a human label are carried over.

predictions_hidden rows are passed through as-is (they may carry rule_pred from a
`sample` round or model_pred from an `active` round; the classifier/accuracy tools
accept either).

Usage:
  python merge_label_sets.py \
      --in ../realtime_runs/white7mp_labels_blind \
           ../realtime_runs/white7mp_labels_active \
      --out ../realtime_runs/white7mp_labels_combined
"""
import argparse
import collections
import csv
import json
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", nargs="+", required=True,
                    help="blind-label dirs to merge, in order")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    windows, label_rows, hidden, wid = [], [], [], 0
    meta = None
    for d in a.inp:
        man = json.load(open(os.path.join(d, "manifest.json")))
        meta = meta or {k: man[k] for k in ("window_s", "clips_dir", "fps") if k in man}
        hl = {}
        with open(os.path.join(d, "human_labels.csv")) as f:
            for r in csv.DictReader(f):
                hl[int(r["window_id"])] = r["behavior"]
        hid = {h["window_id"]: h for h in
               json.load(open(os.path.join(d, "predictions_hidden.json")))}
        kept = 0
        for w in man["windows"]:
            ow = w["window_id"]
            if ow not in hl:                 # not yet labeled: skip
                continue
            nw = dict(w); nw["window_id"] = wid
            windows.append(nw)
            label_rows.append((wid, hl[ow]))
            h = dict(hid.get(ow, {})); h["window_id"] = wid
            hidden.append(h)
            wid += 1; kept += 1
        print(f"  {os.path.basename(d.rstrip('/'))}: {kept} labeled windows")

    os.makedirs(a.out, exist_ok=True)
    json.dump({**meta, "windows": windows},
              open(os.path.join(a.out, "manifest.json"), "w"), indent=2)
    with open(os.path.join(a.out, "human_labels.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["window_id", "behavior"])
        for wd, b in label_rows:
            w.writerow([wd, b])
    json.dump(hidden, open(os.path.join(a.out, "predictions_hidden.json"), "w"), indent=2)

    tok = collections.Counter(
        b for _, s in label_rows for b in s.split(";") if b)
    print(f"merged {len(windows)} windows -> {a.out}")
    print("class tokens:", dict(sorted(tok.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
