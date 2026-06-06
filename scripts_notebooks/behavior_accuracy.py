#!/usr/bin/env python3
"""
behavior_accuracy.py — Score the rule classifier against blind human labels.

Joins the human labels (behavior_label_tool.py `label` output) to the rule
classifier's hidden predictions for the same windows, and reports overall
accuracy, per-behavior precision/recall, and a confusion matrix. This is the
honest accuracy number the rules never had — labels were collected blind, so
agreement is not biased by showing the prediction.

Usage:
  python behavior_accuracy.py --manifest_dir ../realtime_runs/S3_labels_blind
"""

import os
import csv
import json
import argparse
from collections import defaultdict, Counter


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest_dir", required=True)
    args = ap.parse_args()

    md = args.manifest_dir
    with open(os.path.join(md, "predictions_hidden.json")) as f:
        pred = {h["window_id"]: h["rule_pred"] for h in json.load(f)}
    human = {}
    lp = os.path.join(md, "human_labels.csv")
    if not os.path.exists(lp):
        raise SystemExit(f"No human labels yet at {lp} — run the `label` GUI first.")
    with open(lp) as f:
        for r in csv.DictReader(f):
            human[int(r["window_id"])] = r["behavior"]

    pairs = [(human[w], pred[w]) for w in sorted(human) if w in pred]
    if not pairs:
        raise SystemExit("No overlapping labeled+predicted windows.")
    n = len(pairs)
    correct = sum(1 for h, p in pairs if h == p)

    labels = sorted(set([h for h, _ in pairs] + [p for _, p in pairs]))
    conf = defaultdict(Counter)            # conf[human][pred]
    for h, p in pairs:
        conf[h][p] += 1

    print("=" * 64)
    print("BEHAVIOR RULE-CLASSIFIER ACCURACY vs BLIND HUMAN LABELS")
    print("=" * 64)
    print(f"Labeled windows compared: {n}")
    print(f"Overall agreement:        {correct}/{n} ({100*correct/n:.0f}%)")
    print()
    print(f"{'behavior':12s} {'n':>4} {'precision':>10} {'recall':>8}")
    for b in labels:
        tp = conf[b][b]
        human_n = sum(conf[b].values())                 # actually labeled b
        pred_n = sum(conf[h][b] for h in labels)         # predicted b
        prec = tp / pred_n if pred_n else float("nan")
        rec = tp / human_n if human_n else float("nan")
        print(f"{b:12s} {human_n:>4} {prec:>10.2f} {rec:>8.2f}")

    print("\nConfusion matrix (rows = human, cols = rule pred):")
    hdr = "human\\pred  " + " ".join(f"{b[:6]:>7}" for b in labels)
    print(hdr)
    for h in labels:
        row = " ".join(f"{conf[h][p]:>7}" for p in labels)
        print(f"{h:11s} {row}")
    print("=" * 64)
    print("Use this to decide which behaviors need a trained model (low recall/"
          "precision) vs which the rules already handle.")


if __name__ == "__main__":
    main()
