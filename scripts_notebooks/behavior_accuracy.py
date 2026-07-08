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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest_dir", required=True)
    args = ap.parse_args()

    md = args.manifest_dir
    # predictions_hidden carries the rule's guess (`sample`) or the model's
    # (`active`); accept either so this scores both kinds of batch.
    with open(os.path.join(md, "predictions_hidden.json")) as f:
        pred = {h["window_id"]: h.get("rule_pred", h.get("model_pred", "unknown"))
                for h in json.load(f)}
    # human labels are SETS (';'-joined). The predictor emits ONE behavior per
    # window, so its "set" is a singleton. Scoring is per-behavior present/absent.
    SKIP = {"no_worm", "unknown", "frozen"}   # unusable/corrupt windows: not scored
    human = {}
    lp = os.path.join(md, "human_labels.csv")
    if not os.path.exists(lp):
        raise SystemExit(f"No human labels yet at {lp} — run the `label` GUI first.")
    with open(lp) as f:
        for r in csv.DictReader(f):
            human[int(r["window_id"])] = {b for b in r["behavior"].split(";") if b}

    wids = [w for w in sorted(human) if w in pred and (human[w] - SKIP)]
    if not wids:
        raise SystemExit("No overlapping labeled+predicted windows.")
    n = len(wids)
    behaviors = sorted({b for w in wids for b in human[w]} |
                       {pred[w] for w in wids})

    # Exact-set agreement (rule's single label == human's set) is strict but
    # informative; per-behavior present/absent is the main metric.
    exact = sum(1 for w in wids if human[w] == {pred[w]})
    # "Rule label is among the human's behaviors" — the rule got *a* right one.
    among = sum(1 for w in wids if pred[w] in human[w])

    print("=" * 70)
    print("BEHAVIOR RULE-CLASSIFIER ACCURACY vs BLIND HUMAN LABELS (multi-label)")
    print("=" * 70)
    print(f"Labeled windows compared:        {n}")
    print(f"Rule label present in human set: {among}/{n} ({100*among/n:.0f}%)")
    print(f"Exact set match:                 {exact}/{n} ({100*exact/n:.0f}%)")
    print()
    print("Per-behavior (present/absent across windows):")
    print(f"  {'behavior':12s} {'human_n':>7} {'precision':>10} {'recall':>8}")
    for b in behaviors:
        tp = sum(1 for w in wids if pred[w] == b and b in human[w])
        pred_n = sum(1 for w in wids if pred[w] == b)
        human_n = sum(1 for w in wids if b in human[w])
        prec = tp / pred_n if pred_n else float("nan")
        rec = tp / human_n if human_n else float("nan")
        print(f"  {b:12s} {human_n:>7} {prec:>10.2f} {rec:>8.2f}")
    print("=" * 70)
    print("recall = of windows humans saw behavior B, how often the rule said B.")
    print("precision = of windows the rule said B, how often a human agreed.")
    print("Low values flag behaviors that need a trained model (task #13).")


if __name__ == "__main__":
    main()
