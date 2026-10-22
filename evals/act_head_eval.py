#!/usr/bin/env python3
"""Port of research/evals/act_head_eval.py: does `act_probability` rank correct
decisions above incorrect ones on a laya_evals.py report? Tie-correct ROC-AUC
(Mann-Whitney U, mid-ranks). The shipped checkpoints emit ~1.0 everywhere
(#185), so the expected result is a saturated signal - this measures it.

Usage: python3 evals/act_head_eval.py report.json [--control confidence]
"""
import argparse
import json
import math


def roc_auc(pairs):
    """pairs: [(score, correct_bool)]. Mid-rank Mann-Whitney AUC."""
    n_pos = sum(1 for _, k in pairs if k)
    n_neg = len(pairs) - n_pos
    if n_pos == 0:
        return None, "no_positive_class"
    if n_neg == 0:
        return None, "no_negative_class"
    order = sorted(pairs, key=lambda x: x[0])
    ranks = [0.0] * len(order)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1][0] == order[i][0]:
            j += 1
        mid = (i + j) / 2 + 1  # 1-based mid-rank
        for k in range(i, j + 1):
            ranks[k] = mid
        i = j + 1
    pos_rank_sum = sum(r for r, (_, k) in zip(ranks, order) if k)
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--signal", default="act",
                    help="act -> answer.action.act_probability; confidence -> the calibrated control")
    args = ap.parse_args()
    cases = json.load(open(args.report))["cases"]
    buckets = {"paired": [], "no_binary_correctness": 0, "missing_act_probability": 0,
               "missing_confidence": 0, "malformed_case": 0}
    for c in cases:
        if not isinstance(c, dict):
            buckets["malformed_case"] += 1
            continue
        if not isinstance(c.get("correct"), bool):
            buckets["no_binary_correctness"] += 1
            continue
        if args.signal == "act":
            v = (c.get("answer") or {}).get("action", {}).get("act_probability")
        else:
            v = c.get("confidence")
        if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            key = "missing_act_probability" if args.signal == "act" else "missing_confidence"
            buckets[key] += 1
            continue
        buckets["paired"].append((float(v), c["correct"]))

    pairs = buckets.pop("paired")
    out = {"signal": args.signal, "buckets": {**buckets, "paired": len(pairs)}}
    auc, null_reason = roc_auc(pairs)
    out["roc_auc"] = auc
    out["auc_null_reason"] = null_reason
    if pairs:
        scores = sorted({s for s, _ in pairs})
        out["min"] = scores[0]
        out["max"] = scores[-1]
        out["unique_values"] = len(scores)
        out["saturated"] = len(scores) == 1
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
