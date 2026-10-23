#!/usr/bin/env python3
"""MASSIVE-intent eval reproducing the official harness method:
mteb/amazon_massive_intent test split, first 100 rows, Random(13) fresh per
language, 20 options, fixed instructions, labels with underscore->space and
dot->colon-space. Metrics: accuracy, macro-F1, ECE(15), acc@50% coverage.
Compare with the committed sweep (english ckpt, en): acc 0.82 f1 0.7876 ece 0.1789.
"""
import argparse
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import ask, answer_confidence, ece_score, SEED, INSTRUCTIONS, render_label, build_case  # noqa: E402

ECE_BINS = 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lang")
    ap.add_argument("--base", default="http://127.0.0.1:8090")
    ap.add_argument("--per-lang", type=int, default=100)
    ap.add_argument("--n-opts", type=int, default=20)
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("mteb/amazon_massive_intent", args.lang, split="test")
    rows = [{"text": r["text"], "label_text": r["label_text"]} for r in ds]
    labels = sorted({r["label_text"] for r in rows})
    rng = random.Random(SEED)

    correct, confs = [], []
    pred_keys, gold_keys_all = [], []
    for row in rows[: args.per_lang]:
        pool = [x for x in labels if x != row["label_text"]]
        state, question, gold_idx = build_case(row["text"], row["label_text"],
                                                pool, rng, args.n_opts)
        keys = list(question["intent"]["criteria"])
        ans = ask(args.base, state, question)["intent"]
        # our server renders choices as "<key>: <rendered>" (blank criteria give the bare key)
        pred_key = next((k for k in keys if ans["choice"] == f"{k}: {render_label(k)}" or ans["choice"] == k), None)
        gold_key = keys[gold_idx]
        ok = pred_key == gold_key
        correct.append(ok)
        confs.append(max(ans["probabilities"].values()))
        pred_keys.append(pred_key)
        gold_keys_all.append(gold_key)

    n = len(correct)
    acc = sum(correct) / n

    def f1_for(lb):
        tp = sum(1 for p, g in zip(pred_keys, gold_keys_all) if p == lb and g == lb)
        fp = sum(1 for p, g in zip(pred_keys, gold_keys_all) if p == lb and g != lb)
        fn = sum(1 for p, g in zip(pred_keys, gold_keys_all) if p != lb and g == lb)
        if tp == 0:
            return 0.0
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        return 2 * prec * rec / (prec + rec)

    used = set(gold_keys_all)
    macro_f1 = statistics.mean(f1_for(lb) for lb in used) if used else 0.0

    ece = 0.0
    for b in range(ECE_BINS):
        lo, hi = b / ECE_BINS, (b + 1) / ECE_BINS
        sel = [(c, k) for c, k in zip(confs, correct)
               if lo < c <= hi or (b == 0 and c == 0)]
        if sel:
            acc_b = sum(k for _, k in sel) / len(sel)
            conf_b = sum(c for c, _ in sel) / len(sel)
            ece += (len(sel) / n) * abs(acc_b - conf_b)

    ranked = sorted(zip(confs, correct), key=lambda x: -x[0])
    half = max(1, n // 2)
    acc50 = sum(k for _, k in ranked[:half]) / half

    print(args.lang, "n=", n, "acc=", round(acc, 4), "macro_f1=", round(macro_f1, 4),
          "ece=", round(ece, 4), "acc@50%=", round(acc50, 4))
    print(json.dumps({"lang": args.lang, "n": n, "accuracy": acc,
                      "macro_f1": macro_f1, "ece": ece,
                      "acc_at_50_coverage": acc50}, indent=1))


if __name__ == "__main__":
    main()
