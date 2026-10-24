#!/usr/bin/env python3
"""End-to-end eval: run the fixture through laya-serve and through the official
PyTorch reference, compare answers pairwise and against expected labels.

Usage:
  LAYA_MODEL=typed-decisions ./target/release/laya-serve 8095 &
  python3 evals/run_eval.py http://127.0.0.1:8095

Reports: accuracy (laya-burn vs expected), accuracy (reference vs expected),
and pairwise agreement (laya-burn vs reference) - the last one is the port
fidelity signal: identical answers mean the port is lossless on this fixture.
"""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from common import ask  # noqa: E402

FIXTURE = os.environ.get("LAYA_EVAL_FIXTURE",
                         os.environ.get("LAYA_OFFICIAL", "/tmp/laya-official") + "/research/evals/fixture.jsonl")
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8095"


def extract(answer):
    """Reduce an answer to the comparable key: choice label, score bucket, noul bool."""
    t = answer["type"]
    if t == "choice":
        return answer["choice"].split(":")[0].strip()
    if t == "score":
        return round(answer["score"])
    return answer["noul"] >= 0.5


def main():
    items = [json.loads(l) for l in open(FIXTURE) if l.strip()]
    print(f"fixture: {len(items)} items from {FIXTURE}")

    ours = []
    for i, item in enumerate(items):
        ans = ask(BASE, item["state"], item["questions"])
        ours.append({qid: extract(a) for qid, a in ans.items()})
        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{len(items)} via server")

    # Reference: official PyTorch RLAgent over the same fixture.
    sys.path.insert(0, os.environ.get("LAYA_MODEL_DIR", "model"))
    sys.path.insert(0, "model")
    from rl_agent_api import RLAgent  # noqa: E402

    model_dir = os.environ.get("LAYA_MODEL_DIR", "model")
    agent = RLAgent(model_dir, device="cpu")
    refs = []
    for item in items:
        result = agent.system_one(item["state"], item["questions"])
        row = {}
        for qid, a in result["answers"].items():
            if a["type"] == "choice":
                row[qid] = a["choice"]
            elif a["type"] == "score":
                row[qid] = round(a["score"])
            else:
                row[qid] = a["noul"] >= 0.5
        refs.append(row)

    n_q = 0
    acc_ours = acc_ref = agree = 0
    mismatches = []
    # Calibration-metric inputs (official methodology: ECE(15 bins), NLL,
    # Brier, accuracy at 50% coverage) over the top-choice probability.
    import math
    probs_conf = []  # (confidence, correct) for laya-burn answers

    def conf_of(ans):
        if ans["type"] == "choice":
            return max(ans["probabilities"].values())
        if ans["type"] == "score":
            return ans["confidence"]
        return max(ans["noul"], 1.0 - ans["noul"])
    # second server pass capturing full answers for calibration metrics
    full_answers = []
    for item in items:
        full_answers.append(ask(BASE, item["state"], item["questions"]))

    for i, item in enumerate(items):
        for qid, expected in item["expected"].items():
            n_q += 1
            o, r = ours[i][qid], refs[i][qid]
            correct = o == expected
            probs_conf.append((conf_of(full_answers[i][qid]), correct))
            if o == expected:
                acc_ours += 1
            if r == expected:
                acc_ref += 1
            if o == r:
                agree += 1
            else:
                mismatches.append((i, qid, expected, r, o))

    print(f"questions: {n_q}")
    print(f"accuracy laya-burn vs expected : {acc_ours}/{n_q} = {acc_ours / n_q:.3f}")
    print(f"accuracy reference vs expected : {acc_ref}/{n_q} = {acc_ref / n_q:.3f}")
    print(f"agreement  laya-burn vs reference: {agree}/{n_q} = {agree / n_q:.3f}")
    if mismatches:
        print("mismatches (i, qid, expected, reference, ours):")
        for m in mismatches:
            print(" ", m)

    # --- Official calibration metrics (eval/results.json methodology) ---
    if probs_conf:
        # ECE, 15 equal bins
        ece = 0.0
        for b in range(15):
            lo, hi = b / 15.0, (b + 1) / 15.0
            sel = [(c, k) for c, k in probs_conf if lo < c <= hi or (b == 0 and c == 0)]
            if sel:
                acc_b = sum(k for _, k in sel) / len(sel)
                conf_b = sum(c for c, _ in sel) / len(sel)
                ece += (len(sel) / n_q) * abs(acc_b - conf_b)
        # NLL (top-choice, natural log) and Brier
        nll = -sum(math.log(max(c, 1e-12)) if k else math.log(max(1 - c, 1e-12))
                   for c, k in probs_conf) / n_q
        brier = sum((c - (1.0 if k else 0.0)) ** 2 for c, k in probs_conf) / n_q
        # accuracy at 50% coverage (highest-confidence half)
        ranked = sorted(probs_conf, key=lambda x: -x[0])
        half = max(1, n_q // 2)
        acc50 = sum(k for _, k in ranked[:half]) / half
        print(f"ECE(15): {ece:.3f}   NLL: {nll:.3f}   Brier: {brier:.3f}")
        print(f"accuracy at 50% coverage: {acc50:.3f}")


if __name__ == "__main__":
    main()
