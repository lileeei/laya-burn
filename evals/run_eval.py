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

FIXTURE = os.environ.get("LAYA_EVAL_FIXTURE",
                         "/tmp/laya-official/research/evals/fixture.jsonl")
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8095"


def ask_server(item):
    body = json.dumps({"state": item["state"], "questions": item["questions"]}).encode()
    req = urllib.request.Request(f"{BASE}/v1/decide", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.load(resp)["answers"]


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
        ans = ask_server(item)
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
    for i, item in enumerate(items):
        for qid, expected in item["expected"].items():
            n_q += 1
            o, r = ours[i][qid], refs[i][qid]
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


if __name__ == "__main__":
    main()
