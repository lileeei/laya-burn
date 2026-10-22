#!/usr/bin/env python3
"""laya-evals compatible runner for laya-burn: same JSONL dataset format,
metric semantics and report shape as the official `laya.evals` harness, so
results compare directly against research/results/ baselines.

Dataset row: {"state", "questions", "expected": {qid: label|number|bool},
              "tags"?, "language"?, "model"?}  (# and blank lines ignored)

Metrics (official semantics, laya/evals.py):
  choice_accuracy  answer["choice"] == expected            (choice)
  noul_accuracy    (answer["noul"] >= 0.5) == expected      (noul)
  score_mae        |answer["score"] - expected|             (score)
  mean_confidence  answer_confidence -> confidence -> max-prob fallback
  ece (15 bins), brier over (confidence, correct) pairs

Usage:
  python3 evals/laya_evals.py validate data.jsonl
  python3 evals/laya_evals.py run data.jsonl --base http://127.0.0.1:8080 \
      [--model english] [--slice language] [--json report.json] \
      [--min-accuracy 0.8] [--max-ece 0.05] \
      [--baseline baseline.json --tolerance accuracy=0.02,ece=0.03]
"""
import argparse
import json
import math
import sys
import urllib.request


def load_jsonl(path):
    rows = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            for f in ("state", "questions", "expected"):
                if f not in row:
                    raise SystemExit(f"row {i + 1}: missing field {f!r}")
            if not isinstance(row["questions"], dict) or not row["questions"]:
                raise SystemExit(f"row {i + 1}: questions must be a non-empty object")
            rows.append(row)
    return rows


def ask_server(base, state, questions, timeout=300):
    body = json.dumps({"state": state, "questions": questions}).encode()
    req = urllib.request.Request(f"{base}/v1/decide", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)["answers"]


def answer_confidence(answer):
    c = answer.get("answer_confidence")
    if isinstance(c, (int, float)):
        return float(c)
    c = answer.get("confidence")
    if isinstance(c, (int, float)):
        return float(c)
    if answer.get("type") == "noul":
        p = float(answer.get("noul", 0.0))
        return max(p, 1.0 - p)
    probs = answer.get("probabilities")
    if isinstance(probs, dict) and probs:
        return max(float(v) for v in probs.values())
    if isinstance(probs, list) and probs:
        return max(float(v) for v in probs)
    return None


def correct_of(answer, expected):
    if answer.get("type") == "choice" and isinstance(expected, str):
        return answer.get("choice") == expected
    if answer.get("type") == "noul" and isinstance(expected, bool):
        return bool(answer.get("noul", 0.0) >= 0.5) == expected
    return None  # score: no canonical binary correctness (official semantics)


def ece_score(conf, correct, bins=15):
    if not conf:
        return None
    edges = [i / bins for i in range(bins + 1)]
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [(c, k) for c, k in zip(conf, correct) if lo < c <= hi] or \
              [(c, k) for c, k in zip(conf, correct) if c == lo == 0.0]
        if not sel:
            continue
        acc = sum(k for _, k in sel) / len(sel)
        avg = sum(c for c, _ in sel) / len(sel)
        e += len(sel) / len(conf) * abs(acc - avg)
    return e


def summarize(cases):
    conf = [c["confidence"] for c in cases if c["confidence"] is not None]
    corr = [c["correct"] for c in cases if c["confidence"] is not None and c["correct"] is not None]
    choice = [c for c in cases if c["type"] == "choice"]
    noul = [c for c in cases if c["type"] == "noul"]
    score = [c for c in cases if c["type"] == "score"]
    out = {"n": len(cases)}
    if choice:
        out["choice_accuracy"] = sum(c["correct"] for c in choice) / len(choice)
    if noul:
        out["noul_accuracy"] = sum(c["correct"] for c in noul) / len(noul)
    # the official report names the headline column `accuracy`
    marked = [c for c in cases if c["correct"] is not None]
    if marked:
        out["accuracy"] = sum(c["correct"] for c in marked) / len(marked)
    if score and any(c["mae"] is not None for c in score):
        out["score_mae"] = sum(c["mae"] for c in score if c["mae"] is not None) / len(score)
    if conf:
        out["mean_confidence"] = sum(conf) / len(conf)
    if len(conf) == len(corr) and corr:
        out["ece"] = ece_score(conf, corr)
        out["brier"] = sum((c - float(k)) ** 2 for c, k in zip(conf, corr)) / len(conf)
        k = [float(x) for x in corr]
        half = sorted(zip(conf, k), key=lambda x: -x[0])[: max(1, len(conf) // 2)]
        out["acc_at_50_coverage"] = sum(x[1] for x in half) / len(half)
    return out


def run(args):
    rows = load_jsonl(args.data)
    cases = []
    for r_i, row in enumerate(rows):
        answers = ask_server(args.base, row["state"], row["questions"])
        for qid, answer in answers.items():
            expected = row["expected"].get(qid)
            a = {
                "row": r_i,
                "qid": qid,
                "answer": answer,
                "confidence": answer_confidence(answer),
                "correct": correct_of(answer, expected),
                "type": answer.get("type"),
                "language": row.get("language"),
                "tags": row.get("tags", []),
            }
            if answer.get("type") == "score" and isinstance(expected, (int, float)) \
                    and not isinstance(expected, bool):
                a["mae"] = abs(float(answer.get("score", 0.0)) - float(expected))
            cases.append(a)

    slices = {"overall": summarize(cases)}
    for key in ("language", "tags"):
        groups = {}
        for c in cases:
            vals = c.get(key) or []
            vals = [vals] if isinstance(vals, str) else vals
            for v in vals:
                groups.setdefault(v, []).append(c)
        for v, cs in sorted(groups.items()):
            slices[f"{key}:{v}"] = summarize(cs)

    print(json.dumps(slices, indent=1))
    if args.json:
        json.dump({"dataset": args.data, "slices": slices, "cases": cases},
                  open(args.json, "w"), indent=1, ensure_ascii=False)

    failed = []
    if args.min_accuracy is not None:
        acc = slices["overall"].get("choice_accuracy", slices["overall"].get("noul_accuracy"))
        if acc is None or acc < args.min_accuracy:
            failed.append(f"min-accuracy: {acc} < {args.min_accuracy}")
    if args.max_ece is not None:
        e = slices["overall"].get("ece")
        if e is None or e > args.max_ece:
            failed.append(f"max-ece: {e} > {args.max_ece}")
    if args.baseline:
        base = json.load(open(args.baseline))
        base_report = base.get("report", base)
        tol = dict(kv.split("=") for kv in args.tolerance.split(",")) if args.tolerance else {}
        for lang in base.get("languages", ["en"]):
            ref = base_report.get(lang, {})
            ours = slices.get(f"language:{lang}", slices["overall"])
            for metric, t in tol.items():
                if metric in ref and metric in ours:
                    d = abs(ours[metric] - ref[metric])
                    status = "OK" if d <= float(t) else "FAIL"
                    print(f"gate {lang}/{metric}: ours {ours[metric]:.4f} "
                          f"ref {ref[metric]:.4f} delta {d:+.4f} (tol {t}) {status}")
                    if status == "FAIL":
                        failed.append(f"{lang}/{metric} drift {d:.4f} > {t}")
    if failed:
        print("\n".join(f"GATE FAIL: {f}" for f in failed), file=sys.stderr)
        sys.exit(1)
    print("GATE OK")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("data")
    r = sub.add_parser("run")
    r.add_argument("data")
    r.add_argument("--base", default="http://127.0.0.1:8080")
    r.add_argument("--json")
    r.add_argument("--min-accuracy", type=float)
    r.add_argument("--max-ece", type=float)
    r.add_argument("--baseline")
    r.add_argument("--tolerance", default="accuracy=0.02,ece=0.03,mean_confidence=0.03")
    args = ap.parse_args()
    if args.cmd == "validate":
        rows = load_jsonl(args.data)
        print(f"ok: {len(rows)} rows")
    else:
        run(args)


if __name__ == "__main__":
    main()
