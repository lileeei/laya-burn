#!/usr/bin/env python3
"""Latency benchmark mirroring the official HF eval methodology
(eval/results.json "latency_ms"): request batches of 1/10/50 questions,
repeat each N times, report p50/p95 per batch size.

Usage: laya-serve must be running; python3 evals/bench_latency.py [base_url]
"""
import json
import os
import statistics
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8090"
REPEATS = int(os.environ.get("LAYA_BENCH_REPEATS", "20"))


def question(i):
    return {
        "type": "choice",
        "instructions": f"Which team handles case {i}?",
        "criteria": {
            "support": "account help, resets",
            "sales": "pricing, upgrades",
            "billing": "invoices, payments",
            "legal": "contracts, compliance",
        },
    }


def ask(n_questions):
    qs = {f"q{i}": question(i) for i in range(n_questions)}
    body = json.dumps({
        "state": "A customer writes in about being charged twice for March "
                 "and asks for a refund or they will cancel their plan.",
        "questions": qs,
    }).encode()
    req = urllib.request.Request(f"{BASE}/v1/decide", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        json.load(resp)
    return (time.perf_counter() - t0) * 1000.0


def pct(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    k = max(0, min(len(sorted_vals) - 1, int(round((p / 100.0) * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


def main():
    # warmup (kernel compile / autotune / cache)
    ask(1)
    print(f"benchmarking {BASE}, {REPEATS} repeats per batch size")
    out = {}
    for n in (1, 10, 50):
        times = sorted(ask(n) for _ in range(REPEATS))
        p50 = pct(times, 50)
        p95 = pct(times, 95)
        out[f"{n}_questions"] = {"p50_ms": round(p50, 1), "p95_ms": round(p95, 1)}
        print(f"  {n:>2} questions: p50 {p50:7.1f} ms   p95 {p95:7.1f} ms")
    print(json.dumps({"latency_ms": out}, indent=1))


if __name__ == "__main__":
    main()
