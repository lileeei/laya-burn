#!/usr/bin/env python3
"""Run all app suites through a laya-serve and accumulate per-suite metrics."""
import json
import os
import subprocess
import sys
import time

BASE = sys.argv[1]
MODEL = sys.argv[2]          # checkpoint label (english/multilingual)
SUITES_DIR = os.path.join(os.path.dirname(__file__), "app_suites")
OUT = sys.argv[3]

results = {}
for fname in sorted(os.listdir(SUITES_DIR)):
    if not fname.endswith(".jsonl"):
        continue
    suite = fname[:-6]
    t0 = time.time()
    r = subprocess.run(
        ["python3", os.path.join(os.path.dirname(__file__), "laya_evals.py"),
         "run", os.path.join(SUITES_DIR, fname), "--base", BASE, "--json", f"/tmp/app_case_{suite}_{MODEL}.json"],
        capture_output=True, text=True)
    try:
        import json as j
        dec = j.JSONDecoder()
        obj, _ = dec.raw_decode(r.stdout)
        m = obj["overall"]
    except Exception:
        m = {"error": (r.stderr or r.stdout)[-200:]}
    secs = time.time() - t0
    acc = m.get("accuracy") or m.get("choice_accuracy") or m.get("noul_accuracy")
    print(f"{MODEL}/{suite}: acc {acc} ({secs:.0f}s)", flush=True)
    results[suite] = {k: v for k, v in m.items()}
json.dump(results, open(OUT, "w"), indent=1)
print("wrote", OUT)
