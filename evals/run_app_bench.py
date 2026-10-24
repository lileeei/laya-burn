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
    import laya_evals
    import argparse
    args = argparse.Namespace(
        cmd="run", data=os.path.join(SUITES_DIR, fname), base=BASE,
        json=f"/tmp/app_case_{suite}_{MODEL}.json", min_accuracy=None,
        max_ece=None, baseline=None,
        tolerance="accuracy=0.02,ece=0.03,mean_confidence=0.03")
    import io, contextlib
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            laya_evals.run(args)
        import json as j
        dec = j.JSONDecoder()
        obj, _ = dec.raw_decode(buf.getvalue())
        m = obj["overall"]
    except Exception:
        m = {"error": buf.getvalue()[-200:] or "no output"}
    secs = time.time() - t0
    acc = m.get("accuracy") or m.get("choice_accuracy") or m.get("noul_accuracy")
    print(f"{MODEL}/{suite}: acc {acc} ({secs:.0f}s)", flush=True)
    results[suite] = {k: v for k, v in m.items()}
json.dump(results, open(OUT, "w"), indent=1)
print("wrote", OUT)
