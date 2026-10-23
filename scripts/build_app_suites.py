#!/usr/bin/env python3
"""Build the official app-benchmark suites as laya-evals JSONL.

Reuses research/scripts/bench_apps.py's build() VERBATIM (imported from the
official clone) so the sampling sequence - shared random.Random(SEED) across
builders - produces the identical 400 cases per suite the committed
app_benchmark_results.json was measured on.
"""
import json
import os
import sys

OFFICIAL = os.environ.get("LAYA_OFFICIAL", "/tmp/laya-official")
sys.path.insert(0, os.path.join(OFFICIAL, "research", "scripts"))
sys.path.insert(0, OFFICIAL)

import bench_apps  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "evals", "app_suites")
os.makedirs(OUT_DIR, exist_ok=True)

bench_apps.build()

for sname, S in bench_apps.SUITES.items():
    out = []
    # recover the criteria keys per case for choice gold -> key mapping
    for (state, questions), gold_idx in zip(S["cases"], S["gold"]):
        qid, q = next(iter(questions.items()))
        expected = None
        if q["type"] == "choice":
            keys = list(q["criteria"])
            expected = keys[gold_idx]
        else:  # noul: gold is 0/1 -> bool
            expected = bool(gold_idx)
        out.append({
            "state": state,
            "questions": questions,
            "expected": {qid: expected},
            "tags": [sname],
        })
    path = os.path.join(OUT_DIR, f"{sname.replace('/', '_')}.jsonl")
    with open(path, "w") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{sname}: {len(out)} cases -> {path}")
