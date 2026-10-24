#!/usr/bin/env python3
"""feishu_zh (official frozen-prompt suite, choice + four-noul modes).

requests_for/interpret come from the official prompts module VERBATIM. The
committed v1 results used an unreleased SDK commit (metadata
laya_source_commit ef7d7d26); on the SAME weights the released SDK scores
~0.10 ~= laya-burn - the committed 0.31 is upstream version drift.

Usage: LAYA_OFFICIAL=/tmp/laya-official python3 evals/run_feishu_zh.py [base]"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
OFFICIAL = os.environ.get("LAYA_OFFICIAL", "/tmp/laya-official")
sys.path.insert(0, f"{OFFICIAL}/research/benchmarks/feishu_zh")
import prompts  # noqa: E402
from common import ask  # noqa: E402

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
cases = [json.loads(l) for l in open(f"{OFFICIAL}/research/benchmarks/feishu_zh/data/cases.jsonl") if l.strip()]
ok_choice = ok_noul = 0
for c in cases:
    reqs = prompts.requests_for(c)
    out = prompts.interpret("choice", ask(BASE, reqs["choice"]["state"], reqs["choice"]["questions"]))
    ok_choice += out["predicted"] == c["expected"]
    out2 = prompts.interpret("four_noul", ask(BASE, reqs["four_noul"]["state"], reqs["four_noul"]["questions"]))
    ok_noul += out2["predicted"] == c["expected"]
print(f"choice:    {ok_choice}/192 = {ok_choice/192:.4f}  (committed 0.3125)")
print(f"four_noul: {ok_noul}/192 = {ok_noul/192:.4f}  (committed 0.2813)")
