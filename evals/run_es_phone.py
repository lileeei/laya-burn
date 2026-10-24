#!/usr/bin/env python3
"""es_phone_turns (official frozen-prompt suite) through laya-burn serve.

Cases and questions come from the official benchmark VERBATIM (prompts.build
over the frozen data files). Compare against the committed laya results AND
their README's documented zero-shot row: the committed 0.90+ numbers are the
benchmark's own FINE-TUNED model; base multilingual zero-shot is ~0.40 there
and ~0.42 here.

Usage: LAYA_OFFICIAL=/tmp/laya-official python3 evals/run_es_phone.py [base]"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
OFFICIAL = os.environ.get("LAYA_OFFICIAL", "/tmp/laya-official")
sys.path.insert(0, f"{OFFICIAL}/research/benchmarks/es_phone_turns")
import prompts  # noqa: E402
from common import ask  # noqa: E402

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
BENCH = f"{OFFICIAL}/research/benchmarks/es_phone_turns"
COMPARISONS = {
    "cases": "laya-es-phone-v1__cases__es_criteria.json",
    "clinic": "laya-es-phone-v1__clinic__es_criteria.json",
    "frustration": "laya-es-phone-v2__frustration__es_criteria.json",
    "independent_cases": "laya-es-phone-v2__independent_cases__es_criteria.json",
}
for dataset, ref_file in COMPARISONS.items():
    cases = [json.loads(l) for l in open(f"{BENCH}/data/{dataset}.jsonl") if l.strip()]
    ref = json.load(open(f"{BENCH}/results/{ref_file}"))["summary"]
    correct = sum(
        ask(BASE, *prompts.build("es_criteria", c["text"], dataset))["accion"]["choice"] == c["gold"]
        for c in cases
    )
    acc = correct / len(cases)
    print(f"{dataset}: ours {acc:.4f} vs committed {ref['accuracy']:.4f} (d={acc-ref['accuracy']:+.4f}, n={len(cases)})")
