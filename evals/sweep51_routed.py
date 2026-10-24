#!/usr/bin/env python3
"""51-language MASSIVE sweep with AUTO routing: the Router decides per request
(english for Latin scripts, multilingual mmBERT for non-Latin), recording which
checkpoint served each language. Compare against evals/sweep51_results.json
(english-checkpoint-forced) and the official part_a english numbers.
Run with a live laya-serve started WITHOUT LAYA_MODEL."""
import json
import os
import random
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from common import build_case, render_label, ask as ask_answers, write_sweep, load_sweep  # noqa: E402


def ask_routed(state, question):
    out = ask_answers(BASE, state, question, model_field=True)
    return out.get("model", "?"), out["answers"]["intent"]

BASE = os.environ.get("LAYA_SWEEP_BASE", "http://127.0.0.1:8086")
OUT = os.environ.get("LAYA_SWEEP_OUT", "evals/sweep51_routed_results.json")
LANGS = ["af","am","ar","az","bn","cy","da","de","el","en","es","fa","fi","fr",
         "he","hi","hu","hy","id","is","it","ja","jv","ka","km","kn","ko","lv",
         "ml","mn","ms","my","nb","nl","pl","pt","ro","ru","sl","sq","sv","sw",
         "ta","te","th","tl","tr","ur","vi","zh-CN","zh-TW"]

from datasets import load_dataset


def run_lang(lang, n_opts=20, per_lang=100):
    ds = load_dataset("mteb/amazon_massive_intent", lang, split="test")
    rows = [{"text": r["text"], "label_text": r["label_text"]} for r in ds]
    labels = sorted({r["label_text"] for r in rows})
    rng = random.Random(13)
    correct = 0
    routes = {}
    for row in rows[:per_lang]:
        pool = [x for x in labels if x != row["label_text"]]
        state, question, gold_idx = build_case(row["text"], row["label_text"], pool, rng, n_opts)
        keys = list(question["intent"]["criteria"])
        body = json.dumps({"state": state, "questions": question}).encode()
        req = urllib.request.Request(f"{BASE}/v1/decide", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            out = json.load(resp)
        model, ans = out.get("model", "?"), out["answers"]["intent"]
        routes[model] = routes.get(model, 0) + 1
        pred = next((k for k in keys
                     if ans["choice"] == f"{k}: {render_label(k)}" or ans["choice"] == k), None)
        if pred == keys[gold_idx]:
            correct += 1
    return correct, per_lang, routes


def main():
    results = {}
    if os.path.exists(OUT):  # resumable
        results = json.load(open(OUT))
    for i, lang in enumerate(LANGS):
        if lang in results:
            print(lang, "cached, skip", flush=True)
            continue
        t0 = time.time()
        try:
            c, n, routes = run_lang(lang)
        except Exception as e:
            print(lang, "ERROR", repr(e)[:80], flush=True)
            continue
        results[lang] = {"n": n, "accuracy": c / n, "routes": routes}
        print(f"{lang}: {c}/{n} = {c/n:.2f}  routes={routes}  ({time.time()-t0:.0f}s, {i+1}/51)", flush=True)
        json.dump(results, open(OUT, "w"), indent=1)
    if results:
        print("macro accuracy:", round(statistics.mean(v["accuracy"] for v in results.values()), 4),
              "over", len(results), "languages")


if __name__ == "__main__":
    main()
