#!/usr/bin/env python3
"""51-language MASSIVE sweep on laya-burn, comparing per-language accuracy
against the officially committed english-checkpoint numbers.
Run with a live laya-serve (LAYA_MODEL=english forced)."""
import json
import os
import sys
import urllib.request
import random
import statistics
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from massive_eval import build_case, render_label, ask_server  # noqa: E402

BASE = os.environ.get("LAYA_SWEEP_BASE", "http://127.0.0.1:8086")
OUT = os.environ.get("LAYA_SWEEP_OUT", "/tmp/sweep51_ours.json")
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
    for row in rows[:per_lang]:
        pool = [x for x in labels if x != row["label_text"]]
        state, question, gold_idx = build_case(row["text"], row["label_text"], pool, rng, n_opts)
        keys = list(question["intent"]["criteria"])
        ans = ask_server(BASE, state, question)
        pred = next((k for k in keys
                     if ans["choice"] == f"{k}: {render_label(k)}" or ans["choice"] == k), None)
        if pred == keys[gold_idx]:
            correct += 1
    return correct, per_lang

def main():
    results = {}
    for i, lang in enumerate(LANGS):
        t0 = time.time()
        try:
            c, n = run_lang(lang)
        except Exception as e:
            print(lang, "ERROR", repr(e)[:80], flush=True)
            continue
        results[lang] = {"n": n, "accuracy": c / n}
        print(f"{lang}: {c}/{n} = {c/n:.2f}  ({time.time()-t0:.0f}s, {i+1}/51)", flush=True)
        json.dump(results, open(OUT, "w"), indent=1)
    if results:
        print("macro accuracy:", round(statistics.mean(v["accuracy"] for v in results.values()), 4),
              "over", len(results), "languages")

if __name__ == "__main__":
    main()
