#!/usr/bin/env python3
"""Shared eval layer: ONE owner for the server adapter, the official metric
semantics, the MASSIVE case builder, and sweep-result IO. Previously
ask_server existed in 5 scripts, ECE(15) in 3, and three sweep result
shapes drifted until compare_sweep.py raised KeyError on two of them."""
import json
import random
import urllib.request

# ---------------------------------------------------------------- server adapter

def ask(base, state, questions, timeout=300, model_field=False):
    """POST /v1/decide and return the answers object (with the serving
    checkpoint under 'model' when model_field)."""
    body = json.dumps({"state": state, "questions": questions}).encode()
    req = urllib.request.Request(f"{base}/v1/decide", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.load(resp)
    return out if model_field else out["answers"]


# ---------------------------------------------------------------- metrics (official semantics)

def answer_confidence(answer):
    """The calibration column: answer_confidence -> confidence -> noul
    max(p,1-p) -> max(probabilities) (laya.evals order)."""
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


def ece_score(conf, correct, bins=15):
    """Expected Calibration Error over (confidence, correct) pairs."""
    pairs = [(c, k) for c, k in zip(conf, correct) if c is not None]
    if not pairs:
        return None
    edges = [i / bins for i in range(bins + 1)]
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [(c, k) for c, k in pairs if lo < c <= hi] or \
              [(c, k) for c, k in pairs if c == lo == 0.0]
        if not sel:
            continue
        acc = sum(k for _, k in sel) / len(sel)
        avg = sum(c for c, _ in sel) / len(sel)
        e += len(sel) / len(pairs) * abs(acc - avg)
    return e


# ---------------------------------------------------------------- MASSIVE case builder

SEED = 13
INSTRUCTIONS = "What is the user asking for in `utterance`?"


def render_label(key):
    return key.replace("_", " ").replace(".", ": ")


def build_case(text, gold, pool, rng, n_opts):
    keys = [gold] + rng.sample(list(pool), min(n_opts - 1, len(pool)))
    rng.shuffle(keys)
    state = {"utterance": text}
    question = {"intent": {
        "type": "choice",
        "instructions": INSTRUCTIONS,
        "criteria": {k: render_label(k) for k in keys},
    }}
    return state, question, keys.index(gold)


def pred_key(answer, keys):
    """Map a served choice back to its criteria key (key, or rendered text)."""
    return next((k for k in keys
                 if answer["choice"] == f"{k}: {render_label(k)}" or answer["choice"] == k), None)


# ---------------------------------------------------------------- sweep result IO

def write_sweep(path, per_lang, meta=None):
    """One canonical shape: {"_meta": {...}, lang: {n, accuracy, ...}}.
    Written atomically after the run (resumable callers may also write
    incrementally through this shape)."""
    out = {"_meta": meta or {}}
    out.update(per_lang)
    json.dump(out, open(path, "w"), indent=1)


def load_sweep(path):
    """-> (meta, {lang: metrics}); skips the _meta key, tolerating the two
    legacy shapes (bare per-lang dicts without _meta)."""
    d = json.load(open(path))
    meta = d.pop("_meta", {})
    return meta, d
