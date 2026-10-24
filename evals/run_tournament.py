#!/usr/bin/env python3
"""banking77 single vs tournament (group-elimination brackets over
/v1/decide), mirroring the official benchmark strategy: criteria cut into
near-equal groups of group_size (16), one round per shrink, finalist call.

Usage: python3 evals/run_tournament.py [base] [n_rows]"""
import sys, time
sys.path.insert(0, os.path.dirname(__file__))
from common import ask  # noqa: E402
from datasets import load_dataset  # noqa: E402

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 400
GROUP = 16

d = load_dataset("mteb/banking77", split="test")
names = dict((r["label"], r["label_text"]) for r in d)
labels = [names[i].replace("_", " ") for i in sorted(names)]
rows = [(r["text"], r["label_text"].replace("_", " ")) for r in list(d)[:N]]
INSTR = "Which banking intent does `message` express?"

def q_for(subset):
    return {"q": {"type": "choice", "instructions": INSTR,
                  "criteria": {k: None for k in subset}}}

def tournament(state):
    labs = list(labels)
    rounds = 0
    while True:
        parts = -(-len(labs) // GROUP)
        if parts <= 1:
            break
        groups = [labs[i * len(labs) // parts:(i + 1) * len(labs) // parts] for i in range(parts)]
        qs = {f"g{i}": {"type": "choice", "instructions": INSTR,
                        "criteria": {k: None for k in g}} for i, g in enumerate(groups)}
        answers = ask(BASE, {"message": state}, qs)
        labs = [answers[f"g{i}"]["choice"] for i in range(len(groups))]
        rounds += 1
    return ask(BASE, {"message": state}, q_for(labs))["q"]["choice"], rounds + 1

ok_s = ok_t = 0
t0 = time.time()
for text, gold in rows:
    ok_s += ask(BASE, {"message": text}, q_for(labels))["q"]["choice"] == gold
    t_ans, _ = tournament(text)
    ok_t += t_ans == gold
print(f"single:     {ok_s}/{N} = {ok_s/N:.4f}  (official full-split 0.4301)")
print(f"tournament: {ok_t}/{N} = {ok_t/N:.4f}  (official full-split 0.6096)")
print(f"elapsed {time.time()-t0:.0f}s")
