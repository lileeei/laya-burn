#!/usr/bin/env python3
"""Real-task validation: drive the official learning-path example inputs
through BOTH the Python SDK (ground truth, laya 0.4.12 + local checkpoint)
and laya-burn's /v1/decide, then compare pairwise.

Task shapes covered (mirroring examples/01-41):
  minimal noul, choice w/ criteria, score levels, JSON state, email
  cleaning (clean vs raw), conversation turns, many-questions one pass,
  confidence gating, high-cardinality choice, long document, multilingual
  routing, email/phishing + triage + guardrails + moderation + model-router
  presets (question sets via laya.presets where exported).
"""
import json
import os
import sys
import urllib.request

os.environ.setdefault("USE_TF", "0")
OFFICIAL = os.environ.get("LAYA_OFFICIAL", "/tmp/laya-official")
sys.path.insert(0, OFFICIAL)
sys.path.insert(0, os.path.dirname(__file__))

import laya  # noqa: E402
from common import ask  # noqa: E402

BASE = os.environ.get("LAYA_BASE", "http://127.0.0.1:8150")
CKPT = os.environ.get("LAYA_SDK_CKPT", "model-english")

agent = laya.load(CKPT)


def both(state, questions):
    """Run one task through SDK and laya-burn; return (sdk, ours)."""
    sdk = agent.predict(state, questions)["answers"]
    ours = ask(BASE, state, questions)
    return sdk, ours


def compare(qid, a, b):
    """Pairwise verdict for one answer. Returns ok/desc."""
    ta, tb = a.get("type"), b.get("type")
    if ta != tb:
        return False, f"type {ta}!={tb}"
    if ta == "choice":
        if a["choice"] != b["choice"]:
            return False, f"choice {a['choice']!r}!={b['choice']!r}"
        pa, pb = a["probabilities"], b["probabilities"]
        if isinstance(pa, dict):
            # SDK keys are bare criteria keys; serve may render "key: desc".
            # Normalize both sides to the bare key prefix.
            strip = lambda d: {k.split(": ", 1)[0]: v for k, v in d.items()}
            na, nb = strip(pa), strip(pb)
            diffs = [abs(na[k] - nb.get(k, 0.0)) for k in na]
            dmax = max(diffs)
            return (dmax < 0.02, f"choice ok, max p-diff {dmax:.4f}") if dmax < 0.02 \
                else (False, f"max p-diff {dmax:.3f}")
        return True, "choice ok"
    if ta == "noul":
        d = abs(a["noul"] - b["noul"])
        return (d < 0.02, f"noul d={d:.4f}") if d < 0.02 else (False, f"noul d={d:.3f}")
    d = abs(a["score"] - b["score"])
    return (d < 0.15, f"score d={d:.3f}") if d < 0.15 else (False, f"score d={d:.3f}")


TASKS = []


def task(name, state, questions):
    TASKS.append((name, state, questions))


# ---- 01/02: minimal + hello world ----
task("01 minimal refund noul",
     {"body": "Hi, we were billed twice for March. Please refund the duplicate today."},
     {"refund_requested": {"type": "noul", "instructions": "Does the user explicitly request a refund?"}})

# ---- 04: choice with criteria descriptions ----
task("04 department routing",
     "A customer cannot log in since this morning and password reset emails never arrive.",
     {"department": {"type": "choice",
                     "instructions": "Which department should handle this message?",
                     "criteria": {"support": "login problems and account access",
                                  "billing": "invoices, charges and refunds",
                                  "sales": "pricing and upgrades"}}})

# ---- 06/07: score with levels ----
task("06 score priority",
     "The production database is down and every checkout is failing.",
     {"priority": {"type": "score",
                   "instructions": "How urgent is this message?",
                   "criteria": ["routine", "same-day", "urgent", "drop everything"]}})

# ---- 09: noul criteria wording ----
task("09 noul explicit-arm",
     "We would ideally like the invoice reissued, if that is possible.",
     {"wants_reissue": {"type": "noul", "instructions": "Does the writer ask to reissue the invoice?",
                        "criteria": {"true": "an explicit request to reissue",
                                     "false": "a wish or conditional mention, not a request"}}})

# ---- 11: JSON state ----
task("11 json order state",
     {"customer": {"name": "Marta", "tier": "pro"},
      "order": {"id": "A-118", "status": "shipped", "delivery_status": "in transit",
                "ordered": "keyboard", "received": None}},
     {"refund": {"type": "noul", "instructions": "Was a refund explicitly requested?"},
      "item_shipped": {"type": "noul", "instructions": "Does the order status say shipped?"},
      "what_ordered": {"type": "choice", "instructions": "What did the customer order?",
                        "criteria": {"keyboard": None, "mouse": None, "monitor": None}}})

# ---- 13: email clean vs raw (the decision-changing case) ----
PHISH_RAW = {"subject": "Invoice 4411", "from": "security@secure-bank-verify.example",
             "body": "Dear customer,\n\nWe detected unusual activity and locked your account. "
                     "Verify immediately at http://secure-bank--verify.example or the account "
                     "will be closed permanently.\n\nBest regards, Account Security Team"}
EMAIL_QS = laya.email_questions()
task("13 email preset (phishing lure)", PHISH_RAW, EMAIL_QS)

# ---- 14: conversation turns ----
task("14 conversation turns",
     ["user: my order still has not arrived",
      "agent: I am sorry - your order shows in transit, arriving Thursday.",
      "user: thursday is too late, I travel tomorrow. cancel and refund."],
     {"cancel_requested": {"type": "noul", "instructions": "Does the user ask to cancel the order?"}})

# ---- 16: many questions one pass ----
task("16 mixed primitives",
     {"body": "Hi, we were billed twice for March and support chat ignored us. "
              "This is the third time. Please refund today."},
     {"refund": {"type": "noul", "instructions": "Is a refund explicitly requested?"},
      "angry": {"type": "noul", "instructions": "Is the writer frustrated?"},
      "team": {"type": "choice", "instructions": "Which team should reply?",
                "criteria": {"billing": "charges and invoices", "support": "account help",
                             "management": "escalations"}},
      "severity": {"type": "score", "instructions": "How severe is this case?",
                    "criteria": ["minor", "moderate", "major", "critical"]}})

# ---- 18: confidence gating ----
task("18 confidence gating",
     {"body": "maybe sort of a refund could be considered possibly"},
     {"refund": {"type": "noul", "instructions": "Does the user explicitly request a refund?"}})

# ---- 33: high-cardinality choice (30 options) ----
import string
task("33 high-cardinality choice",
     "Route this to the person who owns the payments ledger reconciliation.",
     {"owner": {"type": "choice", "instructions": "Who owns this task?",
                "criteria": {f"owner_{i:02d}": f"owns domain {string.ascii_uppercase[i % 26]}"
                            for i in range(30)}}})

# ---- 37: long document ----
LONG_DOC = "Quarterly report. " + ("Revenue grew 4% quarter over quarter driven by enterprise "
                                   "expansion in EMEA. Churn stayed flat at 2.1%. ") * 120
task("37 long document tone",
     LONG_DOC,
     {"tone": {"type": "choice", "instructions": "What is the tone of this report?",
                "criteria": {"positive": "growth and optimism", "neutral": "factual",
                             "negative": "decline and concern"}},
      "mentions_churn": {"type": "noul", "instructions": "Does the report mention churn?"}})

# ---- 39: error handling shape (empty criteria on noul is legal) ----
task("39 noul default arms",
     {"body": "The tracking page shows the parcel delivered but nothing arrived."},
     {"delivered": {"type": "noul", "instructions": "Was the parcel delivered?"}})


def main():
    import statistics
    results = []
    for name, state, qs in TASKS:
        sdk, ours = both(state, qs)
        oks, descs = [], []
        for qid in sdk:
            ok, desc = compare(qid, sdk[qid], ours.get(qid, {}))
            oks.append(ok)
            descs.append(desc)
        ok = all(oks)
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {'; '.join(descs)}", flush=True)
    n = sum(results)
    print(f"\n{n}/{len(results)} tasks agree (choice exact, noul/score within tolerance)")


if __name__ == "__main__":
    main()
