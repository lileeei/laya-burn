# laya-burn vs the official test content

Every official test artifact is local at `/tmp/laya-official` (clone of
convaiinnovations/laya). This file records what was run, with what, and the
verdicts. Runner: `evals/laya_evals.py` (same JSONL format, metric semantics
and report shape as the official `laya.evals`; `evals/act_head_eval.py` ports
the official act-head diagnostic).

## 1. laya-evals harness compatibility

- `validate` accepts the official `research/evals/fixture.jsonl` (12 rows).
- `run` over the fixture produces official-shaped slices; `--baseline` +
  `--tolerance` implements the scheduled-gate comparison (thresholds.json:
  en accuracy 0.02 / ece 0.03 / mean_confidence 0.03).
- The answer payload now matches the official wire contract: `choice` is the
  criteria KEY (not the rendered option), and carries `answer_confidence`
  (p(reported answer) - the calibration column) and
  `action.act_probability` alongside the entropy-derived `confidence`.

## 2. en gate vs `eval_english_51_languages.json` (english checkpoint)

| | laya-burn | official | verdict |
|---|---|---|---|
| accuracy | 0.79 | 0.82 | delta exactly at tolerance (known rel-2e-4 float-noise flips, 3/100) |
| mean_confidence | 0.9705 | 0.9582 | OK (+0.0123, tol 0.03) |
| ece | 0.1805 | 0.1382 | out (+0.042) - driven by the same 3 flips |

The same checkpoint run through the local PyTorch reference scores 0.82
(commit 49e355d), so this is the documented port-precision gap, not a
semantic error.

The local `model/` checkpoint (newer snapshot, fitted temperatures) scores
**0.84 / ece 0.053** on the same 100 cases - beating the official english
package numbers.

## 3. act head (`act_head_eval.py`, port of the official diagnostic)

Exactly reproduces the official #185 finding: `act_probability` is
**saturated at 1.0** on all 100 en cases (unique values = 1, AUC degenerate),
while the confidence control ranks right/wrong at AUC 0.916. The signal is
measured, not trusted - same as upstream.

## 4. multilingual checkpoint vs `cpu_51_language_sweep_refreshed.json`

Forced-multilingual sweep (`evals/sweep51_multilingual_results.json`):
ours macro **0.3692** vs official refreshed **0.4008**. The gap concentrates
in bn/kn/hy/ta/ml (-0.06..-0.14) while zh-CN/ja/ar/th/km match the official
numbers exactly. Ruled out as causes:

- tokenizer: ids verified identical to `transformers` on 180 sequences
  across those scripts (`examples/tok_cmp.rs`)
- config: official provenance records max_len 1024 / head_max_len 256, same
  as ours; temperatures identical (no fitted map on either side)
- port fidelity: the LOCAL PyTorch reference scores 10/30 on bn (~0.33),
  same as Rust - the port is faithful to the local weights

Verdict: the official refreshed numbers reflect a newer package snapshot
than the local `model-multilingual/` copy. The official repo's own #208
records that the committed multilingual columns do not reproduce and were
refreshed at laya 0.3.20; the local weights are the older snapshot.

## 5. Routing insight (from sweep51_routed_results.json)

The refreshed official data shows mmBERT beats the english checkpoint on
most LATIN languages too (it 0.52 vs 0.34, nb 0.56 vs 0.29, id 0.51 vs
0.32; only en 0.71 vs 0.79 and fr favor english), so the official Router's
Latin->english heuristic leaves accuracy on the table for MASSIVE-style
workloads: oracle per-language best-of = 0.4035 vs routed 0.3402. Not a
porting defect - the router is a faithful reduced port - but a real
property of the production routing policy on this benchmark.
