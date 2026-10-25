# laya-burn

Rust implementation of [Laya](https://huggingface.co/convaiinnovations/laya) — the
421M-parameter decision model (ModernBERT-large encoder + RLCD typed-decision head)
— running natively on the [Burn](https://burn.dev) tensor framework. Answers are
parity-verified against the official PyTorch checkpoint; MASSIVE-intent en accuracy
0.79 vs the official 0.82 (the gap is accumulated rel-2e-4 float noise flipping
near-tied 20-option choices, not semantics; ECE identical to 4 decimals).

## Quick start

```
cargo run --release -- model-english     # demo questions through the real checkpoint
cargo run --release --bin laya-serve -- --port 8080   # HTTP decision API
curl -s localhost:8080/v1/decide -d '{"state": "...", "questions": {...}}'
```

Checkpoint directories (gitignored, ~800MB each): `model-english/` (english,
512 tokens), `model/` (second ModernBERT-large checkpoint), `model-multilingual/`
(mmBERT-base, 1024 tokens, 256k vocab). Each holds `model.safetensors`,
`tokenizer/`, `rl_agent_config.json`, `encoder/config.json` — download from the
HF repo above, e.g.:

```
pip install -U huggingface_hub
hf download convaiinnovations/laya --include "multilingual/*" --local-dir .
mv multilingual model-multilingual
```

(the english checkpoint is the same with `english/*` → `model-english/`; hub
subfolder names mirror the three checkpoint dirs the Router expects. If a
`tokenizer_config.json` is missing, an empty `{}` file works - special-token
names then fall back to the BERT defaults).

With no CLI argument and routed directories present, requests are routed per state
script/language (`src/router.rs`, a port of the official SDK Router):
- `english` — ModernBERT-large, 512 tokens (implemented)
- `typed-decisions` — ModernBERT-large, 1024 tokens (implemented; `model/typed-decisions/`)
- `multilingual` — mmBERT-base, 1024 tokens, 256k vocab, F16 weights (implemented;
  answers parity-verified against the PyTorch reference on en + zh states, all
  question types identical to 4 decimals; MASSIVE zh-CN 0.62 vs english ckpt 0.57)

`LAYA_MODEL=english|typed-decisions|auto` forces a route.

## Q8 quantized serving (recommended)

Export once with `laya quantize <dir>` (writes `model.q8.mpk`, ~513MB, 3.3x smaller than the f32 burnpack; legacy env `LAYA_QUANT=<dir>` still works):

```
cargo run --release -- quantize model-english
```

Block-scaled Q8S (per [1,128] f32 scales) over the encoder + embeddings; the
decision head, scorer/act heads and LayerNorms stay f32. Then serving picks the
pack up automatically (or force with `LAYA_BURNPACK=q8|1`):

| | f32 safetensors | f32 burnpack | q8 burnpack |
|---|---|---|---|
| english artifact | 804 MB | 1685 MB | **513 MB** |
| english load | ~9 s | ~0.9 s | **~1.7 s** (eager dequant) |
| MASSIVE en acc | 0.79 | 0.79 | **0.79** |
| latency 50-question p50 | 884 ms | — | **865 ms** |
| multilingual artifact | 804 MB (f16) | — | **377 MB** |
| MASSIVE zh-CN acc / F1 | 0.62 / 0.518 | — | **0.62 / 0.518** |

Quantized params reload as QFloat tensors that burn's cubecl backend would
dequantize lazily per tensor on first use (~36 s); `laya_burn::quant::
eager_dequantize` instead decodes i8 x scale in one vectorized Rust pass at
load, bit-identical to the kernel path (verified by round-trip). Weights live
dequantized in memory, so runtime compute and latency equal f32; the win is
artifact size and (with `LAYA_QUANT=1`, no artifact) it also works quantize-at-load.

**Device comparison** (evals/metal_vs_flex.json): Metal and flex are
latency-equivalent across all batch sizes on this workload (the encoder's
sliding-window attention is AMX-bound on flex and doesn't gain enough from
GPU parallelism to overcome dispatch overhead). Metal's real advantage is
**load time: 0.6s vs 9.0s (15x)** — use `LAYA_DEVICE=metal` for
cold-start-sensitive deployments. Q8 on Metal gives the best single-question
latency (19.3ms). Accuracy is identical across all three configurations.

## Burnpack (f32)

`LAYA_CONVERT=<dir>` (in `main`) exports `model.mpk` + `model.mpk.json` sidecar;
reloading rebuilds the module tree from a zeros skeleton (no safetensors access)
in ~0.9 s. See `laya_burn::skeleton`.

## Evals

Official-methodology harnesses against a running `laya-serve`:

```
python3 evals/massive_eval.py en --base http://127.0.0.1:8090   # accuracy/F1/ECE
python3 evals/sweep51.py ...                                    # 51-language sweep
python3 evals/bench_latency.py http://127.0.0.1:8090            # p50/p95 latency
```

> **The optional comparison layer**: eval scripts marked `vs official` in
> the list below need a local clone of the upstream
> [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)
> repository (its frozen prompts, committed result JSONs, and in one case
> the Python SDK as a referee). Point them at it with `LAYA_OFFICIAL=<path>`
> (default `/tmp/laya-official`). The library, its tests, and every runner
> NOT marked `vs official` are fully self-contained - a fresh clone passes
> `cargo test` with zero external dependencies.

Committed results:

- `evals/app_benchmark_results_ours.json` *(vs official)* — the official 10-suite application
  benchmark (ag_news / emotion / banking77 / support triage / email spam /
  phishing / jailbreak / toxicity / RAG relevance / model routing; n=400,
  seed 13, suites built by reusing the official `bench_apps.build()`
  verbatim so sampling matches bit-for-bit): laya-burn vs official laya on
  the same checkpoint agrees within ±0.008 on 19/20 cells (mean |delta|
  0.006); vs Jev's published numbers, laya-burn tracks the official laya
  everywhere - ahead on ag_news (0.945 vs 0.91) and emotion (0.59 vs 0.48),
  and behind on banking77 (0.42 vs 0.87), which is the official
  apples-to-oranges caveat: laya scores 77 labels in one pass, Jev
  published 72-label numbers. Suites regenerate via
  `scripts/build_app_suites.py` (needs the HF dataset cache; JSONL inputs
  are gitignored, ~4MB).
- `evals/real_tasks_report.json` + `evals/real_tasks.py` *(vs official)* — real-task
  validation against the official SDK on the 41-example learning path's
  task shapes (email presets, JSON states, conversation turns,
  high-cardinality choice, long documents, confidence gating): f32 10/12
  agree within tolerance, and the suite caught two real port bugs no
  benchmark had (noul criteria arms dropped; compact-vs-python state
  serialization on every dict-state request) - both fixed in jev.rs.
  Robustness: 16-way concurrent serving identical-answer stable, clean 400s
  on malformed requests, language routing correct.
- `evals/official_benchmarks_results.json` *(vs official)* — the official frozen-prompt
  benchmark suites: es_phone_turns (Spanish phone routing), feishu_zh
  (Chinese chat triage, choice + four-noul modes), and the banking77
  tournament (group-elimination over high-cardinality choice, implemented
  client-side over /v1/decide). Every gap vs the committed official
  numbers was refereed to closure: hash verification across three hub
  revisions proves the weights never changed - es_phone's 0.90 rows are the
  benchmark's own FINE-TUNED model (their README's zero-shot row is 0.396,
  which we reproduce), and feishu's committed run used an unreleased SDK
  commit (the released SDK on the same weights scores 0.104 ~= laya-burn
  0.099 ~= checkpoint ref 0.094 - three implementations agree). The suites
  caught a THIRD wire-contract bug: probability keys must be bare criteria
  keys (fixed in jev.rs). Tournament over laya-burn: 0.365 -> 0.595 on
  banking77 (official 0.430 -> 0.610) - the strategy's step-change
  reproduces. Details: evals/official_benchmarks_results.json.
- `evals/sweep51_results.json` — english checkpoint forced on all 51
  languages: macro 0.2247 vs official 0.2269.
- `evals/sweep51_routed_results.json` — AUTO routing (english for Latin,
  mmBERT for non-Latin): **macro 0.3402**, +0.115 over english-only. 27
  languages gain >= 0.10 (hi +0.38, th +0.37, ko +0.35, ar +0.33); only
  fr/es/pt lose slightly (diacritic routing sends part of them to
  multilingual). The official repo has no multilingual-model sweep to
  compare against - these numbers are laya-burn's own contribution.
- `evals/latency_baseline.json` — serving latency + accuracy baselines.

## Environment knobs

| var | effect |
|---|---|
| `LAYA_DEVICE` | `metal` / `wgpu` (default flex CPU); see evals/metal_vs_flex.json |
| `LAYA_DTYPE` | `bf16` / `f16` compute dtype (default f32, parity-verified) |
| `LAYA_MODEL` | force router route (english/typed-decisions/multilingual/auto) |
| `LAYA_BURNPACK` | `q8` / `1`/`f32` / `safetensors` force an artifact; unset prefers q8 pack, then f32 pack, then safetensors |
| `LAYA_QUANT=1` | serve: quantize at load (export scheme). `LAYA_QUANT=<dir>` (binary): legacy alias for `laya quantize` |
| `LAYA_LOAD` | `bytes` / `q8`: exercise the no-filesystem loaders (the wasm path) |
| `LAYA_ATTN` | `manual` / `blocks` / `kernel` attention mode (default fused; kernel requires F32 - panics loudly otherwise) |
| `LAYA_KERNEL` | `v2` selects the second windowed-kernel generation (kernel mode only) |
| `LAYA_ROPE` | `interleaved` rope variant (default half-split) |
| `LAYA_MLP` | `swap` swaps the encoder MLP gate/input halves |
| `LAYA_THETA_SLIDE` | override the sliding rope theta (default 10000) |
| `LAYA_DUMP_DIR` | parity dumps: `all_rows.txt` + model-level tensors, resolved once at load |
| `LAYA_REPLAY=1`, `LAYA_REPLAY_N=k` | graph capture/replay fast path (and its repeat count) |
| `LAYA_BATCH` | demo binary: repeat the question set N times in one forward |
| `LAYA_STATE`, `LAYA_LONG` | demo binary: zh/ru demo state, long-sequence variant |
| `LAYA_BATCH_MS`, `LAYA_BATCH_MAX` | serve micro-batching window (ms) and max batch — routes concurrent requests through one predict_many |
| `LAYA_TRAIN_LOSS=ce`, `LAYA_TRAIN_LOG=1` | finetune: CE-only ablation / per-step loss logging |
| `LAYA_CONVERT`, `LAYA_PARITY_DEBUG` | legacy convert alias; rlcd_core parity debug output |

Env vars resolve at the binary edge (`LoadOptions::from_env` / `ForwardOptions::from_env`, once); library code reads none (Router::new is the documented serving adapter).

## Typed decision surface (the ergonomic API)

`agent.ask(&device, &state)` replaces bare `qtype: 0` literals with a typed
builder, and `Answers` gives typed accessors - the Rust-native answer to
the official SDK's structured.py need (field-mapping validation happens at
compile time via the type system, not at runtime via JSON Schema):

```rust
let out = agent.ask(&device, &json!("Refund invoice 4411"))
    .choice("department", &["billing", "support", "sales"])
    .noul("refund_requested")
    .score("urgency", 0..=3)
    .run()?;

out.choice("department");       // &str
out.noul("refund_requested");   // bool
out.score("urgency");           // i32
out.answer_confidence("urgency"); // f32
```

The confidence gate (official confidence.py semantics, Rust-native): a
per-bucket policy with a THREE-state verdict - `Passed` / `Abstained` /
`Unevaluated` (the gate ran; the answer carried no usable confidence) -
because collapsing the third state into either of the others is the lie
the official design exists to prevent:

```rust
let gate = Gate::new().min("choice:3-5", 0.9).min("noul:2", 0.5);
gate.apply(&mut out);
out.gates(); // [("department", Abstained), ...]
```

End-to-end demo: `cargo run --release --example ask_demo` (cross-checks
every answer against the raw API).

### Task presets

The five official frozen task sets, usable from BOTH surfaces:

- Library: `presets::triage(&agent.ask(...)).run()` (also `email`,
  `guard`, `moderation`, `model_routing`) - typed accessors for every
  field.
- HTTP: `POST /v1/decide {"state": ..., "questions": "triage"}` - the
  preset name expands to its frozen question set; unknown names get a 400
  listing the available five.

Criteria wording is verbatim from the official presets.py, verified
cross-SDK: the same triage input through laya 0.4.12 and laya-burn agrees
on every field (intent exact, noul/score within 0.023 - float noise, no
decision flips).

## Training (direction: trainable)

`cargo run --release --example train_spike` is the trainability gate, and it
passes end to end:

- **Gradients flow** through the full stack (encoder layer 10 grad norm 2.5,
  head grad norm 5e-2) on `Device::default().autodiff()`. Params opt in via
  `Tensor::autodiff().require_grad()` (re-applied after every optimizer step).
- **Official losses**: `train.rs` ports the proper scoring rule
  (`proper_reward_host`, parameterized weights) and both training losses
  (`soft_ce`, `rlcd_core`); the spike runs the SHARED loss (its private
  port with drifted weights - w_sph 0.5 vs 0.75 - is gone). A fixed 6-row
  batch overfits under AdamW on the shared path.
- **Export back**: `laya_burn::export::export_safetensors` writes the official
  PyTorch safetensors layout (206 tensors, linear weights un-transposed,
  `temperature` buffer included) - the reference `RLAgent.load_state_dict`
  accepts it with `strict=True`, and the fine-tuned weights change its
  predictions. The spike's 4/6 training-batch hits is the ceiling: two rows
  are an identical-input/different-gold pair by construction.

The fine-tune loop is COMPLETE (`laya finetune <data.jsonl> [model_dir]
[out_dir] [epochs] [micro_batch]`): laya-evals JSONL in (soft targets,
score interpolation, label smoothing), held-out calibration split,
per-epoch option-order shuffling (opt-in), AdamW, post-training temperature
fitting (golden-section port of fit_one_temperature), and a COMPLETE
checkpoint directory out - servable by laya-serve and loadable by the
PyTorch reference with strict=True. Verified end to end on
zh_short_commands (18 labeled Chinese commands): multilingual 0.611 -> 1.0
accuracy, ECE 0.244 -> 0.004.

The RLCD objective is gradient-verified against the PyTorch reference: a
committed fixed batch (tests/fixtures/grad_parity) pins loss AND
d(loss)/d(logits) to 1e-5 - the parity harness caught two real bugs, the
advantage std measured over the per-row-CENTERED rewards (not the raw ones)
and burn 0.22's repeat_dim backward misrouting through reshape (now tracked
slices + cat). With RLCD restored to the official default, the zh_commands
acceptance run reaches 0.944 accuracy / ECE 0.068 (CE-only memorization
reaches 1.0 - RLCD's exploration noise costs a little on an 18-case set).
Remaining deviation: the official effective batch is 64 via grad
accumulation; this port steps per micro-batch.

## WebAssembly

The full inference stack compiles for wasm32 + WebGPU and now has a runtime
surface (`wasm/` crate, feature `runtime`): `laya_init_q8(pack_bytes, sidecar,
cfg, tokenizer, tokenizer_cfg)` builds the agent from fetched bytes - no
std::fs anywhere on that path (`RlAgent::load_bytes` / `RlAgent::load_q8_bytes`
are the no-filesystem loaders, parity-identical on native via `LAYA_LOAD`) -
and `laya_decide(state, questions_json)` returns /v1/decide-shaped answers.
The `serve` feature (axum/tokio) is default-on for native builds and absent
from wasm builds.

The browser demo RUNS (`wasm/www/index.html`): fetch the 377MB multilingual
Q8 pack, init in ~17s, decide at ~2.2s/question on the flex CPU backend -
answers bit-identical to the native server. Build + serve:

```
cd wasm && wasm-pack build --target web --release --features runtime
python3 -m http.server 8765 --directory ..   # from wasm/
# open http://127.0.0.1:8765/wasm/www/index.html
```

Compute runs on `Device::flex()`: wasm cannot read GPU tensors synchronously
(both the eager dequant and the answer readback need it), and the LLVM CPU
runtime's JIT cannot start in the sandbox - see `wasm/REPORT.md` for the
three wasm-specific findings and what a WebGPU path needs.

## Repository layout

`src/model.rs` (encoder + head port), `src/inference.rs` (`RLAgent.system_one`
port), `src/router.rs` (SDK Router port), `src/quant.rs` (Q8 export + eager
dequant), `src/window_kernel.rs` (cubecl sliding-window kernel, `LAYA_ATTN=kernel`),
`parity/` (tensor dumps + PyTorch reference), `evals/`, `wasm/` (spike).
