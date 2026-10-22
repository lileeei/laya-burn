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

Export once (writes `model.q8.mpk`, ~513MB, 3.3x smaller than the f32 burnpack):

```
cargo run --release -- model-english   # with LAYA_QUANT=model-english
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

Committed results:

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
| `LAYA_MODEL` | force router route |
| `LAYA_BURNPACK` | `q8` / `1` (f32) force a pack; unset prefers `model.q8.mpk` then `model.mpk` then safetensors |
| `LAYA_QUANT=1` | quantize at load (same scheme as the export) |
| `LAYA_QUANT=<dir>` | export `model.q8.mpk` for any checkpoint dir (geometry derived from the model) |
| `LAYA_CONVERT=<dir>` | export f32 burnpack |
| `LAYA_DTYPE` | `bf16` / `f16` compute dtype (default f32, parity-verified) |
| `LAYA_DEVICE` | `metal` / `wgpu` (default flex CPU) |
| `LAYA_REPLAY=1` | graph capture/replay fast path |
| `LAYA_ATTN=manual` | manual sliding-window attention (default: blocked fast path) |
| `LAYA_UNCLAMPED=1` | raw fitted temperatures (official harness mode) |
| `LAYA_LOAD` | `bytes` / `q8`: exercise the no-filesystem loaders (the wasm path); answers identical |
| `LAYA_BATCH_MS` | request micro-batching window in laya-serve |

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
