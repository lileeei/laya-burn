# laya-burn

Rust implementation of [Laya](https://huggingface.co/convaiinnovations/laya) — the 421M
non-autoregressive typed-decision model — running natively on the
[Burn](https://burn.dev) tensor framework with real checkpoint weights.

## What it implements

Ported from the reference implementation (`rl_common.py` / `rl_agent_api.py` in the model repo):

- **ModernBERT-large encoder**: fused Wqkv attention, Llama-style rotate-half RoPE
  (theta 160k global / 10k sliding layers), alternating local-window(128)/global attention
  via additive masks, gated hybrid MLP (`gelu(input) * gate`), unfused residuals,
  no-bias norms, layer 0 without norms, final norm.
- **Typed decision head**: type embedding, 2-layer pre-norm transformer over the sequence,
  per-option scorer on [MASK] marker positions, act head on pooled CLS + answer-distribution
  features, per-bucket temperature calibration.
- **Sequence builder**: `[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]`
  with the head token budget logic, ported from `build_sequence`.
- **Tokenizer**: HuggingFace `tokenizers` crate loading the shipped `tokenizer.json`.
- **Weights**: safetensors F16 → F32 loader with key mapping into Burn modules.

## Setup

```sh
# download the checkpoint (or from huggingface.co directly)
mkdir -p model/tokenizer
curl -L https://hf-mirror.com/convaiinnovations/laya/resolve/main/typed-decisions/model.safetensors -o model/model.safetensors
curl -L https://hf-mirror.com/convaiinnovations/laya/resolve/main/typed-decisions/rl_agent_config.json -o model/rl_agent_config.json
curl -L https://hf-mirror.com/convaiinnovations/laya/resolve/main/typed-decisions/tokenizer/tokenizer.json -o model/tokenizer/tokenizer.json

cargo run --release           # demo questions (choice / score / noul)
```

## Verified behavior

A 12-option "largest number" question (temperature bucket `choice:11+`, temp 0.1) gets a
decisive answer concentrated on the high numbers — confirming the port carries real model
signal, not noise. Empirical A/B checks that pinned down the architecture details:

- RoPE: half-split (Llama style) gives coherent rankings; interleaved does not.
- MLP: `gelu(first half) * second half` keeps signal; the swapped order collapses to uniform.

The high temperature buckets (`noul:2` = 1.98, `choice:3-5` = 1.76) deliberately flatten
distributions — near-uniform outputs there match the shipped calibration.

### Debug switches

- `LAYA_ROPE=interleaved` — use GPT-J style interleaved RoPE (kept for A/B testing)
- `LAYA_MLP=swap` — swap MLP gate order (kept for A/B testing)

## Parity testing

`parity/` contains a test harness against the official PyTorch implementation:

```sh
# rust side: dump intermediates
LAYA_DUMP_DIR=parity/rust_out cargo run --release
# reference side (needs: pip3 install torch transformers safetensors)
python3 parity/run_ref.py
# compare
python3 parity/compare.py parity/rust_out parity/ref_out
```

Latest result: **PASS** — every stage (embedding, encoder layers 0/1/27, final
encoder output, decision head, option logits, act logits) matches the reference
within relative error 1e-5 to 2e-4 (F16->F32 weight conversion noise).

The parity hunt fixed four real bugs, all invisible to the earlier A/B heuristics:

1. RoPE second half used `- q1*sin` instead of `+ q1*sin` (sign error).
2. Sliding window mask was `|i-j| < 128`; ModernBERT uses `|i-j| <= 64`.
3. Layer 0 skipped mlp_norm; only attn_norm is Identity there.
4. Sliding layers ran global attention when seq_len <= window (mask guard).

## Components absorbed from the official SDK

Cross-reviewed against [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)
(`laya-rust/`, Apache-2.0); the following were ported in and adapted:

- `src/calibration.rs` — temperature clamp to [0.5, 5.0] (official issue #42: a fitted
  0.1 bucket sharpened logits and overstated confidence), f64-precision tempered softmax,
  `answer_confidence` (max-probability, the calibrated quantity), Python round-half-even
  `round4`, expected-score readout.
- `src/python_json.rs` — byte-for-byte `json.dumps(ensure_ascii=False)` serialization, so
  dict states tokenize identically to the Python SDK (fixes the key-order/spacing divergence
  found during parity testing).
- `sequence::build_sequence_ex` — `truncate_left` (conversation lists keep the newest
  turns) per the official sequence builder.

Logits-level parity is unaffected by these readout changes (verified: still PASS).

## Status / caveats

- [x] End-to-end inference with real weights on the flex CPU backend
- [x] All 206 checkpoint tensors mapped and loaded
- [x] Parity vs the PyTorch reference: PASS (see "Parity testing")
- [ ] Performance tuning (dense-mask sliding attention, batch inference, quantization)

## macOS performance

Enabled via Cargo features: `apple-amx` (flex CPU matmuls use the Apple Silicon AMX
matrix units through the `gemm` crate) and `metal` (wgpu Metal GPU backend).
Select at runtime with `LAYA_DEVICE` (default `flex`):

```sh
cargo run --release                      # flex + AMX (best for small batches)
LAYA_DEVICE=metal cargo run --release    # Metal GPU
```

Measured on Apple M6, batch of 5 questions (~90-token sequences), warm runs:

| Backend       | Model load | Inference (5 questions) |
|---------------|-----------:|------------------------:|
| flex + AMX    |     0.61 s |                  0.73 s |
| Metal (wgpu)  |     0.63 s |                  0.95 s |

Notes:
- The first Metal run pays one-time CubeCL kernel compilation (~6 s extra); it is
  disk-cached across runs.
- Outputs are identical across both backends (cross-checked on the demo questions).
- At this small batch size the AMX CPU path wins; Metal should overtake on larger
  batches / longer sequences (1024-token states).