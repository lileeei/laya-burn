# Contributing to laya-burn

Thanks for considering a contribution. This document covers the practical
details; the [README](README.md) covers what the project is.

## Getting started

```bash
git clone <your-fork>
cd laya-burn
cargo build --release          # ~3 min; all dependencies from crates.io
cargo test --release           # 38 tests, green without any model download
```

No checkpoint downloads are needed to build or run the test suite. To run
the model itself you need one checkpoint directory (see README "Getting the
checkpoints"); tests that need one skip themselves when it is absent.

## The one rule that governs everything

**Behavior-parity with the official PyTorch implementation is the
contract.** A change to inference numerics is only correct if the parity
anchors hold. Concretely:

- `cargo test --release` must pass (includes the RLCD gradient-parity test
  against a committed PyTorch reference, and the Q8 export round-trip).
- The demo parity anchors must not move: `./target/release/laya-burn
  model-english | grep '^Q'` produces fixed expected output; a PR that
  changes any answer probability needs an explanation for WHY the old one
  was wrong (a port bug) or an explicit note that the change is a
  deliberate semantic deviation from the official model.
- Wire-contract changes (Jev request/response shape) need the same
  treatment: the official SDK is the reference, and deviations are bugs.

## Environment-variable discipline

Library code (`src/`) reads **zero** environment variables. All env knobs
resolve exactly once at the binary edge (`LoadOptions::from_env` /
`ForwardOptions::from_env` in `main.rs` and the serve adapters). If your
change adds a knob, it goes through those types - not `std::env::var`
scattered through the library. The CI warnings gate enforces zero
warnings; `cargo clippy` is advisory but appreciated.

## What lives where

| Path | Role |
|---|---|
| `src/model.rs` | encoder + decision head (the PyTorch port) |
| `src/inference.rs` | loading, forward, decision tail, collate |
| `src/ask.rs` | the typed builder surface (`agent.ask()`) |
| `src/jev.rs` | the wire protocol (request parse / answer render) |
| `src/train.rs` | fine-tune (RLCD, calibration, checkpoint writing) |
| `src/quant.rs` | Q8 export + eager dequant |
| `src/router.rs` | multi-checkpoint routing (script/language detection) |
| `src/calibration.rs` | temperature bucket/clamp/fit - the single home |
| `src/presets.rs` | the five frozen task sets |
| `evals/` | evaluation harnesses (Python; `vs official` ones need `LAYA_OFFICIAL`) |
| `tests/fixtures/` | committed binary fixtures (RLCD gradient parity) |
| `wasm/` | browser spike crate |

## Adding a benchmark or eval

Put the runner in `evals/`, import from `evals/common.py` (the shared
server adapter / metrics / sweep-IO seam - do not re-implement `ask` or
ECE), and mark it `*(vs official)*` in the README if it needs the upstream
repository. Committed result JSONs carry their methodology in `_meta`.

## Commit style

Look at `git log --oneline`: the subject line states the outcome with
numbers where they exist ("Q8 serving: block-scaled quantization recovers
f32 accuracy"), and the body explains what changed and why, including the
verification evidence. A PR that says "fixed bug" without saying which
anchor moved and why will be asked for more detail.

## Filing issues

Include: the checkpoint (english/multilingual/typed-decisions), the exact
command, and the output. For numerical discrepancies, include what the
PyTorch reference produced on the same input - the fastest way to
distinguish a port bug from an upstream difference.
