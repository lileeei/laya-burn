# GLOSSARY

Domain vocabulary for laya-burn. Keep terms stable across code, commits, and
reviews; add a term here the moment code starts relying on it.

- **Checkpoint** — one Laya model directory: `model.safetensors` weights,
  `tokenizer/`, `rl_agent_config.json`, `encoder/config.json`. The three the
  Router knows: `english`, `typed-decisions`, `multilingual`.
- **Burnpack** — Burn-native serialized module record (`.mpk`), loaded with no
  safetensors access by rebuilding the module tree from a zeros skeleton.
- **Sidecar** — `model.mpk.json`: the geometry (`SkeletonSpec`) a burnpack
  reload needs to rebuild the module tree. The Q8 pack shares the f32 sidecar.
- **Q8 pack** — `model.q8.mpk`: block-scaled INT8 burnpack export
  (encoder + embeddings quantized, decision machinery exempt), dequantized
  eagerly in Rust at load.
- **Decision tail** — the pure mapping `(per-question logits, temperature,
  qtype, act_probability) -> Answer`: bucket resolution, temperature
  sharpening, round4, answer_confidence, expected score. Lives in
  `inference::answer_from_logits`; everything the evals measure passes
  through it. *(named during the 2026-10-22 deepening of system_one/run_rows)*
- **Jev** — the wire protocol of the public API: `{state, questions}` in,
  keyed answers out (`choice` = criteria key, `answer_confidence` =
  p(reported answer), `action.act_probability`). Shared by laya-serve and the
  wasm runtime.
- **Calibration split** — the held-out items  reserves BEFORE training; temperatures are fitted on them only.
- **Temperature bucket** — the per-(qtype, option-count) key (e.g. `choice:11+`) whose fitted temperature outranks the qtype base at serving.
- **RLCD** — the official policy-gradient objective: advantage-weighted Gaussian log-density over noise samples plus soft cross-entropy (see train.rs::rlcd_core).
- **Route** — a Router decision mapping a state's script/language to a
  checkpoint (`english` / `typed-decisions` / `multilingual`).
- **Proper scoring rule** — the official training loss family (log score +
  spherical + ordinal RPS); the choice-row port lives in
  `train.rs::proper_reward_host`.
