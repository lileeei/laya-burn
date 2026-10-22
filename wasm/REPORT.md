# wasm32 + WebGPU spike report

Spike crate: `wasm/` (this directory), package `laya-web`. No existing file was modified.
Gate command throughout: `cargo build --target wasm32-unknown-unknown -p laya-web`
run from `wasm/`. Toolchain: stable aarch64-apple-darwin, target wasm32-unknown-unknown.

## Verdict

**The laya-burn stack CAN build for wasm32 + WebGPU, with one precise blocker: the
`tokenizers` dependency's forced `onig` feature.** Everything else — burn 0.22 with
the webgpu backend, and even the exact burn feature set the parent hard-codes —
compiles for wasm32-unknown-unknown.

## What was proven (compile-level)

### Milestone A — burn + WebGPU on wasm32: PASS

`cargo build --target wasm32-unknown-unknown -p laya-web` (default features) exits 0
(~4 min cold). The crate exports `webgpu_tensor_sum()` (wasm-bindgen), which creates a
burn tensor on `Device::webgpu(Default::default())` and returns its sum — the burn
symbol is present in the emitted `laya_web.wasm`.

Feature recipe:

```toml
burn = { version = "0.22", default-features = false, features = ["std", "webgpu"] }
# required additive fix (see "onig is not the only wasm-ism" below)
getrandom = { version = "0.4", features = ["wasm_js"] }
```

One deviation from stock burn wasm setups was needed: burn's chain pulls
`rand` -> `getrandom` 0.4, which refuses to compile for wasm32-unknown-unknown unless
its `wasm_js` feature is enabled. Since cargo features are additive, a dependent can
fix this without touching the parent: `getrandom = { version = "0.4", features =
["wasm_js"] }` as a direct dependency. (Burn's own mnist-inference-web example works
because its workspace adds equivalent plumbing; a standalone dependent must add it.)

### Milestone B — tokenizers on wasm: PASS (with the right feature set)

`cargo build --target wasm32-unknown-unknown -p laya-web --features tokenizer` exits 0.
tokenizers 0.23.2 has a purpose-built wasm path:

```toml
tokenizers = { version = "0.23", default-features = false, features = ["unstable_wasm"] }
```

`unstable_wasm` = `fancy-regex` (pure-Rust regex, replacing onig) +
`getrandom/wasm_js`. The spike's `tokenizer_probe()` parses a tiny embedded
WordPiece tokenizer.json and counts tokens — the whole crate builds and the API is
usable. The other pure-Rust tokenizers deps (esaxx via default features, unigram,
spm_precompiled, daachorse) are not needed for BERT-style WordPiece inference.

### The parent's burn feature set on wasm: PASS (surprise finding)

`cargo build --features parent-burn-features` (adds `flex`, `apple-amx`, `metal`,
`autotune`, `fusion` on top of std+webgpu — exactly what feature unification does
when laya-burn is linked) exits 0 (~10 min). None of these block wasm compilation:

- `metal` / `apple-amx` compile because cubecl/burn-flex gate the native code behind
  `target_os`/`target_arch` cfgs; on wasm32 those paths are compiled out.
- `flex` is explicitly documented as std/no_std/WASM capable.
- `autotune`/`fusion` compile fine for wasm.

They are dead weight in a wasm artifact (and `autotune` benchmarking in a browser is
questionable at runtime), but they do not block the build.

## The blocker

`cargo build --features laya` (links the parent by path) fails:

```
error: failed to run custom build command for `onig_sys v69.9.3`
cargo:warning=oniguruma/src/regint.h:123:10: fatal error: 'stdlib.h' file not found
```

Exact chain: `laya-burn/Cargo.toml` ->
`tokenizers = { version = "0.23", default-features = false, features = ["onig"] }` ->
`onig` -> `onig_sys` (builds the C Oniguruma library with cc/clang
`--target=wasm32-unknown-unknown`; wasm32-unknown-unknown has no libc, so
`stdlib.h` does not exist and the C compilation fails).

Cargo features are additive and a dependent cannot remove them, so **no sub-crate can
build laya-burn for wasm32 as long as its Cargo.toml forces `onig`.** This is the only
hard blocker found. (The parent's own src — `std::fs`, `Device::metal/wgpu` calls —
compiles for wasm32; those are runtime concerns, not compile ones.)

## Recommended fix (one line, in the parent — outside this spike's write scope)

Change the parent's tokenizers dependency to a target-conditional feature set:

```toml
[dependencies]
tokenizers = { version = "0.23", default-features = false, features = ["unstable_wasm"] }
```

or, to keep onig's speed on native:

```toml
[target.'cfg(not(target_family = "wasm"))'.dependencies]
tokenizers = { version = "0.23", default-features = false, features = ["onig"] }

[target.'cfg(target_family = "wasm")'.dependencies]
tokenizers = { version = "0.23", default-features = false, features = ["unstable_wasm"] }
```

plus the additive `getrandom = { version = "0.4", features = ["wasm_js"] }` for the
wasm target. With that, every link of the chain (`--features full`) should build; the
spike's `laya` feature then becomes the gate to re-run.

## Realistic path to a browser demo

1. Apply the parent fix above (target-conditional tokenizers + getrandom wasm_js).
2. Re-run `cargo build --target wasm32-unknown-unknown -p laya-web --features full`.
3. Package with wasm-pack (`wasm-pack build --target web`) or trunk, following
   burn's `examples/mnist-inference-web`: cdylib, `wasm_bindgen(start)` running
   `__wasm_call_ctors` once (cubecl registers static constructors via `inventory`;
   without this, wasm-ld re-runs them on every export and dominates inference time),
   async device init via wasm-bindgen-futures.
4. Ship weights via HTTP fetch into the wasm heap (the parent's `std::fs::read` paths
   must be swapped for fetched bytes on wasm — runtime work, not a build blocker).
5. The 421M-weight question remains a product decision; nothing in the build chain
   forecloses it (safetensors + serde_json compile fine for wasm).

## Caveats

- Compile-level proof only; no browser run was attempted (per task instructions).
- Debug wasm artifact is ~900 MB unstripped; release + wasm-opt is the realistic size.
- `unstable_wasm` is upstream's name; it is the supported wasm path for tokenizers
  0.23 despite the scary name.
- `autotune` at wasm runtime was not evaluated.

## Files

- `wasm/Cargo.toml` — spike crate manifest, feature-gated milestones
- `wasm/src/lib.rs` — `webgpu_tensor_sum()` (A), `tokenizer_probe()` (B)
- `wasm/REPORT.md` — this report
- `wasm/Cargo.lock`, `wasm/target/` — build outputs

---

## Update 2026-10-22: runtime surface compiles; fs is off the hot path

Everything in "Realistic path to a browser demo" steps 1-2 plus half of 4 is
now done at the compile level, in the parent:

- The parent gates its server stack behind a `serve` feature (default on for
  native; `--no-default-features` drops axum/tokio, whose mio dependency was
  the thing that did not compile for wasm32).
- `laya-burn` (lib) builds for wasm32-unknown-unknown with the full inference
  stack: tokenizer, encoder, head, calibration, router, quant/eager dequant.
- New no-filesystem loaders `RlAgent::load_bytes` / `RlAgent::load_q8_bytes`
  (plus `Weights::from_bytes`, `LayaTokenizer::from_json`,
  `ModuleRecord::from_bytes`) remove the std::fs requirement REPORT step 4
  called out. Verified on native: `LAYA_LOAD=bytes|q8` reproduces the
  file-loaded answers bit-identically (f32) and at Q8 noise level (q8 pack).
- The spike crate grew a `runtime` feature: `laya_init_q8` / `laya_decide`
  (wasm-bindgen), thread-local agent on `Device::webgpu`.

Gate (all exit 0):

    cargo build --target wasm32-unknown-unknown -p laya-burn --lib --no-default-features
    cargo build --manifest-path wasm/Cargo.toml --target wasm32-unknown-unknown --features runtime

Still open for a real browser demo: wasm-pack/trunk packaging, async device
init via wasm-bindgen-futures (REPORT step 3), a demo page, and the
runtime-perf question (autotune in a browser, dequant cost on WebGPU).

---

## Update 2026-10-22 (evening): the browser demo RUNS

`wasm/www/index.html` + `wasm-pack build --target web --release --features
runtime` produce a working page: it fetches the 377MB multilingual Q8 pack,
builds the agent in the wasm heap, and answers - bit-identical to the native
server on the same inputs. Measured (WKWebView/IAB, M-series):

- fetch 377MB: ~4s (cached on reload)
- init (skeleton + record + eager dequant, 1.3GB f32): ~17s
- decide: ~2.2s/question warm

Three wasm-specific findings, for anyone following this path:

1. **Device::default() is the LLVM CubeCL CPU runtime on wasm** - its JIT
   service cannot start inside the sandbox ("Service not yet initialized -
   call init() before load()"). The wasm-capable CPU backend is `flex`
   (`Device::flex()`), the same backend the native default uses.
2. **WebGPU tensors cannot be read synchronously on wasm**: into_data /
   into_scalar panic ("try into_data_async"). Both the eager dequant (reads
   QFloat params) and the inference tail (reads probabilities) need sync
   reads, so the demo runs on flex. A WebGPU path needs an async-readback
   pass through the dequant mapper and system_one - plus async device init
   via `Device::wgpu_options().graphics_api(WebGpu).init_async().await`
   (the sync/lazy path parks forever on wasm).
3. wasm-bindgen(start) + `__wasm_call_ctors` + console_error_panic_hook make
   the failure modes visible instead of parked promises.

Serve with `python3 -m http.server 8765 --directory <repo root>` and open
`/wasm/www/index.html` (the page fetches `/model-multilingual/*`).
