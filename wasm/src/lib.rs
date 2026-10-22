//! laya-web: compile-level spike for wasm32 + WebGPU support in the laya-burn stack.
//!
//! Milestone A (default build): an exported function that builds a small burn
//! tensor on the WebGPU device and returns its sum. Compile-level proof that
//! burn 0.22 with default-features = false, features = ["std", "webgpu"]
//! builds for wasm32-unknown-unknown.
//!
//! Milestone B (feature "tokenizer"): parse a tiny embedded tokenizer.json,
//! proving the tokenizers crate builds for wasm when onig is replaced by
//! fancy-regex via the "unstable_wasm" feature.
//!
//! Feature "laya": links the parent crate; see REPORT.md.

use burn::tensor::{Device, Tensor};
use wasm_bindgen::prelude::*;

/// Sum of 1.0 ..= 8.0 computed on the WebGPU device. Expected: 36.
///
/// Device initialization on wasm is async under the hood; in a browser this
/// should be awaited from JS (wasm-bindgen-futures). This spike proves the
/// chain compiles, so a synchronous export is enough.
#[wasm_bindgen]
pub fn webgpu_tensor_sum() -> f32 {
    let device = Device::webgpu(Default::default());
    let tensor = Tensor::<1>::from_floats([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &device);
    tensor.sum().into_scalar()
}

#[cfg(feature = "tokenizer")]
mod tokenizer_probe {
    use std::str::FromStr;

    use tokenizers::Tokenizer;
    use wasm_bindgen::prelude::*;

    /// Minimal WordPiece tokenizer.json (BERT-style, tiny vocab). Proves the
    /// tokenizers crate parses and runs on wasm with fancy-regex.
    const TOKENIZER_JSON: &str = r###"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": { "type": "BertNormalizer" },
        "pre_tokenizer": { "type": "BertPreTokenizer" },
        "post_processor": { "type": "TemplateProcessing" },
        "decoder": { "type": "WordPiece", "prefix": "##", "cleanup": true },
        "model": {
            "type": "WordPiece",
            "vocab": {
                "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3,
                "h": 4, "i": 5, "##i": 6, "hello": 7
            },
            "unk_token": "[UNK]",
            "continuing_subword_prefix": "##"
        }
    }"###;

    /// Token count for "hi hello" under the embedded tokenizer.
    /// Expected: 5 ([CLS] h ##i hello [SEP]).
    #[wasm_bindgen]
    pub fn tokenizer_probe() -> Result<u32, JsError> {
        let tokenizer = Tokenizer::from_str(TOKENIZER_JSON).map_err(|e| JsError::new(&e.to_string()))?;
        let encoding = tokenizer
            .encode("hi hello", true)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(encoding.get_ids().len() as u32)
    }
}

/// Feature "runtime": the real inference surface for a browser host. The page
/// fetches the Q8 burnpack (~380-520MB) plus the sidecar/config/tokenizer
/// jsons, passes the bytes in, then calls decide per request. Weights live
/// dequantized in the wasm heap; compute runs on WebGPU via burn.
///
/// Device init under WebGPU is async under the hood (adapter request); a real
/// page should await it via wasm-bindgen-futures before the first decide -
/// see REPORT.md "Realistic path to a browser demo".
/// Runs once at instantiation: cubecl registers kernels via inventory
/// static constructors, which wasm-ld otherwise re-runs on every export
/// (observed to dominate inference time - see wasm/REPORT.md).
#[cfg(feature = "runtime")]
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    unsafe { __wasm_call_ctors() };
}

#[cfg(feature = "runtime")]
unsafe extern "C" {
    fn __wasm_call_ctors();
}

#[cfg(feature = "runtime")]
mod laya_runtime {
    use std::cell::RefCell;

    use burn::tensor::Device;
    use laya_burn::inference::RlAgent;
    use laya_burn::jev;
    use wasm_bindgen::prelude::*;

    thread_local! {
        static AGENT: RefCell<Option<(Device, RlAgent)>> = const { RefCell::new(None) };
    }

    /// Build the agent from a (Q8 or f32) burnpack + sidecar/config/
    /// tokenizer json bodies. `tok_cfg_json` may be "" when the checkpoint
    /// has no tokenizer_config.json. Returns the number of parameters
    /// dequantized from a quantized pack, 0 for f32.
    /// Async so the page can render a progress state while the 1.3GB of
    /// dequantized weights are built. Runs on the CPU backend: wasm cannot
    /// read GPU tensors synchronously (into_data / into_scalar panic), and
    /// both the eager dequant and the answer readback need sync reads.
    /// WebGPU needs an async-readback pass through the mapper and the
    /// inference tail - see wasm/REPORT.md. Resolves with "ok".
    #[wasm_bindgen]
    pub async fn laya_init_q8(
        pack: Vec<u8>,
        sidecar_json: &str,
        cfg_json: &str,
        tok_json: &str,
        tok_cfg_json: &str,
    ) -> Result<JsValue, JsValue> {
        // flex: the pure-Rust CPU backend (wasm-capable). Device::default()
        // resolves to the LLVM CubeCL CPU runtime on this target, whose JIT
        // service cannot start inside wasm ('Service not yet initialized').
        let device = Device::flex();
        let agent = RlAgent::load_q8_bytes(
            pack,
            sidecar_json,
            cfg_json,
            tok_json,
            (!tok_cfg_json.is_empty()).then_some(tok_cfg_json),
            &device,
        );
        AGENT.with(|a| *a.borrow_mut() = Some((device, agent)));
        Ok(JsValue::from_str("ok"))
    }

    /// One request: `state` string + Jev questions object (json) -> answers
    /// object (json), same shape as POST /v1/decide.
    #[wasm_bindgen]
    pub fn laya_decide(state: &str, questions_json: &str) -> String {
        let body: serde_json::Value =
            serde_json::from_str(&format!(r#"{{"state": {{}}, "questions": {questions_json}}}"#))
                .expect("questions json");
        // The same parse/validate/label path as laya-serve (the wasm copy
        // previously skipped the >=2-option validation).
        let req = jev::parse_request(&body).expect("valid questions");
        AGENT.with(|cell| {
            let mut guard = cell.borrow_mut();
            let (device, agent) = guard.as_mut().expect("call laya_init_q8 first");
            let state = serde_json::Value::String(state.to_string());
            let answers = agent.system_one(device, &state, &req.questions, None);
            let map = jev::answers_json(&req.ids, &req.questions, &answers);
            serde_json::to_string(&serde_json::Value::Object(map)).expect("answers json")
        })
    }
}
