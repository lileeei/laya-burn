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
