//! Thin wrapper over the HuggingFace tokenizers crate for the Laya tokenizer.

use tokenizers::Tokenizer;

pub struct LayaTokenizer {
    tok: Tokenizer,
    pub cls_token_id: i64,
    pub sep_token_id: i64,
    pub mask_token_id: i64,
    pub pad_token_id: i64,
}

fn id_of(tok: &Tokenizer, token: &str) -> i64 {
    tok.token_to_id(token)
        .unwrap_or_else(|| panic!("special token {token} not in tokenizer"))
        as i64
}

impl LayaTokenizer {
    /// Load tokenizer.json next to its tokenizer_config.json (same
    /// directory). Special-token NAMES come from the config: the english
    /// checkpoints use BERT-style [CLS]/[SEP], laya-multilingual uses
    /// <bos>/<eos> (XLM-style) - the ids differ per checkpoint, the ROLES are
    /// stable, so resolve by config name with the BERT name as fallback.
    pub fn load(path: &str) -> Self {
        let cfg = std::path::Path::new(path)
            .parent()
            .map(|d| d.join("tokenizer_config.json"))
            .filter(|p| p.is_file())
            .and_then(|p| std::fs::read_to_string(p).ok());
        Self::from_json(
            &std::fs::read_to_string(path).expect("load tokenizer.json"),
            cfg.as_deref(),
        )
    }

    /// No-filesystem variant of [LayaTokenizer::load]: the tokenizer.json
    /// body plus the optional tokenizer_config.json body (special-token
    /// names), for embedders that fetch bytes (wasm).
    pub fn from_json(json: &str, cfg_json: Option<&str>) -> Self {
        let tok = Tokenizer::from_bytes(json.as_bytes().to_vec()).expect("parse tokenizer.json");
        let cfg: Option<serde_json::Value> =
            cfg_json.and_then(|s| serde_json::from_str(s).ok());
        let name_of = |key: &str, fallback: &str| -> String {
            cfg.as_ref()
                .and_then(|c| c[key].as_str().map(str::to_string))
                .unwrap_or_else(|| fallback.to_string())
        };
        let t = Self {
            cls_token_id: id_of(&tok, &name_of("cls_token", "[CLS]")),
            sep_token_id: id_of(&tok, &name_of("sep_token", "[SEP]")),
            mask_token_id: id_of(&tok, &name_of("mask_token", "[MASK]")),
            pad_token_id: id_of(&tok, &name_of("pad_token", "[PAD]")),
            tok,
        };
        t
    }

    /// Encode without adding special tokens.
    pub fn encode(&self, text: &str) -> Vec<i64> {
        let enc = self.tok.encode(text, false).expect("encode");
        enc.get_ids().iter().map(|&i| i as i64).collect()
    }
}
