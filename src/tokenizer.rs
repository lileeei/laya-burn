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
    pub fn load(path: &str) -> Self {
        let tok = Tokenizer::from_file(path).expect("load tokenizer.json");
        let t = Self {
            cls_token_id: id_of(&tok, "[CLS]"),
            sep_token_id: id_of(&tok, "[SEP]"),
            mask_token_id: id_of(&tok, "[MASK]"),
            pad_token_id: id_of(&tok, "[PAD]"),
            tok,
        }
        // Sanity: ModernBERT ids from the encoder config.
        ;
        t
    }

    /// Encode without adding special tokens.
    pub fn encode(&self, text: &str) -> Vec<i64> {
        let enc = self.tok.encode(text, false).expect("encode");
        enc.get_ids().iter().map(|&i| i as i64).collect()
    }
}
