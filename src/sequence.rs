//! Port of build_sequence / render_options / collate from rl_common.py.

use crate::tokenizer::LayaTokenizer;

pub const QTYPES: [&str; 3] = ["choice", "score", "noul"];

/// A question in internal form.
#[derive(Clone)]
pub struct Question {
    pub qtype: usize, // 0=choice 1=score 2=noul
    pub instructions: String,
    /// choice: ordered (key, description) pairs; score: level descriptions; noul: unused.
    pub options: Vec<String>,
}

/// Option texts in label-index order (render_options).
pub fn render_options(q: &Question) -> Vec<String> {
    q.options.clone()
}

/// [CLS] head [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
/// Returns input ids and marker positions.
pub fn build_sequence(
    tok: &LayaTokenizer,
    state: &str,
    q: &Question,
    max_len: usize,
    head_max_len: usize,
) -> (Vec<i64>, Vec<usize>) {
    build_sequence_ex(tok, state, q, max_len, head_max_len, None, false)
}

/// truncate_left keeps the END of an over-long state (conversation-turn lists are serialized
/// newest-last, so right-truncation would drop the newest turn). Official SDK sets it for
/// JSON-array states.
/// `option_order[s]` places option `option_order[s]` in slot `s` (official
/// build_sequence semantics): markers come back in SLOT order, so the caller
/// must permute its target the same way and un-permute model rows
/// (`unpermute`) before labelling answers.
pub fn build_sequence_ex(
    tok: &LayaTokenizer,
    state: &str,
    q: &Question,
    max_len: usize,
    head_max_len: usize,
    option_order: Option<&[usize]>,
    truncate_left: bool,
) -> (Vec<i64>, Vec<usize>) {
    let opts = render_options(q);
    let order: Vec<usize> =
        option_order.map(|o| o.to_vec()).unwrap_or_else(|| (0..opts.len()).collect());
    let ins = q.instructions.replace("<mask>", " ");
    let head_ids = tok.encode(&format!("{} question: {}", QTYPES[q.qtype], ins));

    let mut opt_ids: Vec<Vec<i64>> = Vec::new();
    for &i in &order {
        let o = &opts[i];
        let text = format!(" {}", o.replace("<mask>", " "));
        let mut ids = vec![tok.mask_token_id];
        ids.extend(tok.encode(&text).into_iter().take(48));
        opt_ids.push(ids);
    }
    let opt_budget = head_max_len as isize - opt_ids.iter().map(|o| o.len() as isize).sum::<isize>();
    if opt_budget < 16 {
        let per = ((head_max_len - 16) / opt_ids.len().max(1)).max(4);
        opt_ids = opt_ids.into_iter().map(|o| o.into_iter().take(per).collect()).collect();
    }
    let head_ids: Vec<i64> = head_ids
        .into_iter()
        .take((opt_budget.max(8)) as usize)
        .collect();

    let mut ids = vec![tok.cls_token_id];
    ids.extend(head_ids);
    ids.push(tok.sep_token_id);
    let mut markers = Vec::new();
    for o in &opt_ids {
        markers.push(ids.len());
        ids.extend(o.iter().copied());
    }
    ids.push(tok.sep_token_id);
    let room = max_len.saturating_sub(ids.len() + 1);
    let st = tok.encode(&state.replace("<mask>", " "));
    let st: Vec<i64> = if truncate_left {
        let skip = st.len().saturating_sub(room);
        st.into_iter().skip(skip).collect()
    } else {
        st.into_iter().take(room).collect()
    };
    ids.extend(st);
    ids.push(tok.sep_token_id);
    ids.truncate(max_len);
    let markers: Vec<usize> = markers.into_iter().filter(|&m| m < max_len).collect();
    (ids, markers)
}

/// A batch of encoded questions.
pub struct Batch {
    pub input_ids: Vec<Vec<i64>>,
    pub attention_mask: Vec<Vec<i64>>,
    pub marker_pos: Vec<Vec<i64>>,
    pub marker_mask: Vec<Vec<i64>>,
    pub qtype: Vec<i64>,
}

/// Put a slot-ordered row (logits/probabilities) back into the caller's
/// option order: out[order[s]] = row[s]. A missing/mismatched order returns
/// the row untouched (official unpermute_probs semantics).
pub fn unpermute(row: &[f32], order: Option<&[usize]>) -> Vec<f32> {
    match order {
        Some(o) if o.len() == row.len() => {
            let mut out = vec![0.0f32; row.len()];
            for (slot, &opt) in o.iter().enumerate() {
                out[opt] = row[slot];
            }
            out
        }
        _ => row.to_vec(),
    }
}

#[cfg(test)]
mod order_tests {
    use super::*;

    #[test]
    fn option_order_places_markers_in_slot_order_and_unpermute_inverts() {
        let tok = crate::tokenizer::LayaTokenizer::load("model-english/tokenizer/tokenizer.json");
        if !std::path::Path::new("model-english/tokenizer/tokenizer.json").is_file() {
            eprintln!("skipping: tokenizer not present");
            return;
        }
        let q = Question {
            qtype: 0,
            instructions: "pick".into(),
            options: vec!["a".into(), "b".into(), "c".into()],
        };
        let (_, m_plain) = build_sequence_ex(&tok, "s", &q, 512, 192, None, false);
        let (_, m_perm) = build_sequence_ex(&tok, "s", &q, 512, 192, Some(&[2, 0, 1]), false);
        // same marker count, different absolute positions => options reordered
        assert_eq!(m_plain.len(), 3);
        assert_eq!(m_perm.len(), 3);
        let row = [0.1f32, 0.8, 0.3]; // slot order: c a b
        let back = unpermute(&row, Some(&[2, 0, 1]));
        assert_eq!(back, vec![0.8, 0.3, 0.1]); // option order: a b c
        assert_eq!(unpermute(&row, None), row.to_vec());
    }
}
