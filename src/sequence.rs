//! Port of build_sequence / render_options / collate from rl_common.py.

use crate::tokenizer::LayaTokenizer;

pub const QTYPES: [&str; 3] = ["choice", "score", "noul"];

/// A question in internal form.
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
    build_sequence_ex(tok, state, q, max_len, head_max_len, false)
}

/// truncate_left keeps the END of an over-long state (conversation-turn lists are serialized
/// newest-last, so right-truncation would drop the newest turn). Official SDK sets it for
/// JSON-array states.
pub fn build_sequence_ex(
    tok: &LayaTokenizer,
    state: &str,
    q: &Question,
    max_len: usize,
    head_max_len: usize,
    truncate_left: bool,
) -> (Vec<i64>, Vec<usize>) {
    let opts = render_options(q);
    let ins = q.instructions.replace("<mask>", " ");
    let head_ids = tok.encode(&format!("{} question: {}", QTYPES[q.qtype], ins));

    let mut opt_ids: Vec<Vec<i64>> = Vec::new();
    for o in &opts {
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
