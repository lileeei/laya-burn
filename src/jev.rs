//! The Jev wire protocol: `{state, questions}` in, keyed answers out.
//!
//! One owner for the request/response contract shared by laya-serve and the
//! wasm runtime - the sorted-question-id ordering and the id<->answer zip
//! are invariants that must not be re-assembled per frontend (the wasm side
//! previously repeated the assembly WITHOUT the >=2-option validation, i.e.
//! the two copies had already diverged).

use serde_json::Value;

use crate::inference::{Answer, QuestionInput};

/// Criterion text must be the PLAIN string, not serde Display (which quotes
/// it); non-strings fall back to their JSON form. Port of rl_common's
/// render_options.
fn render_options(qtype: &str, criteria: &Value) -> Vec<String> {
    let text = |v: &Value| -> String {
        match v {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        }
    };
    match qtype {
        "choice" => criteria
            .as_object()
            .map(|m| {
                m.iter()
                    .map(|(label, desc)| {
                        let blank = matches!(desc, Value::Null)
                            || matches!(desc, Value::String(s) if s.is_empty());
                        if blank {
                            label.clone()
                        } else {
                            format!("{label}: {}", text(desc))
                        }
                    })
                    .collect()
            })
            .unwrap_or_default(),
        "score" => criteria
            .as_array()
            .map(|a| {
                a.iter()
                    .enumerate()
                    .map(|(i, d)| format!("level {i}: {}", text(d)))
                    .collect()
            })
            .unwrap_or_default(),
        // noul: criteria descriptions override the default arm wording
        // (official render_options; the port had dropped this branch, which
        // the real-task suite caught as large noul drift on criteria-carrying
        // questions like laya.email_questions()).
        _ => {
            let arm = |which: &str, fallback: &str| -> String {
                match criteria.get(which) {
                    Some(Value::String(s)) if !s.is_empty() => format!("{which}: {s}"),
                    Some(Value::String(_)) | None => format!("{which}: {fallback}"),
                    _ => format!("{which}: {fallback}"),
                }
            };
            vec![
                arm("false", "no, the statement does not hold"),
                arm("true", "yes, the statement holds"),
            ]
        }
    }
}

/// Jev question shape ({type, instructions, criteria}) -> [QuestionInput].
/// `keys` carries the raw criteria keys: the answer contract reports the
/// KEY as `choice`, not the rendered option text.
pub fn question_from_jev(q: &Value) -> Option<QuestionInput> {
    let qtype = q.get("type")?.as_str()?;
    let idx = match qtype {
        "choice" => 0,
        "score" => 1,
        "noul" => 2,
        _ => return None,
    };
    let instructions = match q.get("instructions") {
        Some(Value::String(s)) => s.clone(),
        Some(other) => crate::python_json::PythonJson::state(other),
        None => String::new(),
    };
    let criteria = q.get("criteria").cloned().unwrap_or(Value::Null);
    let keys = match (qtype, &criteria) {
        ("choice", Value::Object(m)) => m.keys().cloned().collect(),
        ("noul", Value::Object(m)) if m.contains_key("true") || m.contains_key("false") => {
            vec!["false".to_string(), "true".to_string()]
        }
        _ => Vec::new(),
    };
    // Official option_order: a permutation of range(n) placing option
    // option_order[s] in slot s. Invalid orders are ignored (unpermute
    // semantics: mismatched order returns the row untouched).
    let n_options = match qtype {
        "choice" => keys.len(),
        "score" => criteria.as_array().map(|a| a.len()).unwrap_or(0),
        _ => 2,
    };
    let option_order = q
        .get("option_order")
        .and_then(|v| v.as_array())
        .and_then(|a| {
            let order: Vec<usize> = a.iter().filter_map(|x| x.as_u64().map(|u| u as usize)).collect();
            (order.len() == n_options
                && {
                    let mut sorted = order.clone();
                    sorted.sort();
                    sorted == (0..n_options).collect::<Vec<_>>()
                })
            .then_some(order)
        });
    Some(QuestionInput { qtype: idx, instructions, options: render_options(qtype, &criteria), keys, option_order })
}

/// A parsed request: question ids in sorted order, one input each, aligned
/// index-for-index (the invariant [answers_json] relies on).
#[derive(Debug)]
pub struct JevRequest {
    pub state: String,
    pub ids: Vec<String>,
    pub questions: Vec<QuestionInput>,
}

/// Parse and validate a Jev request body.
pub fn parse_request(body: &Value) -> Result<JevRequest, String> {
    // Non-string states serialize PYTHON-style (json.dumps default
    // separators) - the port had serde's compact form here, which the
    // real-task suite caught as drift on every dict-state request.
    let state = match body.get("state") {
        Some(Value::String(s)) => s.clone(),
        Some(other) => crate::python_json::PythonJson::state(other),
        None => return Err("missing state".into()),
    };
    let questions_body = body
        .get("questions")
        .and_then(|q| q.as_object())
        .cloned()
        .unwrap_or_default();

    let mut ids: Vec<String> = questions_body.keys().cloned().collect();
    ids.sort();
    let questions: Vec<QuestionInput> = ids
        .iter()
        .filter_map(|id| question_from_jev(&questions_body[id]))
        .collect();
    if questions.is_empty() {
        return Err("no valid questions (need type in choice|score|noul)".into());
    }
    if questions.iter().any(|q| q.options.len() < 2) {
        return Err("choice/score questions need >= 2 options in criteria".into());
    }
    Ok(JevRequest { state, ids, questions })
}

/// Answer in the official response shape. `answer_confidence` is
/// p(reported answer) - the calibration column - NOT the entropy-derived
/// `confidence`, which moves with option count.
pub fn answer_to_json(q: &QuestionInput, a: &Answer) -> Value {
    match a {
        Answer::Choice { index, probabilities, confidence, act_probability } => serde_json::json!({
            "type": "choice",
            "choice": q.keys.get(*index).or_else(|| q.options.get(*index)).cloned().unwrap_or_default(),
            "probabilities": q.options.iter().zip(probabilities.iter())
                .map(|(o, p)| (o.clone(), serde_json::json!(p)))
                .collect::<serde_json::Map<String, Value>>(),
            "answer_confidence": probabilities.get(*index).copied().unwrap_or(*confidence),
            "confidence": confidence,
            "action": {"act_probability": act_probability},
        }),
        Answer::Score { score, probabilities, confidence, act_probability } => serde_json::json!({
            "type": "score",
            "score": score,
            "probabilities": probabilities,
            "answer_confidence": probabilities
                .iter()
                .cloned()
                .fold(f32::MIN, f32::max),
            "confidence": confidence,
            "action": {"act_probability": act_probability},
        }),
        Answer::Noul { prob, act_probability } => serde_json::json!({
            "type": "noul",
            "noul": prob,
            "answer_confidence": prob.max(1.0 - prob),
            "action": {"act_probability": act_probability},
        }),
    }
}

/// Assemble the keyed answers object from aligned (ids, questions, answers).
pub fn answers_json(
    ids: &[String],
    questions: &[QuestionInput],
    answers: &[Answer],
) -> serde_json::Map<String, Value> {
    let mut map = serde_json::Map::new();
    for ((id, q), a) in ids.iter().zip(questions.iter()).zip(answers.iter()) {
        map.insert(id.clone(), answer_to_json(q, a));
    }
    map
}

#[cfg(test)]
mod jev_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_sorts_ids_and_keeps_alignment() {
        let body = json!({
            "state": "s",
            "questions": {
                "z": {"type": "choice", "instructions": "i", "criteria": {"a": "1", "b": "2"}},
                "a": {"type": "noul", "instructions": "j", "criteria": null},
            }
        });
        let req = parse_request(&body).expect("valid");
        assert_eq!(req.ids, vec!["a", "z"]);
        assert_eq!(req.questions.len(), 2);
        assert_eq!(req.questions[1].keys, vec!["a", "b"]);
    }

    #[test]
    fn parse_rejects_missing_state_and_short_options() {
        assert!(parse_request(&json!({"questions": {}})).is_err());
        let short = json!({
            "state": "s",
            "questions": {"q": {"type": "choice", "instructions": "i", "criteria": {"only": "one"}}}
        });
        let err = parse_request(&short).unwrap_err();
        assert!(err.contains(">= 2 options"), "{err}");
    }

    #[test]
    fn answers_json_labels_by_id_with_key_choice() {
        let ids = vec!["q0".to_string()];
        let qs = vec![question_from_jev(&json!({
            "type": "choice", "instructions": "i", "criteria": {"weather": "查天气", "news": "看新闻"}
        })).unwrap()];
        let answers = vec![Answer::Choice {
            index: 0,
            probabilities: vec![0.9, 0.1],
            confidence: 0.9,
            act_probability: 1.0,
        }];
        let map = answers_json(&ids, &qs, &answers);
        let a = &map["q0"];
        assert_eq!(a["choice"], "weather"); // the KEY, not "weather: 查天气"
        let conf = a["answer_confidence"].as_f64().unwrap();
        assert!((conf - 0.9).abs() < 1e-6);
        assert_eq!(a["action"]["act_probability"], 1.0);
    }
}
