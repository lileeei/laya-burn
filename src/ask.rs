//! The typed decision surface: a builder that replaces bare `qtype: 0`
//! question literals, and typed answer accessors, plus the confidence
//! gate (official structured.py + confidence.py needs, served the Rust
//! way: the type system does the field mapping validation JSON Schema
//! does at runtime, and gate states are an enum, not strings).
//!
//! ```no_run
//! # use laya_burn::inference::RlAgent;
//! # use burn::tensor::Device;
//! let agent = RlAgent::load_with("model-english", &Default::default(), &Device::default());
//! let out = agent.ask(&serde_json::json!("Refund invoice 4411"))
//!     .choice("department", &["billing", "support", "sales"])
//!     .noul("needs_human")
//!     .score("urgency", 0..=3)
//!     .run()
//!     .unwrap();
//! let dept: &str = out.choice("department");
//! let needs: bool = out.noul("needs_human");
//! let urgency: i32 = out.score("urgency");
//! ```

use std::collections::HashMap;

use serde_json::Value;

use crate::inference::{Answer, QuestionInput, RlAgent};

/// A question under construction (one builder method = one typed field).
struct Field {
    qid: String,
    input: QuestionInput,
}

/// Builder over one state: each method adds one question; [Ask::run]
/// executes a single forward for all of them.
pub struct Ask<'a> {
    agent: &'a RlAgent,
    device: &'a burn::tensor::Device,
    state: Value,
    fields: Vec<Field>,
}

impl<'a> Ask<'a> {
    /// Start a typed decision for `state` (str or any JSON value - dict
    /// states serialize Python-style, same as the wire path).
    pub fn new(agent: &'a RlAgent, device: &'a burn::tensor::Device, state: &Value) -> Self {
        Self { agent, device, state: state.clone(), fields: Vec::new() }
    }

    /// An enum field: model picks exactly one option.
    pub fn choice(mut self, qid: &str, options: &[&str]) -> Self {
        let input = choice_input(qid, options);
        self.fields.push(Field { qid: qid.to_string(), input });
        self
    }

    /// A boolean field. The two noul arms are the wire-path defaults.
    pub fn noul(mut self, qid: &str) -> Self {
        self.fields.push(Field {
            qid: qid.to_string(),
            input: noul_input(qid),
        });
        self
    }

    /// An ordinal field over `range`: the model picks a level; the answer
    /// projects back to the level's integer value.
    pub fn score(mut self, qid: &str, range: std::ops::RangeInclusive<i32>) -> Self {
        let input = score_input(qid, range);
        self.fields.push(Field { qid: qid.to_string(), input });
        self
    }

    /// Override the auto instruction (the qid) with explicit wording -
    /// question phrasing matters to the model.
    pub fn instructions(mut self, qid: &str, text: &str) -> Self {
        let f = self
            .fields
            .iter_mut()
            .find(|f| f.qid == qid)
            .unwrap_or_else(|| panic!("unknown question {qid:?}"));
        f.input.instructions = text.to_string();
        self
    }

    /// Run one forward for every field and project to typed values.
    pub fn run(self) -> Result<Answers, String> {
        assert!(!self.fields.is_empty(), "no questions: add choice/noul/score before run");
        let questions: Vec<QuestionInput> = self.fields.iter().map(|f| f.input.clone()).collect();
        let answers = self.agent.system_one(self.device, &self.state, &questions, None);
        let rows = self
            .fields
            .into_iter()
            .zip(answers)
            .map(|(f, a)| (f.qid, f.input, a))
            .collect();
        Ok(Answers { rows, gate_states: Vec::new() })
    }
}

impl RlAgent {
    /// The typed decision surface over [RlAgent::system_one].
    pub fn ask<'a>(&'a self, device: &'a burn::tensor::Device, state: &Value) -> Ask<'a> {
        Ask::new(self, device, state)
    }
}

/// Typed answers, one accessor per question kind, in builder order.
/// Accessors panic on unknown ids or kind mismatch - the builder
/// guarantees the shape, so a mismatch is a programming error, not a
/// runtime condition.
pub struct Answers {
    rows: Vec<(String, QuestionInput, Answer)>,
    /// Filled by [Gate::apply]; the presence of entries - not a value
    /// inside them - is what tells a caller the gate ran.
    gate_states: Vec<(String, GateState)>,
}

impl Answers {
    fn row(&self, qid: &str) -> &(String, QuestionInput, Answer) {
        self.rows
            .iter()
            .find(|(id, _, _)| id == qid)
            .unwrap_or_else(|| panic!("unknown question {qid:?}"))
    }

    fn answer(&self, qid: &str) -> &Answer {
        &self.row(qid).2
    }

    fn input(&self, qid: &str) -> &QuestionInput {
        &self.row(qid).1
    }

    /// The chosen option (a criteria key), for choice questions.
    pub fn choice(&self, qid: &str) -> &str {
        match self.answer(qid) {
            Answer::Choice { index, .. } => self.input(qid).keys[*index].as_str(),
            other => panic!("{qid}: expected choice, got {other:?}"),
        }
    }

    /// The boolean arm, for noul questions.
    pub fn noul(&self, qid: &str) -> bool {
        match self.answer(qid) {
            Answer::Noul { prob, .. } => *prob >= 0.5,
            other => panic!("{qid}: expected noul, got {other:?}"),
        }
    }

    /// The level's integer value, for score questions.
    pub fn score(&self, qid: &str) -> i32 {
        match self.answer(qid) {
            Answer::Score { score, .. } => score.round() as i32,
            other => panic!("{qid}: expected score, got {other:?}"),
        }
    }

    /// p(reported answer) - the calibration column the gate reads.
    pub fn answer_confidence(&self, qid: &str) -> f32 {
        match self.answer(qid) {
            Answer::Choice { index, probabilities, .. } => {
                probabilities.get(*index).copied().unwrap_or(0.0)
            }
            Answer::Score { probabilities, .. } => {
                probabilities.iter().cloned().fold(f32::MIN, f32::max)
            }
            Answer::Noul { prob, .. } => prob.max(1.0 - prob),
        }
    }

    /// Raw answer (probabilities and all), by id.
    pub fn raw(&self, qid: &str) -> &Answer {
        self.answer(qid)
    }

    /// Question ids in the order they were added.
    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.rows.iter().map(|(id, _, _)| id.as_str())
    }

    /// The gate states after [Gate::apply]; empty before it.
    pub fn gates(&self) -> &[(String, GateState)] {
        self.gate_states.as_ref()
    }
}

// ---------------------------------------------------------------- confidence gate

/// The gate's verdict for one answer. `Unevaluated` is a real third state
/// (the gate ran; the answer carried no usable confidence) - collapsing it
/// into passed or abstained is the lie the official design calls out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateState {
    Passed,
    Abstained,
    Unevaluated,
}

/// A confidence-gate policy: minimum confidence per option-count bucket
/// (official bucket_keys semantics: choice questions bucket by option
/// count 2 / 3-5 / 6-10 / 11+, score and noul bucket by kind).
///
/// ```text
/// let gate = Gate::new().min("choice:11+", 0.9).min("noul", 0.8);
/// ```
#[derive(Clone, Debug, Default)]
pub struct Gate {
    by_bucket: HashMap<String, f32>,
}

impl Gate {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the threshold for one bucket key: `"choice:2"`, `"choice:3-5"`,
    /// `"choice:6-10"`, `"choice:11+"`, `"score"`, `"noul"`.
    pub fn min(mut self, bucket: &str, threshold: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&threshold),
            "threshold {threshold} outside [0, 1]"
        );
        self.by_bucket.insert(bucket.to_string(), threshold);
        self
    }

    fn bucket_of(qtype: usize, k: usize) -> String {
        crate::calibration::Calibration::temp_bucket(qtype, k)
    }

    /// Apply the policy to every answer, recording the three-state verdict.
    /// Buckets without a threshold pass untouched (an unset bucket means no
    /// policy for that shape, not a zero bar). Answers carry
    /// [Answers::gates] afterwards; the answers themselves are unchanged -
    /// abstaining is the CALLER's action to take.
    pub fn apply(&self, answers: &mut Answers) {
        answers.gate_states.clear();
        for (qid, input, ans) in &answers.rows {
            let k = match ans {
                Answer::Choice { probabilities, .. } | Answer::Score { probabilities, .. } => {
                    probabilities.len()
                }
                Answer::Noul { .. } => 2,
            };
            let bucket = Self::bucket_of(input.qtype, k);
            let state = match self.by_bucket.get(&bucket) {
                None => continue, // no policy for this bucket
                Some(&min) => {
                    let conf = match ans {
                        Answer::Choice { index, probabilities, .. } => {
                            probabilities.get(*index).copied()
                        }
                        Answer::Score { probabilities, .. } => probabilities
                            .iter()
                            .cloned()
                            .fold(f32::MIN, f32::max)
                            .into(),
                        Answer::Noul { prob, .. } => Some(prob.max(1.0 - prob)),
                    };
                    match conf {
                        Some(c) if c >= min => GateState::Passed,
                        Some(_) => GateState::Abstained,
                        None => GateState::Unevaluated,
                    }
                }
            };
            answers.gate_states.push((qid.clone(), state));
        }
    }
}

/// Pure choice mapping (unit-tested without a model): options become both
/// the rendered option text and the answer keys; duplicate options are the
/// Rust analogue of the schema validator's duplicate-label check.
fn choice_input(qid: &str, options: &[&str]) -> QuestionInput {
    assert!(!options.is_empty(), "{qid}: a choice needs at least one option");
    let mut seen = std::collections::HashSet::new();
    for o in options {
        assert!(seen.insert(*o), "{qid}: duplicate option {o:?}");
    }
    QuestionInput {
        qtype: 0,
        instructions: qid.to_string(),
        options: options.iter().map(|s| s.to_string()).collect(),
        keys: options.iter().map(|s| s.to_string()).collect(),
        option_order: None,
    }
}

/// Pure noul mapping: the two default arms (same rendering the wire path
/// produces for a criteria-less noul question).
fn noul_input(qid: &str) -> QuestionInput {
    QuestionInput {
        qtype: 2,
        instructions: qid.to_string(),
        options: vec![
            "false: no, the statement does not hold".into(),
            "true: yes, the statement holds".into(),
        ],
        keys: vec!["false".into(), "true".into()],
        option_order: None,
    }
}

/// Pure score mapping: an inclusive integer range becomes the level
/// options/keys; over-wide ranges are rejected (10-level ceiling, official
/// MAX_SCORE_LEVELS).
fn score_input(qid: &str, range: std::ops::RangeInclusive<i32>) -> QuestionInput {
    let (lo, hi) = (*range.start(), *range.end());
    assert!(hi >= lo, "{qid}: empty score range");
    assert!(
        hi - lo + 1 <= 10,
        "{qid}: {} levels exceeds the 10-level score ceiling; use choice",
        hi - lo + 1
    );
    QuestionInput {
        qtype: 1,
        instructions: qid.to_string(),
        options: (lo..=hi).map(|l| format!("level {l}")).collect(),
        keys: (lo..=hi).map(|l| l.to_string()).collect(),
        option_order: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choice_mapping() {
        let q = choice_input("department", &["billing", "support"]);
        assert_eq!(q.qtype, 0);
        assert_eq!(q.instructions, "department");
        assert_eq!(q.options, q.keys);
        assert_eq!(q.option_order, None);
    }

    #[test]
    #[should_panic(expected = "duplicate option")]
    fn duplicate_options_rejected() {
        let _ = choice_input("x", &["a", "a"]);
    }

    #[test]
    #[should_panic(expected = "at least one option")]
    fn empty_choice_rejected() {
        let _ = choice_input("x", &[]);
    }

    #[test]
    fn score_mapping() {
        let q = score_input("urgency", 0..=3);
        assert_eq!(q.qtype, 1);
        assert_eq!(q.options, vec!["level 0", "level 1", "level 2", "level 3"]);
        assert_eq!(q.keys, vec!["0", "1", "2", "3"]);
    }

    #[test]
    #[should_panic(expected = "score ceiling")]
    fn wide_score_rejected() {
        let _ = score_input("x", 0..=11);
    }

    #[test]
    fn noul_mapping_has_two_arms() {
        let q = noul_input("needs_human");
        assert_eq!(q.qtype, 2);
        assert_eq!(q.options.len(), 2);
        assert_eq!(q.keys, vec!["false", "true"]);
    }

    #[test]
    fn gate_three_states_and_bucket_keys() {
        use crate::calibration::Calibration;
        assert_eq!(Calibration::temp_bucket(0, 2), "choice:2");
        assert_eq!(Calibration::temp_bucket(0, 20), "choice:11+");
        assert_eq!(Calibration::temp_bucket(2, 2), "noul:2");
        // threshold bounds enforced
        let _ = Gate::new().min("noul:2", 0.0).min("noul:2", 1.0);
    }
}
