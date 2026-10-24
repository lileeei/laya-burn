//! The typed decision surface end to end: builder -> one forward ->
//! typed accessors -> gate. Answers are cross-checked against the same
//! questions asked through the raw QuestionInput API.
fn main() {
    use burn::tensor::Device;
    use laya_burn::ask::Gate;
    use laya_burn::inference::{QuestionInput, RlAgent};
    use serde_json::json;

    let device = Device::default();
    let opts = laya_burn::options::LoadOptions::default();
    let agent = RlAgent::load_with("model-english", &opts, &device);

    let state = json!("A customer writes in about being charged twice for March and asks for a refund.");
    let out = agent
        .ask(&device, &state)
        .choice(
            "department",
            &["billing", "support", "sales"],
        )
        .instructions("department", "Which team should reply?")
        .noul("refund_requested")
        .score("urgency", 0..=3)
        .run()
        .expect("run");

    println!("department:      {}", out.choice("department"));
    println!("refund_requested: {}", out.noul("refund_requested"));
    println!("urgency:          {}", out.score("urgency"));
    println!("confidences:      {:?}", out.ids().map(|id| (id, out.answer_confidence(id))).collect::<Vec<_>>());

    let mut out = out;
    let gate = Gate::new()
        .min("choice:3-5", 0.9) // high bar for the 3-option choice
        .min("noul:2", 0.5)
        .min("score:3-5", 0.5);
    gate.apply(&mut out);
    println!("gates: {:?}", out.gates());

    // Cross-check against the raw API: same questions through
    // system_one must give the same answers.
    let raw = agent.system_one(
        &device,
        &state,
        &[
            QuestionInput {
                qtype: 0,
                instructions: "Which team should reply?".into(),
                options: vec!["billing".into(), "support".into(), "sales".into()],
                keys: vec!["billing".into(), "support".into(), "sales".into()],
                option_order: None,
            },
            QuestionInput {
                qtype: 2,
                instructions: "refund_requested".into(),
                options: vec![
                    "false: no, the statement does not hold".into(),
                    "true: yes, the statement holds".into(),
                ],
                keys: vec!["false".into(), "true".into()],
                option_order: None,
            },
            QuestionInput {
                qtype: 1,
                instructions: "urgency".into(),
                options: vec!["level 0".into(), "level 1".into(), "level 2".into(), "level 3".into()],
                keys: vec!["0".into(), "1".into(), "2".into(), "3".into()],
                option_order: None,
            },
        ],
        None,
    );
    use laya_burn::inference::Answer;
    let raw_choice = match &raw[0] {
        Answer::Choice { index, .. } => ["billing", "support", "sales"][*index],
        _ => unreachable!(),
    };
    assert_eq!(raw_choice, out.choice("department"), "typed vs raw: choice mismatch");
    println!("CROSS-CHECK: typed API == raw API ({raw_choice})");
}
