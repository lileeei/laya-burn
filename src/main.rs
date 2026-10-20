//! Demo: run the real Laya checkpoint on typed questions.

mod calibration;
mod inference;
mod model;
mod sequence;
mod python_json;
mod tokenizer;
mod weights;

use burn::tensor::Device;
use inference::{QuestionInput, RlAgent};

fn pick_device() -> Device {
    match std::env::var("LAYA_DEVICE").as_deref() {
        Ok("metal") => Device::metal(burn::tensor::DeviceKind::DefaultDevice),
        Ok("wgpu") => Device::wgpu(burn::tensor::DeviceKind::DefaultDevice),
        _ => Device::default(), // flex (pure Rust CPU, AMX-accelerated via feature)
    }
}

fn main() {
    let model_dir = std::env::args().nth(1).unwrap_or_else(|| "model".to_string());
    println!("loading laya from {model_dir} ...");
    let t0 = std::time::Instant::now();
    let device = pick_device();
    let agent = RlAgent::load(&model_dir, &device);
    println!("model loaded in {:.2?}", t0.elapsed());

    // LAYA_LONG=1 repeats the state to ~1000 tokens to benchmark long sequences.
    let mut state_text = "The user is deciding whether to run a 421M non-autoregressive decision model in Rust. Facts: Burn is a Rust tensor framework; a checkpoint is available.".to_string();
    if std::env::var("LAYA_LONG").is_ok() {
        state_text = state_text.repeat(20);
    }
    let state = serde_json::Value::String(state_text);

    let questions = vec![
        // choice with 12 options -> temperature bucket "choice:11+" = 0.1 (sharp):
        // a correct port must show a decisive distribution.
        QuestionInput {
            qtype: 0,
            instructions: "Which option is the largest number?".to_string(),
            options: (1..=12).map(|n| n.to_string()).collect(),
        },
        // obvious noul statement.
        QuestionInput {
            qtype: 2,
            instructions: "The word 'apple' contains the letter 'a'.".to_string(),
            options: vec![
                "false: no, the statement does not hold".to_string(),
                "true: yes, the statement holds".to_string(),
            ],
        },
        QuestionInput {
            qtype: 0, // choice
            instructions: "Which runtime should host the model for CPU inference in Rust?".to_string(),
            options: vec![
                "burn".into(),
                "candle".into(),
                "onnxruntime".into(),
                "libtorch".into(),
            ],
        },
        QuestionInput {
            qtype: 2, // noul
            instructions: "The checkpoint runs natively in Rust without any Python dependency.".to_string(),
            options: vec![
                "false: no, the statement does not hold".into(),
                "true: yes, the statement holds".into(),
            ],
        },
        QuestionInput {
            qtype: 1, // score
            instructions: "Rate the usefulness of a zero-token decision model for routing.".to_string(),
            options: vec![
                "level 0: useless".into(),
                "level 1: marginal".into(),
                "level 2: useful".into(),
                "level 3: very useful".into(),
                "level 4: essential".into(),
            ],
        },
    ];

    let t1 = std::time::Instant::now();
    let answers = agent.system_one(&device, &state, &questions);
    println!("inference ({} questions): {:.2?}", questions.len(), t1.elapsed());
    for (i, answer) in answers.iter().enumerate() {
        println!("Q{i}: {answer:?}");
    }
}

#[cfg(test)]
mod pyjson_tests {
    use crate::python_json::PythonJson;
    use serde_json::json;

    #[test]
    fn state_matches_expected_python_dumps() {
        let v = json!({"b": 1, "a": [2.5, null, "x"]});
        // json.dumps(..., ensure_ascii=False, separators=(", ", ": ")) with insertion order
        assert_eq!(PythonJson::state(&v), "{\"b\": 1, \"a\": [2.5, null, \"x\"]}");
        assert_eq!(PythonJson::state(&json!("plain")), "plain");
    }
}
