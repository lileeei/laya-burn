//! Demo: run the real Laya checkpoint on typed questions.

mod calibration;
mod inference;
mod model;
mod sequence;
mod python_json;
mod tokenizer;
mod weights;
mod window_kernel;

use burn::tensor::Device;
use inference::{QuestionInput, RlAgent};

fn pick_device() -> Device {
    match std::env::var("LAYA_DEVICE").as_deref() {
        Ok("metal") => Device::metal(burn::tensor::DeviceKind::DefaultDevice),
        Ok("wgpu") => Device::wgpu(burn::tensor::DeviceKind::DefaultDevice),
        _ => Device::default(), // flex (pure Rust CPU, AMX-accelerated via feature)
    }
}

fn wa_test() {
    use burn::tensor::{Device, Tensor, activation};
    let device = Device::metal(burn::tensor::DeviceKind::DefaultDevice);
    let (b, h, s, d) = (2usize, 3usize, 200usize, 64usize);
    let q = Tensor::<4>::random([b, h, s, d], burn::tensor::Distribution::Default, &device);
    let k = Tensor::<4>::random([b, h, s, d], burn::tensor::Distribution::Default, &device);
    let v = Tensor::<4>::random([b, h, s, d], burn::tensor::Distribution::Default, &device);
    // Half of each batch row is padding (like real batches with varied lengths).
    let mut pm = vec![1.0f32; b * s];
    for bb in 0..b {
        for j in (s / 2)..s {
            pm[bb * s + j] = 0.0;
        }
    }
    let pad = Tensor::<1>::from_floats(pm.as_slice(), &device).reshape([b, s]);
    let _ = &pad;

    let custom = window_kernel::windowed_attention(q.clone(), k.clone(), v.clone(), pad);

    let scores = q.matmul(k.transpose()) * (1.0f32 / 8.0);
    // Reference must apply the same window mask the kernel implements.
    let mut m = vec![-1.0e9f32; b * s * s];
    for bb in 0..b {
        for i in 0..s {
            for j in 0..s {
                if (i as isize - j as isize).unsigned_abs() <= 64 && pm[bb * s + j] == 1.0 {
                    m[(bb * s + i) * s + j] = 0.0;
                }
            }
        }
    }
    let win = burn::tensor::Tensor::<1>::from_floats(m.as_slice(), &device)
        .reshape([b, 1, s, s]);
    let scores = scores + win;
    let probs = activation::softmax(scores, 3);
    let reference = probs.matmul(v);

    // Compare only non-pad query rows: fully-masked rows are 0/0 in any
    // implementation and carry no signal downstream.
    let valid = {
        let mut vi = vec![1.0f32; b * s];
        for bb in 0..b {
            for j in (s / 2)..s {
                vi[bb * s + j] = 0.0;
            }
        }
        Tensor::<1>::from_floats(vi.as_slice(), &device).reshape([b, 1, s, 1])
    };
    let diff: f32 = ((custom - reference).abs() * valid).max().into_scalar::<f32>().into();
    println!("wa_test: max |custom - reference| over valid rows = {diff}");
}

fn main() {
    if std::env::var("LAYA_WA_TEST").is_ok() {
        wa_test();
        return;
    }
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

    // LAYA_BATCH=N repeats the question set N times in one forward (batch scaling probe).
    let reps: usize = std::env::var("LAYA_BATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let big: Vec<_> = questions.iter().cycle().take(questions.len() * reps).cloned().collect();

    // LAYA_REPLAY=1 enables the per-shape graph capture/replay fast path
    // (ReplayEngine). LAYA_REPLAY_N=k calls system_one k times with the SAME
    // questions so one process exercises refresh+replay for the same shape
    // (and doubles as a benchmark); default 1 keeps today's behavior.
    let replay_on = std::env::var("LAYA_REPLAY").as_deref() == Ok("1");
    let replay_n: usize =
        std::env::var("LAYA_REPLAY_N").ok().and_then(|v| v.parse().ok()).filter(|&n| n > 0).unwrap_or(1);
    let mut engine = replay_on.then(|| agent.replay_engine(&device));
    for run in 1..=replay_n {
        let t1 = std::time::Instant::now();
        let answers = agent.system_one(&device, &state, &big, engine.as_mut());
        let label = if replay_n == 1 {
            format!("inference ({} questions)", big.len())
        } else {
            format!("inference run {run}/{replay_n} ({} questions)", big.len())
        };
        println!("{label}: {:.2?}", t1.elapsed());
        for (i, answer) in answers.iter().enumerate() {
            println!("Q{i}: {answer:?}");
        }
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
