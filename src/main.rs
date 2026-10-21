//! Demo: run the real Laya checkpoint on typed questions.
//!
//! With no CLI argument and checkpoint directories present, requests are
//! routed per state (see src/router.rs); LAYA_MODEL=english|typed-decisions|
//! auto (default auto) forces a route. An explicit model directory argument,
//! or no routing directories at all, keeps the original single-model path.

// Declared here (not in lib.rs) so the router ships with the binary without
// touching files outside this task's write scope; move it to lib.rs as
// `pub mod router;` when convenient.

use laya_burn::inference::{QuestionInput, RlAgent};
use laya_burn::router::Router;

use burn::tensor::Device;

fn pick_device() -> Device {
    match std::env::var("LAYA_DEVICE").as_deref() {
        Ok("metal") => Device::metal(burn::tensor::DeviceKind::DefaultDevice),
        Ok("wgpu") => Device::wgpu(burn::tensor::DeviceKind::DefaultDevice),
        _ => Device::default(), // flex (pure Rust CPU, AMX-accelerated via feature)
    }
}

fn demo_questions() -> Vec<QuestionInput> {
    vec![
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
                "false: no, the statement does not hold".to_string(),
                "true: yes, the statement holds".to_string(),
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
    ]
}

fn demo_state() -> serde_json::Value {
    // LAYA_STATE=zh|ru swaps in a non-Latin state to exercise the
    // multilingual stub; LAYA_LONG repeats the state to ~1000 tokens to
    // benchmark long sequences.
    let mut state_text = match std::env::var("LAYA_STATE").as_deref() {
        Ok("zh") => "用户决定是否在 Rust 中运行一个 421M 参数的非自回归决策模型。事实：Burn 是一个 Rust 张量框架；已有可用的检查点。".to_string(),
        Ok("ru") => "Пользователь решает, запускать ли нерегрессивную модель решений на 421M параметров в Rust. Факты: Burn — тензорный фреймворк на Rust; чекпоинт доступен.".to_string(),
        _ => "The user is deciding whether to run a 421M non-autoregressive decision model in Rust. Facts: Burn is a Rust tensor framework; a checkpoint is available.".to_string(),
    };
    if std::env::var("LAYA_LONG").is_ok() {
        state_text = state_text.repeat(20);
    }
    serde_json::Value::String(state_text)
}

fn main() {
    let arg_dir = std::env::args().nth(1);
    let device = pick_device();

    if arg_dir.is_none() {
        let router = Router::new(&device);
        if router.available() {
            run_routed(&router, &device);
            return;
        }
        println!("laya: no routed checkpoint directories found (model/, model-english/); falling back to the single-model path");
    }
    run_single(&arg_dir.unwrap_or_else(|| "model".to_string()), &device);
}

/// Routed path: pick a checkpoint per request and report the chosen route.
fn run_routed(router: &Router, _device: &Device) {
    let questions = demo_questions();
    let state = demo_state();

    // LAYA_BATCH=N repeats the question set N times in one forward.
    let reps: usize = std::env::var("LAYA_BATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let big: Vec<_> = questions.iter().cycle().take(questions.len() * reps).cloned().collect();

    let (route, det) = router.route(&state);
    println!(
        "routing: script={} language={} non_latin_fraction={:.2} -> checkpoint {}",
        det.script,
        det.language.unwrap_or("undecided"),
        det.non_latin_fraction,
        route.name()
    );

    let t1 = std::time::Instant::now();
    let answers = match router.predict(&state, &big) {
        Ok((route, answers)) => {
            println!("route used: {} ({} questions)", route.name(), big.len());
            answers
        }
        Err(laya_burn::router::RouterError::MultilingualUnimplemented(msg)) => {
            eprintln!("route used: multilingual (stub)");
            eprintln!("laya: {msg}");
            std::process::exit(3);
        }
        Err(laya_burn::router::RouterError::Load(msg)) => {
            eprintln!("laya: checkpoint load failed: {msg}");
            std::process::exit(2);
        }
    };
    println!("inference ({} questions): {:.2?}", big.len(), t1.elapsed());
    for (i, answer) in answers.iter().enumerate() {
        println!("Q{i}: {answer:?}");
    }
}

/// Original single-model path, unchanged in output shape.
fn run_single(model_dir: &str, device: &Device) {
    println!("loading laya from {model_dir} ...");
    let t0 = std::time::Instant::now();
    let agent = RlAgent::load(model_dir, device);
    println!("model loaded in {:.2?}", t0.elapsed());

    let questions = demo_questions();
    let state = demo_state();

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
    let mut engine = replay_on.then(|| agent.replay_engine(device));
    for run in 1..=replay_n {
        let t1 = std::time::Instant::now();
        let answers = agent.system_one(device, &state, &big, engine.as_mut());
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
