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
            keys: vec![],
        },
        // obvious noul statement.
        QuestionInput {
            qtype: 2,
            instructions: "The word 'apple' contains the letter 'a'.".to_string(),
            options: vec![
                "false: no, the statement does not hold".to_string(),
                "true: yes, the statement holds".to_string(),
            ],
            keys: vec![],
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
            keys: vec![],
        },
        QuestionInput {
            qtype: 2, // noul
            instructions: "The checkpoint runs natively in Rust without any Python dependency.".to_string(),
            options: vec![
                "false: no, the statement does not hold".to_string(),
                "true: yes, the statement holds".to_string(),
            ],
            keys: vec![],
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
            keys: vec![],
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

/// LAYA_CONVERT=<dir>: checkpoint -> burn-native burnpack round trip.
use burn::module::Module;

/// LAYA_QUANT=<dir>: quantize the loaded model to Q8 (encoder + embeddings;
/// head/norms exempt), export model.q8.mpk, and reload it through the eager
/// dequant path - proving the 4x artifact loads in seconds with answers
/// identical to the in-memory quantized model.
fn quantize_spike(dir: &str, device: &burn::tensor::Device) {
    use laya_burn::inference::{QuestionInput, RlAgent};

    let t0 = std::time::Instant::now();
    let agent = RlAgent::load(dir, device);
    println!("f32 load: {:?}", t0.elapsed());

    let t1 = std::time::Instant::now();
    let q_model = laya_burn::quant::quantize(agent.model.clone(), device);
    println!("quantize (exempt head/norms): {:?}", t1.elapsed());

    let question = QuestionInput {
        qtype: 0,
        instructions: "Which runtime should host the model for CPU inference in Rust?".to_string(),
        options: vec!["burn".into(), "candle".into(), "onnxruntime".into(), "libtorch".into()],
        keys: vec![],
    };
    let state = serde_json::Value::String(
        "The user is deciding whether to run a 421M non-autoregressive decision model in Rust.".to_string(),
    );
    let a1 = agent.system_one(device, &state, &[question.clone()], None);
    println!("f32 answers:  {a1:?}");

    let mut q_agent = RlAgent::load(dir, device);
    q_agent.model = q_model;
    let a2 = q_agent.system_one(device, &state, &[question.clone()], None);
    println!("q8 answers:   {a2:?}");
    let same = format!("{a1:?}") == format!("{a2:?}");
    println!("Q8 ROUND-TRIP: {}", if same { "PASS" } else { "CHECK" });

    // Export: the Q8 model saved as burnpack (i8 + scales), then reloaded
    // through the skeleton + eager-dequant path. The geometry sidecar is
    // shared with the f32 pack; write one when the directory has none (the
    // ModernBERT-large geometry every Q8-quantized checkpoint here shares).
    let sidecar = dir.to_string() + "/model.mpk.json";
    if !std::path::Path::new(&sidecar).is_file() {
        // Derive the geometry from the loaded model so any checkpoint works.
        let spec = laya_burn::skeleton::SkeletonSpec::from_model(&q_agent.model);
        std::fs::write(&sidecar, serde_json::to_string_pretty(&spec).unwrap())
            .expect("write sidecar");
    }
    let qpath = dir.to_string() + "/model.q8.mpk";
    let t2 = std::time::Instant::now();
    q_agent.model.clone().save_file(&qpath).expect("save quantized burnpack");
    println!("saved {}: {:?}", qpath, t2.elapsed());
    let meta = std::fs::metadata(&qpath).expect("stat");
    let f32_size = std::fs::metadata(dir.to_string() + "/model.mpk").map(|m| m.len()).unwrap_or(0);
    println!(
        "quantized burnpack size: {:.0} MB (f32 pack {:.0} MB, {:.2}x)",
        meta.len() as f64 / 1e6,
        f32_size as f64 / 1e6,
        f32_size as f64 / meta.len() as f64,
    );

    let t3 = std::time::Instant::now();
    let r_agent = RlAgent::load_burnpack_named(dir, "model.q8.mpk", device);
    println!("q8 burnpack reload (eager dequant): {:?}", t3.elapsed());
    let a3 = r_agent.system_one(device, &state, &[question], None);
    println!("q8 burnpack answers: {a3:?}");
    let same2 = format!("{a2:?}") == format!("{a3:?}");
    println!(
        "Q8 BURNPACK ROUND-TRIP (eager dequant == lazy kernel): {}",
        if same2 { "PASS" } else { "CHECK" }
    );
}

fn convert_to_burnpack(dir: &str, device: &burn::tensor::Device) {
    use laya_burn::inference::{QuestionInput, RlAgent};
    let t0 = std::time::Instant::now();
    let agent = RlAgent::load(dir, device);
    println!("safetensors load: {:?}", t0.elapsed());
    let out = dir.to_string() + "/model.mpk";
    let t1 = std::time::Instant::now();
    agent.model.clone().save_file(&out).expect("save burnpack");
    // Sidecar: complete geometry derived from the loaded model (the old
    // hardcoded english values would write a wrong sidecar for any other
    // checkpoint, e.g. model-multilingual's 256k vocab).
    let spec = laya_burn::skeleton::SkeletonSpec::from_model(&agent.model);
    std::fs::write(
        dir.to_string() + "/model.mpk.json",
        serde_json::to_string_pretty(&spec).unwrap(),
    )
    .expect("write sidecar");
    println!("saved {} + sidecar: {:?}", out, t1.elapsed());

    // Reload with NO safetensors access: skeleton from the sidecar, weights
    // from the burnpack (eager record path - Module::load_file's lazy wrapper
    // hangs the first forward).
    let t2 = std::time::Instant::now();
    let agent2 = RlAgent::load_burnpack(dir, device);
    println!("burnpack-only reload: {:?}", t2.elapsed());

    // Round-trip proof: the same question through both agents must agree.
    let question = QuestionInput {
        qtype: 0,
        instructions: "Which runtime should host the model for CPU inference in Rust?".to_string(),
        options: vec!["burn".into(), "candle".into(), "onnxruntime".into(), "libtorch".into()],
        keys: vec![],
    };
    let state = serde_json::Value::String(
        "The user is deciding whether to run a 421M non-autoregressive decision model in Rust.".to_string(),
    );
    let a1 = agent.system_one(device, &state, &[question.clone()], None);
    let a2 = agent2.system_one(device, &state, &[question], None);
    println!("round-trip original:  {a1:?}");
    println!("round-trip burnpack:  {a2:?}");
    let same = format!("{a1:?}") == format!("{a2:?}");
    println!("ROUND-TRIP: {}", if same { "PASS" } else { "FAIL" });
}

fn main() {
    if let Ok(dir) = std::env::var("LAYA_QUANT") {
        quantize_spike(&dir, &pick_device());
        return;
    }
    if let Ok(dir) = std::env::var("LAYA_CONVERT") {
        convert_to_burnpack(&dir, &pick_device());
        return;
    }
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
    // LAYA_LOAD=bytes|q8 exercises the no-filesystem loaders (the wasm
    // runtime path) on native: same bytes, answers must be identical.
    let agent = match std::env::var("LAYA_LOAD").as_deref() {
        Ok("bytes") => {
            let weights = std::fs::read(format!("{model_dir}/model.safetensors")).unwrap();
            let cfg = std::fs::read_to_string(format!("{model_dir}/rl_agent_config.json")).unwrap();
            let tok = std::fs::read_to_string(format!("{model_dir}/tokenizer/tokenizer.json")).unwrap();
            let tok_cfg = std::fs::read_to_string(format!("{model_dir}/tokenizer/tokenizer_config.json")).ok();
            RlAgent::load_bytes(weights, &cfg, &tok, tok_cfg.as_deref(), device)
        }
        Ok("q8") => {
            let pack = std::fs::read(format!("{model_dir}/model.q8.mpk"))
                .expect("model.q8.mpk missing: export with LAYA_QUANT first");
            let sidecar = std::fs::read_to_string(format!("{model_dir}/model.mpk.json")).unwrap();
            let cfg = std::fs::read_to_string(format!("{model_dir}/rl_agent_config.json")).unwrap();
            let tok = std::fs::read_to_string(format!("{model_dir}/tokenizer/tokenizer.json")).unwrap();
            let tok_cfg = std::fs::read_to_string(format!("{model_dir}/tokenizer/tokenizer_config.json")).ok();
            RlAgent::load_q8_bytes(pack, &sidecar, &cfg, &tok, tok_cfg.as_deref(), device)
        }
        _ => RlAgent::load(model_dir, device),
    };
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
