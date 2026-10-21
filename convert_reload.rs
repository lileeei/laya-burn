fn convert_to_burnpack(dir: &str, device: &burn::tensor::Device) {
    use burn::module::Module;
    use laya_burn::inference::{QuestionInput, RlAgent};
    let t0 = std::time::Instant::now();
    let agent = RlAgent::load(dir, device);
    println!("safetensors load: {:?}", t0.elapsed());
    let out = dir.to_string() + "/model.mpk";
    let t1 = std::time::Instant::now();
    agent.model.clone().save_file(&out).expect("save burnpack");
    println!("saved {}: {:?}", out, t1.elapsed());

    // Reload: rebuild the module tree, then overwrite every parameter from
    // the burnpack. The structural pass reuses the safetensors path; a
    // zeros-only skeleton would make reload independent of the original
    // checkpoint file (follow-up).
    let t2 = std::time::Instant::now();
    let mut agent2 = RlAgent::load(dir, device);
    agent2.model = agent2.model.load_file(&out);
    println!("burnpack reload: {:?}", t2.elapsed());

    // Round-trip proof: the same question through both agents must agree.
    let question = QuestionInput {
        qtype: 0,
        instructions: "Which runtime should host the model for CPU inference in Rust?".to_string(),
        options: vec!["burn".into(), "candle".into(), "onnxruntime".into(), "libtorch".into()],
    };
    let state = serde_json::Value::String(
        "The user is deciding whether to run a 421M non-autoregressive decision model in Rust.".to_string(),
    );
    let a1 = agent.system_one(device, &state, &[question.clone()], None);
    let a2 = agent2.system_one(device, &state, &[question], None);
    println!("round-trip original: {:?}", a1);
    println!("round-trip burnpack: {:?}", a2);
    let same = format!("{a1:?}") == format!("{a2:?}");
    println!("ROUND-TRIP: {}", if same { "PASS" } else { "FAIL" });
}
