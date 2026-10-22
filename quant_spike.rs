/// LAYA_QUANT=1 spike: quantize the loaded model (Q8 MinMax) and run the demo
/// question through both the f32 and quantized copies, comparing answers.
fn quantize_spike(dir: &str, device: &burn::tensor::Device) {
    use burn::module::{Module, Quantizer};
    use burn::tensor::quantization::{Calibration, QuantValue};
    use laya_burn::inference::{QuestionInput, RlAgent};

    let agent = RlAgent::load(dir, device);
    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S)
        // PackedU32 (the default store) requires the trailing dim to be a
        // multiple of 4 (num_quants); Native i8 storage works for any shape.
        .with_store(burn::tensor::quantization::QuantStore::Native);
    let mut quantizer = Quantizer::new(Calibration::MinMax, scheme);
    let t0 = std::time::Instant::now();
    let q_model = agent.model.clone().quantize_weights(&mut quantizer);
    println!("quantize: {:?}", t0.elapsed());

    let question = QuestionInput {
        qtype: 0,
        instructions: "Which runtime should host the model for CPU inference in Rust?".to_string(),
        options: vec!["burn".into(), "candle".into(), "onnxruntime".into(), "libtorch".into()],
    };
    let state = serde_json::Value::String(
        "The user is deciding whether to run a 421M non-autoregressive decision model in Rust.".to_string(),
    );
    let a1 = agent.system_one(device, &state, &[question.clone()], None);
    println!("f32 answers:  {a1:?}");

    let mut q_agent = RlAgent::load(dir, device);
    q_agent.model = q_model;
    let a2 = q_agent.system_one(device, &state, &[question], None);
    println!("q8 answers:   {a2:?}");
    let same = format!("{a1:?}") == format!("{a2:?}");
    println!("QUANT ROUND-TRIP: {}", if same { "PASS" } else { "CHECK" });
}

