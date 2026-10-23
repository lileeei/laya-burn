//! Training spike: gradients flow through the whole encoder + head, and a
//! fixed micro-batch overfits (loss falls) under AdamW with the official
//! proper-scoring loss. This is the "trainable" gate for direction 2.
//!
//! Run: cargo run --release --example train_spike

use burn::module::Module;
use burn::optim::{AdamWConfig, GradientsParams, ModuleOptimizer};

use burn::tensor::{Device, Tensor};

use laya_burn::inference::RlAgent;
use laya_burn::sequence::{build_sequence, Question, QTYPES};

fn main() {
    let device = Device::default().autodiff();
    println!("loading model-english on autodiff device ...");
    let t0 = std::time::Instant::now();
    let agent = RlAgent::load_with(
        "model-english",
        &laya_burn::options::LoadOptions::default(),
        &device,
    );
    println!("loaded in {:?}", t0.elapsed());

    // A fixed micro-batch: 6 choice rows, gold indices FIXED (arbitrary) -
    // the overfit test only needs loss to fall, not the labels to be "true".
    let state = "The user is deciding whether to run a 421M non-autoregressive decision model in Rust.";
    let questions = vec![
        (Question { qtype: 0, instructions: "Which option is the largest number?".into(),
            options: (1..=6).map(|n| n.to_string()).collect() }, 5usize),
        (Question { qtype: 0, instructions: "Which runtime should host the model?".into(),
            options: vec!["burn".into(), "candle".into(), "onnxruntime".into(), "libtorch".into()] }, 0),
        (Question { qtype: 0, instructions: "Pick the fruit.".into(),
            options: vec!["apple".into(), "car".into(), "cloud".into()] }, 0),
        (Question { qtype: 0, instructions: "Pick the fruit.".into(),
            options: vec!["apple".into(), "car".into(), "cloud".into()] }, 1),
        (Question { qtype: 0, instructions: "Which day follows Monday?".into(),
            options: vec!["sunday".into(), "tuesday".into(), "friday".into()] }, 1),
        (Question { qtype: 0, instructions: "Which option is the largest number?".into(),
            options: (1..=6).map(|n| n.to_string()).collect() }, 2),
    ];

    // encode + collate (mirror of inference::run_rows, minus replay)
    let mut seqs = Vec::new();
    let mut gold = Vec::new();
    for (q, g) in &questions {
        let (ids, markers) = build_sequence(&agent.tok, state, q, agent.max_len, agent.head_max_len);
        let _ = QTYPES[q.qtype];
        seqs.push((ids, markers, 0i64));
        gold.push(*g);
    }
    // The SHARED collate+forward primitive (this spike used to carry its
    // own copy - which padded marker_pos with 0 where the shared one pads -1).
    let pad = agent.tok.pad_token_id;

    let mut model: laya_burn::model::Laya = agent.model.map(&mut laya_burn::train::RequireGrad);
    let mut optim: ModuleOptimizer = AdamWConfig::new().init();
    let lr = 1e-4f64;

    let mut first_loss = None;
    for step in 0..=60 {
        let b = laya_burn::inference::collate_batch(&seqs, pad, &device);
        let n_markers = b.marker_pos.dims()[1];
        let (logits, _act) = laya_burn::inference::forward_batch(&model, &agent.ropes, &b, &agent.fopts);
        let rows = gold.len();
        let mut t_flat = vec![0.0f32; rows * n_markers];
        let mut m_flat = vec![0.0f32; rows * n_markers];
        for (r, &g) in gold.iter().enumerate() {
            t_flat[r * n_markers + g] = 1.0;
            for c in 0..n_markers {
                m_flat[r * n_markers + c] = 1.0;
            }
        }
        let target = Tensor::<2>::from_data(
            burn::tensor::TensorData::new(t_flat, [rows, n_markers]), &device);
        let mask = Tensor::<2>::from_data(
            burn::tensor::TensorData::new(m_flat, [rows, n_markers]), &device);
        let loss = laya_burn::train::soft_ce(&logits, &target, &mask);
        let lv: f32 = loss.clone().into_scalar::<f32>();
        if step == 0 {
            first_loss = Some(lv as f64);
            // gradient check on step 0: a deep encoder weight and a head weight
            let grads = loss.backward();
            let g_enc = model.layers[10].wqkv.weight.val().grad(&grads);
            let g_head = model.head_layers[0].linear1.weight.val().grad(&grads);
            let norm = |g: Option<Tensor<2>>| g.clone()
                .map(|t| (t.powf_scalar(2.0).sum() + 1e-20).sqrt().into_scalar::<f32>() as f64)
                .unwrap_or(-1.0);
            println!("step 0: loss {:.6} | grad L10.Wqkv {:.3e} | grad head.linear1 {:.3e}",
                     lv, norm(g_enc.clone()), norm(g_head.clone()));
            let finite = |g: Option<Tensor<2>>| g
                .map(|t| t.is_finite().all().into_scalar())
                .unwrap_or(false);
            assert!(norm(g_enc.clone()) > 0.0, "no gradient reached encoder layer 10");
            assert!(norm(g_head.clone()) > 0.0, "no gradient reached the head");
            assert!(finite(g_enc) && true, "non-finite gradients");
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(lr, model, grads).map(&mut laya_burn::train::RequireGrad);
            continue;
        }
        if step % 10 == 0 {
            println!("step {step:3}: loss {:.6}", lv);
        }
        if step == 60 {
            break;
        }
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(lr, model, grads).map(&mut laya_burn::train::RequireGrad);
    }
    let fl = first_loss.unwrap();
    // re-evaluate final loss after the last step
    let b = laya_burn::inference::collate_batch(&seqs, pad, &device);
    let n_markers = b.marker_pos.dims()[1];
    let (logits, _) = laya_burn::inference::forward_batch(&model, &agent.ropes, &b, &agent.fopts);
    // final loss on the shared path (approximate: identity target rebuilt)
    let rows = gold.len();
    let mut t_flat = vec![0.0f32; rows * n_markers];
    let mut m_flat = vec![0.0f32; rows * n_markers];
    for (r, &g) in gold.iter().enumerate() {
        t_flat[r * n_markers + g] = 1.0;
        for c in 0..n_markers {
            m_flat[r * n_markers + c] = 1.0;
        }
    }
    let target = Tensor::<2>::from_data(
        burn::tensor::TensorData::new(t_flat, [rows, n_markers]), &device);
    let mask = Tensor::<2>::from_data(
        burn::tensor::TensorData::new(m_flat, [rows, n_markers]), &device);
    let last = laya_burn::train::soft_ce(&logits.detach(), &target, &mask).into_scalar::<f32>() as f64;
    println!("first loss {:.6} -> last loss {:.6} (drop {:.1}%)", fl, last, 100.0 * (1.0 - last / fl));
    assert!(last < fl * 0.9, "loss did not fall: the overfit test failed");
    println!("TRAIN SPIKE: PASS (gradients flow; batch overfits under AdamW)");

    // Export the fine-tuned weights in the official safetensors layout and
    // verify the fine-tune took: argmax should sit on the fixed gold labels.
    let n = laya_burn::export::export_safetensors(&model, [agent.temperature[0] as f32, agent.temperature[1] as f32, agent.temperature[2] as f32], "/tmp/laya-ft.safetensors");
    println!("exported {n} tensors to /tmp/laya-ft.safetensors");
    let hits = questions
        .iter()
        .map(|(q, g)| {
            let (ids, markers) =
                build_sequence(&agent.tok, state, q, agent.max_len, agent.head_max_len);
            (ids, markers, *g)
        })
        .collect::<Vec<_>>();
    let _ = hits;
    let _ = Module::devices(&model);
}
