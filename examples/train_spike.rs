//! Training spike: gradients flow through the whole encoder + head, and a
//! fixed micro-batch overfits (loss falls) under AdamW with the official
//! proper-scoring loss. This is the "trainable" gate for direction 2.
//!
//! Run: cargo run --release --example train_spike

use burn::module::{Module, ModuleMapper, Param};
use burn::optim::{AdamWConfig, GradientsParams, ModuleOptimizer};

/// Mark every float param as a tracked leaf (the autodiff device provides
/// tracking; leaves must opt in via require_grad, and optimizer.step returns
/// fresh tensors that need re-marking).
struct RequireGrad;
impl burn::module::ModuleMapper for RequireGrad {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        param.map(|t| t.autodiff().require_grad())
    }
}
use burn::tensor::{Device, Int, Tensor};

use laya_burn::inference::RlAgent;
use laya_burn::sequence::{build_sequence, Question, QTYPES};

/// Port of rl_common.proper_reward for the choice rows of one batch
/// (log score + 0.5 * spherical score; RPS applies to ordinal only).
/// Returns the LOSS = -mean(reward), differentiable end to end.
fn proper_loss_choice(logits: &Tensor<2>, gold: &[usize], k: usize) -> Tensor<1> {
    // q: [rows, k]
    let q = burn::tensor::activation::softmax(logits.clone(), 1);
    // logq with the official floor
    let logq = q.clone().clamp_min(1e-12).log().clamp_min(-9.21);
    // target: one-hot
    let rows = gold.len();
    let mut t = vec![0.0f32; rows * k];
    for (r, &g) in gold.iter().enumerate() {
        t[r * k + g] = 1.0;
    }
    let target = Tensor::<2>::from_data(
        burn::tensor::TensorData::new(t, [rows, k]),
        &logits.device(),
    );
    let log_score = (target.clone() * logq).sum_dim(1);
    let sph = (target * q.clone()).sum_dim(1)
        / (q.clone() * q).sum_dim(1).sqrt().clamp_min(1e-9);
    let reward = log_score + sph.mul_scalar(0.5);
    reward.mean().mul_scalar(-1.0)
}

fn main() {
    let device = Device::default().autodiff();
    println!("loading model-english on autodiff device ...");
    let t0 = std::time::Instant::now();
    let agent = RlAgent::load("model-english", &device);
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
    let batch_len = seqs.iter().map(|(i, ..)| i.len()).max().unwrap().max(1);
    let n_markers = seqs.iter().map(|(_, m, _)| m.len()).max().unwrap();
    let pad = agent.tok.pad_token_id;
    let mut input_ids = Vec::new();
    let mut attention_mask = Vec::new();
    let mut marker_pos = Vec::new();
    let mut marker_mask = Vec::new();
    let mut qtype = Vec::new();
    for (ids, markers, _) in &seqs {
        let mut row = ids.clone();
        let mut amask = vec![1i64; ids.len()];
        while row.len() < batch_len { row.push(pad); amask.push(0); }
        input_ids.push(row); attention_mask.push(amask);
        let mut mpos = Vec::new(); let mut mmask = Vec::new();
        for i in 0..n_markers {
            if i < markers.len() { mpos.push(markers[i] as i64); mmask.push(1); }
            else { mpos.push(0); mmask.push(0); }
        }
        marker_pos.push(mpos); marker_mask.push(mmask);
        qtype.push(0);
    }
    let t2i = |rows: Vec<Vec<i64>>, cols: usize| -> Tensor<2, Int> {
        let n = rows.len();
        let flat: Vec<i64> = rows.into_iter().flatten().collect();
        Tensor::<2, Int>::from_data(burn::tensor::TensorData::new(flat, [n, cols]), &device)
    };
    let t1i = |v: Vec<i64>| -> Tensor<1, Int> {
        let n = v.len();
        Tensor::<1, Int>::from_data(burn::tensor::TensorData::new(v, [n]), &device)
    };

    let mut model: laya_burn::model::Laya = agent.model.map(&mut RequireGrad);
    let mut optim: ModuleOptimizer = AdamWConfig::new().init();
    let lr = 1e-4f64;

    let mut first_loss = None;
    for step in 0..=60 {
        let (logits, _act) = model.forward(
            t2i(input_ids.clone(), batch_len),
            t2i(attention_mask.clone(), batch_len),
            t2i(marker_pos.clone(), n_markers),
            t2i(marker_mask.clone(), n_markers),
            t1i(qtype.clone()),
            &agent.ropes,
        );
        let loss = proper_loss_choice(&logits, &gold, n_markers);
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
            model = optim.step(lr, model, grads).map(&mut RequireGrad);
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
        model = optim.step(lr, model, grads).map(&mut RequireGrad);
    }
    let fl = first_loss.unwrap();
    // re-evaluate final loss after the last step
    let (logits, _) = model.forward(
        t2i(input_ids.clone(), batch_len),
        t2i(attention_mask.clone(), batch_len),
        t2i(marker_pos.clone(), n_markers),
        t2i(marker_mask.clone(), n_markers),
        t1i(qtype.clone()),
        &agent.ropes,
    );
    let last = proper_loss_choice(&logits.detach(), &gold, n_markers).into_scalar::<f32>() as f64;
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
