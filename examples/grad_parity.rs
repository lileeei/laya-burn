//! RLCD gradient-parity harness: load the fixed batch written by
//! /tmp/grad_parity/gen_and_ref.py, run the Rust rlcd_core on it, and dump
//! loss + d(loss)/d(logits) for comparison with the PyTorch reference.
fn main() {
    use burn::tensor::{Device, Tensor, TensorData};
    // Default: the committed fixture (regenerate with scripts/gen_grad_parity.py;
    // pass a dir to compare against a fresh generation without touching it).
    let dir = std::env::args().nth(1).unwrap_or_else(|| "tests/fixtures/grad_parity".into());
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/meta.json")).unwrap()).unwrap();
    let rows = meta["rows"].as_u64().unwrap() as usize;
    let cols = meta["cols"].as_u64().unwrap() as usize;
    let samples = meta["samples"].as_u64().unwrap() as usize;
    let sigma = meta["sigma"].as_f64().unwrap() as f32;
    let w_sph = meta["w_sph"].as_f64().unwrap() as f32;
    let w_rps = meta["w_rps"].as_f64().unwrap() as f32;
    let qtype: Vec<usize> = meta["qtype"].as_array().unwrap().iter()
        .map(|v| v.as_u64().unwrap() as usize).collect();

    let read2 = |name: &str, r: usize, c: usize| -> Vec<f32> {
        let bytes = std::fs::read(format!("{dir}/{name}.f32")).unwrap();
        bytes.chunks(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).take(r * c).collect()
    };
    let device = Device::default().autodiff();
    let logits_v = read2("logits", rows, cols);
    let target_v = read2("target", rows, cols);
    let mask_v = read2("mask", rows, cols);
    let eps_v = read2("eps", rows * samples, cols);

    let logits = Tensor::<2>::from_data(
        TensorData::new(logits_v.clone(), [rows, cols]), &device).autodiff().require_grad();
    let target_view = target_v.clone();
    let mask_view = mask_v.clone();
    let target = Tensor::<2>::from_data(TensorData::new(target_v, [rows, cols]), &device);
    let mask = Tensor::<2>::from_data(TensorData::new(mask_v, [rows, cols]), &device);
    let eps = Tensor::<2>::from_data(TensorData::new(eps_v, [rows * samples, cols]), &device);

    let loss = if std::env::var("LAYA_PARITY_CE").is_ok() {
        // CE-only isolation: same as rlcd_core's early return
        let z = logits.clone();
        let add = (mask.clone() - 1.0).mul_scalar(1.0e9);
        let logp = burn::tensor::activation::log_softmax(z + add, 1);
        (target.clone() * logp * mask.clone()).sum_dim(1).mean().mul_scalar(-1.0)
    } else {
        laya_burn::train::rlcd_core(
            &logits, &target, &mask, &qtype, &eps, sigma, samples, w_sph, w_rps)
    };
    let lv = loss.clone().into_scalar::<f32>();
    let grads = loss.backward();
    let grad = logits.grad(&grads).expect("grad");
    let gv: Vec<f32> = grad.into_data().as_slice::<f32>().unwrap().to_vec();
    std::fs::write(format!("{dir}/rust_loss.txt"), format!("{lv}")).unwrap();
    let bytes: Vec<u8> = gv.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(format!("{dir}/rust_grad.f32"), bytes).unwrap();
    // reward-level comparison (isolates the advantage path)
    let z_h = logits.detach().into_data();
    let zv: Vec<f32> = z_h.as_slice::<f32>().unwrap().to_vec();
    let eps_h = eps.detach().into_data();
    let ev: Vec<f32> = eps_h.as_slice::<f32>().unwrap().to_vec();
    let mut rewards = Vec::new();
    for r in 0..rows {
        for s_idx in 0..samples {
            let base = (r * samples + s_idx) * cols;
            let z_s: Vec<f32> = (0..cols).map(|c| zv[r * cols + c] + ev[base + c]).collect();
            rewards.push(laya_burn::train::proper_reward_host(
                &z_s, &target_view[r * cols..(r + 1) * cols],
                &mask_view[r * cols..(r + 1) * cols], qtype[r], w_sph, w_rps));
        }
    }
    let bytes: Vec<u8> = rewards.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(format!("{dir}/rust_rewards.f32"), bytes).unwrap(); // gitignored output
    println!("rust loss: {lv} (fixture {dir})");
}
