//! Fine-tuning: a port of the official laya.train recipe reduced to one
//! device - soft cross-entropy plus the RLCD policy-gradient objective,
//! sigma annealing, option-order shuffling, a held-out calibration split,
//! and post-training temperature fitting (calibrate.py's fit_one_temperature).
//!
//! `finetune(model_dir, data, out_dir, ..)` produces a COMPLETE checkpoint
//! directory: safetensors weights (loadable by the PyTorch reference with
//! strict=True), rl_agent_config.json with the fitted temperatures, and the
//! tokenizer/encoder config copied through - servable by laya-serve as-is.

use std::collections::HashMap;

use burn::module::{Module, ModuleMapper, Param};
use burn::optim::{AdamWConfig, GradientsParams, ModuleOptimizer};
use burn::tensor::{Device, Tensor};
use serde_json::Value;

use crate::inference::RlAgent;
use crate::model::Laya;
use crate::sequence::{build_sequence_ex, Question, QTYPES};

/// Training knobs (official TrainConfig defaults where they apply to the
/// single-device reduction). `lr` is a single rate - the official splits
/// encoder/head (2.5e-5 / 1e-4); burn's per-group plumbing exists but one
/// rate keeps the first port honest.
#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub epochs: usize,
    pub micro_batch: usize,
    pub lr: f64,
    pub weight_decay: f64,
    pub grad_clip: Option<f64>,
    pub loss: LossKind,
    pub rl_samples: usize,
    pub sigma_start: f32,
    pub sigma_end: f32,
    pub w_sph: f32,
    pub w_rps: f32,
    /// Question types whose option order is shuffled each epoch
    /// (official default: none - position-balanced training is opt-in).
    pub shuffle_types: Vec<String>,
    pub label_smoothing: f32,
    /// Print per-step losses to stderr (set by the CLI's LAYA_TRAIN_LOG).
    pub log: bool,
    pub calib_max: usize,
    pub calib_frac: f32,
    pub calib_seed: u64,
    pub seed: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LossKind {
    SoftCe,
    Rlcd,
    /// Ablation: soft cross-entropy only (the RLCD policy term removed).
    CeOnly,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 4,
            micro_batch: 8,
            lr: 2.5e-5,
            weight_decay: 0.01,
            grad_clip: Some(1.0),
            loss: LossKind::Rlcd,
            rl_samples: 4,
            sigma_start: 0.4,
            sigma_end: 0.1,
            w_sph: 0.75,
            w_rps: 1.0,
            shuffle_types: vec![],
            label_smoothing: 0.0,
            log: false,
            calib_max: 400,
            calib_frac: 0.1,
            calib_seed: 20260922,
            seed: 0,
        }
    }
}

/// Mark every float param a tracked leaf; optimizer.step returns fresh
/// tensors, so re-apply after every step.
pub struct RequireGrad;
impl ModuleMapper for RequireGrad {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        param.map(|t| t.autodiff().require_grad())
    }
}

/// One labelled training row. `canonical` is the identity-order encoding,
/// computed once at load (fit check) and reused by calibration and by any
/// epoch that does not shuffle - only shuffled epochs re-encode.
pub struct TrainRow {
    pub state: String,
    pub q: Question,
    pub target: Vec<f32>,
    pub canonical: (Vec<i64>, Vec<usize>),
}

impl Clone for TrainRow {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            q: Question {
                qtype: self.q.qtype,
                instructions: self.q.instructions.clone(),
                options: self.q.options.clone(),
            },
            target: self.target.clone(),
            canonical: self.canonical.clone(),
        }
    }
}

/// target_from_expected (train.py): the normalised target distribution in
/// option order. choice: one-hot at the key; noul: bool/0/1/string arm;
/// score: level index or number, fractional values split across the two
/// neighbouring levels. Label smoothing spreads (1-eps) one-hot + eps/K.
pub fn target_from_expected(
    q: &Question,
    keys: &[String],
    expected: &Value,
    label_smoothing: f32,
) -> Result<Vec<f32>, String> {
    let k = q.options.len();
    let idx_of = |expected: &Value| -> Option<usize> {
        if let Value::String(s) = expected {
            if let Some(pos) = keys.iter().position(|key| key == s) {
                return Some(pos);
            }
            if let Some(pos) = q.options.iter().position(|o| o == s) {
                return Some(pos);
            }
        }
        None
    };
    let mut target = match q.qtype {
        0 => {
            let idx = idx_of(expected)
                .ok_or_else(|| format!("expected answer {expected} not in choice options {keys:?}"))?;
            let mut t = vec![0.0f32; k];
            t[idx] = 1.0;
            t
        }
        2 => {
            let idx = match expected {
                Value::Bool(b) => *b as usize,
                Value::Number(n) => match n.as_i64() {
                    Some(0) => 0,
                    Some(1) => 1,
                    _ => return Err(format!("numeric noul expected must be 0 or 1: {expected}")),
                },
                Value::String(s) => match s.trim().to_ascii_lowercase().as_str() {
                    "true" | "1" | "yes" => 1,
                    "false" | "0" | "no" => 0,
                    _ => return Err(format!("cannot parse noul boolean from {expected}")),
                },
                _ => return Err(format!("cannot parse noul boolean from {expected}")),
            };
            let mut t = vec![0.0f32; k.max(2)];
            t[idx.min(1)] = 1.0;
            t
        }
        _ => {
            // score: fractional values interpolate low / low+1
            let mut t = vec![0.0f32; k];
            if let Some(idx) = idx_of(expected) {
                t[idx] = 1.0;
            } else if let Some(v) = expected.as_f64() {
                if !(0.0..=((k - 1) as f64)).contains(&v) {
                    return Err(format!("expected score {v} out of bounds [0, {}]", k - 1));
                }
                let low = v.floor() as usize;
                if low == k - 1 || v == low as f64 {
                    t[low] = 1.0;
                } else {
                    let frac = (v - low as f64) as f32;
                    t[low] = 1.0 - frac;
                    t[low + 1] = frac;
                }
            } else {
                return Err(format!("expected answer {expected} not in score levels"));
            }
            t
        }
    };
    if label_smoothing > 0.0 {
        let k = target.len() as f32;
        for t in &mut target {
            *t = (1.0 - label_smoothing) * *t + label_smoothing / k;
        }
    }
    Ok(target)
}

/// Lay a target out in SLOT order: target_slot[s] = target[order[s]].
fn permute_target(target: &[f32], order: &[usize]) -> Vec<f32> {
    order.iter().map(|&i| target[i]).collect()
}

/// Encode one row-question pair with an optional presentation order; skips
/// pairs whose options do not fit (markers dropped = training on a truncated
/// answer space, official skips).
fn encode(
    agent: &RlAgent,
    state: &str,
    q: &Question,
    order: Option<&[usize]>,
) -> Option<(Vec<i64>, Vec<usize>)> {
    let (ids, markers) = build_sequence_ex(
        &agent.tok,
        state,
        q,
        agent.max_len,
        agent.head_max_len,
        order,
        false,
    );
    if markers.len() != q.options.len() {
        return None; // options did not fit; skip rather than train truncated
    }
    Some((ids, markers))
}

/// items_from_jsonl: laya-evals rows {state, questions, expected} -> items.
/// `expected` keys each question; unlabelled questions are skipped.
pub fn items_from_jsonl(
    agent: &RlAgent,
    path: &str,
) -> Result<(Vec<TrainRow>, HashMap<String, usize>), String> {
    let mut items = Vec::new();
    let mut skipped: HashMap<String, usize> = HashMap::new();
    for line in std::fs::read_to_string(path).map_err(|e| e.to_string())?.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let row: Value = serde_json::from_str(line).map_err(|e| format!("{path}: {e}"))?;
        let state = match &row["state"] {
            Value::String(s) if !s.trim().is_empty() => s.clone(),
            Value::String(_) | Value::Null => {
                *skipped.entry("empty_text".into()).or_default() += 1;
                continue;
            }
            other => crate::python_json::PythonJson::state(other),
        };
        let Some(questions) = row["questions"].as_object() else { continue };
        let expected = row["expected"].as_object().cloned().unwrap_or_default();
        for (qid, question) in questions {
            let Some(expected_val) = expected.get(qid) else { continue };
            let Some(input) = crate::jev::question_from_jev(question) else {
                *skipped.entry("invalid_question".into()).or_default() += 1;
                continue;
            };
            let q = Question::from(&input);
            let target = match target_from_expected(&q, &input.keys, expected_val, 0.0) {
                Ok(t) if t.len() == q.options.len() => t,
                Ok(_) | Err(_) => {
                    *skipped.entry("invalid_target".into()).or_default() += 1;
                    continue;
                }
            };
            match encode(agent, &state, &q, None) {
                Some(canonical) => items.push(TrainRow { state: state.clone(), q, target, canonical }),
                None => {
                    *skipped.entry("options_did_not_fit".into()).or_default() += 1;
                }
            }
        }
    }
    Ok((items, skipped))
}

/// Official split_calibration: hold out the calibration slice BEFORE
/// training; the split depends only on (seed, len) so reruns agree.
pub fn split_calibration(
    n: usize,
    calib_max: usize,
    calib_frac: f32,
    seed: u64,
) -> Vec<usize> {
    let n_calib = calib_max.min((n as f32 * calib_frac) as usize);
    let mut order: Vec<usize> = (0..n).collect();
    // xorshift stand-in for random.Random(seed).shuffle - deterministic.
    let mut state = seed.max(1);
    for i in (1..order.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let j = (state as usize) % (i + 1);
        order.swap(i, j);
    }
    let mut held: Vec<usize> = order.into_iter().take(n_calib).collect();
    held.sort();
    held
}

/// sigma_at: exploration noise, linearly annealed start -> end over epochs.
pub fn sigma_at(epoch: usize, epochs: usize, start: f32, end: f32) -> f32 {
    start + (end - start) * (epoch as f32 / epochs.saturating_sub(1).max(1) as f32)
}


/// Everything the caller needs from a finetune run.
#[derive(Debug)]
pub struct TrainReport {
    pub out_dir: String,
    pub items: usize,
    pub skipped: HashMap<String, usize>,
    pub calibration_items: usize,
    pub first_loss: f32,
    pub last_loss: f32,
    pub temperatures: [f64; 3],
    /// Buckets fitted on the held-out split (needs MIN_BUCKET_N pairs);
    /// empty means the stale temperature_by_options was DROPPED from the
    /// written config - the artifact is single-temperature.
    pub temperature_buckets: Vec<(String, f64)>,
    pub steps: usize,
}

/// The full fine-tune pipeline. `device` should be an autodiff device
/// (`Device::default().autodiff()`); the report's temperatures are written
/// into out_dir/rl_agent_config.json, weights into out_dir/model.safetensors.
pub fn finetune(
    model_dir: &str,
    data_jsonl: &str,
    out_dir: &str,
    cfg: &TrainConfig,
    opts: &crate::options::LoadOptions,
    device: &Device,
) -> Result<TrainReport, String> {
    let agent = RlAgent::load_with(model_dir, opts, device);
    let (all, skipped) = items_from_jsonl(&agent, data_jsonl)?;
    if all.is_empty() {
        return Err(format!("no training items (skipped: {skipped:?})"));
    }
    // Held-out calibration split taken BEFORE training; partition by
    // index so each TrainRow lives exactly once (no all+train+calib
    // triple residency of the canonical encodings).
    let held = split_calibration(all.len(), cfg.calib_max, cfg.calib_frac, cfg.calib_seed);
    let mut train: Vec<TrainRow> = Vec::new();
    let mut calib: Vec<TrainRow> = Vec::new();
    for (i, row) in all.into_iter().enumerate() {
        if held.binary_search(&i).is_ok() {
            calib.push(row);
        } else {
            train.push(row);
        }
    }
    let n_items = train.len() + calib.len();

    std::fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;

    let mut model = agent.model.clone().map(&mut RequireGrad);
    let mut optim: ModuleOptimizer = AdamWConfig::new()
        .with_weight_decay(cfg.weight_decay as f32)
        .init();

    let mut rng_state = cfg.seed.max(1);
    let mut shuffle = |k: usize| -> Vec<usize> {
        let mut order: Vec<usize> = (0..k).collect();
        for i in (1..order.len()).rev() {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let j = (rng_state as usize) % (i + 1);
            order.swap(i, j);
        }
        order
    };

    let mut first_loss: Option<f32> = None;
    let mut last_loss = 0.0f32;
    let mut steps = 0usize;
    for epoch in 0..cfg.epochs {
        let sigma = sigma_at(epoch, cfg.epochs, cfg.sigma_start, cfg.sigma_end);
        let (m, f, l, n) =
            epoch_step(&agent, model.clone(), &mut optim, &train, cfg, sigma, &mut shuffle, device)?;
        model = m;
        if first_loss.is_none() {
            first_loss = f;
        }
        last_loss = l;
        steps += n;
    }

    let (pairs_by_type, pairs_by_bucket) =
        collect_calibration_logits(&agent, &model, &calib, cfg, device)?;
    let temperatures = [
        crate::calibration::fit_one_temperature(&pairs_by_type[0], 10),
        crate::calibration::fit_one_temperature(&pairs_by_type[1], 10),
        crate::calibration::fit_one_temperature(&pairs_by_type[2], 10),
    ];
    // Per-bucket fits (official MIN_BUCKET_N floor: a bucket needs 2000
    // pairs before its fitted value outranks anything). With fewer, the
    // bucket is NOT written - and when no bucket qualifies, the stale
    // pre-finetune temperature_by_options is dropped entirely (the official
    // small-data notebook does the same cfg.pop), so the written artifact
    // is exactly what serving will use.
    const MIN_BUCKET_N: usize = 2000;
    let temperature_buckets: Vec<(String, f64)> = pairs_by_bucket
        .iter()
        .filter(|(_, pairs)| pairs.len() >= MIN_BUCKET_N)
        .map(|(bucket, pairs)| {
            (bucket.clone(), crate::calibration::fit_one_temperature(pairs, MIN_BUCKET_N))
        })
        .collect();

    write_checkpoint(&model, model_dir, out_dir, temperatures, &temperature_buckets)?;

    Ok(TrainReport {
        out_dir: out_dir.to_string(),
        items: n_items,
        skipped,
        calibration_items: calib.len(),
        first_loss: first_loss.unwrap_or(0.0),
        last_loss,
        temperatures,
        temperature_buckets,
        steps,
    })
}




/// One epoch: micro-batch chunks over `train`, optional per-type option
/// shuffling (identity orders reuse the cached canonical encodings), the
/// configured loss, AdamW step. Returns (first_loss, last_loss, steps).
fn epoch_step(
    agent: &RlAgent,
    mut model: Laya,
    optim: &mut ModuleOptimizer,
    train: &[TrainRow],
    cfg: &TrainConfig,
    sigma: f32,
    shuffle: &mut impl FnMut(usize) -> Vec<usize>,
    device: &Device,
) -> Result<(Laya, Option<f32>, f32, usize), String> {
    let mut first_loss: Option<f32> = None;
    let mut last_loss = 0.0f32;
    let mut steps = 0usize;
    {
    for chunk in train.chunks(cfg.micro_batch.max(1)) {
        let mut seqs = Vec::new();
        let mut targets = Vec::new();
        for row in chunk {
            let k = row.q.options.len();
            let shuffled = cfg.shuffle_types.iter().any(|t| t == QTYPES[row.q.qtype]) && k >= 2;
            let order: Vec<usize> = if shuffled { shuffle(k) } else { (0..k).collect() };
            // identity order reuses the canonical encoding cached at
            // load; only shuffled epochs re-encode
            let (ids, markers) = if shuffled {
                encode(&agent, &row.state, &row.q, Some(&order))
                    .ok_or("options did not fit at train time")?
            } else {
                row.canonical.clone()
            };
            seqs.push((ids, markers, row.q.qtype as i64));
            targets.push(permute_target(&row.target, &order));
        }
        let b = crate::inference::collate_batch(&seqs, agent.tok.pad_token_id, device);
        let n_markers = b.marker_pos.dims()[1];
        let (logits, _act) = crate::inference::forward_batch(&model, &agent.ropes, &b, &agent.fopts);
        // target/mask padded to n_markers columns
        let rows = seqs.len();
        let mut t_flat = vec![0.0f32; rows * n_markers];
        let mut m_flat = vec![0.0f32; rows * n_markers];
        for (r, t) in targets.iter().enumerate() {
            for (s, &v) in t.iter().enumerate() {
                t_flat[r * n_markers + s] = v;
                m_flat[r * n_markers + s] = 1.0;
            }
        }
        let target = Tensor::<2>::from_data(
            burn::tensor::TensorData::new(t_flat, [rows, n_markers]),
            device,
        );
        let mask = Tensor::<2>::from_data(
            burn::tensor::TensorData::new(m_flat, [rows, n_markers]),
            device,
        );
        let loss = match cfg.loss {
            LossKind::SoftCe | LossKind::CeOnly => soft_ce(&logits, &target, &mask),
            LossKind::Rlcd => rlcd_loss(
                &logits,
                &target,
                &mask,
                &b.qtype,
                sigma,
                cfg.rl_samples,
                cfg.w_sph,
                cfg.w_rps,
            ),
        };
        let lv = loss.clone().into_scalar::<f32>();
        if cfg.log {
            eprintln!("step {steps}: loss {lv:.6}");
        }
        if first_loss.is_none() {
            first_loss = Some(lv);
        }
        last_loss = lv;
        steps += 1;
        let grads_tensor = loss.backward();
        let grads = GradientsParams::from_grads(grads_tensor, &model);
        model = optim.step(cfg.lr, model, grads).map(&mut RequireGrad);
    }
    }


    Ok((model, first_loss, last_loss, steps))
}


/// Held-out calibration pass in canonical order (the cached encodings):
/// collects (logits, target) pairs keyed by qtype and by temperature bucket.
fn collect_calibration_logits(
    agent: &RlAgent,
    model: &Laya,
    calib: &[TrainRow],
    cfg: &TrainConfig,
    device: &Device,
) -> Result<
    (
        [Vec<(Vec<f32>, Vec<f32>)>; 3],
        std::collections::BTreeMap<String, Vec<(Vec<f32>, Vec<f32>)>>,
    ),
    String,
> {
    let mut pairs_by_type: [Vec<(Vec<f32>, Vec<f32>)>; 3] = Default::default();
    let mut pairs_by_bucket: std::collections::BTreeMap<String, Vec<(Vec<f32>, Vec<f32>)>> =
        Default::default();
    for chunk in calib.chunks(cfg.micro_batch.max(1)) {
    // calibration is always canonical order: the cached encodings
    let seqs: Vec<(Vec<i64>, Vec<usize>, i64)> = chunk
        .iter()
        .map(|row| (row.canonical.0.clone(), row.canonical.1.clone(), row.q.qtype as i64))
        .collect();
    let n_markers = seqs.iter().map(|(_, m, _)| m.len()).max().unwrap();
    let b = crate::inference::collate_batch(&seqs, agent.tok.pad_token_id, device);
    let (logits, _act) = crate::inference::forward_batch(&model, &agent.ropes, &b, &agent.fopts);
    let logits = logits.detach().into_data();
    let vals = logits.as_slice::<f32>().map_err(|e| format!("{e:?}"))?;
    for (r, row) in chunk.iter().enumerate() {
        let k = row.target.len();
        let z: Vec<f32> = (0..k).map(|c| vals[r * n_markers + c]).collect();
        pairs_by_type[row.q.qtype].push((z.clone(), row.target.clone()));
        let bucket = crate::calibration::Calibration::temp_bucket(row.q.qtype, k);
        pairs_by_bucket.entry(bucket).or_default().push((z, row.target.clone()));
    }
    }

    Ok((pairs_by_type, pairs_by_bucket))
}


/// Write the COMPLETE out_dir: weights, config with fitted temperatures
/// (stale temperature_by_options dropped when no bucket qualifies), the
/// tokenizer and encoder config copied through - servable as-is.
#[allow(clippy::type_complexity)]
fn write_checkpoint(
    model: &Laya,
    model_dir: &str,
    out_dir: &str,
    temperatures: [f64; 3],
    temperature_buckets: &[(String, f64)],
) -> Result<(), String> {
    crate::export::export_safetensors(
    &model,
    [temperatures[0] as f32, temperatures[1] as f32, temperatures[2] as f32],
    &format!("{out_dir}/model.safetensors"),
    );
    let mut out_cfg: Value = serde_json::from_str(&std::fs::read_to_string(format!(
    "{model_dir}/rl_agent_config.json"
    ))
    .map_err(|e| e.to_string())?)
    .map_err(|e| e.to_string())?;
    out_cfg["temperature"] = serde_json::json!(temperatures);
    if temperature_buckets.is_empty() {
    if let Some(obj) = out_cfg.as_object_mut() {
        obj.remove("temperature_by_options");
    }
    } else {
    let map = out_cfg
        .as_object_mut()
        .unwrap()
        .entry("temperature_by_options")
        .or_insert_with(|| serde_json::json!({}));
    for (bucket, t) in temperature_buckets.iter() {
        map[bucket] = serde_json::json!(t);
    }
    }
    std::fs::write(
    format!("{out_dir}/rl_agent_config.json"),
    serde_json::to_string_pretty(&out_cfg).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    for f in [
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "encoder/config.json",
    ] {
    if let Ok(bytes) = std::fs::read(format!("{model_dir}/{f}")) {
        let out = format!("{out_dir}/{f}");
        std::fs::create_dir_all(std::path::Path::new(&out).parent().unwrap())
            .map_err(|e| e.to_string())?;
        std::fs::write(out, bytes).map_err(|e| e.to_string())?;
    }
    }

    Ok(())
}

pub fn soft_ce(logits: &Tensor<2>, target: &Tensor<2>, mask: &Tensor<2>) -> Tensor<1> {
    // masked-OUT slots get -1e9 before the softmax: (mask - 1) * 1e9
    let add = (mask.clone() - 1.0).mul_scalar(1.0e9);
    let logp = burn::tensor::activation::log_softmax(logits.clone() + add, 1);
    (target.clone() * logp * mask.clone()).sum_dim(1).mean().mul_scalar(-1.0)
}

/// The RLCD policy-gradient core with the noise INJECTED ([rows*samples, cols]
/// layout, matching the internal generator): advantage-weighted Gaussian
/// log-density of the shared noise plus soft CE. Exposed for the
/// gradient-parity harness against the PyTorch reference.
pub fn rlcd_core(
    logits: &Tensor<2>,
    target: &Tensor<2>,
    mask: &Tensor<2>,
    row_qtypes: &[usize],
    eps_flat: &Tensor<2>,
    sigma: f32,
    samples: usize,
    w_sph: f32,
    w_rps: f32,
) -> Tensor<1> {
    let rows = logits.dims()[0];
    let cols = logits.dims()[1];
    let total = soft_ce(logits, target, mask);
    if samples == 0 || sigma <= 0.0 {
        return total;
    }
    let z_host = logits.clone().detach().into_data();
    let z = z_host.as_slice::<f32>().unwrap_or(&[]).to_vec();
    let t_host = target.clone().detach().into_data();
    let t = t_host.as_slice::<f32>().unwrap_or(&[]).to_vec();
    let m_host = mask.clone().detach().into_data();
    let m = m_host.as_slice::<f32>().unwrap_or(&[]).to_vec();
    let eps_host = eps_flat.clone().detach().into_data();
    let eps_all = eps_host.as_slice::<f32>().unwrap_or(&[]).to_vec();

    // Rewards [rows x samples], then advantages: PER-ROW centering, but the
    // std is GLOBAL over all rows*samples - official `adv.std()` with no dim
    // pools the whole tensor. Per-row std (the earlier port) amplifies
    // numerical noise on rows whose rewards barely differ.
    let mut all_rewards = vec![0.0f32; rows * samples];
    for r in 0..rows {
        for s_idx in 0..samples {
            let base = (r * samples + s_idx) * cols;
            let z_s: Vec<f32> = (0..cols).map(|c| z[r * cols + c] + eps_all[base + c]).collect();
            all_rewards[r * samples + s_idx] = proper_reward_host(
                &z_s,
                &t[r * cols..(r + 1) * cols],
                &m[r * cols..(r + 1) * cols],
                row_qtypes[r],
                w_sph,
                w_rps,
            );
        }
    }
    let mut adv_w = vec![0.0f32; rows * samples];
    {
        // Official: adv = r - r.mean(0); adv / (adv.std() + 1e-6) - the std
        // is over the PER-ROW-CENTERED tensor (global, sample std), which is
        // smaller than the raw rewards' std. Center first, then measure.
        let mut centered = vec![0.0f32; rows * samples];
        for r in 0..rows {
            let mean = (0..samples).map(|s_idx| all_rewards[r * samples + s_idx]).sum::<f32>()
                / samples as f32;
            for s_idx in 0..samples {
                centered[r * samples + s_idx] = all_rewards[r * samples + s_idx] - mean;
            }
        }
        let n = centered.len() as f32;
        let global_var = if centered.len() > 1 {
            let gmean = centered.iter().sum::<f32>() / n;
            centered.iter().map(|x| (x - gmean).powi(2)).sum::<f32>() / (n - 1.0)
        } else {
            0.0
        };
        let global_std = global_var.sqrt() + 1e-6;
        for (a, c) in adv_w.iter_mut().zip(centered) {
            *a = c / global_std;
        }
    }
    let adv = Tensor::<1>::from_data(
        burn::tensor::TensorData::new(adv_w, [rows * samples]),
        &logits.device(),
    );
    // z_s - logits, with z_const = detached logits taken OUTSIDE the graph:
    // the only tracked operand is `logits`, so the backward is exactly
    // d logp / d logits = eps / sigma^2 (verified against the reference).
    // repeat_dim tiles BLOCK-wise; unsqueeze to [R,1,C] so dim-1 tiling
    // interleaves each row's copies, matching the eps layout [r*S+s].
    let z_const = logits.clone().detach();
    // Build [r*S+s, C] by CONCATENATING tracked row slices - repeat_dim's
    // backward misroutes gradients through the reshape (parity-verified:
    // forward exact, gradient wrong; slice+cat is exact in both).
    let mut z_parts = Vec::with_capacity(rows * samples);
    let mut l_parts = Vec::with_capacity(rows * samples);
    for r in 0..rows {
        for _s in 0..samples {
            z_parts.push(z_const.clone().slice([r..r + 1, 0..cols]));
            l_parts.push(logits.clone().slice([r..r + 1, 0..cols]));
        }
    }
    let z_rows = Tensor::cat(z_parts, 0);
    let l_rows = Tensor::cat(l_parts, 0);
    let diff = eps_flat.clone() + z_rows - l_rows;
    // square via self-multiply (trivial backward), not powf
    let sq = diff.clone() * diff;
    let logp = sq.sum_dim(1).mul_scalar(-1.0 / (2.0 * sigma * sigma));
    let logp = logp.reshape([rows * samples]);
    if std::env::var("LAYA_PARITY_DEBUG").is_ok() {
        let av: Vec<f32> = adv.clone().into_data().as_slice::<f32>().unwrap().to_vec();
        let lp: Vec<f32> = logp.clone().into_data().as_slice::<f32>().unwrap().to_vec();
        let avb: Vec<u8> = av.iter().flat_map(|v| v.to_le_bytes()).collect();
        let lpb: Vec<u8> = lp.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write("/tmp/grad_parity/rust_adv.f32", avb).unwrap();
        std::fs::write("/tmp/grad_parity/rust_logp.f32", lpb).unwrap();
    }
    let pg = (adv * logp).mean().mul_scalar(-1.0);
    total + pg
}

/// The full RLCD objective with internally-sampled noise (Box-Muller from a
/// fixed-ish xorshift stream, projected onto each row's options).
fn rlcd_loss(
    logits: &Tensor<2>,
    target: &Tensor<2>,
    mask: &Tensor<2>,
    qtypes: &Tensor<1, burn::tensor::Int>,
    sigma: f32,
    samples: usize,
    w_sph: f32,
    w_rps: f32,
) -> Tensor<1> {
    let rows = logits.dims()[0];
    let cols = logits.dims()[1];
    let q_host = qtypes.clone().into_data();
    let row_qtypes: Vec<usize> = q_host
        .as_slice::<i32>()
        .unwrap_or(&[])
        .iter()
        .map(|&q| q as usize)
        .collect();
    let mut rng: u64 = 0x9E3779B97F4A7C15;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let u1 = ((rng >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
        let u2 = (rng >> 11) as f64 / (1u64 << 53) as f64;
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    };
    let mut eps_all = vec![0.0f32; rows * samples * cols];
    let m_host = mask.clone().detach().into_data();
    let m = m_host.as_slice::<f32>().unwrap_or(&[]).to_vec();
    for r in 0..rows {
        let k = m[r * cols..(r + 1) * cols].iter().sum::<f32>().max(1.0) as usize;
        for s_idx in 0..samples {
            let base = (r * samples + s_idx) * cols;
            let mut sum = 0.0f32;
            for c in 0..cols {
                let v = next() * sigma * m[r * cols + c];
                eps_all[base + c] = v;
                sum += v;
            }
            for c in 0..cols {
                eps_all[base + c] -= sum / k as f32;
                eps_all[base + c] *= m[r * cols + c];
            }
        }
    }
    let eps = Tensor::<2>::from_data(
        burn::tensor::TensorData::new(eps_all, [rows * samples, cols]),
        &logits.device(),
    );
    rlcd_core(logits, target, mask, &row_qtypes, &eps, sigma, samples, w_sph, w_rps)
}

pub fn proper_reward_host(
    z: &[f32],
    target: &[f32],
    mask: &[f32],
    qtype: usize,
    w_sph: f32,
    w_rps: f32,
) -> f32 {
    // log score + spherical (+ RPS for ordinal rows) on a masked softmax;
    // host-side mirror of rl_common.proper_reward (reward is no-grad).
    let max = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = z.iter().zip(mask).map(|(v, &m)| (v - max).exp() * m).collect();
    let sum: f32 = exps.iter().sum();
    let q: Vec<f32> = exps.iter().map(|e| e / sum.max(1e-12)).collect();
    let mut log_score = 0.0f32;
    let mut dot = 0.0f32;
    let mut qnorm2 = 0.0f32;
    for i in 0..z.len() {
        log_score += target[i] * (q[i].max(1e-12)).ln().max(-9.21);
        dot += target[i] * q[i];
        qnorm2 += q[i] * q[i];
    }
    let sph = dot / qnorm2.sqrt().max(1e-9);
    let mut r = log_score + w_sph * sph;
    if qtype == 1 {
        // ranked probability score for ordinal (score) rows
        let k = mask.iter().filter(|&&m| m > 0.5).count().max(2) as f32;
        let mut cdf_q = 0.0f32;
        let mut cdf_t = 0.0f32;
        let mut rps = 0.0f32;
        for i in 0..z.len() {
            cdf_q += q[i];
            cdf_t += target[i];
            rps += (cdf_q - cdf_t).powi(2) * mask[i];
        }
        r -= w_rps * rps / (k - 1.0);
    }
    r
}

#[cfg(test)]
mod train_tests {
    use super::*;

    fn q(qtype: usize, options: Vec<&str>) -> Question {
        Question { qtype, instructions: "i".into(), options: options.into_iter().map(String::from).collect() }
    }

    #[test]
    fn target_from_expected_semantics() {
        // choice: one-hot at the key
        let question = q(0, vec!["a", "b", "c"]);
        let t = target_from_expected(&question, &["b".into(), "a".into(), "c".into()], &serde_json::json!("a"), 0.0).unwrap();
        assert_eq!(t, vec![0.0, 1.0, 0.0]);
        // noul: bool arm
        let question = q(2, vec!["false: no", "true: yes"]);
        let t = target_from_expected(&question, &[], &serde_json::json!(true), 0.0).unwrap();
        assert_eq!(t, vec![0.0, 1.0]);
        // score: fractional interpolates
        let question = q(1, vec!["l0", "l1", "l2", "l3"]);
        let t = target_from_expected(&question, &[], &serde_json::json!(1.5), 0.0).unwrap();
        assert_eq!(t, vec![0.0, 0.5, 0.5, 0.0]);
        // label smoothing spreads mass
        let question = q(0, vec!["a", "b"]);
        let t = target_from_expected(&question, &["a".into(), "b".into()], &serde_json::json!("a"), 0.2).unwrap();
        assert!((t[0] - 0.9).abs() < 1e-6 && (t[1] - 0.1).abs() < 1e-6);
        // unknown label errors
        assert!(target_from_expected(&question, &["a".into(), "b".into()], &serde_json::json!("zzz"), 0.0).is_err());
    }

    #[test]
    fn fit_temperature_pulls_a_sharp_row_toward_its_confidence() {
        // confident correct logits -> fitted T below 1 (sharpen further helps NLL)
        let pairs = vec![(vec![4.0f32, 0.0, 0.0], vec![1.0f32, 0.0, 0.0]); 12];
        let t = crate::calibration::fit_one_temperature(&pairs, 10);
        assert!(t < 1.0, "confident row should sharpen: {t}");
        // under min_n -> neutral
        assert_eq!(crate::calibration::fit_one_temperature(&pairs[..3], 10), 1.0);
        // overconfident-wrong -> T above 1 (soften)
        let pairs = vec![(vec![4.0f32, 0.0, 0.0], vec![0.0f32, 1.0, 0.0]); 12];
        let t = crate::calibration::fit_one_temperature(&pairs, 10);
        assert!(t > 1.0, "wrong-confident row should soften: {t}");
    }

    #[test]
    fn sigma_and_split_are_deterministic() {
        assert!(sigma_at(0, 4, 0.4, 0.1) - 0.4 < 1e-6);
        let end = sigma_at(3, 4, 0.4, 0.1);
        assert!((end - 0.1).abs() < 1e-6);
        let a = split_calibration(100, 400, 0.1, 20260922);
        let b = split_calibration(100, 400, 0.1, 20260922);
        assert_eq!(a, b);
        assert_eq!(a.len(), 10);
    }
}

#[cfg(test)]
mod parity_tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData};

    /// Gradient parity against the PyTorch reference on a committed fixed
    /// batch (tests/fixtures/grad_parity): the RLCD core must match the
    /// official loss AND d(loss)/d(logits) to float tolerance. This is the
    /// regression net for the two bugs the harness caught: advantage std
    /// over the per-row-CENTERED tensor (not the raw rewards), and
    /// repeat_dim's backward misrouting through reshape (replaced by
    /// tracked slices + cat).
    #[test]
    fn rlcd_core_matches_pytorch_loss_and_grad() {
        let dir = "tests/fixtures/grad_parity";
        let meta: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(format!("{dir}/meta.json")).unwrap(),
        )
        .unwrap();
        let rows = meta["rows"].as_u64().unwrap() as usize;
        let cols = meta["cols"].as_u64().unwrap() as usize;
        let samples = meta["samples"].as_u64().unwrap() as usize;
        let sigma = meta["sigma"].as_f64().unwrap() as f32;
        let w_sph = meta["w_sph"].as_f64().unwrap() as f32;
        let w_rps = meta["w_rps"].as_f64().unwrap() as f32;
        let qtype: Vec<usize> = meta["qtype"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let read = |name: &str, n: usize| -> Vec<f32> {
            let b = std::fs::read(format!("{dir}/{name}.f32")).unwrap();
            b.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).take(n).collect()
        };
        // flex: pure-CPU backend, works on headless CI runners
        let device = Device::flex().autodiff();
        let logits = Tensor::<2>::from_data(
            TensorData::new(read("logits", rows * cols), [rows, cols]),
            &device,
        )
        .autodiff()
        .require_grad();
        let target = Tensor::<2>::from_data(
            TensorData::new(read("target", rows * cols), [rows, cols]),
            &device,
        );
        let mask = Tensor::<2>::from_data(
            TensorData::new(read("mask", rows * cols), [rows, cols]),
            &device,
        );
        let eps = Tensor::<2>::from_data(
            TensorData::new(read("eps", rows * samples * cols), [rows * samples, cols]),
            &device,
        );
        let loss = rlcd_core(&logits, &target, &mask, &qtype, &eps, sigma, samples, w_sph, w_rps);
        let lv = loss.clone().into_scalar::<f32>() as f64;
        let py_loss = meta["py_loss"].as_f64().unwrap();
        assert!((lv - py_loss).abs() < 1e-5, "loss {lv} vs py {py_loss}");
        let grads = loss.backward();
        let grad = logits.grad(&grads).expect("grad");
        let gv = grad.into_data().as_slice::<f32>().unwrap().to_vec();
        let py_grad = read("py_grad", rows * cols);
        let max_diff = gv
            .iter()
            .zip(&py_grad)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-5, "grad max|diff| {max_diff}");
    }
}
