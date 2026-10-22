//! Inference pipeline: port of RLAgent.system_one from rl_agent_api.py.

use burn::tensor::{capture, Device, Graph, Int, Tensor};
use serde_json::Value;
use std::collections::HashMap;

use crate::calibration::Calibration;
use crate::model::{EncoderConfig, Laya, Ropes};
use crate::python_json::PythonJson;
use crate::sequence::{build_sequence, Batch, Question};
use crate::tokenizer::LayaTokenizer;
use crate::weights::Weights;

pub struct RlAgent {
    pub model: Laya,
    pub ropes: Ropes,
    pub tok: LayaTokenizer,
    pub max_len: usize,
    pub head_max_len: usize,
    pub temperature: [f64; 3],
    pub temperature_by_options: Vec<(String, f64)>,
    /// Buckets whose fitted temperature was clamped to [0.5, 5.0] at load.
    pub clamped_temperatures: Vec<String>,
}

/// One question in the public API shape.
#[derive(Clone)]
pub struct QuestionInput {
    pub qtype: usize, // 0=choice 1=score 2=noul
    pub instructions: String,
    pub options: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum Answer {
    Choice {
        index: usize,
        probabilities: Vec<f32>,
        confidence: f32,
    },
    Score {
        score: f32,
        probabilities: Vec<f32>,
        confidence: f32,
    },
    Noul {
        prob: f32,
    },
}

fn temp_bucket(qtype: usize, k: usize) -> String {
    let name = ["choice", "score", "noul"][qtype];
    let range = match k {
        2 => "2".to_string(),
        3..=5 => "3-5".to_string(),
        6..=10 => "6-10".to_string(),
        _ => "11+".to_string(),
    };
    format!("{name}:{range}")
}

fn confidence_from_probs(p: &[f32]) -> f32 {
    if p.len() < 2 {
        return 1.0;
    }
    let ent = -p.iter().map(|&x| x * x.max(1e-12).ln()).sum::<f32>();
    1.0 - ent / (p.len() as f32).ln()
}

/// Clamp every fitted temperature to [0.5, 5.0] (official SDK behavior, issue
/// #42): a sub-0.5 temperature would sharpen logits and overstate confidence.
/// LAYA_UNCLAMPED=1 keeps the RAW fitted temperatures - what the official
/// harness uses to reproduce its committed sweep numbers (--unclamped), as
/// opposed to the clamped readout the SDK applies.
fn read_temperatures(cfg: &Value) -> ([f64; 3], Vec<(String, f64)>, Vec<String>) {
    let maybe_clamp = |raw: f64| {
        if std::env::var("LAYA_UNCLAMPED").as_deref() == Ok("1") {
            raw
        } else {
            crate::calibration::Calibration::clamp_temperature(raw)
        }
    };
    let temperature = [
        maybe_clamp(cfg["temperature"][0].as_f64().unwrap_or(1.0)),
        maybe_clamp(cfg["temperature"][1].as_f64().unwrap_or(1.0)),
        maybe_clamp(cfg["temperature"][2].as_f64().unwrap_or(1.0)),
    ];
    let temperature_by_options: Vec<(String, f64)> = cfg["temperature_by_options"]
        .as_object()
        .map(|m| {
            m.iter()
                .map(|(k, v)| (k.clone(), maybe_clamp(v.as_f64().unwrap_or(1.0))))
                .collect()
        })
        .unwrap_or_default();
    let mut clamped_temperatures = Vec::new();
    if let Some(m) = cfg["temperature_by_options"].as_object() {
        for (k, v) in m {
            let raw = v.as_f64().unwrap_or(1.0);
            if crate::calibration::Calibration::clamp_temperature(raw) != raw {
                clamped_temperatures.push(format!("{k}={raw}"));
            }
        }
    }
    for (i, t) in cfg["temperature"].as_array().unwrap_or(&vec![]).iter().enumerate().take(3) {
        let raw = t.as_f64().unwrap_or(1.0);
        if crate::calibration::Calibration::clamp_temperature(raw) != raw {
            clamped_temperatures.push(format!("temperature[{i}]={raw}"));
        }
    }
    if !clamped_temperatures.is_empty() {
        eprintln!("clamped temperatures: {clamped_temperatures:?}");
    }
    (temperature, temperature_by_options, clamped_temperatures)
}

/// Jev request shape ({type, instructions, criteria}) -> [QuestionInput].
/// A port of rl_common.render_options; shared by laya-serve and the wasm
/// runtime so the option text is rendered identically everywhere.
pub fn question_from_jev(q: &Value) -> Option<QuestionInput> {
    let qtype = q.get("type")?.as_str()?;
    let idx = match qtype {
        "choice" => 0,
        "score" => 1,
        "noul" => 2,
        _ => return None,
    };
    let instructions = match q.get("instructions") {
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
        None => String::new(),
    };
    let criteria = q.get("criteria").cloned().unwrap_or(Value::Null);
    Some(QuestionInput { qtype: idx, instructions, options: render_options(qtype, &criteria) })
}

/// Criterion text must be the PLAIN string, not serde Display (which quotes
/// it); non-strings fall back to their JSON form.
fn render_options(qtype: &str, criteria: &Value) -> Vec<String> {
    let text = |v: &Value| -> String {
        match v {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        }
    };
    match qtype {
        "choice" => criteria
            .as_object()
            .map(|m| {
                m.iter()
                    .map(|(label, desc)| {
                        let blank = matches!(desc, Value::Null)
                            || matches!(desc, Value::String(s) if s.is_empty());
                        if blank {
                            label.clone()
                        } else {
                            format!("{label}: {}", text(desc))
                        }
                    })
                    .collect()
            })
            .unwrap_or_default(),
        "score" => criteria
            .as_array()
            .map(|a| {
                a.iter()
                    .enumerate()
                    .map(|(i, d)| format!("level {i}: {}", text(d)))
                    .collect()
            })
            .unwrap_or_default(),
        _ => vec![
            "false: no, the statement does not hold".into(),
            "true: yes, the statement holds".into(),
        ],
    }
}

/// Answer in the official response shape (what POST /v1/decide returns).
pub fn answer_to_json(q: &QuestionInput, a: &Answer) -> Value {
    match a {
        Answer::Choice { index, probabilities, confidence } => serde_json::json!({
            "type": "choice",
            "choice": q.options.get(*index).cloned().unwrap_or_default(),
            "probabilities": q.options.iter().zip(probabilities.iter())
                .map(|(o, p)| (o.clone(), serde_json::json!(p)))
                .collect::<serde_json::Map<String, Value>>(),
            "confidence": confidence,
        }),
        Answer::Score { score, probabilities, confidence } => serde_json::json!({
            "type": "score",
            "score": score,
            "probabilities": probabilities,
            "confidence": confidence,
        }),
        Answer::Noul { prob } => serde_json::json!({
            "type": "noul",
            "noul": prob,
        }),
    }
}

impl RlAgent {
    pub fn load(model_dir: &str, device: &Device) -> Self {
        let weights =
            std::fs::read(format!("{model_dir}/model.safetensors")).expect("read safetensors");
        let cfg_json =
            std::fs::read_to_string(format!("{model_dir}/rl_agent_config.json")).unwrap();
        let tok_json =
            std::fs::read_to_string(format!("{model_dir}/tokenizer/tokenizer.json")).unwrap();
        let tok_cfg = std::fs::read_to_string(format!("{model_dir}/tokenizer/tokenizer_config.json")).ok();
        Self::load_bytes(weights, &cfg_json, &tok_json, tok_cfg.as_deref(), device)
    }

    /// No-filesystem variant of [RlAgent::load] for embedders that fetch
    /// bytes (wasm): safetensors body + rl_agent_config.json +
    /// tokenizer.json + optional tokenizer_config.json bodies.
    pub fn load_bytes(
        weights: Vec<u8>,
        cfg_json: &str,
        tok_json: &str,
        tok_cfg_json: Option<&str>,
        device: &Device,
    ) -> Self {
        let w = Weights::from_bytes(weights);
        let cfg: Value = serde_json::from_str(cfg_json).unwrap();
        let (temperature, temperature_by_options, clamped_temperatures) = read_temperatures(&cfg);

        // Architecture auto-detection from the checkpoint tensor shapes:
        // tok_embeddings [vocab, hidden] gives hidden (and thus heads = hidden/64);
        // the highest encoder.layers.N index gives the layer count. Geometry
        // that is not encoded in shapes (window, thetas, epsilon) matches every
        // shipped checkpoint so far (128 / 160k / 1e-5); mmBERT uses 160k for
        // BOTH thetas where large uses 10k sliding - disambiguate via the
        // encoder/config.json next to the checkpoint when present.
        let (vocab, hidden) = {
            let (data, shape) = w.raw_shape("encoder.embeddings.tok_embeddings.weight");
            let _ = data;
            (shape[0], shape[1])
        };
        let num_layers = w.max_layer_index() + 1;
        // NB: HF's ModernBertAttention reads the legacy top-level
        // local_rope_theta (default 10000), NOT the rope_parameters dict some
        // encoder configs ship - laya-multilingual's config says 160000 there
        // but the weights were trained with 10000 (verified by parity).
        let sliding_theta = 10_000.0f32;
        // Optional override for rope experiments (LAYA_THETA_SLIDE=...).
        let sliding_theta = std::env::var("LAYA_THETA_SLIDE")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(sliding_theta);
        let ecfg = EncoderConfig {
            hidden_size: hidden,
            num_layers,
            num_heads: hidden / 64,
            intermediate_size: if hidden == 1024 { 2624 } else { 1152 },
            window_size: 128,
            global_every: 3,
            max_sequence_length: 1088,
            rope_theta_full: 160_000.0,
            rope_theta_sliding: sliding_theta,
            epsilon: 1e-5,
        };
        let _ = vocab;
        // head_layers comes from the checkpoint config (both ModernBERT-large
        // checkpoints ship 2; kept configurable for future same-architecture
        // variants).
        let head_layers = cfg["head_layers"].as_u64().unwrap_or(2) as usize;
        let model = Laya::load(&w, &ecfg, head_layers, device);
        let ropes = Ropes::new(&ecfg, device);
        let tok = LayaTokenizer::from_json(tok_json, tok_cfg_json);
        let max_len = cfg["max_len"].as_u64().unwrap() as usize;
        let head_max_len = cfg["head_max_len"].as_u64().unwrap() as usize;
        Self { model, ropes, tok, max_len, head_max_len, temperature, temperature_by_options, clamped_temperatures }
    }

    /// Load from a burnpack export (model.mpk + model.mpk.json sidecar) with
    /// NO safetensors access: the sidecar carries the geometry, a zeros-only
    /// skeleton builds the module tree, and the burnpack record overwrites
    /// every parameter. Tokenizer + agent config still come from the
    /// directory (they are configuration, not weights).
    pub fn load_burnpack(model_dir: &str, device: &Device) -> Self {
        Self::load_burnpack_named(model_dir, "model.mpk", device)
    }

    /// Same as [RlAgent::load_burnpack] for an explicitly named pack file
    /// (e.g. the Q8 export model.q8.mpk, which shares the f32 sidecar).
    /// QFloat params from a quantized pack are dequantized eagerly in plain
    /// Rust: burn's lazy per-tensor kernel dequant costs ~36s on first use.
    pub fn load_burnpack_named(model_dir: &str, pack: &str, device: &Device) -> Self {
        let sidecar =
            std::fs::read_to_string(format!("{model_dir}/model.mpk.json")).expect("sidecar");
        let cfg_json =
            std::fs::read_to_string(format!("{model_dir}/rl_agent_config.json")).unwrap();
        let tok_json =
            std::fs::read_to_string(format!("{model_dir}/tokenizer/tokenizer.json")).unwrap();
        let tok_cfg =
            std::fs::read_to_string(format!("{model_dir}/tokenizer/tokenizer_config.json")).ok();
        let pack_bytes =
            std::fs::read(format!("{model_dir}/{pack}")).expect("read burnpack");
        Self::load_q8_bytes(
            pack_bytes,
            &sidecar,
            &cfg_json,
            &tok_json,
            tok_cfg.as_deref(),
            device,
        )
    }

    /// No-filesystem burnpack loader for embedders that fetch bytes (wasm):
    /// the pack body (f32 or Q8 - quantized params dequantize eagerly), the
    /// geometry sidecar, rl_agent_config.json, tokenizer.json and the
    /// optional tokenizer_config.json.
    pub fn load_q8_bytes(
        pack: Vec<u8>,
        sidecar_json: &str,
        cfg_json: &str,
        tok_json: &str,
        tok_cfg_json: Option<&str>,
        device: &Device,
    ) -> Self {
        use crate::skeleton::{SkeletonSpec, SkeletonWeights};
        use burn::module::Module;

        let spec: SkeletonSpec = serde_json::from_str(sidecar_json).expect("parse sidecar");
        let ecfg = crate::model::EncoderConfig {
            hidden_size: spec.hidden,
            num_layers: spec.layers,
            num_heads: spec.hidden / 64,
            intermediate_size: spec.intermediate,
            window_size: 128,
            global_every: 3,
            max_sequence_length: 1088,
            rope_theta_full: 160_000.0,
            rope_theta_sliding: spec.sliding_theta,
            epsilon: 1e-5,
        };
        let skel = SkeletonWeights::new(&spec);
        let model = crate::model::Laya::load(&skel, &ecfg, spec.head_layers, device);
        let record = burn::store::ModuleRecord::from_bytes(
            burn::tensor::Bytes::from_bytes_vec(pack),
        )
        .expect("read burnpack");
        let model = model.load_record(record);
        let (model, ds) = crate::quant::eager_dequantize(model);
        if ds.quantized > 0 {
            eprintln!("laya: dequantized {}/{} params from pack", ds.quantized, ds.params);
        }
        let ropes = crate::model::Ropes::new(&ecfg, device);

        let cfg: Value = serde_json::from_str(cfg_json).unwrap();
        let (temperature, temperature_by_options, clamped_temperatures) = read_temperatures(&cfg);
        let max_len = cfg["max_len"].as_u64().unwrap() as usize;
        let head_max_len = cfg["head_max_len"].as_u64().unwrap() as usize;
        let tok = LayaTokenizer::from_json(tok_json, tok_cfg_json);
        Self {
            model,
            ropes,
            tok,
            max_len,
            head_max_len,
            temperature,
            temperature_by_options,
            clamped_temperatures,
        }
    }

    /// Create the opt-in graph capture/replay engine (LAYA_REPLAY=1).
    pub fn replay_engine(&self, device: &Device) -> ReplayEngine<'_> {
        ReplayEngine::new(self, device)
    }

    /// state + questions -> typed answers, one forward pass for the whole batch.
    /// 'replay' is Some(..) only when LAYA_REPLAY=1: the opt-in graph
    /// capture/replay engine (see ReplayEngine). Pass None for the direct
    /// forward path.
    pub fn system_one(
        &self,
        device: &Device,
        state: &Value,
        questions: &[QuestionInput],
        replay: Option<&mut ReplayEngine<'_>>,
    ) -> Vec<Answer> {
        let state_str = PythonJson::state(state);

        let mut seqs = Vec::new();
        for q in questions {
            let (ids, markers) = build_sequence(
                &self.tok,
                &state_str,
                &Question { qtype: q.qtype, instructions: q.instructions.clone(), options: q.options.clone() },
                self.max_len,
                self.head_max_len,
            );
            seqs.push((ids, markers, q.qtype as i64));
        }

        let batch_len: usize = seqs.iter().map(|(ids, _, _)| ids.len()).max().unwrap();
        let n_markers: usize = seqs.iter().map(|(_, m, _)| m.len()).max().unwrap();
        let b = collate(&seqs, self.tok.pad_token_id, batch_len, n_markers);
        let Batch { input_ids: b_input_ids, attention_mask: b_attention_mask, marker_pos: b_marker_pos, marker_mask: b_marker_mask, qtype: b_qtype } = b;

        if let Ok(dir) = std::env::var("LAYA_DUMP_DIR") {
            let _ = std::fs::create_dir_all(&dir);
            let rows: Vec<String> = b_input_ids
                .iter()
                .map(|r| r.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","))
                .collect();
            let _ = std::fs::write(format!("{dir}/all_rows.txt"), rows.join("
"));
        }
        // LAYA_REPLAY=1 routes the forward through the per-shape graph
        // cache; when unset (replay == None) the direct path below runs
        // exactly as before.
        let key: ShapeKey = (seqs.len(), batch_len, n_markers);
        let (logits, act_logits) = match replay {
            None => self.model.forward(
                t2i(b_input_ids, batch_len, device),
                t2i(b_attention_mask, batch_len, device),
                t2i(b_marker_pos, n_markers, device),
                t2i(b_marker_mask, n_markers, device),
                t1i(b_qtype, device),
                &self.ropes,
            ),
            Some(engine) => engine.run(key, || StableInputs {
                input_ids: t2i(b_input_ids, batch_len, device),
                attention_mask: t2i(b_attention_mask, batch_len, device),
                marker_pos: t2i(b_marker_pos, n_markers, device),
                marker_mask: t2i(b_marker_mask, n_markers, device),
                qtype: t1i(b_qtype, device),
            }),
        };

        let logits = logits.to_data().as_slice::<f32>().unwrap().to_vec();
        let act_row = act_logits.to_data().as_slice::<f32>().unwrap().to_vec();
        let act = Calibration::softmax(&act_row[..2]);

        let mut answers = Vec::new();
        for (r, q) in questions.iter().enumerate() {
            let k = seqs[r].1.len();
            let bucket = Calibration::temp_bucket(q.qtype, k);
            let temp = self
                .temperature_by_options
                .iter()
                .find(|(key, _)| *key == bucket)
                .map(|&(_, t)| t)
                .unwrap_or(self.temperature[q.qtype]);
            let z: Vec<f32> = (0..k).map(|c| logits[r * n_markers + c]).collect();
            let p = Calibration::probabilities(&z, k, temp);
            let answer = match q.qtype {
                0 => Answer::Choice {
                    index: Calibration::arg_max(&p),
                    probabilities: p.iter().map(|v| Calibration::round4(*v) as f32).collect(),
                    confidence: Calibration::round4(Calibration::answer_confidence(&p, k)) as f32,
                },
                1 => Answer::Score {
                    score: Calibration::round4(Calibration::expected_score(&p)) as f32,
                    probabilities: p.iter().map(|v| Calibration::round4(*v) as f32).collect(),
                    confidence: Calibration::round4(Calibration::answer_confidence(&p, k)) as f32,
                },
                _ => Answer::Noul {
                    prob: Calibration::round4(p[1]) as f32,
                },
            };
            let _ = act;
            answers.push(answer);
        }
        answers
    }


    /// Batched variant for the server: each item carries its OWN state and
    /// questions; all rows are collated into ONE forward pass. Returns one
    /// Vec<Answer> per input item, in order.
    pub fn system_one_many(
        &self,
        device: &Device,
        items: &[(Value, Vec<QuestionInput>)],
        replay: Option<&mut ReplayEngine<'_>>,
    ) -> Vec<Vec<Answer>> {
        let rows: Vec<(String, &QuestionInput)> = items
            .iter()
            .flat_map(|(state, qs)| {
                let s = PythonJson::state(state);
                qs.iter().map(move |q| (s.clone(), q)).collect::<Vec<_>>()
            })
            .collect();
        let flat = self.run_rows(device, &rows, replay);
        let mut out = Vec::with_capacity(items.len());
        let mut at = 0;
        for (_, qs) in items {
            let n = qs.len();
            out.push(flat[at..at + n].to_vec());
            at += n;
        }
        out
    }

    /// Row-level core shared by system_one / system_one_many.
    fn run_rows(
        &self,
        device: &Device,
        rows: &[(String, &QuestionInput)],
        replay: Option<&mut ReplayEngine<'_>>,
    ) -> Vec<Answer> {
        let mut seqs = Vec::new();
        for (state_str, q) in rows {
            let (ids, markers) = build_sequence(
                &self.tok,
                state_str,
                &Question { qtype: q.qtype, instructions: q.instructions.clone(), options: q.options.clone() },
                self.max_len,
                self.head_max_len,
            );
            seqs.push((ids, markers, q.qtype as i64));
        }
        let batch_len: usize = seqs.iter().map(|(ids, _, _)| ids.len()).max().unwrap();
        let n_markers: usize = seqs.iter().map(|(_, m, _)| m.len()).max().unwrap();
        let b = collate(&seqs, self.tok.pad_token_id, batch_len, n_markers);
        let Batch { input_ids: b_input_ids, attention_mask: b_attention_mask, marker_pos: b_marker_pos, marker_mask: b_marker_mask, qtype: b_qtype } = b;

        if let Ok(dir) = std::env::var("LAYA_DUMP_DIR") {
            let _ = std::fs::create_dir_all(&dir);
            let rows: Vec<String> = b_input_ids
                .iter()
                .map(|r| r.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","))
                .collect();
            let _ = std::fs::write(format!("{dir}/all_rows.txt"), rows.join("
"));
        }
        // LAYA_REPLAY=1 routes the forward through the per-shape graph
        // cache; when unset (replay == None) the direct path below runs
        // exactly as before.
        let key: ShapeKey = (seqs.len(), batch_len, n_markers);
        let (logits, act_logits) = match replay {
            None => self.model.forward(
                t2i(b_input_ids, batch_len, device),
                t2i(b_attention_mask, batch_len, device),
                t2i(b_marker_pos, n_markers, device),
                t2i(b_marker_mask, n_markers, device),
                t1i(b_qtype, device),
                &self.ropes,
            ),
            Some(engine) => engine.run(key, || StableInputs {
                input_ids: t2i(b_input_ids, batch_len, device),
                attention_mask: t2i(b_attention_mask, batch_len, device),
                marker_pos: t2i(b_marker_pos, n_markers, device),
                marker_mask: t2i(b_marker_mask, n_markers, device),
                qtype: t1i(b_qtype, device),
            }),
        };

        let logits = logits.to_data().as_slice::<f32>().unwrap().to_vec();
        let act_row = act_logits.to_data().as_slice::<f32>().unwrap().to_vec();
        let act = Calibration::softmax(&act_row[..2]);

        let mut answers = Vec::new();
        for (r, (_, q)) in rows.iter().enumerate() {
            let k = seqs[r].1.len();
            let bucket = Calibration::temp_bucket(q.qtype, k);
            let temp = self
                .temperature_by_options
                .iter()
                .find(|(key, _)| *key == bucket)
                .map(|&(_, t)| t)
                .unwrap_or(self.temperature[q.qtype]);
            let z: Vec<f32> = (0..k).map(|c| logits[r * n_markers + c]).collect();
            let p = Calibration::probabilities(&z, k, temp);
            let answer = match q.qtype {
                0 => Answer::Choice {
                    index: Calibration::arg_max(&p),
                    probabilities: p.iter().map(|v| Calibration::round4(*v) as f32).collect(),
                    confidence: Calibration::round4(Calibration::answer_confidence(&p, k)) as f32,
                },
                1 => Answer::Score {
                    score: Calibration::round4(Calibration::expected_score(&p)) as f32,
                    probabilities: p.iter().map(|v| Calibration::round4(*v) as f32).collect(),
                    confidence: Calibration::round4(Calibration::answer_confidence(&p, k)) as f32,
                },
                _ => Answer::Noul {
                    prob: Calibration::round4(p[1]) as f32,
                },
            };
            let _ = act;
            answers.push(answer);
        }

        answers
    }
}



/// Opt-in graph capture/replay fast path (env LAYA_REPLAY=1).
///
/// For every distinct collated shape we capture the model forward ONCE with
/// burn's runtime graph API and re-run it with Graph::replay afterwards.
/// The graph has no explicit inputs: it reads exactly the device buffers the
/// captured closure touched, so the engine owns STABLE input tensors per shape
/// and refreshes them IN PLACE between replays (Tensor::inplace +
/// slice_assign); burn tensor clones alias the same buffer, so the closure
/// cloning them still reads the refreshed bytes.
///
/// The closures borrow the agent's model and ropes, so a ReplayEngine must not
/// outlive the RlAgent it was created from (the 'a lifetime enforces this).
pub struct ReplayEngine<'a> {
    device: Device,
    model: &'a Laya,
    ropes: &'a Ropes,
    entries: HashMap<ShapeKey, ReplayEntry<'a>>,
    /// Set when graph capture failed once (e.g. a host read mid-forward is
    /// rejected inside a hardware capture window); further calls then use the
    /// direct forward so results stay correct.
    capture_disabled: bool,
}

/// The stable inputs shared between the engine (which refreshes them) and the
/// captured closure (which clones them per invocation). The per-invocation
/// clones are transient: between replays the slot holds the only tensor
/// handle, so an in-place refresh writes the same buffer the captured graph
/// reads.
type SharedInputs = std::rc::Rc<std::cell::RefCell<StableInputs>>;

/// (rows, padded seq len, padded marker slots) identifying a collated shape.
type ShapeKey = (usize, usize, usize);

type ForwardOutput = (Tensor<2>, Tensor<2>);

/// Stable device input buffers for one shape, held across replays.
#[derive(Clone)]
struct StableInputs {
    input_ids: Tensor<2, Int>,
    attention_mask: Tensor<2, Int>,
    marker_pos: Tensor<2, Int>,
    marker_mask: Tensor<2, Int>,
    qtype: Tensor<1, Int>,
}

impl StableInputs {
    /// Overwrite every stable buffer in place with a freshly collated batch.
    /// Shapes match by construction of the ShapeKey lookup.
    fn refresh(&mut self, fresh: StableInputs) {
        let [r, c] = self.input_ids.dims();
        self.input_ids.inplace(|t| t.slice_assign([0..r, 0..c], fresh.input_ids));
        self.attention_mask.inplace(|t| t.slice_assign([0..r, 0..c], fresh.attention_mask));
        self.marker_pos.inplace(|t| t.slice_assign([0..r, 0..c], fresh.marker_pos));
        self.marker_mask.inplace(|t| t.slice_assign([0..r, 0..c], fresh.marker_mask));
        let n = self.qtype.dims()[0];
        self.qtype.inplace(|t| t.slice_assign([0..n], fresh.qtype));
    }
}

struct ReplayEntry<'a> {
    inputs: SharedInputs,
    graph: Graph<ForwardOutput, Box<dyn FnMut() -> ForwardOutput + 'a>>,
}

impl<'a> ReplayEngine<'a> {
    pub fn new(agent: &'a RlAgent, device: &Device) -> Self {
        Self {
            device: device.clone(),
            model: &agent.model,
            ropes: &agent.ropes,
            entries: HashMap::new(),
            capture_disabled: false,
        }
    }

    fn direct(&self, inputs: &StableInputs) -> ForwardOutput {
        self.model.forward(
            inputs.input_ids.clone(),
            inputs.attention_mask.clone(),
            inputs.marker_pos.clone(),
            inputs.marker_mask.clone(),
            inputs.qtype.clone(),
            self.ropes,
        )
    }

    /// Run the forward for 'key': capture on first encounter (capture itself
    /// warms up the kernels), refresh + replay afterwards.
    fn run(&mut self, key: ShapeKey, fresh: impl FnOnce() -> StableInputs) -> ForwardOutput {
        if self.capture_disabled {
            return self.direct(&fresh());
        }
        if let Some(entry) = self.entries.get_mut(&key) {
            {
                // Refresh the stable buffers in place. Scoped: on software
                // replay the closure re-borrows the slot, so the RefCell
                // borrow must be released before replay.
                entry.inputs.borrow_mut().refresh(fresh());
            }
            // SAFETY: replaying is sound here because the three clauses of
            // Graph::replay's contract all hold for this engine:
            // - Liveness: the only tensors the captured closure read are the
            //   stable input buffers, owned by entry.inputs (the Rc slot
            //   shares them with the closure, whose per-invocation clones are
            //   dropped before control returns here), and its own
            //   intermediates and outputs, owned by the graph itself. Both
            //   live exactly as long as the graph, inside this entry.
            // - No concurrent use: this engine, the agent, and its device are
            //   used single-threaded here; no other stream or thread reads or
            //   writes these tensors while a replay executes.
            // - Same-stream refreshes: the in-place refresh write above and
            //   the to_data() read the caller performs after this returns are
            //   issued on the same device/stream the graph was captured on,
            //   so they are ordered before/after the replay rather than
            //   racing it with stale or torn data. Because the slot held the
            //   only handle at refresh time, Tensor::inplace wrote the very
            //   buffers the recorded graph reads.
            let (logits, act_logits) = unsafe { entry.graph.replay() };
            (logits.clone(), act_logits.clone())
        } else {
            let inputs: SharedInputs = std::rc::Rc::new(std::cell::RefCell::new(fresh()));
            // Each invocation clones the stable tensors out of the slot
            // (aliasing the same buffers); the clones are consumed by forward
            // and dropped, so between invocations the slot holds the only
            // handle — the precondition for the in-place refresh above, and
            // for software replay (flex/WGPU/Metal re-run the closure) it
            // always reads the current contents.
            let slot = inputs.clone();
            let model = self.model;
            let ropes = self.ropes;
            // The forward still host-reads a mask mid-graph (its removal is in
            // flight elsewhere). Hardware capture windows (the wgpu/Metal
            // runtime selected by Device::default() here) record dispatches
            // only, so that read is rejected and the forward's unwrap()
            // panics — but ONLY during the capture run; the warmup runs read
            // normally. Catch the panic INSIDE the closure and return a dummy
            // output so `capture` still reaches graph_stop_capture and the
            // device leaves capture mode cleanly. A panicked-out closure
            // would skip the stop and poison the device for the rest of the
            // process. If the flag is set afterwards, discard the (useless)
            // graph and fall back to direct execution; results are identical
            // either way, so LAYA_REPLAY stays correct independently of the
            // model.rs change.
            let read_failed = std::rc::Rc::new(std::cell::Cell::new(false));
            let flag = read_failed.clone();
            // Pre-allocated dummy outputs for the failure path: allocating
            // fresh memory inside an open capture window is illegal, so the
            // closure only clones these handles (burn clones alias buffers).
            let dummy = (
                Tensor::<1>::from_floats([0.0f32], &self.device).reshape([1, 1]),
                Tensor::<1>::from_floats([0.0f32], &self.device).reshape([1, 1]),
            );
            let closure: Box<dyn FnMut() -> ForwardOutput + 'a> = Box::new(move || {
                let inputs = slot.borrow().clone();
                let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    model.forward(
                        inputs.input_ids,
                        inputs.attention_mask,
                        inputs.marker_pos,
                        inputs.marker_mask,
                        inputs.qtype,
                        ropes,
                    )
                }));
                match attempt {
                    Ok(out) => out,
                    Err(e) => {
                        let msg = e
                            .downcast_ref::<String>()
                            .cloned()
                            .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                            .unwrap_or_else(|| "<non-string panic>".to_string());
                        eprintln!("CAPTURE-PANIC: {msg}");
                        flag.set(true);
                        (dummy.0.clone(), dummy.1.clone())
                    }
                }
            });
            // Silence the default panic hook for the expected mid-capture
            // read failure above; the read_failed flag reports it cleanly.
            let prev_hook = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            let graph = capture(&self.device, closure);
            std::panic::set_hook(prev_hook);
            if read_failed.get() {
                eprintln!(
                    "laya: graph capture failed (host read mid-forward); \
                     falling back to direct execution"
                );
                self.capture_disabled = true;
                return self.direct(&inputs.borrow());
            }
            let entry = ReplayEntry { inputs, graph };
            // The capture run already produced the output for THIS batch.
            let (logits, act_logits) = entry.graph.output();
            let out = (logits.clone(), act_logits.clone());
            self.entries.insert(key, entry);
            out
        }
    }
}

fn t2i(rows: Vec<Vec<i64>>, cols: usize, device: &Device) -> Tensor<2, Int> {
    let n_rows = rows.len();
    let flat: Vec<i64> = rows.into_iter().flatten().collect();
    Tensor::<2, Int>::from_data(burn::tensor::TensorData::new(flat, [n_rows, cols]), device)
}

fn t1i(v: Vec<i64>, device: &Device) -> Tensor<1, Int> {
    let n = v.len();
    Tensor::<1, Int>::from_data(burn::tensor::TensorData::new(v, [n]), device)
}

#[cfg(test)]
mod replay_tests {
    use super::*;
    use burn::tensor::TensorData;

    /// The replay contract depends on the shared-slot pattern: the closure
    /// clones the stable tensors per invocation and drops the clones, so at
    /// refresh time the slot holds the only handle and Tensor::inplace
    /// rewrites the same buffer the captured graph reads. A RETAINED clone
    /// (held across the refresh) would break that: with another handle
    /// alive, inplace+slice_assign swaps in a fresh buffer and the replay
    /// would read stale bytes.
    #[test]
    fn refresh_through_shared_slot_is_visible_to_next_invocation() {
        let device = Device::default();
        let slot: std::rc::Rc<std::cell::RefCell<StableInputs>> = std::rc::Rc::new(std::cell::RefCell::new(StableInputs {
            input_ids: Tensor::<2, Int>::from_data(TensorData::new(vec![1i64, 2, 3, 4], [2, 2]), &device),
            attention_mask: Tensor::<2, Int>::from_data(TensorData::new(vec![1i64, 1, 1, 1], [2, 2]), &device),
            marker_pos: Tensor::<2, Int>::from_data(TensorData::new(vec![0i64, 1, 0, 1], [2, 2]), &device),
            marker_mask: Tensor::<2, Int>::from_data(TensorData::new(vec![1i64, 1, 1, 1], [2, 2]), &device),
            qtype: Tensor::<1, Int>::from_data(TensorData::new(vec![0i64, 0], [2]), &device),
        }));
        let fresh = StableInputs {
            input_ids: Tensor::<2, Int>::from_data(TensorData::new(vec![9i64, 8, 7, 6], [2, 2]), &device),
            attention_mask: Tensor::<2, Int>::from_data(TensorData::new(vec![0i64, 0, 0, 0], [2, 2]), &device),
            marker_pos: Tensor::<2, Int>::from_data(TensorData::new(vec![2i64, 3, 2, 3], [2, 2]), &device),
            marker_mask: Tensor::<2, Int>::from_data(TensorData::new(vec![0i64, 0, 0, 0], [2, 2]), &device),
            qtype: Tensor::<1, Int>::from_data(TensorData::new(vec![2i64, 2], [2]), &device),
        };
        // Invocation-style transient clone, immediately dropped.
        {
            let view = slot.borrow().clone();
            assert_eq!(view.input_ids.to_data().as_slice::<i32>().unwrap(), &[1, 2, 3, 4]);
        }
        slot.borrow_mut().refresh(fresh);
        // The next invocation must observe the refreshed bytes.
        let view = slot.borrow().clone();
        assert_eq!(view.input_ids.to_data().as_slice::<i32>().unwrap(), &[9, 8, 7, 6]);
        assert_eq!(view.qtype.to_data().as_slice::<i32>().unwrap(), &[2, 2]);
    }
}

fn collate(seqs: &[(Vec<i64>, Vec<usize>, i64)], pad: i64, batch_len: usize, n_markers: usize) -> Batch {
    let mut input_ids = Vec::new();
    let mut attention_mask = Vec::new();
    let mut marker_pos = Vec::new();
    let mut marker_mask = Vec::new();
    let mut qtype = Vec::new();
    for (ids, markers, qt) in seqs {
        let mut row = ids.clone();
        let mut amask = vec![1i64; ids.len()];
        while row.len() < batch_len {
            row.push(pad);
            amask.push(0);
        }
        input_ids.push(row);
        attention_mask.push(amask);
        let mut mpos = Vec::new();
        let mut mmask = Vec::new();
        for i in 0..n_markers {
            if i < markers.len() {
                mpos.push(markers[i] as i64);
                mmask.push(1);
            } else {
                mpos.push(-1);
                mmask.push(0);
            }
        }
        marker_pos.push(mpos);
        marker_mask.push(mmask);
        qtype.push(*qt);
    }
    Batch { input_ids, attention_mask, marker_pos, marker_mask, qtype }
}
