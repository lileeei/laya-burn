//! Inference pipeline: port of RLAgent.system_one from rl_agent_api.py.

use burn::tensor::{Device, Int, Tensor};
use serde_json::Value;

use crate::model::{EncoderConfig, Laya, Ropes};
use crate::sequence::{build_sequence, Batch, Question};
use crate::tokenizer::LayaTokenizer;
use crate::weights::Weights;

pub struct RlAgent {
    pub model: Laya,
    pub ropes: Ropes,
    pub tok: LayaTokenizer,
    pub max_len: usize,
    pub head_max_len: usize,
    pub temperature: [f32; 3],
    pub temperature_by_options: Vec<(String, f32)>,
}

/// One question in the public API shape.
pub struct QuestionInput {
    pub qtype: usize, // 0=choice 1=score 2=noul
    pub instructions: String,
    pub options: Vec<String>,
}

#[derive(Debug)]
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

impl RlAgent {
    pub fn load(model_dir: &str, device: &Device) -> Self {
        let w = Weights::open(&format!("{model_dir}/model.safetensors"));
        let cfg: Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{model_dir}/rl_agent_config.json")).unwrap())
                .unwrap();
        let max_len = cfg["max_len"].as_u64().unwrap() as usize;
        let head_max_len = cfg["head_max_len"].as_u64().unwrap() as usize;
        let temperature = [
            cfg["temperature"][0].as_f64().unwrap() as f32,
            cfg["temperature"][1].as_f64().unwrap() as f32,
            cfg["temperature"][2].as_f64().unwrap() as f32,
        ];
        let temperature_by_options = cfg["temperature_by_options"]
            .as_object()
            .map(|m| {
                m.iter()
                    .map(|(k, v)| (k.clone(), v.as_f64().unwrap() as f32))
                    .collect()
            })
            .unwrap_or_default();

        let ecfg = EncoderConfig::laya();
        let model = Laya::load(&w, &ecfg, 2, device);
        let ropes = Ropes::new(&ecfg, device);
        let tok = LayaTokenizer::load(&format!("{model_dir}/tokenizer/tokenizer.json"));
        Self { model, ropes, tok, max_len, head_max_len, temperature, temperature_by_options }
    }

    /// state + questions -> typed answers, one forward pass for the whole batch.
    pub fn system_one(&self, device: &Device, state: &Value, questions: &[QuestionInput]) -> Vec<Answer> {
        let state_str = match state {
            Value::String(s) => s.clone(),
            v => serde_json::to_string(v).unwrap(),
        };

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

        if let Ok(dir) = std::env::var("LAYA_DUMP_DIR") {
            let _ = std::fs::create_dir_all(&dir);
            let rows: Vec<String> = b
                .input_ids
                .iter()
                .map(|r| r.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","))
                .collect();
            let _ = std::fs::write(format!("{dir}/all_rows.txt"), rows.join("
"));
        }
        let (logits, act_logits) = self.model.forward(
            t2i(b.input_ids, batch_len, device),
            t2i(b.attention_mask, batch_len, device),
            t2i(b.marker_pos, n_markers, device),
            t2i(b.marker_mask, n_markers, device),
            t1i(b.qtype, device),
            &self.ropes,
        );

        let logits = logits.to_data().as_slice::<f32>().unwrap().to_vec();
        let act = softmax_host(
            &act_logits.to_data().as_slice::<f32>().unwrap().to_vec()[..2.min(questions.len()) * 2],
        );

        let mut answers = Vec::new();
        for (r, q) in questions.iter().enumerate() {
            let k = seqs[r].1.len();
            let bucket = temp_bucket(q.qtype, k);
            let temp = self
                .temperature_by_options
                .iter()
                .find(|(key, _)| *key == bucket)
                .map(|&(_, t)| t)
                .unwrap_or(self.temperature[q.qtype]);
            let z: Vec<f32> = (0..k).map(|c| logits[r * n_markers + c] / temp).collect();
            let p = softmax_host(&z);
            let answer = match q.qtype {
                0 => Answer::Choice {
                    index: p
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(i, _)| i)
                        .unwrap(),
                    probabilities: round4(&p),
                    confidence: (confidence_from_probs(&p) * 10000.0).round() / 10000.0,
                },
                1 => Answer::Score {
                    score: (p.iter().enumerate().map(|(i, &v)| i as f32 * v).sum::<f32>()
                        * 10000.0)
                        .round()
                        / 10000.0,
                    probabilities: round4(&p),
                    confidence: (confidence_from_probs(&p) * 10000.0).round() / 10000.0,
                },
                _ => Answer::Noul {
                    prob: (p[1] * 10000.0).round() / 10000.0,
                },
            };
            let _ = act;
            answers.push(answer);
        }
        answers
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

fn round4(p: &[f32]) -> Vec<f32> {
    p.iter().map(|v| (v * 10000.0).round() / 10000.0).collect()
}

fn softmax_host(z: &[f32]) -> Vec<f32> {
    let zmax = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = z.iter().map(|v| (v - zmax).exp()).collect();
    let s: f32 = exp.iter().sum();
    exp.into_iter().map(|e| e / s).collect()
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
