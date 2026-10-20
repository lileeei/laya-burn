//! Laya decision model: ModernBERT encoder + typed-decision head, ported from the
//! reference PyTorch implementation (rl_common.py) to Burn.

use burn::module::{Module, Param, ParamId, ParameterValue};
use burn::nn::{Embedding, LayerNorm, LayerNormConfig, Linear};
use burn::tensor::activation::{gelu, relu, softmax};
use burn::tensor::{Bool, Device, Int, Tensor, module};

use crate::weights::Weights;

/// Write a float tensor as little-endian f32 with a .shape sidecar when LAYA_DUMP_DIR is set.
pub fn dump_tensor<const D: usize>(name: &str, t: &Tensor<D>) {
    let Ok(dir) = std::env::var("LAYA_DUMP_DIR") else { return };
    let _ = std::fs::create_dir_all(&dir);
    let vals = t.to_data().as_slice::<f32>().unwrap_or(&[]).to_vec();
    let mut bytes = Vec::with_capacity(vals.len() * 4);
    for v in vals {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let _ = std::fs::write(format!("{dir}/{name}.f32"), bytes);
    let _ = std::fs::write(format!("{dir}/{name}.shape"), format!("{:?}", t.dims()));
}

/// Write an int tensor as text lines when LAYA_DUMP_DIR is set.
pub fn dump_ids<const D: usize>(name: &str, t: &Tensor<D, Int>) {
    let Ok(dir) = std::env::var("LAYA_DUMP_DIR") else { return };
    let _ = std::fs::create_dir_all(&dir);
    let text = match t.to_data().as_slice::<i64>() {
        Ok(v) => v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","),
        Err(_) => t
            .to_data()
            .as_slice::<i32>()
            .unwrap()
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(","),
    };
    let _ = std::fs::write(format!("{dir}/{name}.txt"), text);
}

/// Encoder (ModernBERT-large) hyperparameters from encoder/config.json.
pub struct EncoderConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub window_size: usize,
    pub global_every: usize,
    pub max_sequence_length: usize,
    pub rope_theta_full: f32,
    pub rope_theta_sliding: f32,
    pub epsilon: f64,
}

impl EncoderConfig {
    pub fn laya() -> Self {
        Self {
            hidden_size: 1024,
            num_layers: 28,
            num_heads: 16,
            intermediate_size: 2624,
            window_size: 128,
            global_every: 3,
            max_sequence_length: 1088,
            rope_theta_full: 160_000.0,
            rope_theta_sliding: 10_000.0,
            epsilon: 1e-5,
        }
    }
}

fn param<T: ParameterValue>(t: T) -> Param<T> {
    Param::initialized(ParamId::new(), t)
}

/// Linear from checkpoint tensors. Checkpoint stores PyTorch [out, in];
/// Burn Linear expects [in, out].
fn linear_from(w: &Weights, wt: Tensor<2>, bias: Option<Tensor<1>>) -> Linear {
    let _ = w;
    Linear {
        weight: param(wt.transpose()),
        bias: bias.map(param),
    }
}

fn linear(w: &Weights, prefix: &str, device: &Device) -> Linear {
    let wt = w.t2(&format!("{prefix}.weight"), device);
    let b = w
        .has(&format!("{prefix}.bias"))
        .then(|| w.t1(&format!("{prefix}.bias"), device));
    linear_from(w, wt, b)
}

/// LayerNorm without bias (encoder norms).
fn norm_no_bias(w: &Weights, prefix: &str, cfg: &EncoderConfig, device: &Device) -> LayerNorm {
    let mut ln = LayerNormConfig::new(cfg.hidden_size)
        .with_epsilon(cfg.epsilon)
        .with_bias(false)
        .init(device);
    ln.gamma = param(w.t1(&format!("{prefix}.weight"), device));
    ln
}

/// LayerNorm with bias (head and scorer norms).
fn norm_bias(wt: Tensor<1>, b: Tensor<1>, d_model: usize, device: &Device) -> LayerNorm {
    let mut ln = LayerNormConfig::new(d_model).init(device);
    ln.gamma = param(wt);
    ln.beta = Some(param(b));
    ln
}

/// One ModernBERT encoder layer. Layer 0 carries no attn/mlp norms.
#[derive(Module, Debug)]
pub struct EncoderLayer {
    pub attn_norm: Option<LayerNorm>,
    pub wqkv: Linear,
    pub attn_out: Linear,
    pub mlp_norm: Option<LayerNorm>,
    pub wi: Linear,
    pub mlp_out: Linear,
    pub is_global: bool,
}

/// One torch nn.TransformerEncoderLayer (norm_first=True, relu) of the head.
#[derive(Module, Debug)]
pub struct HeadLayer {
    pub norm1: LayerNorm,
    pub in_proj: Linear,
    pub out_proj: Linear,
    pub norm2: LayerNorm,
    pub linear1: Linear,
    pub linear2: Linear,
}

/// The Laya DecisionModel.
#[derive(Module, Debug)]
pub struct Laya {
    // encoder
    pub tok_embeddings: Embedding,
    pub emb_norm: LayerNorm,
    pub layers: Vec<EncoderLayer>,
    pub final_norm: LayerNorm,
    // typed decision head
    pub head_layers: Vec<HeadLayer>,
    pub type_emb: Embedding,
    pub scorer_norm: LayerNorm,
    pub scorer_fc1: Linear,
    pub scorer_fc2: Linear,
    pub act_fc1: Linear,
    pub act_fc2: Linear,
    // config
    pub window_size: usize,
    pub num_heads: usize,
}

/// Llama-style rotate_half RoPE (what HF ModernBERT uses): the first and second
/// halves of the head dim rotate as (q1 cos - q2 sin, q2 cos + q1 sin).
#[derive(Debug)]
pub struct HalfSplitRope {
    cos: Tensor<2>, // [max_seq, head_dim/2]
    sin: Tensor<2>,
    cos_pair: Tensor<2>, // [max_seq, head_dim] interleaved duplicates
    sin_pair: Tensor<2>,
    head_dim: usize,
}

impl HalfSplitRope {
    pub fn new(max_seq: usize, head_dim: usize, theta_base: f32, device: &Device) -> Self {
        let half = head_dim / 2;
        // inv_freq_i = 1 / theta^(2i/head_dim), i in [0, half)
        let mut inv = Vec::with_capacity(half);
        for i in 0..half {
            inv.push(1.0 / theta_base.powf((2.0 * i as f32) / head_dim as f32));
        }
        let mut cos = Vec::with_capacity(max_seq * half);
        let mut sin = Vec::with_capacity(max_seq * half);
        for pos in 0..max_seq {
            for &f in &inv {
                cos.push((pos as f32 * f).cos());
                sin.push((pos as f32 * f).sin());
            }
        }
        let mut cos_pair = Vec::with_capacity(max_seq * head_dim);
        let mut sin_pair = Vec::with_capacity(max_seq * head_dim);
        for pos in 0..max_seq {
            for i in 0..half {
                cos_pair.push(cos[pos * half + i]);
                cos_pair.push(cos[pos * half + i]);
                sin_pair.push(sin[pos * half + i]);
                sin_pair.push(sin[pos * half + i]);
            }
        }
        Self {
            cos: Tensor::<1>::from_floats(cos.as_slice(), device).reshape([max_seq, half]),
            sin: Tensor::<1>::from_floats(sin.as_slice(), device).reshape([max_seq, half]),
            cos_pair: Tensor::<1>::from_floats(cos_pair.as_slice(), device).reshape([max_seq, head_dim]),
            sin_pair: Tensor::<1>::from_floats(sin_pair.as_slice(), device).reshape([max_seq, head_dim]),
            head_dim,
        }
    }

    /// Interleaved (GPT-J style) variant: pairs (x[2i], x[2i+1]) rotate.
    /// Selected with env LAYA_ROPE=interleaved; default is half-split (Llama style).
    pub fn forward_interleaved<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let seq = x.dims()[D - 2];
        let shape = x.shape();
        let flat: Tensor<3> = x.reshape([-1isize, seq as isize, self.head_dim as isize]);
        let [n, _, _] = flat.dims();
        let half = self.head_dim / 2;
        let cos = self
            .cos_pair
            .clone()
            .slice([0..seq, 0..self.head_dim])
            .reshape([1, seq, self.head_dim]);
        let sin = self
            .sin_pair
            .clone()
            .slice([0..seq, 0..self.head_dim])
            .reshape([1, seq, self.head_dim]);
        // rotate_interleave: [-x2, x1] per pair
        let x1 = flat.clone().slice([0..n, 0..seq, 0..half]);
        let x2 = flat.clone().slice([0..n, 0..seq, half..self.head_dim]);
        let neg_x2 = x2.clone() * -1.0;
        let rot = Tensor::cat(vec![neg_x2, x1], 2);
        // interleave columns back: reorder [a1,b1,a2,b2,...] from [a...,b...]
        // build index permutation
        let mut perm = Vec::with_capacity(self.head_dim);
        for i in 0..half {
            perm.push(i as i64);
            perm.push((half + i) as i64);
        }
        use burn::tensor::Int;
        let idx = Tensor::<1, Int>::from_data(burn::tensor::TensorData::new(perm, [self.head_dim]), &flat.device());
        let rot = rot.gather(2, idx.reshape([1, 1, self.head_dim]).repeat_dim(0, n).repeat_dim(1, seq));
        (flat * cos + rot * sin).reshape(shape)
    }

    /// x: [B, H, S, head_dim] (also works for other leading dims).
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let seq = x.dims()[D - 2];
        let half = self.head_dim / 2;
        let device = x.device();
        let shape = x.shape();
        let flat: Tensor<3> = x.reshape([-1isize, seq as isize, self.head_dim as isize]);
        let [n, _, _] = flat.dims();
        let q1 = flat.clone().slice([0..n, 0..seq, 0..half]);
        let q2 = flat.clone().slice([0..n, 0..seq, half..self.head_dim]);
        let cos = self.cos.clone().slice([0..seq, 0..half]).reshape([1, seq, half]);
        let sin = self.sin.clone().slice([0..seq, 0..half]).reshape([1, seq, half]);
        let _ = device;
        let out = Tensor::cat(
            vec![
                q1.clone() * cos.clone() - q2.clone() * sin.clone(),
                q2 * cos + q1 * sin,
            ],
            2,
        );
        out.reshape(shape)
    }
}


/// The two RoPE caches used by the encoder (global vs sliding layers).
pub struct Ropes {
    pub full: HalfSplitRope,
    pub sliding: HalfSplitRope,
    /// Precomputed input-independent masks over max_sequence_length:
    /// bool [1,1,S,S] (true = masked by window) and additive float twin.
    pub window_bool: Tensor<4, Bool>,
    pub window_add: Tensor<4>,
}

impl Ropes {
    pub fn new(ecfg: &EncoderConfig, device: &Device) -> Self {
        let hd = ecfg.hidden_size / ecfg.num_heads;
        let ms = ecfg.max_sequence_length;
        let half_win = ecfg.window_size / 2;
        let mut mb = Vec::with_capacity(ms * ms);
        let mut mf = Vec::with_capacity(ms * ms);
        for i in 0..ms {
            for j in 0..ms {
                let masked = (i as isize - j as isize).unsigned_abs() > half_win;
                mb.push(masked);
                mf.push(if masked { -1.0e9 } else { 0.0 });
            }
        }
        Self {
            full: HalfSplitRope::new(ecfg.max_sequence_length, hd, ecfg.rope_theta_full, device),
            sliding: HalfSplitRope::new(ecfg.max_sequence_length, hd, ecfg.rope_theta_sliding, device),
            window_bool: Tensor::<2, Bool>::from_data(
                burn::tensor::TensorData::new(mb, [ms, ms]),
                device,
            )
            .reshape([1, 1, ms, ms]),
            window_add: Tensor::<1>::from_floats(mf.as_slice(), device)
                .reshape([1, 1, ms, ms]),
        }
    }
}

/// Attention over one layer's q/k/v with additive mask.
enum AttnMask {
    Manual(Tensor<4>),      // additive float mask (fallback comparison path)
    Bool(Tensor<4, Bool>),  // boolean mask for the fused module::attention kernel
    None,
}

fn attention(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    n_heads: usize,
    mask: &AttnMask,
) -> Tensor<3> {
    let [batch, _h, seq, _d] = q.dims();
    let head_dim = q.dims()[3];
    if std::env::var("LAYA_ATTN").as_deref() != Ok("manual") {
        let (m, bias) = match mask {
            AttnMask::Bool(b) => (Some(b.clone()), None),
            AttnMask::Manual(add) => (None, Some(add.clone())), // additive as bias: same softmax result
            AttnMask::None => (None, None),
        };
        let out = module::attention(
            q.clone(),
            k.clone(),
            v.clone(),
            m,
            bias,
            burn::tensor::ops::AttentionModuleOptions::default(),
        );
        return out.swap_dims(1, 2).reshape([batch, seq, n_heads * head_dim]);
    }
    let add = match mask {
        AttnMask::Manual(add) => add.clone(),
        AttnMask::Bool(_) => panic!("bool mask on manual attention path"),
        AttnMask::None => {
            let z = Tensor::<1>::from_floats([0.0], &q.device());
            z.reshape([1, 1, 1, 1]).repeat_dim(2, seq).repeat_dim(3, seq)
        }
    };
    let scores = q.matmul(k.transpose()) / (head_dim as f32).sqrt() + add;
    softmax(scores, 3)
        .matmul(v)
        .swap_dims(1, 2)
        .reshape([batch, seq, n_heads * head_dim])
}

/// Split fused qkv [B,S,3D] into head-laid-out [B,H,S,hd] triple.
fn split_qkv(qkv: Tensor<3>, d: usize, n_heads: usize) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
    let [batch, seq, _] = qkv.dims();
    let head_dim = d / n_heads;
    let split = |idx: usize| {
        qkv.clone()
            .slice([0..batch, 0..seq, idx * d..(idx + 1) * d])
            .reshape([batch, seq, n_heads, head_dim])
            .swap_dims(1, 2)
    };
    (split(0), split(1), split(2))
}

impl Laya {
    /// Load the model from checkpoint tensors (typed-decisions layout keys).
    pub fn load(w: &Weights, ecfg: &EncoderConfig, head_layers: usize, device: &Device) -> Self {
        let d = ecfg.hidden_size;
        let layers = (0..ecfg.num_layers)
            .map(|i| {
                let p = format!("encoder.layers.{i}");
                let first = i == 0;
                EncoderLayer {
                    attn_norm: (!first).then(|| norm_no_bias(w, &format!("{p}.attn_norm"), ecfg, device)),
                    wqkv: linear(w, &format!("{p}.attn.Wqkv"), device),
                    attn_out: linear(w, &format!("{p}.attn.Wo"), device),
                    mlp_norm: Some(norm_no_bias(w, &format!("{p}.mlp_norm"), ecfg, device)),
                    wi: linear(w, &format!("{p}.mlp.Wi"), device),
                    mlp_out: linear(w, &format!("{p}.mlp.Wo"), device),
                    is_global: i % ecfg.global_every == 0,
                }
            })
            .collect();

        let head = (0..head_layers)
            .map(|i| {
                let p = format!("head.layers.{i}");
                let in_proj = Linear {
                    weight: param(w.t2(&format!("{p}.self_attn.in_proj_weight"), device).transpose()),
                    bias: Some(param(w.t1(&format!("{p}.self_attn.in_proj_bias"), device))),
                };
                HeadLayer {
                    norm1: norm_bias(
                        w.t1(&format!("{p}.norm1.weight"), device),
                        w.t1(&format!("{p}.norm1.bias"), device),
                        d,
                        device,
                    ),
                    in_proj,
                    out_proj: linear(w, &format!("{p}.self_attn.out_proj"), device),
                    norm2: norm_bias(
                        w.t1(&format!("{p}.norm2.weight"), device),
                        w.t1(&format!("{p}.norm2.bias"), device),
                        d,
                        device,
                    ),
                    linear1: linear(w, &format!("{p}.linear1"), device),
                    linear2: linear(w, &format!("{p}.linear2"), device),
                }
            })
            .collect();

        let tok_w = w.t2("encoder.embeddings.tok_embeddings.weight", device);
        let type_w = w.t2("type_emb.weight", device);

        Self {
            tok_embeddings: Embedding { weight: param(tok_w) },
            emb_norm: norm_no_bias(w, "encoder.embeddings.norm", ecfg, device),
            layers,
            final_norm: norm_no_bias(w, "encoder.final_norm", ecfg, device),
            head_layers: head,
            type_emb: Embedding { weight: param(type_w) },
            scorer_norm: norm_bias(
                w.t1("scorer.0.weight", device),
                w.t1("scorer.0.bias", device),
                d,
                device,
            ),
            scorer_fc1: linear(w, "scorer.1", device),
            scorer_fc2: linear(w, "scorer.3", device),
            act_fc1: linear(w, "act_head.0", device),
            act_fc2: linear(w, "act_head.2", device),
            window_size: ecfg.window_size,
            num_heads: ecfg.num_heads,
        }
    }

    /// Full forward mirroring DecisionModel::forward.
    /// Returns per-option logits [B, M] (invalid options at -1e4) and act logits [B, 2].
    pub fn forward(
        &self,
        input_ids: Tensor<2, Int>,
        attention_mask: Tensor<2, Int>,
        marker_pos: Tensor<2, Int>,
        marker_mask: Tensor<2, Int>,
        qtype: Tensor<1, Int>,
        ropes: &Ropes,
    ) -> (Tensor<2>, Tensor<2>) {
        let device = input_ids.device();
        let [batch, seq] = input_ids.dims();
        let d = self.tok_embeddings.weight.dims()[1];
        let n_heads = self.num_heads;

        // Additive padding mask [B,1,1,S]: 0 where the key is a real token.
        let pad_add = {
            let m = attention_mask.clone().float().reshape([batch, 1, 1, seq]);
            (m - 1.0) * 1.0e9
        };

        let use_manual = std::env::var("LAYA_ATTN").as_deref() == Ok("manual");
        let pad_bool = attention_mask.clone().equal_scalar(0).reshape([batch, 1, 1, seq]);

        // Masks are identical for every layer of one batch: build once, reuse
        // across the 28 encoder layers (the sliding-window mask is the same
        // bool tensor for all windowed layers).
        // Device-side mask composition (no host readback: keeps the forward
        // capturable for graph replay). Window masks are input-independent and
        // precomputed in Ropes over max_sequence_length; pad comes from the
        // attention mask. bool_or needs matching shapes, so both sides are
        // materialized to [B,1,S,S].
        let window_add = ropes
            .window_add
            .clone()
            .slice([0..1, 0..1, 0..seq, 0..seq]);
        let window_bool = ropes
            .window_bool
            .clone()
            .slice([0..1, 0..1, 0..seq, 0..seq])
            .repeat_dim(0, batch);
        let pad_full = pad_bool
            .clone()
            .reshape([batch, 1, 1, seq])
            .repeat_dim(2, seq);
        let window_bool: Tensor<4, Bool> = window_bool.bool_or(pad_full);

        // ---- encoder ----
        let mut h = self.emb_norm.forward(self.tok_embeddings.forward(input_ids.clone()));
        dump_tensor("emb_out", &h.clone().slice([0..1, 0..seq, 0..d]).reshape([seq, d]));
        dump_ids("input_ids", &input_ids.clone().slice([0..1, 0..seq]));
        for (li, layer) in self.layers.iter().enumerate() {
            let _ = li;
            let x1 = match &layer.attn_norm {
                Some(n) => n.forward(h.clone()),
                None => h.clone(),
            };
            let (q, k, v) = split_qkv(layer.wqkv.forward(x1), d, n_heads);
            let rope = if layer.is_global { &ropes.full } else { &ropes.sliding };
            let interleaved = std::env::var("LAYA_ROPE").as_deref() == Ok("interleaved");
            let (q, k) = if interleaved {
                (rope.forward_interleaved(q), rope.forward_interleaved(k))
            } else {
                (rope.forward(q), rope.forward(k))
            };
            if li == 0 {
                {
            }
            }

            let amask = if use_manual {
                let mut mask = pad_add.clone();
                if !layer.is_global {
                    mask = mask + window_add.clone();
                }
                AttnMask::Manual(mask)
            } else if !layer.is_global {
                AttnMask::Bool(window_bool.clone())
            } else {
                AttnMask::Bool(pad_bool.clone())
            };
            let attn_res = attention(q, k, v, n_heads, &amask);
            let attn_mod_out = layer.attn_out.forward(attn_res);
            let h2 = h + attn_mod_out;

            let x2 = match &layer.mlp_norm {
                Some(n) => n.forward(h2.clone()),
                None => h2.clone(),
            };
            let wi = layer.wi.forward(x2); // [B,S,2*I]: input | gate halves
            let i_dim = wi.dims()[2] / 2;
            let a = wi.clone().slice([0..batch, 0..seq, 0..i_dim]);
            let b = wi.slice([0..batch, 0..seq, i_dim..2 * i_dim]);
            let mlp = if std::env::var("LAYA_MLP").as_deref() == Ok("swap") {
                a * gelu(b)
            } else {
                gelu(a) * b
            };
            h = h2 + layer.mlp_out.forward(mlp);
            if li == 0 || li == 1 || li == self.layers.len() - 1 {
                dump_tensor(&format!("enc_layer_{li}"), &h.clone().slice([0..1, 0..seq, 0..d]).reshape([seq, d]));
            }
        }
        let mut h = self.final_norm.forward(h);
        dump_tensor("encoder_out", &h.clone().slice([0..1, 0..seq, 0..d]).reshape([seq, d]));

        // ---- typed decision head ----
        h = h + self
            .type_emb
            .forward(qtype.reshape([batch, 1]))
            .reshape([batch, 1, d]);

        let pad_add3 = {
            let m = attention_mask.clone().float().reshape([batch, 1, 1, seq]);
            (m - 1.0) * 1.0e9
        };
        let pad_bool3 = attention_mask.equal_scalar(0).reshape([batch, 1, 1, seq]);
        for hl in &self.head_layers {
            let (q, k, v) = split_qkv(hl.in_proj.forward(hl.norm1.forward(h.clone())), d, n_heads);
            let amask3 = if use_manual {
                AttnMask::Manual(pad_add3.clone())
            } else {
                AttnMask::Bool(pad_bool3.clone())
            };
            let h2 = h + hl.out_proj.forward(attention(q, k, v, n_heads, &amask3));
            h = h2.clone() + hl.linear2.forward(relu(hl.linear1.forward(hl.norm2.forward(h2))));
        }

        dump_tensor("head_out", &h.clone().slice([0..1, 0..seq, 0..d]).reshape([seq, d]));

        // gather marker positions -> [B,M,d]
        let n_markers = marker_pos.dims()[1];
        let idx = marker_pos
            .clamp_min(0)
            .reshape([batch, n_markers, 1])
            .repeat_dim(2, d);
        let m = h.clone().gather(1, idx);

        // scorer: LayerNorm -> Linear -> GELU -> Linear -> [B,M]
        let mut logits = self
            .scorer_fc2
            .forward(gelu(self.scorer_fc1.forward(self.scorer_norm.forward(m))))
            .reshape([batch, n_markers]);
        logits = logits + (marker_mask.clone().float() - 1.0) * 1.0e4;

        // act head: pooled CLS + detached answer-distribution features
        let pooled = h.slice([0..batch, 0..1, 0..d]).reshape([batch, d]);
        let feats = act_features(&logits, &marker_mask, &device); // [B,4]
        let act_logits = self
            .act_fc2
            .forward(gelu(self.act_fc1.forward(Tensor::cat(vec![pooled, feats], 1))));

        let mut lg = logits.clone().slice([0..1, 0..n_markers]).reshape([1, n_markers]);
        dump_tensor("logits", &lg.clone());
        lg = lg.slice([0..1, 0..n_markers]).reshape([1, n_markers]);
        let _ = lg;
        let ag = act_logits.clone().slice([0..1, 0..2]);
        dump_tensor("act_logits", &ag);

        (logits, act_logits)
    }
}

/// Device-side [top1, top1-top2, normalized entropy, k/255] features.
///
/// Softmax is computed over ALL n marker columns instead of the per-row valid
/// subsets: invalid markers carry the additive -1e4 mask applied in forward(),
/// so their softmax mass is ~exp(-1e4) = 0 relative to typical logit scales.
/// That makes top-2 over all columns match the valid-subset top-2 (and match
/// the host version's top2 = 0.0 fallback when a row has a single valid
/// marker), and the entropy contribution of ~0-mass entries stays ~0 because
/// clamp_min(1e-9) bounds p*ln(p) at ~1e-4*ln(1e-9) there. No host readback.
fn act_features(logits: &Tensor<2>, marker_mask: &Tensor<2, Int>, device: &Device) -> Tensor<2> {
    let [batch, n] = logits.dims();
    let p = softmax(logits.clone(), 1);
    // Top-1 via max + argmax; top-2 by masking the FIRST maximal position
    // (argmax ties resolve to the lowest index, matching NumPy/torch topk
    // semantics for tied values). Avoids sort: the backend-default sort
    // implementation round-trips through the host, which breaks graph
    // capture on wgpu/cubecl backends.
    let (top1, top1_idx) = p.clone().max_dim_with_indices(1);
    let top2 = if n >= 2 {
        // Device-native one-hot via broadcast equality (Tensor::one_hot
        // round-trips through the host, which breaks graph capture).
        let js = Tensor::<1, Int>::arange(0..n as i64, device)
            .reshape([1, n])
            .repeat_dim(0, batch);
        let onehot = js
            .equal(top1_idx.clone().repeat_dim(1, n))
            .float();
        // p is in [0,1]; subtracting 2 at the first-max position pushes only
        // that position below every other probability (ties keep their value).
        (p.clone() - onehot * 2.0).max_dim(1)
    } else {
        Tensor::zeros([batch, 1], device)
    };
    // k = max(valid_marker_count, 2)
    let k = marker_mask.clone().float().sum_dim(1).clamp_min(2.0);
    let pc = p.clamp_min(1e-9);
    let ent = -(pc.clone() * pc.log()).sum_dim(1) / k.clone().log();
    Tensor::cat(vec![top1.clone(), top1 - top2, ent, k / 255.0], 1)
}
