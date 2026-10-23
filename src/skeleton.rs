//! Skeleton weights: build the Laya module tree from a geometry spec alone
//! (all-zero tensors), so a burnpack reload needs no safetensors file.

use std::collections::BTreeMap;

use burn::tensor::{Device, Tensor, TensorData};

use crate::weights::WeightSource;

/// Geometry sufficient to derive every checkpoint tensor shape AND the
/// encoder hyperparameters - the single source of truth a burnpack reload
/// needs. The trailing fields default to the shipped-checkpoint constants,
/// so sidecars written before they existed still parse identically.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SkeletonSpec {
    pub vocab: usize,
    pub hidden: usize,
    pub layers: usize,
    pub intermediate: usize,
    pub head_layers: usize,
    pub sliding_theta: f32,
    #[serde(default = "d_window")]
    pub window: usize,
    #[serde(default = "d_global_every")]
    pub global_every: usize,
    #[serde(default = "d_max_seq")]
    pub max_sequence_length: usize,
    #[serde(default = "d_theta_full")]
    pub rope_theta_full: f32,
    #[serde(default = "d_epsilon")]
    pub epsilon: f32,
}

fn d_window() -> usize { 128 }
fn d_global_every() -> usize { 3 }
fn d_max_seq() -> usize { 1088 }
fn d_theta_full() -> f32 { 160_000.0 }
fn d_epsilon() -> f32 { 1e-5 }

impl SkeletonSpec {
    /// The EncoderConfig this spec fully determines. The ONLY place that
    /// turns checkpoint geometry into an EncoderConfig: safetensors sniffing
    /// and sidecar parsing both go through here.
    pub fn encoder_config(&self) -> crate::model::EncoderConfig {
        crate::model::EncoderConfig {
            hidden_size: self.hidden,
            num_layers: self.layers,
            num_heads: self.hidden / 64,
            intermediate_size: self.intermediate,
            window_size: self.window,
            global_every: self.global_every,
            max_sequence_length: self.max_sequence_length,
            rope_theta_full: self.rope_theta_full,
            rope_theta_sliding: self.sliding_theta,
            epsilon: self.epsilon as f64,
        }
    }

    /// Spec from a live loaded model (the export side: sidecar writing).
    /// Thetas/window/max_seq are the shipped-checkpoint constants; sliding is
    /// 10_000 because every checkpoint was trained that way (verified by
    /// parity - see inference.rs), whatever its encoder config claims.
    pub fn from_model(model: &crate::model::Laya) -> Self {
        let (vocab, hidden) = {
            let d = model.tok_embeddings.weight.val().dims();
            (d[0], d[1])
        };
        Self {
            vocab,
            hidden,
            layers: model.layers.len(),
            // NB burn Linear stores [in, out]; Wi is [h, 2*intermediate].
            intermediate: model.layers[0].wi.weight.val().dims()[1] / 2,
            head_layers: model.head_layers.len(),
            sliding_theta: 10_000.0,
            window: model.window_size,
            global_every: d_global_every(),
            max_sequence_length: d_max_seq(),
            rope_theta_full: d_theta_full(),
            epsilon: d_epsilon(),
        }
    }

    /// Spec by sniffing a safetensors checkpoint's tensor shapes (load side).
    pub fn from_weights(w: &crate::weights::Weights, head_layers: usize) -> Self {
        let (_, shape) = w.raw_shape("encoder.embeddings.tok_embeddings.weight");
        let hidden = shape[1];
        Self {
            vocab: shape[0],
            hidden,
            layers: w.max_layer_index() + 1,
            // Shape sniffing cannot see the MLP width; the shipped
            // checkpoints pair hidden 1024->2624 and 768->1152.
            intermediate: if hidden == 1024 { 2624 } else { 1152 },
            head_layers,
            sliding_theta: 10_000.0,
            window: d_window(),
            global_every: d_global_every(),
            max_sequence_length: d_max_seq(),
            rope_theta_full: d_theta_full(),
            epsilon: d_epsilon(),
        }
    }
}

/// Every key Laya::load asks for, with its checkpoint shape (PyTorch [out,in]
/// orientation for linears, like the safetensors).
pub fn skeleton_shapes(spec: &SkeletonSpec) -> BTreeMap<String, Vec<usize>> {
    let (v, h, n, i, hl) = (
        spec.vocab,
        spec.hidden,
        spec.layers,
        spec.intermediate,
        spec.head_layers,
    );
    let mut m = BTreeMap::new();
    let mut put = |name: String, shape: Vec<usize>| {
        m.insert(name, shape);
    };

    put("encoder.embeddings.tok_embeddings.weight".into(), vec![v, h]);
    put("encoder.embeddings.norm.weight".into(), vec![h]);
    for l in 0..n {
        let p = format!("encoder.layers.{l}");
        if l != 0 {
            put(format!("{p}.attn_norm.weight"), vec![h]);
        }
        put(format!("{p}.attn.Wqkv.weight"), vec![3 * h, h]);
        put(format!("{p}.attn.Wo.weight"), vec![h, h]);
        put(format!("{p}.mlp.Wi.weight"), vec![2 * i, h]);
        put(format!("{p}.mlp.Wo.weight"), vec![h, i]);
        put(format!("{p}.mlp_norm.weight"), vec![h]);
    }
    put("encoder.final_norm.weight".into(), vec![h]);
    for j in 0..hl {
        let p = format!("head.layers.{j}");
        put(format!("{p}.norm1.weight"), vec![h]);
        put(format!("{p}.norm1.bias"), vec![h]);
        put(format!("{p}.self_attn.in_proj_weight"), vec![3 * h, h]);
        put(format!("{p}.self_attn.in_proj_bias"), vec![3 * h]);
        put(format!("{p}.self_attn.out_proj.weight"), vec![h, h]);
        put(format!("{p}.self_attn.out_proj.bias"), vec![h]);
        put(format!("{p}.norm2.weight"), vec![h]);
        put(format!("{p}.norm2.bias"), vec![h]);
        put(format!("{p}.linear1.weight"), vec![4 * h, h]);
        put(format!("{p}.linear1.bias"), vec![4 * h]);
        put(format!("{p}.linear2.weight"), vec![h, 4 * h]);
        put(format!("{p}.linear2.bias"), vec![h]);
    }
    put("type_emb.weight".into(), vec![3, h]);
    put("scorer.0.weight".into(), vec![h]);
    put("scorer.0.bias".into(), vec![h]);
    put("scorer.1.weight".into(), vec![h, h]);
    put("scorer.1.bias".into(), vec![h]);
    put("scorer.3.weight".into(), vec![1, h]);
    put("scorer.3.bias".into(), vec![1]);
    put("act_head.0.weight".into(), vec![256, h + 4]);
    put("act_head.0.bias".into(), vec![256]);
    put("act_head.2.weight".into(), vec![2, 256]);
    put("act_head.2.bias".into(), vec![2]);
    m
}

/// WeightSource over the skeleton table: zeros everywhere; the burnpack
/// record overwrites every parameter afterwards.
pub struct SkeletonWeights {
    shapes: BTreeMap<String, Vec<usize>>,
}

impl SkeletonWeights {
    pub fn new(spec: &SkeletonSpec) -> Self {
        Self { shapes: skeleton_shapes(spec) }
    }

    fn zeros<const R: usize>(&self, name: &str, device: &Device) -> Tensor<R> {
        let shape = self
            .shapes
            .get(name)
            .unwrap_or_else(|| panic!("skeleton: unknown tensor {name}"))
            .clone();
        let n: usize = shape.iter().product();
        Tensor::from_data(TensorData::new(vec![0.0f32; n], shape), device)
    }
}

impl WeightSource for SkeletonWeights {
    fn t1(&self, name: &str, device: &Device) -> Tensor<1> {
        self.zeros::<1>(name, device)
    }
    fn t2(&self, name: &str, device: &Device) -> Tensor<2> {
        self.zeros::<2>(name, device)
    }
    fn has(&self, name: &str) -> bool {
        self.shapes.contains_key(name)
    }
    fn max_layer_index(&self) -> usize {
        self.shapes
            .keys()
            .filter_map(|k| {
                k.strip_prefix("encoder.layers.")
                    .and_then(|r| r.split('.').next())
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .max()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod spec_tests {
    use super::*;

    /// Sidecars written before the complete-geometry fields existed (the
    /// original 6-field form) must parse with the constants defaulted.
    #[test]
    fn old_six_field_sidecar_parses_with_defaults() {
        let old = r#"{"vocab":256000,"hidden":768,"layers":22,"intermediate":1152,
                      "head_layers":2,"sliding_theta":10000.0}"#;
        let spec: SkeletonSpec = serde_json::from_str(old).expect("old sidecar");
        assert_eq!((spec.vocab, spec.layers), (256000, 22));
        assert_eq!(spec.window, 128);
        assert_eq!(spec.global_every, 3);
        assert_eq!(spec.max_sequence_length, 1088);
        assert_eq!(spec.rope_theta_full, 160_000.0);
        assert!((spec.epsilon - 1e-5).abs() < 1e-9);
    }

    /// A complete spec round-trips through JSON field-for-field.
    #[test]
    fn complete_spec_round_trips() {
        let spec = SkeletonSpec {
            vocab: 7, hidden: 128, layers: 2, intermediate: 64, head_layers: 1,
            sliding_theta: 550.0, window: 17, global_every: 5,
            max_sequence_length: 99, rope_theta_full: 42.0, epsilon: 3e-4,
        };
        let json = serde_json::to_string(&spec).unwrap();
        let back: SkeletonSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.window, 17);
        assert_eq!(back.max_sequence_length, 99);
        assert_eq!(back.rope_theta_full, 42.0);
    }

    /// encoder_config() is the single mapping spec -> EncoderConfig.
    #[test]
    fn encoder_config_maps_every_field() {
        let spec = SkeletonSpec {
            vocab: 50368, hidden: 1024, layers: 28, intermediate: 2624,
            head_layers: 2, sliding_theta: 10_000.0, window: 128,
            global_every: 3, max_sequence_length: 1088,
            rope_theta_full: 160_000.0, epsilon: 1e-5,
        };
        let c = spec.encoder_config();
        assert_eq!(c.num_heads, 16); // 1024 / 64
        assert_eq!(c.rope_theta_sliding, 10_000.0);
        assert_eq!(c.window_size, 128);
        assert_eq!(c.max_sequence_length, 1088);
    }
}
