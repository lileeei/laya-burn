//! Skeleton support: a Weights variant that answers every checkpoint key
//! with zeros of the recorded shape, letting Laya::load build the module
//! tree without the original safetensors file.

use std::collections::BTreeMap;

use burn::tensor::{Device, FloatDType, Tensor, TensorData};

use crate::weights::Weights;

/// Geometry needed to derive every checkpoint tensor shape.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SkeletonSpec {
    pub vocab: usize,
    pub hidden: usize,
    pub layers: usize,
    pub intermediate: usize,
    pub head_layers: usize,
    pub sliding_theta: f32,
}

/// Every tensor key Laya::load asks Weights for, with its checkpoint shape
/// (PyTorch [out, in] orientation for linears, exactly like the safetensors).
pub fn skeleton_shapes(spec: &SkeletonSpec) -> BTreeMap<String, Vec<usize>> {
    let (v, h, n, i) = (spec.vocab, spec.hidden, spec.layers, spec.intermediate);
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
    for j in 0..spec.head_layers {
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

/// A Weights implementation backed by the skeleton shape table: every tensor
/// is zeros; loading overwrites everything via ModuleRecord.
pub struct SkeletonWeights {
    shapes: BTreeMap<String, Vec<usize>>,
}

impl SkeletonWeights {
    pub fn new(spec: &SkeletonSpec) -> Self {
        Self { shapes: skeleton_shapes(spec) }
    }

    fn zeros(&self, name: &str, device: &Device, rank: usize) -> Tensor {
        let shape = self
            .shapes
            .get(name)
            .unwrap_or_else(|| panic!("skeleton: unknown tensor {name}"))
            .clone();
        let n: usize = shape.iter().product();
        Tensor::from_data(TensorData::new(vec![0.0f32; n], shape), device).cast(FloatDType::F32)
            .reshape_generic(rank)
    }
}

impl WeightsLike for SkeletonWeights {}

/// Trait mirroring the four methods Laya::load calls on Weights.
pub trait WeightsLike {
    fn t1s(&self, name: &str, device: &Device) -> Tensor;
    fn t2s(&self, name: &str, device: &Device) -> Tensor;
    fn hass(&self, name: &str) -> bool;
    fn max_layers(&self) -> usize;
}

impl WeightsLike for SkeletonWeights {
    fn t1s(&self, name: &str, device: &Device) -> Tensor {
        self.zeros(name, device, 1)
    }
    fn t2s(&self, name: &str, device: &Device) -> Tensor {
        self.zeros(name, device, 2)
    }
    fn hass(&self, name: &str) -> bool {
        self.shapes.contains_key(name)
    }
    fn max_layers(&self) -> usize {
        self.shapes
            .keys()
            .filter_map(|k| k.strip_prefix("encoder.layers.").and_then(|r| r.split('.').next()).and_then(|s| s.parse::<usize>().ok()))
            .max()
            .unwrap_or(0)
    }
}

// Silence unused warnings for the trait object form.
#[allow(dead_code)]
fn _w(_: &dyn WeightsLike) {}
#[allow(dead_code)]
fn _f(_: &Weights) {}
