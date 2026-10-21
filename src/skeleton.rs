//! Skeleton weights: build the Laya module tree from a geometry spec alone
//! (all-zero tensors), so a burnpack reload needs no safetensors file.

use std::collections::BTreeMap;

use burn::tensor::{Device, FloatDType, Tensor, TensorData};

use crate::weights::WeightSource;

/// Geometry sufficient to derive every checkpoint tensor shape.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SkeletonSpec {
    pub vocab: usize,
    pub hidden: usize,
    pub layers: usize,
    pub intermediate: usize,
    pub head_layers: usize,
    pub sliding_theta: f32,
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
