//! Export a Laya module back to the official PyTorch safetensors layout, so
//! a fine-tuned laya-burn checkpoint loads into the reference implementation
//! (RLAgent.load_state_dict strict=True) and the released SDK.

use std::collections::BTreeMap;
use std::collections::HashMap;

use burn::module::{Module, ModuleMapper, Param};
use burn::tensor::Tensor;
use safetensors::tensor::TensorView;

use crate::model::Laya;

/// Rust module-tree path -> checkpoint name. Linear weights live transposed
/// in burn ([in, out] vs PyTorch [out, in]) - flagged here and undone at write.
fn to_checkpoint_name(rust_path: &str) -> Option<(String, bool)> {
    // head: head_layers.N.<field> -> head.layers.N.<ckpt field>
    if let Some(rest) = rust_path.strip_prefix("head_layers.") {
        let (n, field) = rest.split_once('.')?;
        let (f, transpose) = match field {
            "norm1.gamma" => ("norm1.weight", false),
            "norm1.beta" => ("norm1.bias", false),
            "norm2.gamma" => ("norm2.weight", false),
            "norm2.beta" => ("norm2.bias", false),
            "in_proj.weight" => ("self_attn.in_proj_weight", true),
            "in_proj.bias" => ("self_attn.in_proj_bias", false),
            "out_proj.weight" => ("self_attn.out_proj.weight", true),
            "out_proj.bias" => ("self_attn.out_proj.bias", false),
            "linear1.weight" => ("linear1.weight", true),
            "linear1.bias" => ("linear1.bias", false),
            "linear2.weight" => ("linear2.weight", true),
            "linear2.bias" => ("linear2.bias", false),
            _ => return None,
        };
        return Some((format!("head.layers.{n}.{f}"), transpose));
    }
    // encoder layers
    if let Some(rest) = rust_path.strip_prefix("layers.") {
        let (n, field) = rest.split_once('.')?;
        let (f, transpose) = match field {
            "wqkv.weight" => ("attn.Wqkv.weight", true),
            "attn_out.weight" => ("attn.Wo.weight", true),
            "wi.weight" => ("mlp.Wi.weight", true),
            "mlp_out.weight" => ("mlp.Wo.weight", true),
            "attn_norm.gamma" => ("attn_norm.weight", false),
            "mlp_norm.gamma" => ("mlp_norm.weight", false),
            _ => return None,
        };
        return Some((format!("encoder.layers.{n}.{f}"), transpose));
    }
    Some(match rust_path {
        "tok_embeddings.weight" => ("encoder.embeddings.tok_embeddings.weight".into(), false),
        "emb_norm.gamma" => ("encoder.embeddings.norm.weight".into(), false),
        "final_norm.gamma" => ("encoder.final_norm.weight".into(), false),
        "type_emb.weight" => ("type_emb.weight".into(), false),
        "scorer_norm.gamma" => ("scorer.0.weight".into(), false),
        "scorer_norm.beta" => ("scorer.0.bias".into(), false),
        "scorer_fc1.weight" => ("scorer.1.weight".into(), true),
        "scorer_fc1.bias" => ("scorer.1.bias".into(), false),
        "scorer_fc2.weight" => ("scorer.3.weight".into(), true),
        "scorer_fc2.bias" => ("scorer.3.bias".into(), false),
        "act_fc1.weight" => ("act_head.0.weight".into(), true),
        "act_fc1.bias" => ("act_head.0.bias".into(), false),
        "act_fc2.weight" => ("act_head.2.weight".into(), true),
        "act_fc2.bias" => ("act_head.2.bias".into(), false),
        _ => return None,
    })
}

#[derive(Default)]
struct Collector {
    path: Vec<String>,
    tensors: BTreeMap<String, (Vec<usize>, Vec<f32>, bool)>,
}

impl ModuleMapper for Collector {
    fn enter_module(&mut self, name: &str, _t: &str) {
        self.path.push(name.to_string());
    }
    fn exit_module(&mut self, _name: &str, _t: &str) {
        self.path.pop();
    }
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let rust_path = self.path.join(".");
        if let Some((name, transpose)) = to_checkpoint_name(&rust_path) {
            let t = param.val().detach();
            let data = t.into_data();
            if let Ok(vals) = data.as_slice::<f32>() {
                let shape: Vec<usize> = data.shape().to_vec();
                self.tensors.insert(name, (shape, vals.to_vec(), transpose && D == 2));
            }
        }
        param
    }
}

/// Write `model` to `path` in the official safetensors layout. Returns the
/// number of tensors written (206 for a full Laya checkpoint, including the
/// `temperature` buffer the official DecisionModel state_dict expects).
pub fn export_safetensors(model: &Laya, temperature: [f32; 3], path: &str) -> usize {
    let mut collector = Collector::default();
    let _ = model.clone().map(&mut collector);
    // Buffers must outlive the TensorViews borrowing them.
    let mut buffers: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    for (name, (shape, vals, transpose)) in collector.tensors {
        let (shape, vals) = if transpose {
            let [r, c] = [shape[0], shape[1]];
            let mut out = vec![0.0f32; r * c];
            for i in 0..r {
                for j in 0..c {
                    out[j * r + i] = vals[i * c + j];
                }
            }
            (vec![c, r], out)
        } else {
            (shape, vals)
        };
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        buffers.push((name, shape, bytes));
    }
    let views: HashMap<String, TensorView> = buffers
        .iter()
        .filter_map(|(n, s, b)| {
            TensorView::new(safetensors::Dtype::F32, s.clone(), b)
                .ok()
                .map(|v| (n.clone(), v))
        })
        .collect();
    let mut views = views;
    let tbytes: Vec<u8> = temperature.iter().flat_map(|v| v.to_le_bytes()).collect();
    let tview = TensorView::new(safetensors::Dtype::F32, vec![3], &tbytes)
        .expect("temperature view");
    views.insert("temperature".to_string(), tview);
    let n = views.len();
    safetensors::serialize_to_file(&views, None::<HashMap<String, String>>, std::path::Path::new(path))
        .expect("write safetensors");
    n
}
