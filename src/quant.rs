//! Q8 deployment path: export-side group exemption and load-side eager
//! dequantization.
//!
//! Burn's own reload of a Q8 burnpack leaves params as QFloat tensors that a
//! cubecl kernel dequantizes lazily, per tensor, on first use (~36s for the
//! 421M model). `eager_dequantize` replaces that with a plain-Rust pass at
//! load: read the stored i8 + scales (no kernel), multiply in one vectorized
//! sweep, hand the module plain f32 tensors so serving hits no quantization
//! at all. The artifact stays ~4x smaller than f32; memory after load is the
//! same as f32 (weights live dequantized).

use burn::module::{Module, ModuleMapper, Param, ParamGroup, Quantizer};
use burn::tensor::quantization::{Calibration, QuantStore, QuantValue, ScaleDtype, QuantizedBytes};
use burn::tensor::{Device, DType, FloatDType, Tensor, TensorData};

use crate::model::Laya;

/// Rust field-name paths kept in f32 when exporting Q8: the decision head
/// (the small transformer the encoder exists to feed), the scorer/act heads,
/// every LayerNorm, and the type embedding. Together ~5% of the weights; the
/// encoder linears and embeddings (95%) still quantize.
const EXEMPT: &[&str] = &["head_layers", "scorer", "act_fc", "type_emb", "norm"];

/// Quantize the encoder + embeddings to Q8S (MinMax, Native store), leaving
/// the decision machinery in f32. PackedU32 store would need trailing dims
/// divisible by 4, which the head violates; Native i8 works for any shape.
/// Scales are per [1, 128] block in f32: per-tensor scaling costs ~2 points
/// of MASSIVE en accuracy, block scaling keeps 8 scales per 1024-wide row
/// (+3% bytes) and recovers it.
pub fn quantize(model: Laya, device: &Device) -> Laya {
    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S)
        .with_store(QuantStore::Native)
        .per_block([1, 128], ScaleDtype::F32);
    let mut quantizer = Quantizer::new(Calibration::MinMax, scheme);
    model.quantize_weights_group(
        &mut quantizer,
        ParamGroup::all().exclude(ParamGroup::from_any_predicates(EXEMPT.to_vec())),
    )
}


struct CastToCompute;

impl ModuleMapper for CastToCompute {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        // The safetensors path casts per tensor at load (weights.rs); the
        // burnpack path builds an f32 skeleton and overwrites from an f32
        // record, so every param needs one cast here for LAYA_DTYPE=bf16/f16
        // (mixed dtypes panic inside the graph builder).
        match crate::weights::compute_dtype() {
            FloatDType::F32 => param,
            dtype => param.map(|t| t.cast(dtype)),
        }
    }
}

/// Cast every float param of `model` to [crate::weights::compute_dtype].
/// No-op walk for f32.
pub fn cast_compute_dtype<M: Module>(model: M) -> M {
    model.map(&mut CastToCompute)
}

/// How much work `eager_dequantize` did.
#[derive(Debug, Default)]
pub struct DequantStats {
    pub params: usize,
    pub quantized: usize,
}

struct EagerDequant {
    stats: DequantStats,
}

impl ModuleMapper for EagerDequant {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        self.stats.params += 1;
        if !matches!(param.val().dtype(), DType::QFloat(_)) {
            return param;
        }
        self.stats.quantized += 1;
        param.map(dequantize_q8)
    }
}

/// Replace every QFloat param of `model` with its dequantized f32 tensor,
/// computed in plain Rust. A no-op walk (no tensor touched) for f32 packs.
pub fn eager_dequantize<M: Module>(model: M) -> (M, DequantStats) {
    let mut mapper = EagerDequant { stats: DequantStats::default() };
    let model = model.map(&mut mapper);
    (model, mapper.stats)
}

/// Decode one Q8S tensor: values as stored i8, scales appended to the bytes
/// (per-tensor, or per block with a global normalizer). Symmetric scheme, so
/// dequantized = i8 * scale (two-level: i8 * block_scale * global).
fn dequantize_q8<const D: usize>(tensor: Tensor<D>) -> Tensor<D> {
    let device = tensor.device();
    let shape = tensor.shape();
    let data = tensor.into_data(); // raw bytes; quantized tensors are not converted here
    let scheme = match data.dtype() {
        DType::QFloat(scheme) => scheme,
        _ => unreachable!("checked by the mapper"),
    };
    let quantized = QuantizedBytes {
        bytes: data.into_bytes(),
        scheme: scheme.clone(),
        shape: shape.clone(),
    };
    let (values, scales) = quantized.into_vec_i8();
    let out: Vec<f32> = match scheme.block_size() {
        None => {
            let scale = scales.block[0];
            values.into_iter().map(|v| v as f32 * scale).collect()
        }
        Some(block) => {
            let layout = burn::tensor::quantization::BlockLayout::new(&shape, &block);
            let global = scales.global.unwrap_or(1.0);
            values
                .into_iter()
                .enumerate()
                .map(|(i, v)| v as f32 * scales.block[layout.block_of(i)] * global)
                .collect()
        }
    };
    let tensor = Tensor::<D>::from_data(TensorData::new(out, shape), &device);
    match crate::weights::compute_dtype() {
        FloatDType::F32 => tensor,
        dtype => tensor.cast(dtype),
    }
}

/// Report from [export_q8]: everything a caller (binary or test) needs to
/// assert - the round-trip checks are data, not console lines.
#[derive(Debug)]
pub struct PackExport {
    pub pack_path: String,
    pub pack_bytes: u64,
    pub quantize_ms: u128,
    pub reload_ms: u128,
    /// reload(eager dequant) answers == in-memory Q8 answers.
    pub round_trip_ok: bool,
}

/// The Q8 export pipeline (old LAYA_QUANT spike): load the f32 checkpoint,
/// quantize with the exempted group, write model.q8.mpk plus a complete
/// sidecar, reload through the eager-dequant path, verify bit-identical
/// answers on a fixed probe question.
pub fn export_q8(dir: &str, device: &Device) -> PackExport {
    use crate::inference::{QuestionInput, RlAgent};
    use burn::module::Module;

    let agent = RlAgent::load(dir, device);
    let t0 = std::time::Instant::now();
    let q_model = quantize(agent.model.clone(), device);
    let quantize_ms = t0.elapsed().as_millis();

    let probe = QuestionInput {
        qtype: 0,
        instructions: "Which runtime should host the model for CPU inference in Rust?".to_string(),
        options: vec!["burn".into(), "candle".into(), "onnxruntime".into(), "libtorch".into()],
        keys: vec![],
        option_order: None,
    };
    let state = serde_json::Value::String(
        "The user is deciding whether to run a 421M non-autoregressive decision model in Rust."
            .to_string(),
    );
    let mut q_agent = RlAgent::load(dir, device);
    q_agent.model = q_model;
    let in_memory = q_agent.system_one(device, &state, &[probe.clone()], None);

    // Sidecar: complete geometry from the live model (works for every
    // checkpoint; the old hardcoded english values were wrong for others).
    let sidecar = format!("{dir}/model.mpk.json");
    if !std::path::Path::new(&sidecar).is_file() {
        std::fs::write(
            &sidecar,
            serde_json::to_string_pretty(&crate::skeleton::SkeletonSpec::from_model(&q_agent.model))
                .unwrap(),
        )
        .expect("write sidecar");
    }
    let pack_path = format!("{dir}/model.q8.mpk");
    q_agent.model.clone().save_file(&pack_path).expect("save quantized burnpack");

    let t1 = std::time::Instant::now();
    let reloaded = RlAgent::load_burnpack_named(dir, "model.q8.mpk", device);
    let reload_ms = t1.elapsed().as_millis();
    let reloaded_answers = reloaded.system_one(device, &state, &[probe], None);
    PackExport {
        pack_path,
        pack_bytes: std::fs::metadata(format!("{dir}/model.q8.mpk")).map(|m| m.len()).unwrap_or(0),
        quantize_ms,
        reload_ms,
        round_trip_ok: format!("{in_memory:?}") == format!("{reloaded_answers:?}"),
    }
}

#[cfg(test)]
mod export_tests {
    /// Full Q8 export round-trip as a TEST (the old pipeline printed
    /// PASS/CHECK to stdout from the demo binary). Skips when the checkpoint
    /// directory is absent (CI); runs for real wherever model-english exists.
    #[test]
    fn q8_export_round_trip_when_checkpoint_present() {
        let dir = "model-english";
        for f in ["model.safetensors", "rl_agent_config.json"] {
            if !std::path::Path::new(&format!("{dir}/{f}")).exists() {
                eprintln!("skipping: {dir}/{f} not present");
                return;
            }
        }
        let device = burn::tensor::Device::default();
        let report = super::export_q8(dir, &device);
        assert!(report.pack_bytes > 300_000_000, "pack suspiciously small");
        assert!(report.round_trip_ok, "eager-dequant reload != in-memory Q8");
    }
}
