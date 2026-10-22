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
