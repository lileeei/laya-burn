//! Load Laya checkpoint tensors from a safetensors file into Burn tensors.

use burn::tensor::{Device, FloatDType, Tensor, TensorData};

/// The tensor surface Laya::load consumes: implemented by the safetensors
/// reader (Weights) and by the burnpack skeleton (crate::skeleton).

/// Compute dtype for loaded weights: LAYA_DTYPE=bf16 selects BF16 compute,
/// anything else stays F32 (the parity-verified default).
pub fn compute_dtype() -> FloatDType {
    match std::env::var("LAYA_DTYPE").as_deref() {
        Ok("bf16") => FloatDType::BF16,
        Ok("f16") => FloatDType::F16,
        _ => FloatDType::F32,
    }
}
use safetensors::SafeTensors;

pub trait WeightSource {
    fn t1(&self, name: &str, device: &Device) -> Tensor<1>;
    fn t2(&self, name: &str, device: &Device) -> Tensor<2>;
    fn has(&self, name: &str) -> bool;
    fn max_layer_index(&self) -> usize;
}

/// A safetensors file held in memory with lazy tensor views.
pub struct Weights {
    bytes: Vec<u8>,
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = (((h >> 15) & 1) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = match (exp, frac) {
        (0, 0) => sign,
        (0, _) => {
            // Subnormal half: normalize manually.
            let mut e = -1i32;
            let mut f = frac as i32;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            sign | (((113 + e) as u32) << 23) | ((f as u32 & 0x3ff) << 13)
        }
        (0x1f, _) => sign | 0x7f80_0000 | (frac << 13),
        _ => sign | ((exp - 15 + 127) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

impl WeightSource for Weights {
    fn t1(&self, name: &str, device: &Device) -> Tensor<1> {
        Weights::t1(self, name, device)
    }
    fn t2(&self, name: &str, device: &Device) -> Tensor<2> {
        Weights::t2(self, name, device)
    }
    fn has(&self, name: &str) -> bool {
        Weights::has(self, name)
    }
    fn max_layer_index(&self) -> usize {
        Weights::max_layer_index(self)
    }
}

impl Weights {
    pub fn open(path: &str) -> Self {
        Self {
            bytes: std::fs::read(path).expect("read safetensors file"),
        }
    }

    /// Raw F32 data of a tensor in stored (PyTorch) orientation.
    fn data_f32(&self, name: &str) -> (Vec<f32>, Vec<usize>) {
        let st = SafeTensors::deserialize(&self.bytes).expect("deserialize safetensors");
        let view = st
            .tensor(name)
            .unwrap_or_else(|_| panic!("tensor {name} missing from checkpoint"));
        let shape: Vec<usize> = view.shape().to_vec();
        let n: usize = shape.iter().product();
        let mut out = Vec::with_capacity(n);
        match view.dtype() {
            safetensors::Dtype::F16 => {
                let bytes = view.data();
                for i in 0..n {
                    let h = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
                    out.push(f16_bits_to_f32(h));
                }
            }
            safetensors::Dtype::F32 => {
                let bytes = view.data();
                for i in 0..n {
                    out.push(f32::from_le_bytes([
                        bytes[4 * i],
                        bytes[4 * i + 1],
                        bytes[4 * i + 2],
                        bytes[4 * i + 3],
                    ]));
                }
            }
            other => panic!("unsupported dtype {other:?} for tensor {name}"),
        }
        (out, shape)
    }

    /// 1-D tensor as stored, cast to the compute dtype.
    pub fn t1(&self, name: &str, device: &Device) -> Tensor<1> {
        let (data, shape) = self.data_f32(name);
        let t: Tensor<1> = Tensor::from_data(TensorData::new(data, shape), device);
        t.cast(compute_dtype())
    }

    /// 2-D tensor as stored (PyTorch [out, in] orientation), cast to the compute dtype.
    pub fn t2(&self, name: &str, device: &Device) -> Tensor<2> {
        let (data, shape) = self.data_f32(name);
        let t: Tensor<2> = Tensor::from_data(TensorData::new(data, shape), device);
        t.cast(compute_dtype())
    }

    /// Shape of a tensor without materializing it (header only).
    pub fn raw_shape(&self, name: &str) -> (Vec<f32>, Vec<usize>) {
        let st = SafeTensors::deserialize(&self.bytes).expect("deserialize safetensors");
        let view = st
            .tensor(name)
            .unwrap_or_else(|_| panic!("tensor {name} missing from checkpoint"));
        (Vec::new(), view.shape().to_vec())
    }

    /// Highest N in encoder.layers.N.* keys (layer count - 1).
    pub fn max_layer_index(&self) -> usize {
        let st = SafeTensors::deserialize(&self.bytes).expect("deserialize safetensors");
        st.names()
            .iter()
            .filter_map(|n| {
                let rest = n.strip_prefix("encoder.layers.")?;
                let idx = rest.split('.').next()?;
                idx.parse::<usize>().ok()
            })
            .max()
            .unwrap_or(0)
    }

    pub fn has(&self, name: &str) -> bool {
        let st = SafeTensors::deserialize(&self.bytes).expect("deserialize safetensors");
        st.tensor(name).is_ok()
    }
}
