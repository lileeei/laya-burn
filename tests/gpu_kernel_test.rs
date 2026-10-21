//! GPU differential test for the custom windowed-attention kernel.
//!
//! Requires a CubeCL-capable device (Metal on macOS, wgpu elsewhere), so it is
//! #[ignore]d by default; run explicitly with:
//!     cargo test --release -- --ignored

use burn::tensor::{Device, Tensor, activation};
use laya_burn::window_kernel;

#[test]
#[ignore = "requires a GPU (Metal/wgpu)"]
fn windowed_attention_kernel_matches_masked_reference() {
    let device = Device::metal(burn::tensor::DeviceKind::DefaultDevice);
    let (b, h, s, d) = (2usize, 3usize, 200usize, 64usize);
    let q = Tensor::<4>::random([b, h, s, d], burn::tensor::Distribution::Default, &device);
    let k = Tensor::<4>::random([b, h, s, d], burn::tensor::Distribution::Default, &device);
    let v = Tensor::<4>::random([b, h, s, d], burn::tensor::Distribution::Default, &device);

    // Half of each batch row is padding, like real batches with varied lengths.
    let mut pm = vec![1.0f32; b * s];
    for bb in 0..b {
        for j in (s / 2)..s {
            pm[bb * s + j] = 0.0;
        }
    }
    let pad = Tensor::<1>::from_floats(pm.as_slice(), &device).reshape([b, s]);

    let custom = window_kernel::windowed_attention(q.clone(), k.clone(), v.clone(), pad.clone());

    // Reference: dense attention with the same window + pad mask.
    let scores = q.matmul(k.transpose()) * (1.0f32 / 8.0);
    let mut m = vec![-1.0e9f32; b * s * s];
    for bb in 0..b {
        for i in 0..s {
            for j in 0..s {
                if (i as isize - j as isize).unsigned_abs() <= 64 && pm[bb * s + j] == 1.0 {
                    m[(bb * s + i) * s + j] = 0.0;
                }
            }
        }
    }
    let win = Tensor::<1>::from_floats(m.as_slice(), &device).reshape([b, 1, s, s]);
    let probs = activation::softmax(scores + win, 3);
    let reference = probs.matmul(v);

    // Compare only non-pad query rows (fully-masked rows carry no signal).
    let valid = Tensor::<1>::from_floats(pm.as_slice(), &device).reshape([b, 1, s, 1]);
    let diff: f32 = ((custom - reference).abs() * valid)
        .max()
        .into_scalar::<f32>()
        .into();
    assert!(diff < 1e-4, "kernel diverged from reference: {diff}");
}
