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

    // Shapes cover: window boundaries (s around 64/65), row-block boundaries
    // (s around multiples of 8), the plain mid-size case, a long sequence, and
    // degenerate single-row/batch cases.
    let cases: Vec<(usize, usize, usize)> = vec![
        (2, 3, 200),
        (1, 2, 1000),
        (1, 1, 1),
        (1, 1, 8),
        (1, 1, 9),
        (1, 2, 63),
        (1, 2, 64),
        (1, 2, 65),
        (2, 1, 71),  // i_base + r block edge crossing the pad boundary
        (1, 1, 1000),
    ];

    for &(b, h, s) in &cases {
        run_case(&device, b, h, s, PadPattern::SecondHalfPad);
    }
    // No padding at all, and a fully-padded batch row.
    run_case(&device, 2, 2, 1000, PadPattern::NoPad);
    run_case(&device, 2, 2, 130, PadPattern::OneBatchAllPad);
}

enum PadPattern {
    NoPad,
    SecondHalfPad,
    OneBatchAllPad,
}

fn run_case(device: &Device, b: usize, h: usize, s: usize, pattern: PadPattern) {
    let d = window_kernel::HEAD_DIM;
    let q = Tensor::<4>::random([b, h, s, d], burn::tensor::Distribution::Default, device);
    let k = Tensor::<4>::random([b, h, s, d], burn::tensor::Distribution::Default, device);
    let v = Tensor::<4>::random([b, h, s, d], burn::tensor::Distribution::Default, device);

    let mut pm = vec![1.0f32; b * s];
    match pattern {
        PadPattern::NoPad => {}
        PadPattern::SecondHalfPad => {
            for bb in 0..b {
                for j in (s / 2)..s {
                    pm[bb * s + j] = 0.0;
                }
            }
        }
        PadPattern::OneBatchAllPad => {
            for j in 0..s {
                pm[j] = 0.0;
            }
        }
    }
    let pad = Tensor::<1>::from_floats(pm.as_slice(), device).reshape([b, s]);

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
    let win = Tensor::<1>::from_floats(m.as_slice(), device).reshape([b, 1, s, s]);
    let probs = activation::softmax(scores + win, 3);
    let reference = probs.matmul(v);

    // Compare only non-pad query rows (fully-masked rows carry no signal).
    let valid = Tensor::<1>::from_floats(pm.as_slice(), device).reshape([b, 1, s, 1]);
    let diff: f32 = ((custom - reference).abs() * valid)
        .max()
        .into_scalar::<f32>()
        .into();
    assert!(diff < 1e-4, "kernel diverged from reference: {diff}");
}
