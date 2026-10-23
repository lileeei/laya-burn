//! Custom windowed-attention kernel via #[backend_extension] - no patched
//! dependencies, everything from crates.io.
//!
//! Two kernel generations live here:
//!
//! * v1 (`windowed_attention_kernel_v1`, the default): one cube (plane of 32
//!   units) per (batch, head, query) row. Each unit owns two head-dim
//!   components (d = unit and d = unit + 32, for the 64-wide heads of the
//!   Laya encoder). The score dot product is reduced with a plane sum; softmax
//!   runs online (flash-style) over the query's window keys only.
//! * v2 (`windowed_attention_kernel_v2`, behind LAYA_KERNEL=v2): one cube
//!   processes ROWS_PER_CUBE adjacent query rows of the same (head, batch).
//!   The K/V union window [i_min - W, i_max + W] is streamed once and every
//!   K/V element read is reused across all in-window rows of the block,
//!   cutting K/V global traffic by roughly the row-block overlap factor.
//!   Per-row online-softmax state (m / l / acc) lives in registers as
//!   fixed-size arrays indexed by compile-time-unrolled row loops.
//!
//! Selection: v1 is the default entry. On Apple Silicon (M-series, Metal) the
//! row-blocked v2 measured no faster than v1 (long run: tie; short run: up to
//! ~6% slower at R=8) - the kernel is bounded by plane-reduction compute and
//! cube parallelism rather than K/V bandwidth, and R sets of softmax state add
//! register pressure - so the simpler v1 stays the default. Set LAYA_KERNEL=v2
//! to select v2.
// Historical note (from the deleted examples/kernel_spike.rs probe): custom
// cubecl kernels plug in via burn-backend-extension (`BackendExtension`),
// with `Kernel` impls launched through `cubecl::launch`. The differential
// test in tests/gpu_kernel_test.rs is the live harness for this kernel.

use burn::{
    backend::{Dispatch, backend_extension, tensor::FloatTensor},
    tensor::Tensor,
};

/// Plane width this kernel is specialized for (SIMD group size on Metal/CUDA).
pub const PLANE: usize = 32;
/// Head dimension this kernel is specialized for (components per unit = D / PLANE).
pub const HEAD_DIM: usize = 64;
/// Window radius (|i-j| <= WINDOW keeps the pair).
pub const WINDOW: usize = 64;
/// Adjacent query rows processed by one cube in the v2 kernel.
pub const ROWS_PER_CUBE: usize = 8;

#[backend_extension(Cube, Fusion)]
pub trait WindowAttentionBackend: burn::backend::Backend {
    #[fusion(dtype = query, shape = output_shape(query))]
    fn windowed_attention(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        pad_mask: FloatTensor<Self>,
    ) -> FloatTensor<Self>;
}

fn output_shape(query: &burn::tensor::Shape) -> burn::tensor::Shape {
    query.clone()
}

/// True when the caller opted into the v2 row-blocked kernel with LAYA_KERNEL=v2.
fn use_v2() -> bool {
    use std::sync::OnceLock;
    static USE_V2: OnceLock<bool> = OnceLock::new();
    *USE_V2.get_or_init(|| std::env::var("LAYA_KERNEL").ok().as_deref() == Some("v2"))
}

/// Public entry: windowed attention with the Laya encoder's window (64).
/// Inputs q/k/v: [B, H, S, D]; pad_mask: [B, S] float (1 = real, 0 = pad).
/// Output: [B, H, S, D].
pub fn windowed_attention(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>, pad_mask: Tensor<2>) -> Tensor<4> {
    let output = Dispatch::windowed_attention(
        q.into_dispatch(),
        k.into_dispatch(),
        v.into_dispatch(),
        pad_mask.into_dispatch(),
    );
    Tensor::from_dispatch(output)
}

mod kernel {
    use super::{HEAD_DIM, PLANE, ROWS_PER_CUBE, WINDOW};
    use cubecl::{cube, prelude::*};

    #[cube(launch)]
    pub fn windowed_attention_kernel_v1<F: Float>(
        query: &Tensor<F>,
        key: &Tensor<F>,
        value: &Tensor<F>,
        pad_mask: &Tensor<F>,
        output: &mut Tensor<F>,
        seq_len: u32,
        #[define(F)] _dtype: ElemType,
    ) {
        // One cube per (query, head, batch); 32 units per cube own the head dim.
        let i = CUBE_POS_X as usize; // query position (cube index, not unit!)
        let h = CUBE_POS_Y as usize; // head
        let b = CUBE_POS_Z as usize; // batch
        let u = UNIT_POS_X as usize; // component offset inside the head dim

        let heads = output.shape(1);

        if i >= seq_len as usize {
            terminate!();
        }

        let s = seq_len as usize;
        let q_base = ((b * heads + h) * s + i) * HEAD_DIM;
        // Own components: u, u + PLANE.
        let q0 = query[q_base + u];
        let q1 = query[q_base + u + PLANE];

        let mask_base = b * s;

        let scale = F::new(1.0 / (HEAD_DIM as f32).sqrt());
        let neg_inf = F::new(-1.0e9_f32);

        // Online softmax state (per unit; scalars identical across units).
        let mut m = F::new(-3.0e38_f32);
        let mut l = F::new(0.0_f32);
        let mut acc0 = F::new(0.0_f32);
        let mut acc1 = F::new(0.0_f32);

        let j_lo = i.saturating_sub(WINDOW);
        let j_hi = (i + WINDOW + 1).min(s);

        for j in j_lo..j_hi {
            let kv_base = ((b * heads + h) * s + j) * HEAD_DIM;
            // Dot product over this unit's components, then across the plane.
            let partial = q0 * key[kv_base + u] + q1 * key[kv_base + u + PLANE];
            let dot = plane_sum(partial);
            let keep = pad_mask[mask_base + j];
            // keep=1 (real) adds 0; keep=0 (pad) adds -1e9.
            let s = dot * scale + (F::new(1.0_f32) - keep) * neg_inf;

            let m_new = m.max(s);
            let alpha = (m - m_new).exp();
            let p = (s - m_new).exp();

            l = alpha * l + p;
            acc0 = alpha * acc0 + p * value[kv_base + u];
            acc1 = alpha * acc1 + p * value[kv_base + u + PLANE];
            m = m_new;
        }

        output[q_base + u] = acc0 / l;
        output[q_base + u + PLANE] = acc1 / l;
    }

    #[cube(launch)]
    pub fn windowed_attention_kernel_v2<F: Float>(
        query: &Tensor<F>,
        key: &Tensor<F>,
        value: &Tensor<F>,
        pad_mask: &Tensor<F>,
        output: &mut Tensor<F>,
        seq_len: u32,
        #[define(F)] _dtype: ElemType,
    ) {
        // One cube per (row block of ROWS_PER_CUBE queries, head, batch).
        let block = CUBE_POS_X as usize;
        let h = CUBE_POS_Y as usize; // head
        let b = CUBE_POS_Z as usize; // batch
        let u = UNIT_POS_X as usize; // component offset inside the head dim

        let heads = output.shape(1);
        let s = seq_len as usize;
        let i_base = block * ROWS_PER_CUBE;

        if i_base >= s {
            terminate!();
        }

        let scale = F::new(1.0 / (HEAD_DIM as f32).sqrt());
        let neg_inf = F::new(-1.0e9_f32);

        // Per-row state in registers: query components plus online-softmax
        // accumulators. Rows are indexed by compile-time-unrolled loops, so
        // each element stays a scalar register.
        let mut q0 = Array::<F>::new(ROWS_PER_CUBE);
        let mut q1 = Array::<F>::new(ROWS_PER_CUBE);
        let mut m = Array::<F>::new(ROWS_PER_CUBE);
        let mut l = Array::<F>::new(ROWS_PER_CUBE);
        let mut acc0 = Array::<F>::new(ROWS_PER_CUBE);
        let mut acc1 = Array::<F>::new(ROWS_PER_CUBE);
        for r in 0..ROWS_PER_CUBE {
            let i = i_base + r;
            // Rows past the sequence end keep zero state and are never written.
            if i < s {
                let q_row = ((b * heads + h) * s + i) * HEAD_DIM;
                q0[r] = query[q_row + u];
                q1[r] = query[q_row + u + PLANE];
            } else {
                q0[r] = F::new(0.0_f32);
                q1[r] = F::new(0.0_f32);
            }
            m[r] = F::new(-3.0e38_f32);
            l[r] = F::new(0.0_f32);
            acc0[r] = F::new(0.0_f32);
            acc1[r] = F::new(0.0_f32);
        }

        // Union window of the whole row block: read each K/V element once.
        let i_last = (i_base + ROWS_PER_CUBE - 1).min(s - 1);
        let j_lo = i_base.saturating_sub(WINDOW);
        let j_hi = (i_last + WINDOW + 1).min(s);

        for j in j_lo..j_hi {
            let kv_base = ((b * heads + h) * s + j) * HEAD_DIM;
            let k0 = key[kv_base + u];
            let k1 = key[kv_base + u + PLANE];
            let v0 = value[kv_base + u];
            let v1 = value[kv_base + u + PLANE];
            let keep = pad_mask[b * s + j];
            let pad_add = (F::new(1.0_f32) - keep) * neg_inf;

            for r in 0..ROWS_PER_CUBE {
                let i = i_base + r;
                let in_window = i < s && j + WINDOW >= i && i + WINDOW >= j;
                // plane_sum stays in straight-line (plane-uniform) code; only
                // the state update is guarded, by a condition identical across
                // the plane since it depends on i/j only.
                let partial = q0[r] * k0 + q1[r] * k1;
                let dot = plane_sum(partial);
                let score = dot * scale + pad_add;
                if in_window {
                    let m_new = m[r].max(score);
                    let alpha = (m[r] - m_new).exp();
                    let p = (score - m_new).exp();
                    l[r] = alpha * l[r] + p;
                    acc0[r] = alpha * acc0[r] + p * v0;
                    acc1[r] = alpha * acc1[r] + p * v1;
                    m[r] = m_new;
                }
            }
        }

        for r in 0..ROWS_PER_CUBE {
            let i = i_base + r;
            if i < s {
                let out_row = ((b * heads + h) * s + i) * HEAD_DIM;
                output[out_row + u] = acc0[r] / l[r];
                output[out_row + u + PLANE] = acc1[r] / l[r];
            }
        }
    }

    const _: () = assert!(HEAD_DIM % PLANE == 0, "components must divide evenly");
}


impl WindowAttentionBackend for burn_cubecl::CubeBackend {
    fn windowed_attention(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        pad_mask: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        use burn::backend::cubecl::dtype_to_storage_type;
        use burn_cubecl::{kernel::into_contiguous, tensor::CubeTensor};
        use cubecl::{CubeCount, CubeDim};

        let dtype = query.dtype;
        let cube_dim = CubeDim::new_1d(PLANE as u32);

        let q = into_contiguous(query);
        let k = into_contiguous(key);
        let v = into_contiguous(value);
        let mask = into_contiguous(pad_mask);

        let [batch, heads, seq, dim] = q.meta.shape().dims::<4>();
        assert_eq!(dim, HEAD_DIM, "kernel is specialized for 64-wide heads");
        assert_eq!(
            dtype, burn::backend::DType::F32,
            "the kernel path requires F32 compute (LAYA_DTYPE unset)"
        );

        let shape_out = q.meta.shape().clone();
        let buffer = q.client.empty(shape_out.num_elements() * dtype.size());
        let output = CubeTensor::new_contiguous(
            q.client.clone(),
            q.device.clone(),
            shape_out,
            buffer,
            dtype,
        );

        if use_v2() {
            let blocks = seq.div_ceil(ROWS_PER_CUBE);
            let cube_count = CubeCount::Static(blocks as u32, heads as u32, batch as u32);
            kernel::windowed_attention_kernel_v2::launch(
                &output.client,
                cube_count,
                cube_dim,
                q.into_tensor_arg(),
                k.into_tensor_arg(),
                v.into_tensor_arg(),
                mask.into_tensor_arg(),
                output.clone().into_tensor_arg(),
                seq as u32,
                dtype_to_storage_type(dtype),
            );
        } else {
            let cube_count = CubeCount::Static(seq as u32, heads as u32, batch as u32);
            kernel::windowed_attention_kernel_v1::launch(
                &output.client,
                cube_count,
                cube_dim,
                q.into_tensor_arg(),
                k.into_tensor_arg(),
                v.into_tensor_arg(),
                mask.into_tensor_arg(),
                output.clone().into_tensor_arg(),
                seq as u32,
                dtype_to_storage_type(dtype),
            );
        }

        output
    }
}
