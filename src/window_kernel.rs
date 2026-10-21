//! Custom windowed-attention kernel via #[backend_extension] - no patched
//! dependencies, everything from crates.io.
//!
//! Layout: one cube (plane of 32 units) per (batch, head, query) row. Each
//! unit owns two head-dim components (d = unit and d = unit + 32, for the
//! 64-wide heads of the Laya encoder). The score dot product is reduced with
//! a plane sum; softmax runs online (flash-style) over the query's window
//! keys only - keys outside |i - j| <= W2 are never read at all.

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
    use super::{HEAD_DIM, PLANE, WINDOW};
    use cubecl::{cube, prelude::*};

    #[cube(launch)]
    pub fn windowed_attention_kernel<F: Float>(
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
        let d_per_unit = HEAD_DIM / PLANE; // 2

        if i >= seq_len as usize {
            terminate!();
        }

        let q_base = ((b * heads + h) * seq_len as usize + i) * HEAD_DIM;
        // Own components: u, u + PLANE.
        let mut q0 = F::new(0.0);
        let mut q1 = F::new(0.0);
        if u < PLANE {
            q0 = query[q_base + u];
            q1 = query[q_base + u + PLANE];
        }

        let kv_head_stride = seq_len as usize * HEAD_DIM;
        let mask_base = b * seq_len as usize;

        let scale = F::new(1.0 / (HEAD_DIM as f32).sqrt());
        let neg_inf = F::new(-1.0e9);

        // Online softmax state (per unit; scalars identical across units).
        let mut m = F::new(-3.0e38);
        let mut l = F::new(0.0);
        let mut acc0 = F::new(0.0);
        let mut acc1 = F::new(0.0);

        let j_lo = i.saturating_sub(WINDOW);
        let j_hi = (i + WINDOW + 1).min(seq_len as usize);

        for j in j_lo..j_hi {
            let kv_base = ((b * heads + h) * seq_len as usize + j) * HEAD_DIM;
            // Dot product over this unit's components, then across the plane.
            let partial = q0 * key[kv_base + u] + q1 * key[kv_base + u + PLANE];
            let dot = plane_sum(partial);
            let keep = pad_mask[mask_base + j];
            // keep=1 (real) adds 0; keep=0 (pad) adds -1e9.
            let s = dot * scale + (F::new(1.0) - keep) * neg_inf;

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

        let shape_out = q.meta.shape().clone();
        let buffer = q.client.empty(shape_out.num_elements() * dtype.size());
        let output = CubeTensor::new_contiguous(
            q.client.clone(),
            q.device.clone(),
            shape_out,
            buffer,
            dtype,
        );

        let cube_count = CubeCount::Static(seq as u32, heads as u32, batch as u32);
        kernel::windowed_attention_kernel::launch(
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

        output
    }
}
