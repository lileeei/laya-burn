//! Spike: custom CubeCL kernel via #[backend_extension], crates.io deps only.
//! Flat structure mirroring burn's examples/custom-cubecl-kernel (lib.rs level).

use burn::{
    backend::{Dispatch, backend_extension, tensor::FloatTensor},
    tensor::{Shape, Tensor, activation},
};

#[backend_extension(Cube, Fusion)]
pub trait SpikeBackend: burn::backend::Backend {
    #[fusion(dtype = lhs, shape = output_shape(lhs, rhs, bias))]
    fn fused_matmul_add_relu(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FloatTensor<Self>;
}

pub fn matmul_add_relu_custom(lhs: Tensor<3>, rhs: Tensor<3>, bias: Tensor<3>) -> Tensor<3> {
    let output = Dispatch::fused_matmul_add_relu(
        lhs.into_dispatch(),
        rhs.into_dispatch(),
        bias.into_dispatch(),
    );
    Tensor::from_dispatch(output)
}

pub fn matmul_add_relu_reference(lhs: Tensor<3>, rhs: Tensor<3>, bias: Tensor<3>) -> Tensor<3> {
    activation::relu(lhs.matmul(rhs) + bias)
}

fn output_shape(lhs: &Shape, rhs: &Shape, bias: &Shape) -> Shape {
    assert!(lhs.num_dims() >= 2, "matmul needs at least two dimensions");
    let shape =
        burn::backend::calculate_matmul_output(lhs, rhs).expect("compatible matmul shapes");
    assert_eq!(&shape, bias, "kernel requires bias to match the output shape");
    shape
}

mod kernel {
    use cubecl::{cube, prelude::*};

    #[cube(launch)]
    pub fn fused_matmul_add_relu_kernel<F: Float>(
        lhs: &Tensor<F>,
        rhs: &Tensor<F>,
        bias: &Tensor<F>,
        output: &mut Tensor<F>,
        #[define(F)] _dtype: ElemType,
    ) {
        let row = ABSOLUTE_POS_X as usize;
        let col = ABSOLUTE_POS_Y as usize;
        let batch = ABSOLUTE_POS_Z as usize;

        let n_rows = output.shape(output.rank() - 2);
        let n_cols = output.shape(output.rank() - 1);
        let dim_k = rhs.shape(rhs.rank() - 2);

        if row >= n_rows || col >= n_cols {
            terminate!();
        }

        let offset_output = batch * n_rows * n_cols;
        let mut offset_lhs = 0;
        let mut offset_rhs = 0;

        let batch_dims = output.rank() - 2;
        for dim in 0..batch_dims {
            offset_lhs += offset_output / output.stride(dim) % lhs.shape(dim) * lhs.stride(dim);
            offset_rhs += offset_output / output.stride(dim) % rhs.shape(dim) * rhs.stride(dim);
        }

        let mut sum = F::new(0.0);
        for k in 0..dim_k {
            let lhs_index = row * dim_k + k;
            let rhs_index = k * n_cols + col;
            sum += lhs[offset_lhs + lhs_index] * rhs[offset_rhs + rhs_index];
        }

        let out_index = row * n_cols + col;
        let index = offset_output + out_index;
        output[index] = F::max(sum + bias[index], F::new(0.0));
    }
}

impl SpikeBackend for burn_cubecl::CubeBackend {
    fn fused_matmul_add_relu(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        use burn::backend::cubecl::dtype_to_storage_type;
        use burn_cubecl::{kernel::into_contiguous, tensor::CubeTensor};
        use cubecl::{CubeCount, CubeDim};

        let dtype = lhs.dtype;
        let cube_dim = CubeDim::new_3d(16, 16, 1);

        lhs.assert_is_on_same_device(&rhs);
        lhs.assert_is_on_same_device(&bias);

        let lhs = into_contiguous(lhs);
        let rhs = into_contiguous(rhs);
        let bias = into_contiguous(bias);

        let shape_out = output_shape(lhs.meta.shape(), rhs.meta.shape(), bias.meta.shape());
        let ndims = lhs.meta.num_dims();
        let num_rows = lhs.meta.shape()[ndims - 2];
        let num_cols = rhs.meta.shape()[ndims - 1];
        let num_batches: usize = (0..ndims - 2).map(|i| shape_out[i]).product();

        let buffer = lhs.client.empty(shape_out.num_elements() * dtype.size());
        let output = CubeTensor::new_contiguous(
            lhs.client.clone(),
            lhs.device.clone(),
            shape_out,
            buffer,
            dtype,
        );

        let cubes_needed_in_x = f32::ceil(num_rows as f32 / cube_dim.x as f32) as u32;
        let cubes_needed_in_y = f32::ceil(num_cols as f32 / cube_dim.y as f32) as u32;
        let cube_count =
            CubeCount::Static(cubes_needed_in_x, cubes_needed_in_y, num_batches as u32);

        kernel::fused_matmul_add_relu_kernel::launch(
            &output.client,
            cube_count,
            cube_dim,
            lhs.into_tensor_arg(),
            rhs.into_tensor_arg(),
            bias.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            dtype_to_storage_type(dtype),
        );

        output
    }
}

fn main() {
    let device = burn::tensor::Device::metal(burn::tensor::DeviceKind::DefaultDevice);

    let lhs = Tensor::<3>::random([2, 32, 64], burn::tensor::Distribution::Default, &device);
    let rhs = Tensor::<3>::random([2, 64, 48], burn::tensor::Distribution::Default, &device);
    let bias = Tensor::<3>::random([2, 32, 48], burn::tensor::Distribution::Default, &device);

    let custom = matmul_add_relu_custom(lhs.clone(), rhs.clone(), bias.clone());
    let reference = matmul_add_relu_reference(lhs, rhs, bias);

    let diff: f32 = (custom - reference).abs().max().into_scalar::<f32>().into();
    println!("SPIKE OK; max |custom - reference| = {diff:?}");
}
