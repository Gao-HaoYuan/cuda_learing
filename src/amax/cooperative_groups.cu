#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <limits>
#include <algorithm>

#include "util.h"
#include "inline_ops.cuh"

using torch::Tensor;
namespace cg = cooperative_groups;

template <typename scalar_t>
__global__ void quantize_bias_per_channel_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    scalar_t min_val,
    scalar_t max_val,
    int numel
) {
    cg::grid_group grid = cg::this_grid();

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    scalar_t local_max = neg_inf<scalar_t>();
    scalar_t val = input[idx];
    local_max = abs(val) > local_max ? abs(val) : local_max;

    scalar_t block_max = blockReduceMax<scalar_t>(local_max);

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        output[0] = neg_inf<scalar_t>();
    }
    grid.sync();

    if (threadIdx.x == 0) {
        atomicMax(&output[0], block_max);
    }
    grid.sync();

    scalar_t abs_max = max(output[0], (scalar_t)1e-8);
    scalar_t scale = max_val / abs_max;

    val = val * scale / static_cast<scalar_t>(128.0);
    val = static_cast<scalar_t>(qround_rn(val)) * static_cast<scalar_t>(128.0);
    val = max(min_val, min(max_val, val));
    output[idx] = val;
}

void quantize_bias_per_channel_cuda(
    const Tensor& input,
    Tensor& output,
    int bias_row_n
) {
    int numel = input.numel();

    int coop = 0;
    cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, input.get_device());
    TORCH_CHECK(coop, "GPU does not support cooperative launch");

    dim3 block(512);
    dim3 grid((numel + block.x - 1) / block.x);
    
    at::cuda::CUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "quantize_per_channel_one_kernel", [&]{
        // 估算可同时驻留的最大 blocks，避免 grid.sync 死锁
        int maxBlocksPerSM = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSM,
            quantize_bias_per_channel_kernel<scalar_t>,
            block.x, 0);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, input.get_device());
        int maxGrid = maxBlocksPerSM * prop.multiProcessorCount;
        TORCH_CHECK(grid.x <= (unsigned)maxGrid, "grid too large for cooperative launch");

        scalar_t* in_ptr  = input.data_ptr<scalar_t>();
        scalar_t* out_ptr = output.data_ptr<scalar_t>();

        scalar_t min_val = static_cast<scalar_t>(-128 * bias_row_n);
        scalar_t max_val = static_cast<scalar_t>( 127 * bias_row_n);
        int num = static_cast<int>(numel);

        void* args[] = {
            &in_ptr,
            &out_ptr,
            &min_val,
            &max_val,
            &num
        };

        cudaError_t err = cudaLaunchCooperativeKernel(
            (void*)quantize_bias_per_channel_kernel<scalar_t>,
            grid,
            block,
            args,
            0,
            stream
        );
    });

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
}

Tensor quantize_bias_per_channel(
    const Tensor& input,
    int bias_row_n
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    Tensor output = empty_like(input);
    quantize_bias_per_channel_cuda(input, output, bias_row_n);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_bias_per_channel", &quantize_bias_per_channel,
          "Per-channel bias quantization (single-kernel, CUDA, float/double only)");
}
