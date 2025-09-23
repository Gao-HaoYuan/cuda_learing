#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <limits>

template <typename T> __device__ __forceinline__ T neg_inf();
template <> __device__ __forceinline__ float neg_inf<float>()  { return -INFINITY; }
template <> __device__ __forceinline__ double neg_inf<double>() { return -INFINITY; }

template <typename T>
__device__ __forceinline__ T blockReduceMax(T val) {
    unsigned mask = __activemask();
    int lane = threadIdx.x & (warpSize - 1);
    int active = __popc(mask);
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        T other = __shfl_down_sync(mask, val, offset);
        if (lane + offset < active) {
            val = val > other ? val : other;
        }
    }

    __shared__ T shared[32];
    int wid = threadIdx.x / warpSize;
    if (lane == 0) shared[wid] = val;
    __syncthreads();


    T warp_val = neg_inf<T>();
    int warp_count = (blockDim.x + warpSize - 1) / warpSize;
    if (wid == 0) {
        if (lane < warp_count) warp_val = shared[lane];

        unsigned mask0 = __activemask();
        int active0 = __popc(mask0);
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            T other = __shfl_down_sync(mask0, warp_val, offset);
            if (lane + offset < active0) {
                warp_val = warp_val > other ? warp_val : other;
            }
        }
    }
    return warp_val;
}

/**
 * amax(dim=axis, keepdim=True)
 * 将输入张量视为 (outer, reduce, inner)，
 * 每个 block 处理一个 (outer_idx, inner_idx) 对，沿 reduce 轴取最大值。
 * 输出张量视为 (outer, 1, inner)，保持 keepdim=True。
 *
 * 线性索引：
 *   in_idx  = ((outer * reduce_size) + r) * inner_size + inner
 *   out_idx = outer * inner_size + inner   // 因为输出维度是 (outer, 1, inner)
 */
template <typename scalar_t>
__global__ void amax_axis_keepdim_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t outer_size,
    int64_t reduce_size,
    int64_t inner_size)
{
    int64_t b = blockIdx.x;
    if (b >= outer_size * inner_size) return;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    int64_t outer = b / inner_size;
    int64_t inner = b % inner_size;

    scalar_t local_max = neg_inf<scalar_t>();
    for (int64_t r = tid; r < reduce_size; r += nthreads) {
        int64_t idx = (outer * reduce_size + r) * inner_size + inner;
        scalar_t v = input[idx];
        local_max = v > local_max ? v : local_max;
    }

    scalar_t block_max = blockReduceMax<scalar_t>(local_max);

    if (threadIdx.x == 0) {
        output[b] = block_max;
    }
}

torch::Tensor amax_axis_keepdim_cuda(torch::Tensor input, int axis) {
    // 计算 outer / reduce / inner
    int64_t outer_size = 1;
    for (int i = 0; i < axis; ++i) outer_size *= input.size(i);
    int64_t reduce_size = input.size(axis);
    int64_t inner_size = 1;
    for (int i = axis + 1; i < input.dim(); ++i) inner_size *= input.size(i);

    // 输出尺寸（keepdim=True）
    std::vector<int64_t> out_sizes(input.sizes().begin(), input.sizes().end());
    out_sizes[axis] = 1;
    auto output = torch::empty(out_sizes, input.options());

    int64_t blocks = outer_size * inner_size;
    int threads = std::min<int64_t>(reduce_size, 512);

    dim3 grid(blocks);
    dim3 block(threads);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "amax_axis_keepdim_kernel", [&]{
        amax_axis_keepdim_kernel<scalar_t><<<grid, block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size, reduce_size, inner_size);
    });

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    return output;
}

torch::Tensor amax_axis_keepdim(torch::Tensor input, int axis) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(axis >= 0 && axis < input.dim(), "axis out of range");
    TORCH_CHECK(input.scalar_type() == torch::kFloat || input.scalar_type() == torch::kDouble,
                "only float32/float64 are supported in this build");

    return amax_axis_keepdim_cuda(input, axis);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("amax_axis_keepdim", &amax_axis_keepdim,
          "amax along axis with keepdim=True (CUDA, float/double)");
}