// kernel.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

using at::Tensor;

__global__ void curand_normal_kernel(
    float* __restrict__ out,
    int64_t N,
    unsigned long long seed,
    float mean,
    float std)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    // 每个线程独立的 RNG 状态
    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, 0, &state);  // (seed, sequence, offset, state)

    for (int64_t i = tid; i < N; i += stride) {
        float r = curand_normal(&state); // mean=0, std=1
        out[i] = mean + std * r;
    }
}

void launch_curand_normal_kernel(at::Tensor out, unsigned long long seed, float mean, float std)
{
    TORCH_CHECK(out.is_cuda(), "output must be CUDA tensor");
    TORCH_CHECK(out.scalar_type() == at::kFloat, "only float32 supported");

    const int64_t N = out.numel();
    if (N == 0) return;

    const int threads = 256;
    const int blocks = std::min<int>((N + threads - 1) / threads, 4096);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    curand_normal_kernel<<<blocks, threads, 0, stream>>>(
        out.data_ptr<float>(), N, seed, mean, std);

    AT_CUDA_CHECK(cudaGetLastError());
}
