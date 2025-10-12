// kernel_strided.cu
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t, typename acc_t>
__global__ void mean_strided_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    int64_t out_elems,                       // = 乘积(keep_sizes)
    const int64_t* __restrict__ keep_sizes,  // [K]
    const int64_t* __restrict__ keep_strides,// [K]
    int K,                                   // keep dims count
    const int64_t* __restrict__ red_sizes,   // [R]
    const int64_t* __restrict__ red_strides, // [R]
    int R,                                   // reduce dims count
    int64_t reduce_elems)                    // = 乘积(red_sizes)
{
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= out_elems) return;

    // 1) 由 out_idx 解出 keep 维 multi-index，并计算 base offset
    int64_t tmp = out_idx;
    int64_t base_off = 0;
    #pragma unroll
    for (int i = K - 1; i >= 0; --i) {
        int64_t size_i = keep_sizes[i];
        int64_t idx_i  = tmp % size_i;
        tmp /= size_i;
        base_off += idx_i * keep_strides[i];
    }

    // 2) 遍历 reduce 维（混合进制），求和
    acc_t sum = acc_t(0);
    // 让每个线程以 grid-stride 方式遍历 reduce 线性索引
    for (int64_t ridx = threadIdx.y; ridx < reduce_elems; ridx += blockDim.y) {
        int64_t rtmp = ridx;
        int64_t off = base_off;
        #pragma unroll
        for (int j = R - 1; j >= 0; --j) {
            int64_t sz = red_sizes[j];
            int64_t ij = rtmp % sz;
            rtmp /= sz;
            off += ij * red_strides[j];
        }
        sum += static_cast<acc_t>(x[off]);
    }

    // 3) block 内在 y 方向（threadIdx.y）做归约
    extern __shared__ unsigned char smem_raw[]; // 足够小的共享内存
    acc_t* smem = reinterpret_cast<acc_t*>(smem_raw);
    int lane = threadIdx.y;
    smem[threadIdx.x * blockDim.y + lane] = sum;
    __syncthreads();

    // 简单串行归约（blockDim.y 一般不大）
    if (lane == 0) {
        acc_t s = acc_t(0);
        for (int t = 0; t < blockDim.y; ++t) {
            s += smem[threadIdx.x * blockDim.y + t];
        }
        y[out_idx] = static_cast<scalar_t>(s / static_cast<acc_t>(reduce_elems));
    }
}

void launch_mean_strided(
    const at::Tensor& x, at::Tensor& y,
    at::Tensor keep_sizes, at::Tensor keep_strides,
    at::Tensor red_sizes,  at::Tensor red_strides)
{
    TORCH_CHECK(x.is_cuda() && y.is_cuda(), "x/y must be CUDA tensors");
    TORCH_CHECK(keep_sizes.dtype() == at::kLong && keep_strides.dtype() == at::kLong, "keep meta must be int64");
    TORCH_CHECK(red_sizes.dtype()  == at::kLong && red_strides.dtype()  == at::kLong, "reduce meta must be int64");

    const int K = keep_sizes.numel();
    const int R = red_sizes.numel();

    // 计算 out_elems / reduce_elems（在 host 已算好也行）
    int64_t out_elems = 1, reduce_elems = 1;
    auto h_keep_sizes = keep_sizes.cpu();
    auto h_red_sizes  = red_sizes.cpu();
    for (int i = 0; i < K; ++i) out_elems  *= h_keep_sizes.data_ptr<int64_t>()[i];
    for (int j = 0; j < R; ++j) reduce_elems *= h_red_sizes.data_ptr<int64_t>()[j];

    if (out_elems == 0) return;

    dim3 threads(32, 8); // x 方向负责不同输出y元素，y 方向分摊 reduce 维的求和工作
    dim3 blocks( (out_elems + threads.x - 1) / threads.x );

    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "mean_strided", [&]{
        using acc_t = typename std::conditional<std::is_same<scalar_t,double>::value,double,float>::type;
        
        size_t shmem_bytes = sizeof(acc_t) * threads.x * threads.y;  // 32*8

        mean_strided_kernel<scalar_t, acc_t><<<blocks, threads, shmem_bytes, stream>>>(
            x.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            out_elems,
            keep_sizes.data_ptr<int64_t>(),
            keep_strides.data_ptr<int64_t>(),
            K,
            red_sizes.data_ptr<int64_t>(),
            red_strides.data_ptr<int64_t>(),
            R,
            reduce_elems
        );
    });
    AT_CUDA_CHECK(cudaGetLastError());
}
