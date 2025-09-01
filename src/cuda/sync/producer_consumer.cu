#include <vector>
#include <cstdio>
#include <cuda/barrier>
#include <cooperative_groups.h>

#include "minitest.hpp"

using barrier = cuda::barrier<cuda::thread_scope_block>;

__device__ void producer(barrier ready[], barrier filled[], float* buffer, float* in, int N, int buffer_len)
{
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    for (int i = 0; i < (N / buffer_len); ++i) {
        // 等待 buffer 可写
        ready[i%2].arrive_and_wait();

        // 拷贝数据到 buffer
        for (int j = tid; j < buffer_len; j += num_threads) {
            buffer[(i%2)*buffer_len + j] = in[i*buffer_len + j];
        }

        // 通知 buffer 已填充
        barrier::arrival_token token = filled[i%2].arrive();
    }
}

__device__ void consumer(barrier ready[], barrier filled[], float* buffer, float* out, int N, int buffer_len)
{
    int tid = threadIdx.x - warpSize;
    int num_threads = blockDim.x;

    barrier::arrival_token token1 = ready[0].arrive(); /* buffer_0 is ready for initial fill */
    barrier::arrival_token token2 = ready[1].arrive(); /* buffer_1 is ready for initial fill */

    for (int i = 0; i < (N / buffer_len); ++i) {
        // 等待 buffer 填充完成
        filled[i%2].arrive_and_wait();
        // 从 buffer 拷贝到输出
        for (int j = tid; j < buffer_len; j += num_threads) {
            out[i*buffer_len + j] = buffer[(i%2)*buffer_len + j];
        }

        barrier::arrival_token token = ready[i%2].arrive(); /* buffer_(i%2) is ready to be re-filled */
    }
}

//N is the total number of float elements in arrays in and out
__global__ void producer_consumer_pattern(int N, int buffer_len, float* in, float* out) {
    extern __shared__ float buffer[]; // 双缓冲共享内存
    __shared__ barrier bar[4];        // ready0/1 + filled0/1

    auto block = cooperative_groups::this_thread_block();

    // 初始化 barrier
    if (block.thread_rank() < 4)
        init(bar + block.thread_rank(), block.size());

    block.sync(); // bootstrap

    // warp specialization
    if (block.thread_rank() < warpSize)
        producer(bar, bar+2, buffer, in, N, buffer_len);
    else
        consumer(bar, bar+2, buffer, out, N, buffer_len);
}

// -------------------- host main 函数 --------------------
TEST(CUDA, PROCUDE) {
    const int N = 16;
    const int buffer_len = 4;
    const int threadsPerBlock = 64;

    std::vector<float> host_in(N), host_out(N, 0);
    for (int i = 0; i < N; ++i) host_in[i] = i + 1;

    float *dev_in = nullptr, *dev_out = nullptr;
    cudaMalloc(&dev_in, N * sizeof(float));
    cudaMalloc(&dev_out, N * sizeof(float));
    cudaMemcpy(dev_in, host_in.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    size_t shared_mem_size = 2 * buffer_len * sizeof(float);
    producer_consumer_pattern<<<1, threadsPerBlock, shared_mem_size>>>(N, buffer_len, dev_in, dev_out);
    cudaDeviceSynchronize();

    cudaMemcpy(host_out.data(), dev_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output array:\n");
    for (int i = 0; i < N; ++i) printf("%.1f ", host_out[i]);
    printf("\n");

    cudaFree(dev_in);
    cudaFree(dev_out);
}
