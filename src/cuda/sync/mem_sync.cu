#include <cstdio>
#include <cuda_awbarrier_primitives.h>

#include "minitest.hpp"

__global__ void mbarrier_drop_demo(int* data) {
    extern __shared__ int smem[]; // shared memory buffer
    typedef __mbarrier_t barrier_t;
    __shared__ barrier_t bar;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // 初始化 barrier
    if (tid == 0) __mbarrier_init(&bar, num_threads);
    __syncthreads();

    // 假设偶数线程提前退出
    if (tid % 2 == 0) {
        __mbarrier_arrive_and_drop(&bar); // 当前阶段完成，同时不参与下一阶段
        return; // 提前退出循环
    }

    // 其他线程正常工作
    smem[tid] = data[tid] * 2;
    __mbarrier_token_t token = __mbarrier_arrive(&bar);

    // 等待所有参与线程到达
    while (!__mbarrier_test_wait(&bar, token));

    printf("smem val is %d\n", smem[tid]);

    // 写回 global memory
    data[tid] = smem[tid];

    // 失效 barrier
    if (tid == 0) __mbarrier_inval(&bar);
}

TEST(CUDA, Mem_Sync) {
    const int N = 4;
    int host_data[N] = {1,2,3,4};

    int* dev_data;
    cudaMalloc(&dev_data, N * sizeof(int));
    cudaMemcpy(dev_data, host_data, N * sizeof(int), cudaMemcpyHostToDevice);

    mbarrier_drop_demo<<<1, 4, 4 * sizeof(int)>>>(dev_data);
    cudaDeviceSynchronize();

    cudaMemcpy(host_data, dev_data, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Result:\n");
    for (int i = 0; i < N; ++i) printf("%d ", host_data[i]);
    printf("\n");

    cudaFree(dev_data);
}
