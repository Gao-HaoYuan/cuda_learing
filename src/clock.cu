#include <stdio.h>
#include <cuda_runtime.h>

#include "minitest.hpp"

__global__ void test_clock(unsigned int* clock_result_32, unsigned long long* clock_result_64) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 32-bit clock
    clock_t start_32 = clock();

    // 64-bit clock
    unsigned long long start_64 = clock64();

    // 做点简单的工作：空转循环
    volatile int dummy = 0;
    for (int i = 0; i < 1000; ++i) {
        dummy += i % 7;
    }

    // 结束时记录
    clock_t end_32 = clock();
    unsigned long long end_64 = clock64();

    // 存结果
    clock_result_32[tid] = end_32 - start_32;
    clock_result_64[tid] = end_64 - start_64;
}

TEST(CUDA, Clock) {
    const int threads = 256;
    const int blocks = 1;
    const int total = threads * blocks;

    // 分配 host 和 device 内存
    unsigned int* h_result_32 = new unsigned int[total];
    unsigned long long* h_result_64 = new unsigned long long[total];
    unsigned int* d_result_32;
    unsigned long long* d_result_64;

    cudaMalloc(&d_result_32, total * sizeof(unsigned int));
    cudaMalloc(&d_result_64, total * sizeof(unsigned long long));

    // 启动 kernel
    test_clock<<<blocks, threads>>>(d_result_32, d_result_64);
    cudaDeviceSynchronize();

    // 拷回 host
    cudaMemcpy(h_result_32, d_result_32, total * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_result_64, d_result_64, total * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // 打印前几个线程的结果
    for (int i = 0; i < 10; ++i) {
        printf("Thread %2d | clock(): %10u cycles | clock64(): %10llu cycles\n",
               i, h_result_32[i], h_result_64[i]);
    }

    // 清理
    delete[] h_result_32;
    delete[] h_result_64;
    cudaFree(d_result_32);
    cudaFree(d_result_64);
}
