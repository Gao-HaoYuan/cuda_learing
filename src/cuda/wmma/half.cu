#include <mma.h>         // WMMA API
#include <cuda_fp16.h>   // half 类型
#include <iostream>

#include "minitest.hpp"

using namespace nvcuda::wmma;

__global__ void wmma_demo_kernel(const half *A, const half *B, float *C, float *D) {
    // 定义 fragment
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag, d_frag;

    // 初始化
    fill_fragment(c_frag, 0.0f);

    // 加载
    load_matrix_sync(a_frag, A, 16);
    load_matrix_sync(b_frag, B, 16);

    // 矩阵乘
    mma_sync(d_frag, a_frag, b_frag, c_frag);

    float alpha = 0.5f;
    if (threadIdx.x == 1)
        for(int t = 0; t < d_frag.num_elements; t++)
            d_frag.x[t] *= alpha;

    printf("num_elements: %d\n", d_frag.num_elements);
    printf("num_storage_elements: %d\n", d_frag.num_storage_elements);
    // 存储
    store_matrix_sync(D, d_frag, 16, mem_row_major);
}

TEST(WMMA, Half) {
    const int M = 16, N = 16, K = 16;
    half *d_A, *d_B;
    float *d_C, *d_D;

    cudaMalloc(&d_A, M*K*sizeof(half));
    cudaMalloc(&d_B, K*N*sizeof(half));
    cudaMalloc(&d_C, M*N*sizeof(float));
    cudaMalloc(&d_D, M*N*sizeof(float));

    // 初始化矩阵 A, B, C
    half h_A[M*K], h_B[K*N];
    for (int i=0; i<M*K; i++) h_A[i] = __float2half(1.0f);
    for (int i=0; i<K*N; i++) h_B[i] = __float2half(1.0f);

    cudaMemcpy(d_A, h_A, M*K*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K*N*sizeof(half), cudaMemcpyHostToDevice);

    // 启动 kernel（每个 warp 完成一个 16x16 tile）
    wmma_demo_kernel<<<1, 32>>>(d_A, d_B, d_C, d_D);

    // 拷回结果
    float h_D[M*N];
    cudaMemcpy(h_D, d_D, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    // 打印部分结果
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            std::cout << h_D[i * 16 + j] << " ";
        }
        std::cout << std::endl;
    }
        

    cudaFree(d_A); cudaFree(d_B);
    cudaFree(d_C); cudaFree(d_D);
}