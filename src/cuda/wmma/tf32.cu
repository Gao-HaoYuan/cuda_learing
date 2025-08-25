#include <cuda.h>
#include <mma.h>
#include <stdio.h>

#include "minitest.hpp"

using namespace nvcuda::wmma;

__device__ float to_tf32(float x) {
    return  __float_to_tf32(x);
}

__global__ void test_wmma_tf32(const float *a, const float *b, float *c) {
    fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a_frag;
    fragment<matrix_b, 16, 16, 8, precision::tf32, col_major> b_frag;
    fragment<accumulator, 16, 16, 8, float> c_frag;

    fill_fragment(c_frag, 0.0f);

    // 临时 buffer，用于存放 tf32 转换后的矩阵
    __shared__ float A_tf32[16*8];
    __shared__ float B_tf32[8*16];

    int tid = threadIdx.x;
    if (tid < 16*8) A_tf32[tid] = to_tf32(a[tid]);
    if (tid < 8*16) B_tf32[tid] = to_tf32(b[tid]);
    __syncthreads();

    // 加载转换后的矩阵
    load_matrix_sync(a_frag, A_tf32, 16);
    load_matrix_sync(b_frag, B_tf32, 8);

    // 执行 Tensor Core 矩阵乘法
    mma_sync(c_frag, a_frag, b_frag, c_frag);

    // 保存结果
    store_matrix_sync(c, c_frag, 16, mem_row_major);
}

TEST(WMMA, TF32) {
    const int M = 16, N = 16, K = 8;
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float *hA = new float[M * K];
    float *hB = new float[K * N];
    float *hC = new float[M * N];

    for (int i = 0; i < M*K; i++) hA[i] = (float)(i % 3 + 1);
    for (int i = 0; i < K*N; i++) hB[i] = (float)(i % 5 + 1);

    float *dA, *dB, *dC;
    cudaMalloc(&dA, sizeA);
    cudaMalloc(&dB, sizeB);
    cudaMalloc(&dC, sizeC);

    cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice);

    test_wmma_tf32<<<1, 32>>>(dA, dB, dC);
    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, sizeC, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 8; i++) {
        printf("%f ", hC[i]);
    }
    printf("\n");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA; delete[] hB; delete[] hC;
}
