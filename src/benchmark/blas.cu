#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "perf.h"

#define TILE_SIZE 16

__device__ __forceinline__
float gemm_bit_plane(
    const int8_t* __restrict__ a,  // [m, k]
    const float* __restrict__ b,  // [k, n]
    int m, int n, int k,
    int row, int col
) {
    __shared__ float shared_a[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_b[TILE_SIZE][TILE_SIZE];

    float bit_sum = 0.0f;
    for (int tile = 0; tile < k; tile += TILE_SIZE) {
        int k_a = tile + threadIdx.x;
        int k_b = tile + threadIdx.y;

        if (k_a < k)
            shared_a[threadIdx.y][threadIdx.x] = a[row * k + k_a];
        else
            shared_a[threadIdx.y][threadIdx.x] = 0.0f;

        if (k_b < k)
            shared_b[threadIdx.y][threadIdx.x] = b[k_b * n + col];
        else
            shared_b[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        float block_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            block_sum += shared_a[threadIdx.y][i] * shared_b[i][threadIdx.x];
        }

        bit_sum += block_sum;
        __syncthreads();
    }

    return bit_sum;
}

__global__ void matmul_ex(
    const int8_t* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m, int n, int k,
    float alpha,
    float beta
) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (row >= m || col >= n) return;

    float sum = gemm_bit_plane(a, b, m, n, k, row, col);
    sum *= alpha;

    c[row * n + col] = sum + beta * c[row * n + col];
}

#define TILE_M 128
#define TILE_N 128
#define TILE_K 32
#define THREADS_X 16
#define THREADS_Y 16

__global__ void gemm_kernel(
    const int8_t* __restrict__ A,  // [M, K]
    const float* __restrict__ B,  // [K, N]
    float* __restrict__ C,        // [M, N]
    int M, int N, int K,
    float alpha,
    float beta
) {
    // 每个 block 负责输出 C 的 TILE_M x TILE_N 块
    const int block_row = blockIdx.y * TILE_M;
    const int block_col = blockIdx.x * TILE_N;

    // 每个线程处理 (TILE_M / THREADS_Y) × (TILE_N / THREADS_X) 个输出元素
    const int thread_row_base = threadIdx.y * (TILE_M / THREADS_Y);
    const int thread_col_base = threadIdx.x * (TILE_N / THREADS_X);

    // 本地寄存器缓存
    float acc[TILE_M / THREADS_Y][TILE_N / THREADS_X] = {0.0f};

    // Shared memory
    __shared__ float Asub[TILE_M][TILE_K];
    __shared__ float Bsub[TILE_K][TILE_N];

    for (int tile_k = 0; tile_k < K; tile_k += TILE_K) {
        // Load Asub [TILE_M x TILE_K]
        for (int i = 0; i < TILE_M / THREADS_Y; ++i) {
            for (int j = 0; j < TILE_K / THREADS_X; ++j) {
                int row = thread_row_base + i;
                int col = threadIdx.x * (TILE_K / THREADS_X) + j;
                if ((block_row + row) < M && (tile_k + col) < K) {
                    Asub[row][col] = A[(block_row + row) * K + (tile_k + col)];
                } else {
                    Asub[row][col] = 0.0f;
                }
            }
        }

        // Load Bsub [TILE_K x TILE_N]
        for (int i = 0; i < TILE_K / THREADS_Y; ++i) {
            for (int j = 0; j < TILE_N / THREADS_X; ++j) {
                int row = threadIdx.y * (TILE_K / THREADS_Y) + i;
                int col = thread_col_base + j;
                if ((tile_k + row) < K && (block_col + col) < N) {
                    Bsub[row][col] = B[(tile_k + row) * N + (block_col + col)];
                } else {
                    Bsub[row][col] = 0.0f;
                }
            }
        }

        __syncthreads();

        // Compute partial result
        for (int k_inner = 0; k_inner < TILE_K; ++k_inner) {
            for (int i = 0; i < TILE_M / THREADS_Y; ++i) {
                float a = Asub[thread_row_base + i][k_inner];
                for (int j = 0; j < TILE_N / THREADS_X; ++j) {
                    float b = Bsub[k_inner][thread_col_base + j];
                    acc[i][j] += a * b;
                }
            }
        }

        __syncthreads();
    }

    // Write back to C
    #pragma unroll
    for (int i = 0; i < TILE_M / THREADS_Y; ++i) {
        int row = block_row + thread_row_base + i;
        if (row >= M) continue;
        for (int j = 0; j < TILE_N / THREADS_X; ++j) {
            int col = block_col + thread_col_base + j;
            if (col >= N) continue;
            C[row * N + col] = alpha * acc[i][j] + beta * C[row * N + col];
        }
    }
}

__global__ void int8_to_float_kernel(const int8_t* A_int8, float* A_fp32, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A_fp32[idx] = static_cast<float>(A_int8[idx]);
    }
}

int main() {
    // 矩阵尺寸定义
    const int M = 2048;
    const int K = 4096;
    const int N = 1040;
    const int B = 14;

    // Host-side data
    std::vector<int8_t> h_A(B * M * K);
    std::vector<float> h_B(B * K * N);
    std::vector<float> h_W(B * B);
    std::vector<float> h_C(M * N, 0.f);

    // 随机初始化
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (auto& x : h_A) x = 2.0;
    for (auto& x : h_B) x = 4.0;
    for (auto& x : h_W) x = 3.0;

    // 设备内存分配
    int8_t *A_d;
    float *B_d, *C_d, *W_d;
    CHECK_CUDA(cudaMalloc(&A_d, B * M * K * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&B_d, B * K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&W_d, B * B * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&C_d, M * N * sizeof(float)));

    // 拷贝到设备
    CHECK_CUDA(cudaMemcpy(A_d, h_A.data(), B * M * K * sizeof(int8_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(B_d, h_B.data(), B * K * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(W_d, h_W.data(), B * B * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(C_d, 0, M * N * sizeof(float))); // 清零 C_d

    // cuBLAS 初始化
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // 创建 CUDA stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    cublasSetStream(handle, stream);

    const int tile = TILE_SIZE;
    int gm = (M + tile - 1) / tile;
    int gn = (N + tile - 1) / tile;

    // 设置 block 和 grid 尺寸（32×32 block）
    dim3 bDim(tile, tile);
    dim3 gDim(gm, gn);

    dim3 blockDim(THREADS_X, THREADS_Y);
    dim3 gridDim((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);

    // warm up
    gpu_warmup();

    float *A_t;
    CHECK_CUDA(cudaMalloc(&A_t, M * K * sizeof(float)));

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, stream));

    // 叠加矩阵乘法
    for (int a_bit = 0; a_bit < B; ++a_bit) {
        const int8_t* T_A = A_d + a_bit * M * K;
        // int8_to_float_kernel<<<512, 1024, 0, stream>>>(T_A, A_t, M * K);

        for (int b_bit = 0; b_bit < B; ++b_bit) {
            const float* T_B = B_d + b_bit * K * N;
            float alpha = h_W[a_bit * B + b_bit];
            float beta = 1.f;

            // CHECK_CUBLAS(cublasGemmEx(
            //     handle,
            //     CUBLAS_OP_N, CUBLAS_OP_N,
            //     N, M, K,
            //     &alpha,
            //     T_B, CUDA_R_32F, N,
            //     A_t, CUDA_R_32F, K,
            //     &beta,
            //     C_d, CUDA_R_32F, N,
            //     CUDA_R_32F,
            //     CUBLAS_GEMM_DEFAULT
            // ));

            // matmul_ex<<<gDim, bDim, 0, stream>>>(T_A, T_B, C_d, M, N, K, alpha, beta);
            gemm_kernel<<<gridDim, blockDim, 0, stream>>>(T_A, T_B, C_d, M, N, K, alpha, beta);
        }
    }

    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    std::cout << "Total GEMM time: " << elapsed_ms << " ms" << std::endl;

    // 拷贝结果回 CPU
    CHECK_CUDA(cudaMemcpy(h_C.data(), C_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // 打印 C 的前 5x5 矩阵
    std::cout << "C matrix (2x2 block):" << std::endl;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            std::cout << h_C[i * N + j] << "\t";
        }
        std::cout << std::endl;
    }

    // 释放资源
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(A_d));
    CHECK_CUDA(cudaFree(B_d));
    CHECK_CUDA(cudaFree(C_d));
    CHECK_CUDA(cudaFree(W_d));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaStreamDestroy(stream));

    return 0;
}
