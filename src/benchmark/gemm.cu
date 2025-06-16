#include <vector>
#include <random>

#include "perf.h"

__global__ void bit_matmul_3d_kernel(
    const int8_t* a,
    const float_t* b,
    const float_t* w,
    float* c,
    int m,
    int n,
    int k,
    int bit_nums
) {
    // 使用共享内存缓存数据
    __shared__ float shared_a[32][32];  // 调整为32x32以匹配block大小
    __shared__ float shared_b[32][32];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.0f;
        // 遍历每个bit位置
        for (int a_bit = 0; a_bit < bit_nums; a_bit++) {
            for (int b_bit = 0; b_bit < bit_nums; b_bit++) {
                float bit_sum = 0.0f;
                // 分块计算
                for (int tile = 0; tile < (k + 31) / 32; tile++) {  // 调整为32以匹配block大小
                    // 加载数据到共享内存
                    if (row < m && tile * 32 + threadIdx.x < k) {
                        int a_idx = a_bit * m * k + row * k + tile * 32 + threadIdx.x;
                        if (a_idx < bit_nums * m * k) {  // 添加边界检查
                            shared_a[threadIdx.y][threadIdx.x] = static_cast<float>(a[a_idx]);
                        } else {
                            shared_a[threadIdx.y][threadIdx.x] = 0.0f;
                        }
                    } else {
                        shared_a[threadIdx.y][threadIdx.x] = 0.0f;
                    }
                    if (col < n && tile * 32 + threadIdx.y < k) {
                        int b_idx = b_bit * k * n + (tile * 32 + threadIdx.y) * n + col;
                        if (b_idx < bit_nums * k * n) {  // 添加边界检查
                            shared_b[threadIdx.y][threadIdx.x] = b[b_idx];
                        } else {
                            shared_b[threadIdx.y][threadIdx.x] = 0.0f;
                        }
                    } else {
                        shared_b[threadIdx.y][threadIdx.x] = 0.0f;
                    }
                    __syncthreads();
                    // 计算当前块的结果
                    float block_sum = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < 32; i++) {  // 调整为32以匹配block大小
                        block_sum += shared_a[threadIdx.y][i] * shared_b[i][threadIdx.x];
                    }
                    bit_sum += block_sum;
                    __syncthreads();
                }
                // 应用权重
                int w_idx = a_bit * bit_nums + b_bit;
                if (w_idx < bit_nums * bit_nums) {  // 添加边界检查
                    float w_val = w[w_idx];
                    sum += bit_sum * w_val;
                }
            }
        }
        // 写入结果
        if (row < m && col < n) {
            c[row * n + col] = sum;
        }
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

    dim3 blockDim(32, 32);
    dim3 gridDim((N + 31) / 32, (M + 31) / 32);

    // warm up
    gpu_warmup();

    perf_performance(bit_matmul_3d_kernel, "origin matmul", gridDim, blockDim, A_d, B_d, W_d, C_d, M, N, K, B);

    // 拷贝结果回 CPU
    CHECK_CUDA(cudaMemcpy(h_C.data(), C_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    // 打印 C 的前 5x5 矩阵
    std::cout << "C matrix (2x2 block):" << std::endl;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            std::cout << h_C[i * N + j] << "\t";
        }
        std::cout << std::endl;
    }

    // 释放资源
    CHECK_CUDA(cudaFree(A_d));
    CHECK_CUDA(cudaFree(B_d));
    CHECK_CUDA(cudaFree(C_d));
    CHECK_CUDA(cudaFree(W_d));

    return 0;
}