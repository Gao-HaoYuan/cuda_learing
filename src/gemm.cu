#include <vector>

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




// 初始化函数
void init_and_run(int m, int n, int k, int bit_nums) {
    int size_a = bit_nums * m * k;
    int size_b = bit_nums * k * n;
    int size_w = bit_nums * bit_nums;
    int size_c = m * n;

    // 分配 host 内存
    std::vector<int8_t> h_a(size_a);
    std::vector<float> h_b(size_b);
    std::vector<float> h_w(size_w);
    std::vector<float> h_c(size_c, 0.0f);

    // 随机初始化
    for (auto& v : h_a) v = rand() % 2;
    for (auto& v : h_b) v = static_cast<float>(rand() % 100) / 100.0f;
    for (auto& v : h_w) v = static_cast<float>(rand() % 100) / 100.0f;

    // 分配 device 内存
    int8_t* d_a;
    float* d_b;
    float* d_w;
    float* d_c;
    CHECK_CUDA(cudaMalloc(&d_a, size_a * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&d_b, size_b * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w, size_w * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, size_c * sizeof(float)));

    // 拷贝到 device
    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), size_a * sizeof(int8_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), size_b * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w, h_w.data(), size_w * sizeof(float), cudaMemcpyHostToDevice));

    // 设置 block 和 grid 尺寸（32×32 block）
    dim3 blockDim(32, 32);
    dim3 gridDim((n + 31) / 32, (m + 31) / 32);

    // 调用 kernel
    gpu_warmup();
    perf_performance(bit_matmul_3d_kernel, "origin gemm", gridDim, blockDim, d_a, d_b, d_w, d_c, m, n, k, bit_nums);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // 拷贝结果回 host
    CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, size_c * sizeof(float), cudaMemcpyDeviceToHost));

    // 简单验证输出（打印前 5×5）
    std::cout << "Output C (first 5x5 block):" << std::endl;
    for (int i = 0; i < std::min(m, 5); ++i) {
        for (int j = 0; j < std::min(n, 5); ++j) {
            std::cout << h_c[i * n + j] << "\t";
        }
        std::cout << "\n";
    }

    // 清理
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_w));
    CHECK_CUDA(cudaFree(d_c));
}

int main() {
    // 示例输入维度
    int m = 512;
    int n = 512;
    int k = 1024;
    int bit_nums = 16;

    init_and_run(m, n, k, bit_nums);

    return 0;
}