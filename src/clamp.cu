#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

constexpr int N = 1024 * 1024;
constexpr int ITER = 100;

template<typename T>
__device__ __forceinline__
T clamp_if(T t1, T min_val, T max_val, int8_t* clamp_flag) {
    if (t1 < min_val) {
        *clamp_flag = static_cast<int8_t>(1);
        return min_val;
    } else if (t1 > max_val) {
        *clamp_flag = static_cast<int8_t>(1);
        return max_val;
    } else {
        *clamp_flag = static_cast<int8_t>(0);
        return t1;
    }
}

template<typename T>
__device__ __forceinline__
T clamp_minmax(T t1, T min_val, T max_val, int8_t* clamp_flag) {
    T clip = min(max(t1, min_val), max_val);
    *clamp_flag = (t1 == clip) ? 0 : 1;
    return clip;
}

__global__ void kernel_clamp_if(float* out, const float* in, int8_t* flags, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float val = in[idx];
    int8_t flag = 0;
    #pragma unroll
    for (int i = 0; i < ITER; ++i) {
        val = clamp_if(val, min_val, max_val, &flag);
    }
    out[idx] = val;
    flags[idx] = flag;
}

__global__ void kernel_clamp_minmax(float* out, const float* in, int8_t* flags, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float val = in[idx];
    int8_t flag = 0;
    #pragma unroll
    for (int i = 0; i < ITER; ++i) {
        val = clamp_minmax(val, min_val, max_val, &flag);
    }
    out[idx] = val;
    flags[idx] = flag;
}

template<typename KernelFunc, typename... Args>
float benchmark_performance(KernelFunc&& kernel, const char* name, Args&&... args) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    std::forward<KernelFunc>(kernel)<<<(N + 255) / 256, 256>>>(std::forward<Args>(args)...);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("%s time: %.3f ms\n", name, ms);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

int main() {
    float *d_in, *d_out;
    int8_t* d_flag;

    cudaMalloc(&d_in,  N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMalloc(&d_flag, N * sizeof(int8_t));

    // 生成一些值 [-150, 150]
    float* h_in = new float[N];
    for (int i = 0; i < N; ++i) h_in[i] = (rand() % 300) - 150;
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    printf("Benchmarking clamp implementations...\n");
    benchmark_performance(kernel_clamp_if,     "clamp_if",     d_out, d_in, d_flag, 0.0f, 127.0f);
    benchmark_performance(kernel_clamp_minmax, "clamp_minmax", d_out, d_in, d_flag, 0.0f, 127.0f);

    benchmark_performance(kernel_clamp_if,     "clamp_if",     d_out, d_in, d_flag, 0.0f, 127.0f);
    benchmark_performance(kernel_clamp_minmax, "clamp_minmax", d_out, d_in, d_flag, 0.0f, 127.0f);

    // 结果比较
    float* h_out_if = new float[N];
    float* h_out_mm = new float[N];
    int8_t* h_flag_if = new int8_t[N];
    int8_t* h_flag_mm = new int8_t[N];

    // 分别重新计算两个版本结果
    kernel_clamp_if<<<(N + 255) / 256, 256>>>(d_out, d_in, d_flag, 0.0f, 127.0f);
    cudaMemcpy(h_out_if, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_flag_if, d_flag, N * sizeof(int8_t), cudaMemcpyDeviceToHost);

    kernel_clamp_minmax<<<(N + 255) / 256, 256>>>(d_out, d_in, d_flag, 0.0f, 127.0f);
    cudaMemcpy(h_out_mm, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_flag_mm, d_flag, N * sizeof(int8_t), cudaMemcpyDeviceToHost);

    // 对比输出
    int err_count = 0;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_out_if[i] - h_out_mm[i]) > 1e-6f || h_flag_if[i] != h_flag_mm[i]) {
            if (err_count < 10) {
                printf("Mismatch at %d: out_if=%.3f, out_mm=%.3f | flag_if=%d, flag_mm=%d\n",
                    i, h_out_if[i], h_out_mm[i], h_flag_if[i], h_flag_mm[i]);
            }
            err_count++;
        }
    }

    if (err_count == 0)
        printf("✅ clamp_if 和 clamp_minmax 输出一致！\n");
    else
        printf("❌ clamp 结果有 %d 处不同（前 10 处已列出）\n", err_count);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_flag);
    delete[] h_in;

    // 清理
    delete[] h_out_if;
    delete[] h_out_mm;
    delete[] h_flag_if;
    delete[] h_flag_mm;

    return 0;
}
