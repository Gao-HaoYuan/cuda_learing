#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include "perf.h"

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


int main() {
    float *d_in, *d_out1, *d_out2;
    int8_t* d_flag1, *d_flag2;

    cudaMalloc(&d_in,  N * sizeof(float));
    cudaMalloc(&d_out1, N * sizeof(float));
    cudaMalloc(&d_out2, N * sizeof(float));
    cudaMalloc(&d_flag1, N * sizeof(int8_t));
    cudaMalloc(&d_flag2, N * sizeof(int8_t));


    // 生成一些值 [-150, 150]
    float* h_in = new float[N];
    for (int i = 0; i < N; ++i) {
        h_in[i] = (rand() % 300) - 150;
    }
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    gpu_warmup();
    perf_performance(kernel_clamp_if,     "clamp_if",     (N + 255)/256, 256, d_out1, d_in, d_flag1, 0.0f, 127.0f);
    perf_performance(kernel_clamp_minmax, "clamp_minmax", (N + 255)/256, 256, d_out2, d_in, d_flag2, 0.0f, 127.0f);
    perf_accuracy(d_out1, d_out2, N);

    cudaFree(d_in);
    cudaFree(d_out1);
    cudaFree(d_out2);
    cudaFree(d_flag1);
    cudaFree(d_flag2);
    delete[] h_in;

    return 0;
}
