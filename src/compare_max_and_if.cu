#include <cuda_runtime.h>
#include <cstdio>

template<typename T>
__device__ __forceinline__
T max0_custom(T t1, int8_t* flag) {
    return (t1 > 0) * t1;
}

template<typename T>
__device__ __forceinline__
T max0_if(T t1) {
    return (t1 > 0) ? t1 : T(0);
}

constexpr int N = 1024 * 1024;
constexpr int ITER = 1000;

__global__ void kernel_custom(float* out, const float* in, int8_t* flag) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float val = in[idx];
    #pragma unroll
    for (int i = 0; i < ITER; ++i) {
        val = max0_custom(val, flag);
    }
    out[idx] = val;
}

__global__ void kernel_if(float* out, const float* in) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float val = in[idx];
    #pragma unroll
    for (int i = 0; i < ITER; ++i) {
        val = max0_if(val);
    }
    out[idx] = val;
}

float benchmark_custom(float* out, const float* in, int8_t* flag) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    kernel_custom<<<(N + 255) / 256, 256>>>(out, in, flag);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

float benchmark_if(float* out, const float* in) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    kernel_if<<<(N + 255) / 256, 256>>>(out, in);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

int main() {
    float *d_in, *d_out;
    int8_t *d_flag;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMalloc(&d_flag, N * sizeof(int8_t));

    float time1 = benchmark_custom(d_out, d_in, d_flag);
    float time2 = benchmark_if(d_out, d_in);

    printf("max0_custom time: %.3f ms\n", time1);
    printf("max0_if     time: %.3f ms\n", time2);

    time1 = benchmark_custom(d_out, d_in, d_flag);
    time2 = benchmark_if(d_out, d_in);

    printf("max0_custom time: %.3f ms\n", time1);
    printf("max0_if     time: %.3f ms\n", time2);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_flag);
    return 0;
}
