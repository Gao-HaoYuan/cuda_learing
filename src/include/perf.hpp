/**
 *
    特性	            static	                 inline
    每个.cpp 副本	    ✅ 是	                ❌ 编译器会避免重复定义
    链接可见性	        只对当前 .cpp 可见	       所有使用者可见
    推荐用于	        局部优化或防止命名冲突	    头文件中定义函数
 *
 */

#pragma once

#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " code=" << err << " \"" << cudaGetErrorString(err)   \
                      << "\"" << std::endl;                                    \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

#define CHECK_CUBLAS(call)                                                     \
    do {                                                                       \
        cublasStatus_t status = call;                                          \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            std::cerr << "cuBLAS error " << status << " at " << __FILE__       \
                      << ":" << __LINE__ << std::endl;                         \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

template <typename T>
static bool is_device_pointer(const T *ptr) {
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);

#if CUDART_VERSION >= 10000
    if (err == cudaSuccess && attr.type == cudaMemoryTypeDevice)
#else
    if (err == cudaSuccess && attr.memoryType == cudaMemoryTypeDevice)
#endif
        return true;

    return false;
}

__global__ static void warmup_kernel(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] = data[idx] * 2.0f + 1.0f;
}

inline void gpu_warmup(int repeat = 3, int N = 1024 * 1024) {
    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemset(d_data, 0, N * sizeof(float));

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    for (int i = 0; i < repeat; ++i) {
        warmup_kernel<<<grid, block>>>(d_data, N);
    }

    cudaDeviceSynchronize();
    cudaFree(d_data);
}

template <typename KernelFunc, typename... Args>
inline float perf_performance(KernelFunc &&kernel,
                              const char *name,
                              dim3 grid,
                              dim3 block,
                              Args &&...args) {
    cudaEvent_t start, stop;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);

    std::forward<KernelFunc>(kernel)<<<grid, block, 0, stream>>>(
        std::forward<Args>(args)...);

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("%s time: %.3f ms\n", name, ms);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    return ms;
}

template <typename T>
inline void perf_accuracy(const T *out_major,
                          const T *out_minor,
                          const int total) {
    int err_count = 0;
    const float epsilon = 1e-5f;
    for (int i = 0; i < total; ++i) {
        float a = static_cast<float>(out_major[i]);
        float b = static_cast<float>(out_minor[i]);

        float diff = fabs(a - b);
        float denom = std::max(fabs(a), 1e-6f);
        if (diff / denom > epsilon || std::isnan(a) || std::isnan(b)
            || std::isinf(a) || std::isinf(b)) {
            if (err_count < 10) {
                printf(
                    "Mismatch at %d: out_major=%.3f, out_minor=%.3f, error=%.3f\n",
                    i,
                    a,
                    b,
                    a - b);
            }
            err_count++;
        }
    }

    if (err_count == 0)
        printf("✅ 输出一致！\n");
    else
        printf("❌ 结果有 %d 处不同（前 10 处已列出）\n", err_count);

    // delete[] out_major;
    // delete[] out_minor;

    // 二维矩阵析构
    // for (int i = 0; i < rows; ++i) {
    //     delete[] data[i];
    // }
    // delete[] data;
}
