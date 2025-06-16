#pragma once

#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void warmup_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        data[idx] = data[idx] * 2.0f + 1.0f;
}

void gpu_warmup() {
    int N = 1024 * 1024;
    
    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemset(d_data, 0, N * sizeof(float));

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    // 可以多次调用以更充分 warmup
    for (int i = 0; i < 3; ++i) {
        warmup_kernel<<<grid, block>>>(d_data, N);
    }

    cudaDeviceSynchronize();  // 确保执行完毕
    cudaFree(d_data);
}

template<typename KernelFunc, typename... Args>
float perf_performance(KernelFunc&& kernel, const char* name, dim3 grid, dim3 block, Args&&... args) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    std::forward<KernelFunc>(kernel)<<<grid, block>>>(std::forward<Args>(args)...);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("%s time: %.3f ms\n", name, ms);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

template<typename T>
void perf_accuracy(const T* out1, const T* out2, const int total) {
    T* out_major = new T[total];
    T* out_minor = new T[total];

    cudaMemcpy(out_major, out1, total * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_minor, out2, total * sizeof(T), cudaMemcpyDeviceToHost);

    int err_count = 0;
    const double epsilon = 1e-6f;
    for (int i = 0; i < total; ++i) {
        double a = static_cast<double>(out_major[i]);
        double b = static_cast<double>(out_minor[i]);

        if (fabs(a - b) > epsilon) {
            if (err_count < 10) {
                printf("Mismatch at %d: out_major=%.3f, out_minor=%.3f \n", i, out_major[i], out_minor[i]);
            }
            err_count++;
        }
    }

    
    if (err_count == 0)
        printf("✅ 输出一致！\n");
    else
        printf("❌ 结果有 %d 处不同（前 10 处已列出）\n", err_count);

    delete[] out_major;
    delete[] out_minor;
    
    // 二维矩阵析构
    // for (int i = 0; i < rows; ++i) {
    //     delete[] data[i];
    // }
    // delete[] data;
}
