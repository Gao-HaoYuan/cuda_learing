#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "add.h"

template <typename T>
__global__ void my_add_kernel(const T* __restrict__ input,
                              T* __restrict__ output,
                              int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] + static_cast<T>(1);
    }
}

template<typename  T> void my_add_cuda(const T* in, 
                                    T* out, 
                                    size_t tot, 
                                    dim3 grid, 
                                    dim3 block) {
    my_add_kernel<<<grid, block>>>(in, out, tot);
}

template void my_add_cuda<float>(const float* in, float* out, size_t tot, dim3 grid, dim3 block);
template void my_add_cuda<double>(const double* in, double* out, size_t tot, dim3 grid, dim3 block);
