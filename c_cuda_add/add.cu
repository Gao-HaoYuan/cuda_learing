#include <cuda_runtime.h>

__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

void my_add_cuda(const float* a, const float* b, float* out, size_t tot, dim3 grid, dim3 block) {
    add_kernel<<<grid, block>>>(a, b, out, tot);
}