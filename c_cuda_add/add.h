#include <cuda_runtime.h>

void my_add_cuda(const float* a, const float* b, float* out, size_t tot, dim3 grid, dim3 block);