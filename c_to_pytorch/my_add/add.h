#include <cuda_runtime.h>

template<typename  T> 
void my_add_cuda(const T* in, T* out, size_t tot, dim3 grid, dim3 block);