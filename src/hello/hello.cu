// 这是 test.cu文件的内容
#include <cstdio>
#include <cuda_runtime.h>

__host__ __device__ void say_hello() {
#ifdef __CUDA_ARCH__
    printf("Hello, world from GPU architecture %d!\n", __CUDA_ARCH__);
#else
    printf("Hello, world from CPU!\n");
#endif
}

__global__ void kernel() {
    say_hello();
}

int main() {
    kernel<<<1, 2>>>();
    cudaDeviceSynchronize();
    say_hello();
    return 0;
}
