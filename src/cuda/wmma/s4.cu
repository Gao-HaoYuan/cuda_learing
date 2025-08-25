#include <mma.h>
#include <cuda.h>
#include <cstdio>

#include "minitest.hpp"

using namespace nvcuda::wmma;
using namespace nvcuda::wmma::experimental;

// GPU kernel
__global__ void s4_bmma_kernel() {
    // 声明 fragment
    fragment<matrix_a, 8, 8, 32, experimental::precision::s4, row_major> a_frag;
    fragment<matrix_b, 8, 8, 32, experimental::precision::s4, col_major> b_frag;
    fragment<accumulator, 8, 8, 32, int> c_frag;

    // 初始化 fragment
    fill_fragment(a_frag, 1);   // A 全 +1
    fill_fragment(b_frag, -1);  // B 全 -1
    fill_fragment(c_frag, 0);   // C 全 0

    // 使用 bmma_sync 做 bit-matrix multiply-accumulate
    // 这里用 XOR + POPC 累加（统计 1 的数量）
    mma_sync(c_frag, a_frag, b_frag, c_frag);

    printf("num_elements: %d\n", a_frag.num_elements);
    printf("num_storage_elements: %d\n", a_frag.num_storage_elements);
}

TEST(WMMA, s4) {
    // 启动 kernel
    s4_bmma_kernel<<<1,32>>>();
    cudaDeviceSynchronize();
}
