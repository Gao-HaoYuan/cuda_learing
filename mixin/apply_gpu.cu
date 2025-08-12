#include <iostream>
#include <chrono>
#include <utility>

#include <thrust/tuple.h>
#include <cuda_runtime.h>

// 简单加减乘算子struct（设备端）
struct Add {
    __host__ __device__ 
    int operator()(int a, int b) const { return a + b; }
};
struct Sub {
    __host__ __device__ 
    int operator()(int a, int b) const { return a - b; }
};
struct Mul {
    __host__ __device__ 
    int operator()(int a, int b) const { return a * b; }
};

// apply_all 和 ParallelThenCombine 在设备端实现
template<typename Tuple, std::size_t... I, typename... Args>
__host__ __device__
auto apply_all_impl(const Tuple& funcs, std::index_sequence<I...>, Args&&... args) {
    return thrust::make_tuple(thrust::get<I>(funcs)(std::forward<Args>(args)...)...);
}

template<typename Tuple, typename... Args>
__host__ __device__
auto apply_all(const Tuple& funcs, Args&&... args) {
    return apply_all_impl(funcs, std::make_index_sequence<thrust::tuple_size<Tuple>::value>{}, std::forward<Args>(args)...);
}

template <typename Combiner, typename... Fused>
struct ParallelThenCombine {
    thrust::tuple<Fused...> functors;
    Combiner combiner;

    __host__ __device__
    ParallelThenCombine(Combiner c, Fused... fs)
        : combiner(c), functors(fs...) {}

    template <typename... Args>
    __host__ __device__
    auto operator()(Args&&... args) const {
        auto results = apply_all(functors, std::forward<Args>(args)...);
        return apply_combiner(results, combiner, std::make_index_sequence<sizeof...(Fused)>{});
    }

private:
    template <typename Tuple, typename C, std::size_t... I>
    __host__ __device__
    static auto apply_combiner(const Tuple& t, const C& c, std::index_sequence<I...>) {
        return c(thrust::get<I>(t)...);
    }
};

__global__ void kernel_direct(int* out, const int* a, const int* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = (a[idx] + b[idx]) * (a[idx] - b[idx]);
}

template <typename FusedOp>
__global__ void kernel_fused(int* out, const int* a, const int* b, int n, FusedOp fused) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fused(a[idx], b[idx]);
}

int main() {
    constexpr int N = 1 << 20;
    int *h_a = new int[N], *h_b = new int[N];
    for (int i = 0; i < N; ++i) { 
        h_a[i] = i % 1000; 
        h_b[i] = (i * 7) % 1000;
    }

    int *d_a, *d_b, *d_out_direct, *d_out_fused;
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMalloc(&d_out_direct, N * sizeof(int));
    cudaMalloc(&d_out_fused, N * sizeof(int));
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256, blocks = (N + threads - 1) / threads;
    cudaEvent_t start, stop; 
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);

    ParallelThenCombine<Mul, Add, Sub> fusedOp(Mul{}, Add{}, Sub{});

    kernel_direct<<<blocks, threads>>>(d_out_direct, d_a, d_b, N);
    kernel_fused<<<blocks, threads>>>(d_out_fused, d_a, d_b, N, fusedOp);

    // Fused kernel
    cudaEventRecord(start);
    kernel_fused<<<blocks, threads>>>(d_out_fused, d_a, d_b, N, fusedOp);
    cudaEventRecord(stop); 
    cudaEventSynchronize(stop);
    float time_fused; 
    cudaEventElapsedTime(&time_fused, start, stop);

    // Direct kernel
    cudaEventRecord(start);
    kernel_direct<<<blocks, threads>>>(d_out_direct, d_a, d_b, N);
    cudaEventRecord(stop); 
    cudaEventSynchronize(stop);
    float time_direct; 
    cudaEventElapsedTime(&time_direct, start, stop);

    std::cout << "Direct: " << time_direct << " ms\n";
    std::cout << "Fused : " << time_fused  << " ms\n";

    // 验证结果
    int *h_out_direct = new int[N], *h_out_fused = new int[N];
    cudaMemcpy(h_out_direct, d_out_direct, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_fused, d_out_fused, N * sizeof(int), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (int i = 0; i < N; ++i) 
        if (h_out_direct[i] != h_out_fused[i]) { ok = false; break; }
    std::cout << (ok ? "Results match!\n" : "Results differ!\n");

    delete[] h_a; 
    delete[] h_b; 
    delete[] h_out_direct; 
    delete[] h_out_fused;

    cudaFree(d_a); 
    cudaFree(d_b); 
    cudaFree(d_out_direct); 
    cudaFree(d_out_fused);
    cudaEventDestroy(start); 
    cudaEventDestroy(stop);

    return 0;
}
