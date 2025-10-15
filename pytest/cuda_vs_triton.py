# add_benchmark.py
import os
import time
import math
import argparse
import torch

assert torch.cuda.is_available(), "需要一块可用的 NVIDIA GPU（torch.cuda.is_available() 为 False）"

# -----------------------------
# Triton kernel（可选）
# -----------------------------
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

if TRITON_AVAILABLE:
    # @triton.jit
    # def add_kernel_triton(A_ptr, B_ptr, C_ptr, N,
    #                       BLOCK_SIZE: tl.constexpr):
    #     pid = tl.program_id(axis=0)
    #     offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    #     mask = offs < N
    #     a = tl.load(A_ptr + offs, mask=mask)
    #     b = tl.load(B_ptr + offs, mask=mask)
    #     c = a + b
    #     tl.store(C_ptr + offs, c, mask=mask)

    # def triton_add(a: torch.Tensor, b: torch.Tensor, block_size=1024):
    #     assert a.is_cuda and b.is_cuda and a.dtype == b.dtype and a.numel() == b.numel()
    #     c = torch.empty_like(a)
    #     grid = lambda meta: (triton.cdiv(a.numel(), meta['BLOCK_SIZE']),)
    #     add_kernel_triton[grid](a, b, c, a.numel(), BLOCK_SIZE=block_size)
    #     return c

    # 自动调参：不同 BLOCK/VEC/warps/stages 的组合跑一次，记录最快的
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK': 1024, 'VEC': 4}, num_warps=4,  num_stages=2),
            triton.Config({'BLOCK': 2048, 'VEC': 4}, num_warps=4,  num_stages=2),
            triton.Config({'BLOCK': 4096, 'VEC': 4}, num_warps=8,  num_stages=2),
            triton.Config({'BLOCK': 4096, 'VEC': 8}, num_warps=8,  num_stages=2),
            triton.Config({'BLOCK': 8192, 'VEC': 4}, num_warps=8,  num_stages=3),
            triton.Config({'BLOCK': 8192, 'VEC': 8}, num_warps=8,  num_stages=3),
        ],
        key=['N']  # 针对不同 N 记住最快配置
    )
    @triton.jit
    def add_vec_kernel(A, B, C, N,
                    BLOCK: tl.constexpr, VEC: tl.constexpr):
        # program 维度：每个 program 负责 BLOCK*VEC 个元素（向量化）
        pid = tl.program_id(axis=0)
        base = pid * BLOCK * VEC

        idx  = base + tl.arange(0, BLOCK * VEC)
        mask = idx < N
        tl.multiple_of(idx, VEC)
        tl.max_contiguous(idx, VEC)
        
        a = tl.load(A + idx, mask=mask)
        b = tl.load(B + idx, mask=mask)
        c = a + b
        tl.store(C + idx, c, mask=mask)

    def triton_add(a: torch.Tensor, b: torch.Tensor):
        assert a.is_cuda and b.is_cuda and a.dtype == b.dtype
        assert a.is_contiguous() and b.is_contiguous()
        N = a.numel()
        c = torch.empty_like(a)
        # 一个 program 处理 BLOCK*VEC 个元素
        def grid(meta):
            elems_per_prog = meta['BLOCK'] * meta['VEC']
            return (triton.cdiv(N, elems_per_prog),)
        add_vec_kernel[grid](a, b, c, N)
        return c

# -----------------------------
# CUDA kernel（通过 PyTorch C++/CUDA 扩展 JIT）
# -----------------------------
from torch.utils.cpp_extension import load_inline

# cuda_src = r"""
# #include <cuda_runtime.h>
# #include <ATen/cuda/CUDAContext.h>

# extern "C" __global__
# void add_kernel_cuda(const float* __restrict__ A,
#                      const float* __restrict__ B,
#                      float* __restrict__ C,
#                      long long N) {
#     long long idx = blockDim.x * (long long)blockIdx.x + threadIdx.x;
#     if (idx < N) C[idx] = A[idx] + B[idx];
# }

# extern "C" void add_cuda_launcher_raw(const float* A,
#                                       const float* B,
#                                       float* C,
#                                       long long N) {
#     const int threads = 256;
#     const int blocks  = (int)((N + threads - 1) / threads);
#     // 用 PyTorch 当前流，避免与 torch 运算不同步
#     cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#     add_kernel_cuda<<<blocks, threads, 0, stream>>>(A, B, C, N);
# }

# """

# cpp_src = r"""
# #include <torch/extension.h>

# extern "C" void add_cuda_launcher_raw(const float* A,
#                                       const float* B,
#                                       float* C,
#                                       long long N);

# void add_cuda_launcher(const at::Tensor& A,
#                        const at::Tensor& B,
#                        at::Tensor& C) {
#     TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(), "tensors must be CUDA");
#     TORCH_CHECK(A.scalar_type() == at::kFloat && B.scalar_type() == at::kFloat && C.scalar_type() == at::kFloat,
#                 "this demo only supports float32");
#     TORCH_CHECK(A.numel() == B.numel() && A.numel() == C.numel(), "numel mismatch");

#     add_cuda_launcher_raw(
#         A.data_ptr<float>(),
#         B.data_ptr<float>(),
#         C.data_ptr<float>(),
#         static_cast<long long>(A.numel())
#     );
# }

# PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#     m.def("add_cuda_launcher", &add_cuda_launcher, "Add two tensors (CUDA)");
# }
# """

cuda_src = r"""
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

static inline bool is_aligned_16(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 0xF) == 0;
}

// ================= float32: float4 向量化 =================

extern "C" __global__
void add_f32_vec4_kernel(const float* __restrict__ A,
                         const float* __restrict__ B,
                         float* __restrict__ C,
                         long long vecN) { // vecN = N / 4
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    auto A4 = reinterpret_cast<const float4*>(A);
    auto B4 = reinterpret_cast<const float4*>(B);
    auto C4 = reinterpret_cast<float4*>(C);

    for (long long i = idx; i < vecN; i += (long long)gridDim.x * blockDim.x) {
        float4 a = A4[i];
        float4 b = B4[i];
        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        C4[i] = c;
    }
}

extern "C" __global__
void add_f32_tail_kernel(const float* __restrict__ A,
                         const float* __restrict__ B,
                         float* __restrict__ C,
                         long long start,
                         long long N) {
    long long i = start + blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

extern "C" __global__
void add_f32_scalar_kernel(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           long long N) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    for (; i < N; i += (long long)gridDim.x * blockDim.x) {
        C[i] = A[i] + B[i];
    }
}

// 通过占用率 API 给一个 blockSize 建议值
static inline void suggest_block_size(int& gridSize, int& blockSize, long long work_items, const void* func) {
    // int minGrid = 0;
    // cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, func, 0, 0);
    // gridSize = (int)((work_items + blockSize - 1) / blockSize);
    blockSize = 256;
    gridSize  = (int)((work_items + blockSize - 1) / blockSize);
}

// 统一的 launcher（float32）
extern "C" void add_f32_vec_launcher(const float* A,
                                     const float* B,
                                     float* C,
                                     long long N) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // 可向量化的条件：对齐到 16B 且 N>=4
    bool can_vec = is_aligned_16(A) && is_aligned_16(B) && is_aligned_16(C) && (N >= 4);
    if (can_vec) {
        long long vecN = N / 4;
        long long tail_start = vecN * 4;

        // 主体：float4
        int gridV = 0, blockV = 0;
        suggest_block_size(gridV, blockV, vecN, (const void*)add_f32_vec4_kernel);
        add_f32_vec4_kernel<<<gridV, blockV, 0, stream>>>(A, B, C, vecN);

        // 尾部（最多 3 个元素）
        if (tail_start < N) {
            int gridT = 0, blockT = 0;
            long long tailN = N - tail_start;
            suggest_block_size(gridT, blockT, tailN, (const void*)add_f32_tail_kernel);
            add_f32_tail_kernel<<<gridT, blockT, 0, stream>>>(A, B, C, tail_start, N);
        }
        return;
    }

    // 退化到标量 kernel（未对齐或 N<4）
    {
        int gridS = 0, blockS = 0;
        suggest_block_size(gridS, blockS, N, (const void*)add_f32_scalar_kernel);
        add_f32_scalar_kernel<<<gridS, blockS, 0, stream>>>(A, B, C, N);
    }
}

/* ================= half: __half2（可选）=================
#include <cuda_fp16.h>

extern "C" __global__
void add_f16_h2_kernel(const __half* __restrict__ A,
                       const __half* __restrict__ B,
                       __half* __restrict__ C,
                       long long h2N) { // h2N = N / 2
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    auto A2 = reinterpret_cast<const __half2*>(A);
    auto B2 = reinterpret_cast<const __half2*>(B);
    auto C2 = reinterpret_cast<__half2*>(C);

    for (long long i = idx; i < h2N; i += (long long)gridDim.x * blockDim.x) {
        __half2 c = __hadd2(A2[i], B2[i]);
        C2[i] = c;
    }
}

extern "C" __global__
void add_f16_tail_kernel(const __half* __restrict__ A,
                         const __half* __restrict__ B,
                         __half* __restrict__ C,
                         long long start,
                         long long N) {
    long long i = start + blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < N) C[i] = __hadd(A[i], B[i]);
}

extern "C" void add_f16_vec_launcher(const __half* A,
                                     const __half* B,
                                     __half* C,
                                     long long N) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    bool can_vec = (((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B) | reinterpret_cast<uintptr_t>(C)) & 0x3) == 0) && (N >= 2);
    if (can_vec) {
        long long h2N = N / 2;
        long long tail = h2N * 2;
        int grid = 0, block = 0;
        suggest_block_size(grid, block, h2N, (const void*)add_f16_h2_kernel);
        add_f16_h2_kernel<<<grid, block, 0, stream>>>(A, B, C, h2N);
        if (tail < N) {
            int gridT = 0, blockT = 0;
            suggest_block_size(gridT, blockT, N - tail, (const void*)add_f16_tail_kernel);
            add_f16_tail_kernel<<<gridT, blockT, 0, stream>>>(A, B, C, tail, N);
        }
    } else {
        int gridS = 0, blockS = 0;
        suggest_block_size(gridS, blockS, N, (const void*)add_f16_tail_kernel);
        add_f16_tail_kernel<<<gridS, blockS, 0, stream>>>(A, B, C, 0, N);
    }
}
*/
"""

cpp_src = r"""
#include <torch/extension.h>

extern "C" void add_f32_vec_launcher(const float* A, const float* B, float* C, long long N);
// extern "C" void add_f16_vec_launcher(const __half* A, const __half* B, __half* C, long long N); // 若启用 half2 路径

void add_cuda_launcher(const at::Tensor& A,
                  const at::Tensor& B,
                  at::Tensor& C) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(A.scalar_type() == B.scalar_type() && A.scalar_type() == C.scalar_type(), "dtype mismatch");
    TORCH_CHECK(A.numel() == B.numel() && A.numel() == C.numel(), "numel mismatch");

    const auto N = static_cast<long long>(A.numel());
    switch (A.scalar_type()) {
        case at::kFloat:
            add_f32_vec_launcher(
                A.data_ptr<float>(),
                B.data_ptr<float>(),
                C.data_ptr<float>(),
                N
            );
            break;
        // 若要启用 half 路径，取消下方注释并确保已在 .cu 中实现：
        // case at::kHalf:
        //     add_f16_vec_launcher(
        //         reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        //         reinterpret_cast<const __half*>(B.data_ptr<at::Half>()),
        //         reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        //         N
        //     );
        //     break;
        default:
            TORCH_CHECK(false, "Only float32 is enabled in this demo. (Enable half2 path if needed)");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_cuda_launcher", &add_cuda_launcher, "Vectorized add (float4, tail-safe)");
}
"""

ext = load_inline(
    name="add_cuda_ext",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    extra_cuda_cflags=["-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)

def cuda_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda and a.dtype == b.dtype and a.numel() == b.numel()
    # 这里为了简洁固定 float32；需要其它 dtype 可扩展 kernel
    assert a.dtype == torch.float32, "当前示例 CUDA kernel 仅实现了 float32"
    c = torch.empty_like(a)
    ext.add_cuda_launcher(a, b, c)
    return c

# -----------------------------
# 计时工具
# -----------------------------
def bench(fn, warmup=1, repeat=5, mode="mean"):
    # 热身
    for _ in range(warmup):
        out = fn()
        torch.cuda.synchronize()
    # 正式计时
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    if mode == "min":
        t = min(times)
    elif mode == "mean":
        t = sum(times) / len(times)
    elif mode == "median":
        t = sorted(times)[len(times) // 2]

    return t, out  # 取最小时间更稳健

def gbps(numel, bytes_per_elem, seconds):
    # A + B -> C：读 A（N*e）、读 B（N*e）、写 C（N*e） => 3 * N * e 字节
    total_bytes = 3.0 * numel * bytes_per_elem
    return (total_bytes / seconds) / (1024**3)

# -----------------------------
# 主流程
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=64 * 1024 * 1024, help="元素个数（默认 64M）")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32", "bfloat16"], help="数据类型")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--triton_block", type=int, default=1024, help="Triton BLOCK_SIZE")
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    device = torch.device("cuda")
    torch.manual_seed(0)

    a = torch.randn(args.numel, device=device, dtype=dtype)
    b = torch.randn(args.numel, device=device, dtype=dtype)

    print(f"Device       : {torch.cuda.get_device_name(0)}")
    print(f"Numel        : {args.numel}")
    print(f"Dtype        : {dtype}")
    print(f"Triton avail?: {TRITON_AVAILABLE}")
    print("-" * 60)

    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()

    # 1) PyTorch 内置
    t_pt, out_pt = bench(lambda: a + b, warmup=args.warmup, repeat=args.repeat)
    gbps_pt = gbps(args.numel, bytes_per_elem, t_pt)
    print(f"[PyTorch]   time: {t_pt*1e3:.3f} ms   bandwidth: {gbps_pt:,.2f} GB/s")

    # 2) CUDA kernel
    # 注意：演示内核仅实现 float32；其它 dtype 做 cast 以便公平比较
    a32 = a.float()
    b32 = b.float()
    t_cuda, out_cuda = bench(lambda: cuda_add(a32, b32), warmup=args.warmup, repeat=args.repeat)
    gbps_cuda = gbps(args.numel, 4, t_cuda)  # float32 = 4 bytes
    # 和 PyTorch 输出比对（都转回 float32 再比）
    ok_cuda = torch.allclose(out_cuda, (a + b).float(), atol=1e-3 if dtype != torch.float32 else 1e-6)
    print(f"[CUDA Kern] time: {t_cuda*1e3:.3f} ms   bandwidth: {gbps_cuda:,.2f} GB/s   correct: {ok_cuda}")

    # 3) Triton kernel（若可用）
    if TRITON_AVAILABLE:
        if dtype == torch.float16:
            # Triton fp16 没问题；bfloat16 也可，但对比时用相同 dtype
            pass
        t_triton, out_triton = bench(lambda: triton_add(a, b),
                                     warmup=args.warmup, repeat=args.repeat)
        gbps_triton = gbps(args.numel, bytes_per_elem, t_triton)
        ok_triton = torch.allclose(out_triton, a + b, atol=1e-3 if dtype != torch.float32 else 1e-6)
        print(f"[Triton]    time: {t_triton*1e3:.3f} ms   bandwidth: {gbps_triton:,.2f} GB/s   correct: {ok_triton}")
    else:
        print("[Triton]    未安装或导入失败，跳过 Triton 测试。")

    print("-" * 60)

if __name__ == "__main__":
    main()
