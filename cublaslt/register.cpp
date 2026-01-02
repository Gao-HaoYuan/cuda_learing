#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/extension.h>

#include "blas_gemm.h"

namespace py = pybind11;

// 核心函数：从 torch.Tensor 注册 GEMM，只传 A/B
template<typename T>
torch::Tensor py_register_gemm_from_torch(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const c10::optional<torch::Tensor>& bias = c10::nullopt
){
    // 1️⃣ 检查 Tensor 是否在 CUDA
    if(!A.is_cuda() || !B.is_cuda()){
        throw std::runtime_error("All input tensors must be CUDA tensors");
    }

    // 2️⃣ 获取维度
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // 3️⃣ 获取指针
    void* d_A = A.data_ptr();
    void* d_B = B.data_ptr();

    // 4️⃣ 创建输出 Tensor
    torch::Tensor D = at::zeros({M,N}, A.options());
    void* d_D = D.data_ptr();

    void* d_bias = nullptr;
    bool use_bias = false;
    if(bias.has_value()){
        auto b = bias.value();
        if(!b.is_cuda()){
            throw std::runtime_error("Bias tensor must be CUDA tensor");
        }
        d_bias = b.data_ptr();
        use_bias = true;
    }

    // 5️⃣ 创建 GEMM Runner
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    std::unique_ptr<CublasLtGemmBase> gemm = std::make_unique<CublasLtGemm<T>>(
        N, M, K, 
        CUBLAS_OP_N, 
        CUBLAS_OP_N,
        stream
    );

    float alpha = 1.f, beta = 0.f; // 没有 C 输入，beta = 0
    if(use_bias){
        gemm->run_with_bias(d_B, d_A, nullptr, d_D, d_bias, &alpha, &beta);
    }else{
        gemm->run(d_B, d_A, nullptr, d_D, &alpha, &beta);
    }

    return D;
}

// pybind11 模块
PYBIND11_MODULE(cublaslt_gemm, m){
    m.def("register_gemm_float", &py_register_gemm_from_torch<float>,
          py::arg("A"), py::arg("B"), py::arg("bias"));
}
