#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/CUDAContext.h>

#include <vector>
#include <cuda_runtime.h>
#include <cublasLt.h>

#include "witin_cublaslt.h"

template<typename T>
cudaDataType_t CublasLtGemmRunner<T>::get_cuda_dtype() const {
    if (std::is_same<T, double>::value) return CUDA_R_64F;
    if (std::is_same<T, float>::value) return CUDA_R_32F;
    if (std::is_same<T, half>::value) return CUDA_R_16F;
    throw std::runtime_error("Unsupported data type T.");
}

template<typename T>
cublasComputeType_t CublasLtGemmRunner<T>::get_compute_type() const {
    if (std::is_same<T, half>::value) {
        return CUBLAS_COMPUTE_32F;
    }
    else if (std::is_same<T, float>::value) {
        return CUBLAS_COMPUTE_32F_FAST_TF32; // 使用 tensorcore 加速
    }
    else if (std::is_same<T, double>::value) {
        return CUBLAS_COMPUTE_64F;
    }
    else {
        throw std::runtime_error("Unsupported data type for cuBLASLt compute type.");
    }
}

template<typename T>
CublasLtGemmRunner<T>::CublasLtGemmRunner(
    bool transpose_matA,
    bool transpose_matB,
    int M, 
    int N, 
    int K,
    size_t max_workspace_bytes
) : M_(M), N_(N), K_(K), max_workspace_bytes_(max_workspace_bytes)
{
    opA_ = transpose_matA ? CUBLAS_OP_T : CUBLAS_OP_N;
    opB_ = transpose_matB ? CUBLAS_OP_T : CUBLAS_OP_N;
    lt_handle_ = at::cuda::getCurrentCUDABlasLtHandle();

    int ldA = (opA_ == CUBLAS_OP_N) ? M_ : K_;
    int ldB = (opB_ == CUBLAS_OP_N) ? K_ : N_;
    int ldC = M_;
    int ldD = M_;

    cudaDataType_t dtype = get_cuda_dtype();
    cublasLtMatmulDescCreate(&matmul_desc_, get_compute_type(), dtype);
    cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_TRANSA, &opA_, sizeof(opA_));
    cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_TRANSB, &opB_, sizeof(opB_));

    cublasLtMatrixLayoutCreate(&layoutA_, dtype, M, K, ldA);
    cublasLtMatrixLayoutCreate(&layoutB_, dtype, K, N, ldB);
    cublasLtMatrixLayoutCreate(&layoutC_, dtype, M, N, ldC);
    cublasLtMatrixLayoutCreate(&layoutD_, dtype, M, N, ldD);

    determine_best_algorithm();

    size_t required_ws = best_algo_.workspaceSize;
    if (required_ws > 0) {
        workspace_ptr_ = c10::cuda::CUDACachingAllocator::raw_alloc(required_ws);
    }
}

template<typename T>
CublasLtGemmRunner<T>::~CublasLtGemmRunner() {
    if (layoutD_) cublasLtMatrixLayoutDestroy(layoutD_);
    if (layoutC_) cublasLtMatrixLayoutDestroy(layoutC_);
    if (layoutB_) cublasLtMatrixLayoutDestroy(layoutB_);
    if (layoutA_) cublasLtMatrixLayoutDestroy(layoutA_);
    if (matmul_desc_) cublasLtMatmulDescDestroy(matmul_desc_);

    if (workspace_ptr_) {
        c10::cuda::CUDACachingAllocator::raw_delete(workspace_ptr_);
    }
}

template<typename T>
void CublasLtGemmRunner<T>::determine_best_algorithm() {
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(
        pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &max_workspace_bytes_,
        sizeof(max_workspace_bytes_)
    );

    long long size_M = M_, size_N = N_, size_K = K_;
    std::vector<cublasLtMatmulHeuristicResult_t> results(1);
    int returned_algo_count = 0;

    cublasLtMatmulAlgoGetHeuristic(
        lt_handle_,
        matmul_desc_,
        layoutA_,
        layoutB_,
        layoutC_,
        layoutD_,
        pref,
        1,
        results.data(),
        &returned_algo_count
    );

    cublasLtMatmulPreferenceDestroy(pref);

    if (returned_algo_count == 0) {
        throw std::runtime_error("cuBLASLt failed to find any suitable algorithm.");
    }

    best_algo_ = results[0];
}

template<typename T>
void CublasLtGemmRunner<T>::run(
    const T* A, 
    const T* B, 
    const T* C, 
    T* D, 
    const T* alpha, 
    const T* beta, 
    const cudaStream_t stream
) {
    cublasLtMatmul(
        lt_handle_,
        matmul_desc_,
        alpha,
        A, layoutA_,
        B, layoutB_,
        beta,
        C, layoutC_,
        D, layoutD_,
        &best_algo_.algo,
        workspace_ptr_,
        best_algo_.workspaceSize,
        stream
    );
}

template<typename T>
void CublasLtGemmRunner<T>::run_with_bias(
    const T* A, 
    const T* B, 
    const T* C, 
    T* D, 
    const T* bias, 
    const T* alpha, 
    const T* beta, 
    const cudaStream_t stream 
) {
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
    cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));

    cublasLtMatmul(
        lt_handle_,
        matmul_desc_,
        alpha,
        A, layoutA_,
        B, layoutB_,
        beta,
        C, layoutC_,
        D, layoutD_,
        &best_algo_.algo,
        workspace_ptr_,
        best_algo_.workspaceSize,
        stream
    );
}

template class CublasLtGemmRunner<double>;
template class CublasLtGemmRunner<float>;
template class CublasLtGemmRunner<half>;