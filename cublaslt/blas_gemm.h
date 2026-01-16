#pragma once

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <type_traits>
#include <cstdint>

template <typename T>
struct CublasLtTraits;

template <>
struct CublasLtTraits<float> {
    static constexpr cudaDataType_t kDataType = CUDA_R_32F;
    static constexpr cublasComputeType_t kComputeType = CUBLAS_COMPUTE_32F_FAST_16F;
};

template <>
struct CublasLtTraits<double> {
    static constexpr cudaDataType_t kDataType = CUDA_R_64F;
    static constexpr cublasComputeType_t kComputeType = CUBLAS_COMPUTE_64F;
};

template <>
struct CublasLtTraits<at::Half> {
    static constexpr cudaDataType_t kDataType = CUDA_R_16F;
    static constexpr cublasComputeType_t kComputeType = CUBLAS_COMPUTE_32F;
};

template <>
struct CublasLtTraits<at::BFloat16> {
    static constexpr cudaDataType_t kDataType = CUDA_R_16BF;
    static constexpr cublasComputeType_t kComputeType = CUBLAS_COMPUTE_32F;
};

struct CublasLtGemmBase {
    virtual ~CublasLtGemmBase() = default;

    virtual void run(
        const void* A, const void* B, const void* C, void* D,
        const void* alpha, const void* beta
    ) = 0;

    virtual void run_with_bias(
        const void* A, const void* B, const void* C, void* D,
        const void* bias,
        const void* alpha, const void* beta
    ) = 0;
};

// cublaslt 指定为行主序时, ReLU, dReLu, GELU, dGELU and Bias epilogue 不被支持
// 为更好的使用 cublaslt 本类只支持 Col Major 模式
// torch tensor 默认是行主序，使用时，请手动做 track ：D = A @ B + C ===> D^T = B^T @ A^T + C^T
template <typename T>
class CublasLtGemm : public CublasLtGemmBase {
private:
    using Traits = CublasLtTraits<T>;
    
public:
    CublasLtGemm(
        int64_t M, int64_t N, int64_t K,
        cublasOperation_t opA,
        cublasOperation_t opB,
        cudaStream_t stream
    );

    ~CublasLtGemm();

    // 禁止拷贝和移动
    CublasLtGemm(const CublasLtGemm&) = delete;
    CublasLtGemm& operator=(const CublasLtGemm&) = delete;

    /**
     * @brief 执行 GEMM：D = alpha * op(A) * op(B) + beta * C
     */
    void run(
        const void* A, const void* B, const void* C, void* D,
        const void* alpha, const void* beta
    );
    /**
     * @brief 执行 GEMM 并加 bias：D = alpha * op(A) * op(B) + beta * C + bias
     * @param bias 列向量，长度 N（广播到每行）
     */
    void run_with_bias(
        const void* A, const void* B, const void* C, void* D,
        const void* bias,
        const void* alpha, const void* beta
    );

private:
    void create_descriptors_();
    void select_algo_();

    void set_epilogue_default_();
    void set_epilogue_bias_(const void* bias);

private:
    int64_t M_, N_, K_;
    cublasOperation_t opA_, opB_;
    cudaStream_t stream_;

    void* workspace_ptr_;
    size_t workspace_bytes_;
  
    cublasLtHandle_t lt_handle_{};
    cublasLtMatmulDesc_t matmul_desc_{};
    cublasLtMatrixLayout_t layoutA_{}, layoutB_{}, layoutC_{}, layoutD_{};
    cublasLtMatmulHeuristicResult_t best_algo_{};

    // default col-major layout order
    const cublasLtOrder_t order_{CUBLASLT_ORDER_COL};
};