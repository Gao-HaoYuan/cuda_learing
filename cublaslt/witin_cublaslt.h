#include <cublasLt.h>

template<typename T>
class CublasLtGemmRunner {
public:
    CublasLtGemmRunner(
        bool transpose_matA,
        bool transpose_matB,
        int M, 
        int N, 
        int K,
        size_t max_workspace_bytes = 64 * 1024 * 1024
    );

    ~CublasLtGemmRunner();

    // 禁止拷贝和移动
    CublasLtGemmRunner(const CublasLtGemmRunner&) = delete;
    CublasLtGemmRunner& operator=(const CublasLtGemmRunner&) = delete;
    CublasLtGemmRunner(CublasLtGemmRunner&&) = delete;
    CublasLtGemmRunner& operator=(CublasLtGemmRunner&&) = delete;

    /**
     * @brief 执行 GEMM：D = alpha * op(A) * op(B) + beta * C
     */
    void run(
        const T* A, const T* B, const T* C, T* D,
        const T* alpha, const T* beta,
        const cudaStream_t stream
    );

    /**
     * @brief 执行 GEMM 并加 bias：D = alpha * op(A) * op(B) + beta * C + bias
     * @param bias 列向量，长度 N（广播到每行）
     */
    void run_with_bias(
        const T* A, const T* B, const T* C, T* D,
        const T* bias,
        const T* alpha, const T* beta,
        const cudaStream_t stream
    );

private:
    void determine_best_algorithm();
    cudaDataType_t get_cuda_dtype() const;
    cublasComputeType_t get_compute_type() const;

private:
    cublasOperation_t opA_;
    cublasOperation_t opB_;
    int M_, N_, K_;

    cublasLtHandle_t lt_handle_ = nullptr;
    cublasLtMatmulDesc_t matmul_desc_ = nullptr;
    cublasLtMatrixLayout_t layoutA_ = nullptr;
    cublasLtMatrixLayout_t layoutB_ = nullptr;
    cublasLtMatrixLayout_t layoutC_ = nullptr;
    cublasLtMatrixLayout_t layoutD_ = nullptr;

    cublasLtMatmulHeuristicResult_t best_algo_;
    size_t max_workspace_bytes_;
    void* workspace_ptr_ = nullptr;
};