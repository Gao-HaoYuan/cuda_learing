#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/CUDAContext.h>

#include <vector>

#include "blas_gemm.h"
#include "workspace.h"

template <typename T>
static inline int64_t ld_major(int64_t rows, int64_t cols /*COL MAJOR*/) {
    (void)cols;
    return rows;
}

template<typename T>
CublasLtGemm<T>::CublasLtGemm(
    int64_t M, int64_t N, int64_t K,
    cublasOperation_t opA,
    cublasOperation_t opB,
    cudaStream_t stream
) : M_(M), N_(N), K_(K), 
    opA_(opA), opB_(opB), 
    stream_(stream)
{
    // lt_ = at::cuda::getCurrentCUDABlasHandle();
    CHECK_LT(cublasLtCreate(&lt_handle_));
    create_descriptors_();
    select_algo_();
}

template<typename T>
CublasLtGemm<T>::~CublasLtGemm() {
    if (layoutD_) cublasLtMatrixLayoutDestroy(layoutD_);
    if (layoutC_) cublasLtMatrixLayoutDestroy(layoutC_);
    if (layoutB_) cublasLtMatrixLayoutDestroy(layoutB_);
    if (layoutA_) cublasLtMatrixLayoutDestroy(layoutA_);
    if (matmul_desc_) cublasLtMatmulDescDestroy(matmul_desc_);
    if (lt_handle_) cublasLtDestroy(lt_handle_);

    // c10::cuda::CUDACachingAllocator::raw_delete(workspace_ptr_);
}

template <typename T>
void CublasLtGemm<T>::create_descriptors_() {
    CHECK_LT(cublasLtMatmulDescCreate(&matmul_desc_, Traits::kComputeType, Traits::kDataType));
    CHECK_LT(cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_TRANSA, &opA_, sizeof(opA_)));
    CHECK_LT(cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_TRANSB, &opB_, sizeof(opB_)));

    int64_t a_rows = (opA_ == CUBLAS_OP_N) ? M_ : K_;
    int64_t a_cols = (opA_ == CUBLAS_OP_N) ? K_ : M_;

    int64_t b_rows = (opB_ == CUBLAS_OP_N) ? K_ : N_;
    int64_t b_cols = (opB_ == CUBLAS_OP_N) ? N_ : K_;

    int64_t c_rows = M_;
    int64_t c_cols = N_;

    int64_t d_rows = M_;
    int64_t d_cols = N_;

    int64_t ldA = ld_major<T>(a_rows, a_cols);
    int64_t ldB = ld_major<T>(b_rows, b_cols);
    int64_t ldC = ld_major<T>(c_rows, c_cols);
    int64_t ldD = ld_major<T>(d_rows, d_cols);

    CHECK_LT(cublasLtMatrixLayoutCreate(&layoutA_, Traits::kDataType, a_rows, a_cols, ldA));
    CHECK_LT(cublasLtMatrixLayoutCreate(&layoutB_, Traits::kDataType, b_rows, b_cols, ldB));
    CHECK_LT(cublasLtMatrixLayoutCreate(&layoutC_, Traits::kDataType, c_rows, c_cols, ldC));
    CHECK_LT(cublasLtMatrixLayoutCreate(&layoutD_, Traits::kDataType, d_rows, d_cols, ldD));

    // Explicitly set Lead-Order
    CHECK_LT(cublasLtMatrixLayoutSetAttribute(layoutA_, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_, sizeof(order_)));
    CHECK_LT(cublasLtMatrixLayoutSetAttribute(layoutB_, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_, sizeof(order_)));
    CHECK_LT(cublasLtMatrixLayoutSetAttribute(layoutC_, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_, sizeof(order_)));
    CHECK_LT(cublasLtMatrixLayoutSetAttribute(layoutD_, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_, sizeof(order_)));
}

template<typename T>
void CublasLtGemm<T>::select_algo_() {
    workspace_ptr_ =  global_ws_sm().template get<void>(stream_);
    workspace_bytes_ = global_ws_sm().get_bytes();

    cublasLtMatmulPreference_t preference_ = nullptr;
    CHECK_LT(cublasLtMatmulPreferenceCreate(&preference_));
    CHECK_LT(cublasLtMatmulPreferenceSetAttribute(
        preference_,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_bytes_,
        sizeof(workspace_bytes_)
    ));

    constexpr int kNumHeur = 16;
    std::vector<cublasLtMatmulHeuristicResult_t> heurs(kNumHeur);
    int returned_algo_count = 0;

    CHECK_LT(cublasLtMatmulAlgoGetHeuristic(
        lt_handle_,
        matmul_desc_,
        layoutA_,
        layoutB_,
        layoutC_,
        layoutD_,
        preference_,
        kNumHeur,
        heurs.data(),
        &returned_algo_count
    ));

    CHECK_LT(cublasLtMatmulPreferenceDestroy(preference_));

    for (int i = 0; i < returned_algo_count; ++i) {
        if (heurs[i].workspaceSize <= workspace_bytes_) {
            best_algo_ = heurs[i];
            return;
        }
    }

    throw std::runtime_error("cuBLASLt failed to find any suitable algorithm.");
    // c10::cuda::CUDACachingAllocator::raw_delete(workspace_ptr_);
}

template <typename T>
void CublasLtGemm<T>::set_epilogue_default_() {
    cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_DEFAULT;
    CHECK_LT(cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
    
    // Clear bias pointer just in case
    const void* null_bias = nullptr;
    CHECK_LT(cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &null_bias, sizeof(null_bias)));
}

template <typename T>
void CublasLtGemm<T>::set_epilogue_bias_(const void* bias) {
    cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
    CHECK_LT(cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
    
    // bias pointer device address
    CHECK_LT(cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
}

template<typename T>
void CublasLtGemm<T>::run(
    const void* A, 
    const void* B, 
    const void* C, 
    void* D, 
    const void* alpha, 
    const void* beta
) {
    set_epilogue_default_();

    CHECK_LT(cublasLtMatmul(
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
        workspace_bytes_,
        stream_
    ));
}

template<typename T>
void CublasLtGemm<T>::run_with_bias(
    const void* A, 
    const void* B, 
    const void* C, 
    void* D, 
    const void* bias, 
    const void* alpha, 
    const void* beta
) {
    set_epilogue_bias_(bias);

    CHECK_LT(cublasLtMatmul(
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
        workspace_bytes_,
        stream_
    ));
}

template class CublasLtGemm<float>;
template class CublasLtGemm<double>;
template class CublasLtGemm<c10::Half>;
