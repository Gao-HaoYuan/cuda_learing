#pragma once

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <stdexcept>
#include <string>

inline const char* cublaslt_status_to_string(cublasStatus_t s) {
    switch (s) {
        case CUBLAS_STATUS_SUCCESS:         return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:    return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:   return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:   return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:   return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:  return "CUBLAS_STATUS_INTERNAL_ERROR";
#if defined(CUBLAS_STATUS_NOT_SUPPORTED)
        case CUBLAS_STATUS_NOT_SUPPORTED:   return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
        default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

#define CHECK_CUDA(expr) do {                                                   \
    cudaError_t _e = (expr);                                                    \
    if (_e != cudaSuccess) {                                                    \
        throw std::runtime_error(std::string("CUDA error: ") +                  \
            cudaGetErrorString(_e) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    }                                                                           \
} while (0)

#define CHECK_LT(expr) do {                                                     \
    cublasStatus_t _s = (expr);                                                 \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                          \
        throw std::runtime_error(std::string("cuBLASLt error: ") +              \
            cublaslt_status_to_string(_s) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    }                                                                           \
} while (0)
