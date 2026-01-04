#pragma once
#include <cuda_runtime.h>
#include <cublasLt.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <stdexcept>

#include "inline_check.h"

struct ThreadBoundCublasLt {
    static constexpr int kMaxDevicesCap = 16;

    static int device_count() {
        int n = 0;
        CHECK_CUDA(cudaGetDeviceCount(&n));

        if (n < 0) n = 0;
        return std::min(n, kMaxDevicesCap);
    }

    static int current_device() {
        int dev = 0;
        CHECK_CUDA(cudaGetDevice(&dev));
        return dev;
    }

    struct DeviceGuard {
        int prev = 0;
        explicit DeviceGuard(int dev) {
            CHECK_CUDA(cudaGetDevice(&prev));
            if (prev != dev) CHECK_CUDA(cudaSetDevice(dev));
        }

        ~DeviceGuard() noexcept {
            (void)cudaSetDevice(prev);
        }
    };

    struct TLS {
        std::array<cublasLtHandle_t, kMaxDevicesCap> h{};
        std::array<uint8_t,         kMaxDevicesCap> inited{};
        int cached_dev_count = -1;

        void refresh_dev_count_if_needed() {
            if (cached_dev_count < 0) {
                cached_dev_count = device_count();
            }
        }

        ~TLS() {
            for (int d = 0; d < cached_dev_count; ++d) {
                if (inited[d] && h[d]) {
                    int prev = 0;
                    (void)cudaGetDevice(&prev);
                    if (prev != d) (void)cudaSetDevice(d);  // 切换到目标 device

                    (void)cublasLtDestroy(h[d]);
                    h[d] = nullptr;
                    inited[d] = 0;

                    if (prev != d) (void)cudaSetDevice(prev); // 恢复到原始的 device
                }
            }
        }
    };

    static TLS& tls() {
        thread_local TLS t;
        return t;
    }

    static cublasLtHandle_t get() {
        return get(current_device());
    }

    static cublasLtHandle_t get(int dev) {
        TLS& t = tls();
        t.refresh_dev_count_if_needed();

        const int n = t.cached_dev_count;
        if (dev < 0 || dev >= n) {
            throw std::out_of_range("ThreadBoundCublasLt::get(dev): invalid device id");
        }

        if (t.inited[dev] && t.h[dev]) return t.h[dev];

        DeviceGuard g(dev); // 确保在 dev 上 create
        cublasLtHandle_t handle = nullptr;
        cublasStatus_t st = cublasLtCreate(&handle);
        if (st != CUBLAS_STATUS_SUCCESS || handle == nullptr) {
            throw std::runtime_error("cublasLtCreate failed (thread-bound)");
        }

        t.h[dev] = handle;
        t.inited[dev] = 1;
        return handle;
    }
};

inline cublasLtHandle_t getBlasLtHandle() {
    return ThreadBoundCublasLt::get();
}