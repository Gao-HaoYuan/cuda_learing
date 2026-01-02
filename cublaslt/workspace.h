#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <array>
#include <mutex>

#include "inline_check.h"

struct WorkspaceMutexSM {
    static constexpr int kMaxDevices = 16;

    enum class State : uint8_t { Empty, Allocated };

    struct PerDevice {
        std::mutex m;
        void*  ptr = nullptr;
        size_t bytes = 0;
        bool   used_async = false;
        State  st = State::Empty;
    };

    size_t bytes_ = 0;
    std::array<PerDevice, kMaxDevices> dev_{};

    explicit WorkspaceMutexSM(size_t bytes) : bytes_(bytes) {}
    WorkspaceMutexSM() = default;
    WorkspaceMutexSM(const WorkspaceMutexSM&) = delete;
    WorkspaceMutexSM& operator=(const WorkspaceMutexSM&) = delete;

    static void* alloc_bytes(size_t bytes, cudaStream_t stream, bool& used_async_out) {
        if (bytes == 0) return nullptr;
        void* p = nullptr;

#if CUDART_VERSION >= 11020
        cudaError_t e = cudaMallocAsync(&p, bytes, stream);
        if (e == cudaSuccess) { used_async_out = true; return p; }
        cudaGetLastError();
#endif
        used_async_out = false;
        CHECK_CUDA(cudaMalloc(&p, bytes));
        return p;
    }

    static void free_ptr(void* p, bool used_async, cudaStream_t stream) {
        if (!p) return;
#if CUDART_VERSION >= 11020
        if (used_async) {
            cudaError_t e = cudaFreeAsync(p, stream);
            if (e == cudaSuccess) return;
            cudaGetLastError();
        }
#endif
        CHECK_CUDA(cudaFree(p));
    }

    static int current_device() {
        int dev = 0;
        CHECK_CUDA(cudaGetDevice(&dev));
        if (dev < 0 || dev >= kMaxDevices) std::abort();
        return dev;
    }

    // 可重入：如果 bytes_ 不一样，就 free + 重新分配
    void* get_raw(cudaStream_t stream = 0) {
        int dev = current_device();
        auto& d = dev_[dev];

        std::lock_guard<std::mutex> lk(d.m);

        if (d.st == State::Allocated) {
            if (d.bytes == bytes_) {
                return d.ptr; // 复用
            }

            free_ptr(d.ptr, d.used_async, stream);
            d.ptr = nullptr;
            d.bytes = 0;
            d.used_async = false;
            d.st = State::Empty;
        }

        // 走到这里一定是 Empty：分配
        d.ptr = alloc_bytes(bytes_, stream, d.used_async);
        d.bytes = bytes_;
        d.st = State::Allocated;
        return d.ptr;
    }

    template <typename T>
    T* get(cudaStream_t stream = 0) { return reinterpret_cast<T*>(get_raw(stream)); }

    // 手动释放：释放后未来 get 会重新分配
    void release_current(cudaStream_t stream = 0) {
        int dev = current_device();
        auto& d = dev_[dev];
        std::lock_guard<std::mutex> lk(d.m);

        if (d.st == State::Allocated) {
            free_ptr(d.ptr, d.used_async, stream);
            d.ptr = nullptr;
            d.bytes = 0;
            d.used_async = false;
            d.st = State::Empty;
        }
    }

    // resize：下一次 get 会按新大小重分配（如果不同）
    void set_bytes(size_t new_bytes) { bytes_ = new_bytes; }

    size_t get_bytes(cudaStream_t stream = 0) {
        int dev = current_device();
        auto& d = dev_[dev];

        std::lock_guard<std::mutex> lk(d.m);

        size_t bytes = 0;
        if (d.st == State::Allocated) {
            bytes = d.bytes;
        }

        return bytes;
    }
};

// SIOF-safe 单例
// 全局单例，cuda driver 在进程结束的时候回收内存
inline WorkspaceMutexSM& global_ws_sm() {
    static WorkspaceMutexSM ws(1ull << 22);
    return ws;
}
