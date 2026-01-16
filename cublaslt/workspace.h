#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <mutex>

#include "inline_check.h"

struct WorkspaceMutexSM {
    static constexpr int kMaxDevices = 16;

    enum class Backend : uint8_t { Empty, TorchBorrowed, Owned };

    struct PerDevice {
        std::mutex m;
        void*   ptr = nullptr;
        size_t  bytes = 0;
        bool    used_async = false;  // 只对 Owned 有意义
        Backend be = Backend::Empty;
    };

    // fallback 期望的 workspace 大小（Torch 模式下会被忽略）
    size_t bytes_req_ = 0;
    std::array<PerDevice, kMaxDevices> dev_{};

    explicit WorkspaceMutexSM(size_t bytes) { set_bytes(bytes); }
    WorkspaceMutexSM() = default;
    WorkspaceMutexSM(const WorkspaceMutexSM&) = delete;
    WorkspaceMutexSM& operator=(const WorkspaceMutexSM&) = delete;

    static int current_device() {
        int dev = 0;
        CHECK_CUDA(cudaGetDevice(&dev));
        if (dev < 0 || dev >= kMaxDevices) std::abort();
        return dev;
    }

    // ---- Torch provider（仅在 >=2.9.0 可用）----
    static inline bool torch_supported() {
#if TORCH_VERSION_GE(2, 9, 0)
        return true;
#else
        return false;
#endif
    }

    static inline void torch_query(void*& ptr_out, size_t& bytes_out) {
#if TORCH_VERSION_GE(2, 9, 0)
        bytes_out = at::cuda::getCUDABlasLtWorkspaceSize();
        ptr_out   = (bytes_out ? at::cuda::getCUDABlasLtWorkspace() : nullptr);
#else
        (void)ptr_out; (void)bytes_out;
        ptr_out = nullptr; bytes_out = 0;
#endif
    }

    // ---- fallback allocator（Owned）----
    static void* alloc_owned(size_t bytes, cudaStream_t stream, bool& used_async_out) {
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

    static void free_owned(void* p, bool used_async, cudaStream_t stream) {
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

    // ---- PerDevice 统一“释放/清空”（根据 backend）----
    static void clear_pd(PerDevice& d, cudaStream_t stream) {
        if (d.be == Backend::Owned) {
            free_owned(d.ptr, d.used_async, stream);
        }
        d.ptr = nullptr;
        d.bytes = 0;
        d.used_async = false;
        d.be = Backend::Empty;
    }

    void* get_raw(cudaStream_t stream = 0) {
        int dev = current_device();
        auto& d = dev_[dev];
        std::lock_guard<std::mutex> lk(d.m);

        // TorchBorrowed 模式：PerDevice 也保存 ptr/bytes，但不拥有、不释放
        if (torch_supported()) {
            void*  p = nullptr;
            size_t b = 0;
            torch_query(p, b);

            // 如果之前是 Owned，先清理掉（避免泄露）
            if (d.be == Backend::Owned) clear_pd(d, stream);

            d.ptr = p;
            d.bytes = b;
            d.used_async = false;
            d.be = Backend::TorchBorrowed;
            return d.ptr;
        }

        // fallback Owned 模式：按 bytes_req_ 缓存复用
        if (d.be == Backend::Owned) {
            if (d.bytes == bytes_req_) {
                return d.ptr;
            }

            clear_pd(d, stream);
        }

        d.ptr = alloc_owned(bytes_req_, stream, d.used_async);
        d.bytes = bytes_req_;
        d.be = Backend::Owned;
        return d.ptr;
    }

    template <typename T>
    T* get(cudaStream_t stream = 0) { return reinterpret_cast<T*>(get_raw(stream)); }

    void release_current(cudaStream_t stream = 0) {
        int dev = current_device();
        auto& d = dev_[dev];
        std::lock_guard<std::mutex> lk(d.m);

        if (d.be != Backend::Empty) clear_pd(d, stream);
    }

    void set_bytes(size_t new_bytes) {
        bytes_req_ = new_bytes; // Torch 模式下会被忽略；fallback 模式会用
    }

    size_t get_bytes(cudaStream_t stream = 0) {
        int dev = current_device();
        auto& d = dev_[dev];
        std::lock_guard<std::mutex> lk(d.m);

        return d.bytes;
    }
};

// SIOF-safe 单例
// 全局单例，cuda driver 在进程结束的时候回收内存
inline WorkspaceMutexSM& global_ws_sm() {
    static WorkspaceMutexSM ws(1ull << 24);
    return ws;
}
