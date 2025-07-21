#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>

#include "add.h"

namespace py = pybind11;

py::array_t<float> add(py::array_t<float> a_np, py::array_t<float> b_np) {
    py::buffer_info a_buf = a_np.request(), b_buf = b_np.request();

    if (a_buf.size != b_buf.size) {
        throw std::runtime_error("Mismatched sizes");
    }

    int n = a_buf.size;
    float *a_h = static_cast<float*>(a_buf.ptr);
    float *b_h = static_cast<float*>(b_buf.ptr);

    // 分配 GPU 内存
    float *a_d, *b_d, *out_d;
    cudaMalloc(&a_d, n * sizeof(float));
    cudaMalloc(&b_d, n * sizeof(float));
    cudaMalloc(&out_d, n * sizeof(float));

    // 拷贝输入数据到 GPU
    cudaMemcpy(a_d, a_h, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, n * sizeof(float), cudaMemcpyHostToDevice);

    // 启动 kernel
    const int blocks = 256;
    const int grids = (n + blocks - 1) / blocks;
    my_add_cuda(a_d, b_d, out_d, n, blocks, grids);

    // 拷贝输出数据到 Host
    auto result = py::array_t<float>(n);
    cudaMemcpy(result.mutable_data(), out_d, n * sizeof(float), cudaMemcpyDeviceToHost);

    // 释放 GPU 内存
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(out_d);

    return result;
}

PYBIND11_MODULE(cuda_add, m) {
    m.def("add", &add, "Add two arrays on GPU");
}
