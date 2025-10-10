// binding.cpp
#include <torch/extension.h>

void launch_curand_normal_kernel(at::Tensor out, unsigned long long seed, float mean, float std);

void curand_normal_(at::Tensor out, unsigned long long seed, float mean, float std) {
    launch_curand_normal_kernel(out, seed, mean, std);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // 用 py::arg 显式声明关键字参数
    m.def(
        "curand_normal_",
        &curand_normal_,
        py::arg("out"),
        py::arg("seed") = 1234ULL,
        py::arg("mean") = 0.0f,
        py::arg("std") = 1.0f,
        "Generate Gaussian noise with cuRAND (CUDA)"
    );
}
