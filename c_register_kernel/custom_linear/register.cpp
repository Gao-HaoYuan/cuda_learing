#include "linear.h"

TORCH_LIBRARY(custom_linear, m) {
    m.def("linear(Tensor a, Tensor b, float c) -> Tensor");
}

TORCH_LIBRARY_IMPL(custom_linear, CPU, m) {
    m.impl("linear", linear_cpu);
}

TORCH_LIBRARY_IMPL(custom_linear, CUDA, m) {
    m.impl("linear", linear_cuda);
}
