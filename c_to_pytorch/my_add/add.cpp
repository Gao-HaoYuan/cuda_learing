#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "add.h"
#include "support.h"

torch::Tensor my_add(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    const int blocks = 256;
    const int grids = (size + blocks - 1) / blocks;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "my_add", ([&] {
                                   my_add_cuda(ptr<scalar_t>(input),
                                               ptr<scalar_t>(output),
                                               size,
                                               grids,
                                               blocks);
                               }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_add", &my_add, "Add 1 to tensor (CUDA)");
}
