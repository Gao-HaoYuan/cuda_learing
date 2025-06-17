#pragma once
#include <torch/extension.h>

at::Tensor linear_cpu(const at::Tensor &a, const at::Tensor &b, double c);
at::Tensor linear_cuda(const at::Tensor &a, const at::Tensor &b, double c);
