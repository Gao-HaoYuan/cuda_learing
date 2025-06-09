#include <ATen/ATen.h>
#include "linear.h"

__global__ void linear_kernel(int n, const float* a, const float* b, float c, float* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] * b[i] + c;
  }
}

at::Tensor linear_cuda(const at::Tensor& a, const at::Tensor& b, double c) {
  TORCH_CHECK(a.sizes() == b.sizes(), "Tensors must be the same size");
  TORCH_CHECK(a.dtype() == at::kFloat && b.dtype() == at::kFloat, "Both tensors must be float32");
  TORCH_INTERNAL_ASSERT(a.is_cuda() && b.is_cuda());

  auto a_contig = a.contiguous();
  auto b_contig = b.contiguous();
  auto result = torch::empty_like(a_contig);

  int numel = a.numel();
  const int threads = 256;
  const int blocks = (numel + threads - 1) / threads;

  linear_kernel<<<blocks, threads>>>(
      numel,
      a_contig.data_ptr<float>(),
      b_contig.data_ptr<float>(),
      static_cast<float>(c),
      result.data_ptr<float>());

  return result;
}
