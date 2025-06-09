#include "linear.h"

at::Tensor linear_cpu(const at::Tensor& a, const at::Tensor& b, double c) {
  TORCH_CHECK(a.sizes() == b.sizes(), "Tensors must be the same size");
  TORCH_CHECK(a.dtype() == at::kFloat && b.dtype() == at::kFloat, "Both tensors must be float32");
  TORCH_INTERNAL_ASSERT(a.device().is_cpu() && b.device().is_cpu());

  auto a_contig = a.contiguous();
  auto b_contig = b.contiguous();
  auto result = torch::empty_like(a_contig);

  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();

  int64_t numel = a.numel();
  for (int64_t i = 0; i < numel; ++i) {
    result_ptr[i] = a_ptr[i] * b_ptr[i] + static_cast<float>(c);
  }

  return result;
}
