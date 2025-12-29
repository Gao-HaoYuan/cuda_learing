#include <torch/extension.h>

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/saved_variable.h>

using at::Tensor;
using torch::autograd::SavedVariable;
using torch::autograd::variable_list;
using torch::autograd::TraceableFunction;

// ----------------------
// 1) 真实 CPU kernel（不走 autograd）
// y = a * b * c
// ----------------------
static Tensor scale_mul_cpu_kernel(const Tensor& a, const Tensor& b, double c) {
  TORCH_CHECK(a.device().is_cpu() && b.device().is_cpu(), "CPU only demo");
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), "dtype mismatch");
  TORCH_CHECK(a.sizes() == b.sizes(), "demo requires same shape");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "demo requires contiguous");

  auto out = at::empty_like(a);
  const auto n = a.numel();

  AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "scale_mul_cpu_kernel", [&] {
    const scalar_t* pa = a.data_ptr<scalar_t>();
    const scalar_t* pb = b.data_ptr<scalar_t>();
    scalar_t* po = out.data_ptr<scalar_t>();
    const scalar_t cc = static_cast<scalar_t>(c);
    for (int64_t i = 0; i < n; ++i) {
      po[i] = pa[i] * pb[i] * cc;
    }
  });

  return out;
}

// ----------------------
// 2) 手写 backward Node（TraceableFunction 风格）
// 注意：你这版 torch 没有 compiled_args/apply_with_saved 虚函数 => 不写
// ----------------------
struct TORCH_API ScaleMulBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;

  SavedVariable a_;
  SavedVariable b_;
  double c_{0.0};
  bool need_a_{false};
  bool need_b_{false};

  variable_list apply(variable_list&& grads) override {
    TORCH_CHECK(grads.size() == 1, "ScaleMulBackward0 expects 1 grad output");
    const auto& grad_out = grads[0];

    Tensor grad_a, grad_b;
    if (need_a_) {
      auto b = b_.unpack(shared_from_this());
      grad_a = grad_out * b * c_;
    }
    if (need_b_) {
      auto a = a_.unpack(shared_from_this());
      grad_b = grad_out * a * c_;
    }

    // forward 输入 (a, b, c) -> backward 返回 (grad_a, grad_b, None)
    return {grad_a, grad_b};
  }

  std::string name() const override { return "ScaleMulBackward0"; }

  void release_variables() override {
    a_.reset_data();
    b_.reset_data();
  }
};

// ----------------------
// 3) Autograd wrapper：创建 grad_fn + set_history
// ----------------------
static Tensor scale_mul_autograd_impl(const Tensor& a, const Tensor& b, double c) {
  const Tensor& a_ = a;
  const Tensor& b_ = b;

  std::shared_ptr<ScaleMulBackward0> grad_fn;

  // 是否需要构建 grad_fn
  bool executable = torch::autograd::GradMode::is_enabled();
  if (executable) {
    // any_variable_requires_grad 要吃 variable_list
    variable_list inputs;
    inputs.reserve(2);
    inputs.push_back(a_);
    inputs.push_back(b_);
    executable = torch::autograd::any_variable_requires_grad(inputs);
  }

  if (executable) {
    grad_fn = std::make_shared<ScaleMulBackward0>();

    // collect_next_edges 是 variadic：collect_next_edges(a_, b_)
    grad_fn->set_next_edges(torch::autograd::collect_next_edges(a_, b_));

    grad_fn->need_a_ = a_.requires_grad();
    grad_fn->need_b_ = b_.requires_grad();
    grad_fn->c_ = c;

    // 需要哪个梯度就存另一个（和你 python setup_context 的思路一致）
    if (grad_fn->need_a_) grad_fn->b_ = SavedVariable(b_, /*is_output*/false);
    if (grad_fn->need_b_) grad_fn->a_ = SavedVariable(a_, /*is_output*/false);
  }

  Tensor out;
  {
    at::AutoDispatchBelowAutograd guard;
    out = scale_mul_cpu_kernel(a_, b_, c);
  }

  if (grad_fn) {
    // 你这版没有 flatten_tensor_args，单输出直接 set_history(out, grad_fn)
    torch::autograd::set_history(out, grad_fn);
  }

  return out;
}

// ----------------------
// 4) schema + CPU + Autograd 注册
// ----------------------
TORCH_LIBRARY(myops, m) {
  m.def("scale_mul(Tensor a, Tensor b, float c) -> Tensor");
}

TORCH_LIBRARY_IMPL(myops, CPU, m) {
  m.impl("scale_mul", &scale_mul_cpu_kernel);
}

TORCH_LIBRARY_IMPL(myops, Autograd, m) {
  m.impl("scale_mul", &scale_mul_autograd_impl);
}

// pybind 模块空壳：extension 需要它生成 .so
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
