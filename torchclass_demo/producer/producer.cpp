#include <torch/extension.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/util/intrusive_ptr.h>

struct MyState : torch::CustomClassHolder {
    int64_t bias;
    MyState(int64_t b = 0) : bias(b) {}
    int64_t get_bias() const { return bias; }
    void set_bias(int64_t b) { bias = b; }
};

// --------- 核心计算：y = x * w + (state.bias) ---------
static torch::Tensor infer_cpu_impl(const c10::intrusive_ptr<MyState>& st, torch::Tensor x, torch::Tensor w) {
    // 简单起见：都放 CPU 上算
    x = x.contiguous();
    w = w.contiguous();
    auto y = x * w + st->bias;
    return y;
}

// --------- Autograd：forward 调用 infer_cpu_impl，backward 返回 dx, dw ---------
static torch::Tensor infer_autograd_impl(const c10::intrusive_ptr<MyState>& st, torch::Tensor x, torch::Tensor w) {
    printf("infer by autograd\n");
    return infer_cpu_impl(st, x, w);
}

// --------- 注册 schema + class ---------
TORCH_LIBRARY(producer, m) {
    // 暴露 custom class 到 torch.classes.producer.MyState
    m.class_<MyState>("MyState")
        .def(torch::init<int64_t>())
        .def("get_bias", &MyState::get_bias)
        .def("set_bias", &MyState::set_bias);

    // op：创建 state（返回 custom class）
    m.def("make_state(int bias) -> __torch__.torch.classes.producer.MyState");

    // op：infer(state, x, w) -> y
    m.def("infer(__torch__.torch.classes.producer.MyState state, Tensor x, Tensor w) -> Tensor");
}

// make_state 的实现（CPU）
static c10::intrusive_ptr<MyState> make_state_impl(int64_t bias) {
    return c10::make_intrusive<MyState>(bias);
}

TORCH_LIBRARY_IMPL(producer, CPU, m) {
    m.impl("infer", TORCH_FN(infer_cpu_impl));
}

TORCH_LIBRARY_IMPL(producer, CompositeExplicitAutograd, m) {
    m.impl("make_state", TORCH_FN(make_state_impl));
}

// autograd key 下的实现（如果你有自写 backward，可在这里接入）
TORCH_LIBRARY_IMPL(producer, Autograd, m) {
    m.impl("infer", TORCH_FN(infer_autograd_impl));
}

// 生成 pybind11 的符号
PYBIND11_MODULE(producer, m) {}
