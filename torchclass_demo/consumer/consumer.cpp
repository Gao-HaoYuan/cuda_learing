#include <torch/extension.h>
#include <torch/torch.h>

#include <ATen/core/dispatch/Dispatcher.h>
#include <dlfcn.h>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_set>

static std::mutex g_mu;
static std::unordered_set<std::string> g_loaded;
static std::vector<void*> g_handles;

static void dlopen_keepalive_once(const std::string& path) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_loaded.count(path)) return;

    void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        throw std::runtime_error(std::string("dlopen failed: ") + dlerror());
    }
    g_handles.push_back(h);
    g_loaded.insert(path);

    std::cerr << "[consumer] dlopen ok: " << path << "\n";
}

torch::Tensor
infer_by_producer_so(
    std::string path,
    int64_t bias,
    const torch::Tensor& x,
    const torch::Tensor& w
) {
    // 1) load producer.so once
    // python 代码 import 后不需要再调用
    // dlopen_keepalive_once(path);

    auto& disp = c10::Dispatcher::singleton();

    // 2) lookup ops
    auto op_make  = disp.findSchemaOrThrow("producer::make_state", "");
    auto op_infer = disp.findSchemaOrThrow("producer::infer", "");

    auto schema = op_infer.schema();
    size_t nret = schema.returns().size();
    std::cout << "return size is:" << nret << std::endl;

    // 3) make_state(bias)
    std::vector<c10::IValue> st_stack;
    st_stack.emplace_back(bias);
    disp.callBoxed(op_make, &st_stack);

    if (st_stack.empty())
        throw std::runtime_error("make_state returned empty stack");

    c10::IValue st = st_stack.back();

    // 4) infer(state, x, w)
    std::vector<c10::IValue> stack;
    stack.emplace_back(st);
    stack.emplace_back(x);
    stack.emplace_back(w);

    disp.callBoxed(op_infer, &stack);

    if (stack.empty() || !stack.back().isTensor())
        throw std::runtime_error("infer did not return Tensor");

    // 5) return safe tensor
    return stack.back().toTensor().contiguous().clone();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "infer_by_producer_so",
        &infer_by_producer_so,
        py::arg("path"),
        py::arg("bias"),
        py::arg("x"),
        py::arg("w")
    );
}
