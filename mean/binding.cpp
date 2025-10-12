#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>

// kernel 启动声明
void launch_mean_strided(
            const at::Tensor& x, at::Tensor& y,
            at::Tensor keep_sizes, at::Tensor keep_strides,
            at::Tensor red_sizes,  at::Tensor red_strides);

// 将任意形状 + 任意 dim(单/多) 的 mean 转换为 [outer, reduce, inner] 的视图，再调用 GPU kernel
at::Tensor mean_highdim_cuda_impl(const at::Tensor& x, at::OptionalIntArrayRef dim_opt) {
    TORCH_CHECK(x.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(x.dim() >= 1, "input must have at least 1 dim");

    // 1) 解析 reduce 维
    std::vector<int64_t> reduce_axes;
    if (dim_opt.has_value()) {
        auto dref = dim_opt.value();
        reduce_axes.assign(dref.begin(), dref.end());
        for (auto& d : reduce_axes) { if (d < 0) d += x.dim(); }
        std::sort(reduce_axes.begin(), reduce_axes.end());
        reduce_axes.erase(std::unique(reduce_axes.begin(), reduce_axes.end()), reduce_axes.end());
        for (auto d : reduce_axes) TORCH_CHECK(0 <= d && d < x.dim(), "dim out of range");
    } else {
        reduce_axes.resize(x.dim());
        std::iota(reduce_axes.begin(), reduce_axes.end(), 0);
    }

    // 全部约简 => 标量
    if ((int)reduce_axes.size() == x.dim()) {
        return x.mean(); // 这里直接用原生，不会重排数据
    }

    // 2) 组装 keep/reduce 的 sizes/strides（保持原 layout，不做 permute/contig）
    std::vector<int64_t> keep_axes; keep_axes.reserve(x.dim() - reduce_axes.size());
    {
        int p = 0;
        for (int64_t i = 0; i < x.dim(); ++i) {
            if (p < (int)reduce_axes.size() && reduce_axes[p] == i) { ++p; }
            else keep_axes.push_back(i);
        }
    }

    auto sizes   = x.sizes().vec();
    auto strides = x.strides().vec();

    std::vector<int64_t> keep_sizes, keep_strides, red_sizes, red_strides;
    keep_sizes.reserve(keep_axes.size());
    keep_strides.reserve(keep_axes.size());
    for (auto a : keep_axes) { keep_sizes.push_back(sizes[a]); keep_strides.push_back(strides[a]); }

    red_sizes.reserve(reduce_axes.size());
    red_strides.reserve(reduce_axes.size());
    for (auto a : reduce_axes) { red_sizes.push_back(sizes[a]); red_strides.push_back(strides[a]); }

    // 3) 输出形状：保留维的 sizes（顺序保持与原张量一致）
    std::vector<int64_t> out_shape;
    for (auto a : keep_axes) out_shape.push_back(sizes[a]);
    auto y = at::empty(out_shape, x.options());

    // 4) 把 meta 拷到 GPU（很小的张量）
    auto opts = x.options().dtype(at::kLong).device(x.device());
    at::Tensor keep_sizes_t   = at::from_blob(keep_sizes.data(), {(int64_t)keep_sizes.size()}, at::TensorOptions().dtype(at::kLong)).clone().to(opts.device());
    at::Tensor keep_strides_t = at::from_blob(keep_strides.data(), {(int64_t)keep_strides.size()}, at::TensorOptions().dtype(at::kLong)).clone().to(opts.device());
    at::Tensor red_sizes_t    = at::from_blob(red_sizes.data(),  {(int64_t)red_sizes.size()},  at::TensorOptions().dtype(at::kLong)).clone().to(opts.device());
    at::Tensor red_strides_t  = at::from_blob(red_strides.data(),{(int64_t)red_strides.size()},at::TensorOptions().dtype(at::kLong)).clone().to(opts.device());

    // 5) 启动 kernel（零重排，按 stride 访问）
    launch_mean_strided(x, y, keep_sizes_t, keep_strides_t, red_sizes_t, red_strides_t);
    return y;
}

static at::Tensor mean_highdim_cuda_wrap(const at::Tensor& x, py::object dim_obj) {
    if (dim_obj.is_none()) {
        return mean_highdim_cuda_impl(x, c10::optional<at::IntArrayRef>());  // None => 全部约简
    }

    std::vector<int64_t> dims;
    if (py::isinstance<py::int_>(dim_obj)) {
        dims.push_back(dim_obj.cast<int64_t>());
    } else if (py::isinstance<py::tuple>(dim_obj)) {
        auto t = dim_obj.cast<py::tuple>();
        dims.reserve(t.size());
        for (auto item : t) dims.push_back(py::cast<int64_t>(item));
    } else if (py::isinstance<py::list>(dim_obj)) {
        auto l = dim_obj.cast<py::list>();
        dims.reserve(l.size());
        for (auto item : l) dims.push_back(py::cast<int64_t>(item));
    } else {
        throw std::runtime_error("dim must be int, tuple[int], list[int], or None");
    }

    // 把 vector 包装成 IntArrayRef，再喂给 impl
    at::IntArrayRef dim_ref(dims);
    // 注意：impl 内部要把 dim_ref 的内容复制出来使用（我之前给的实现就是先复制到 std::vector 再处理），
    // 这样不会出现生命周期问题。
    return mean_highdim_cuda_impl(x, dim_ref);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mean_highdim_cuda", &mean_highdim_cuda_wrap,
          "High-dimensional mean over arbitrary dims (CUDA)",
          py::arg("input"),
          py::arg("dim") = py::none());
}
