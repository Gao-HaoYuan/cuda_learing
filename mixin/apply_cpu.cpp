#include <iostream>
#include <tuple>
#include <utility>
#include <chrono>

// 下面是你之前的示例算子（加、减、乘）
struct Add {
    int operator()(int a, int b) const { return a + b; }
};
struct Sub {
    int operator()(int a, int b) const { return a - b; }
};
struct Mul {
    int operator()(int a, int b) const { return a * b; }
};

// 简单的 apply_all 模板辅助
template <typename Tuple, std::size_t... I, typename... Args>
auto apply_all_impl(const Tuple& funcs, std::index_sequence<I...>, Args&&... args) {
    return std::make_tuple(std::get<I>(funcs)(std::forward<Args>(args)...)...);
}

template <typename Tuple, typename... Args>
auto apply_all(const Tuple& funcs, Args&&... args) {
    return apply_all_impl(funcs,
                          std::make_index_sequence<std::tuple_size<Tuple>::value>{},
                          std::forward<Args>(args)...);
}

// 你的 ParallelThenCombine 结构体
template<typename Combiner, typename... Fused>
struct ParallelThenCombine {
    std::tuple<Fused...> functors;
    Combiner combiner;

    ParallelThenCombine(Combiner c, Fused... fs)
        : combiner(std::move(c)), functors(std::move(fs)...) {}

    template <typename... Args>
    auto operator()(Args&&... args) const {
        auto results = apply_all(functors, std::forward<Args>(args)...);
        return std::apply(combiner, results);
    }
};

int main() {
    constexpr int N = 10000000;
    int a = 5, b = 3;

    // 直接调用三个算子的循环
    Add add;
    Sub sub;
    Mul mul;

    auto start1 = std::chrono::high_resolution_clock::now();
    volatile int result1 = 0;
    for (int i = 0; i < N; ++i) {
        int sum = add(a, b);
        int diff = sub(a, b);
        result1 = mul(sum, diff);
    }
    auto end1 = std::chrono::high_resolution_clock::now();

    // 用 ParallelThenCombine 融合算子
    ParallelThenCombine<Mul, Add, Sub> fusedOp(Mul{}, Add{}, Sub{});
    auto start2 = std::chrono::high_resolution_clock::now();
    volatile int result2 = 0;
    for (int i = 0; i < N; ++i) {
        result2 = fusedOp(a, b);
    }
    auto end2 = std::chrono::high_resolution_clock::now();

    std::cout << "Direct result: " << result1 << "\n";
    std::cout << "Fused result: " << result2 << "\n";

    auto dur1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    auto dur2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);

    std::cout << "Direct call time: " << dur1.count() << " us\n";
    std::cout << "Fused call time: " << dur2.count() << " us\n";

    return 0;
}
