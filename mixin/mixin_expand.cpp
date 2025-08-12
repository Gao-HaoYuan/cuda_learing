#include <iostream>
#include <utility>
#include <type_traits>
#include <chrono>

template<typename First, typename... Rest>
struct OpInvokerChain : OpInvokerChain<Rest...> {
    First first;

    OpInvokerChain(First f, Rest... rest) : first(std::move(f)), OpInvokerChain<Rest...>(rest...) {}

    template<typename... Args>
    auto operator()(Args&&... args) const {
        auto intermediate = first(std::forward<Args>(args)...);
        return OpInvokerChain<Rest...>::operator()(intermediate);
    }
};

template<typename Last>
struct OpInvokerChain<Last> {
    Last last;

    OpInvokerChain(Last l) : last(std::move(l)) {}

    template<typename... Args>
    auto operator()(Args&&... args) const {
        return last(std::forward<Args>(args)...);
    }
};

// MixinClass 包装调用者
template<typename... Mixins>
struct MixinClass : OpInvokerChain<Mixins...> {
    MixinClass(Mixins... mixins) : OpInvokerChain<Mixins...>(std::move(mixins)...) {}

    using OpInvokerChain<Mixins...>::operator();
};


struct Add {
    int operator()(int a, int b) const { return a + b; }
};

struct Sub {
    int operator()(int a, int b) const { return a - b; }
};

struct Relu {
    int operator()(int x) const { return x > 0 ? x : 0; }
};

struct Mul {
    int operator()(int a, int b) const { return a * b; }
};

int main() {
    constexpr int N = 10000000;
    int a = 5, b = 3;

    // 直接调用三个算子的循环
    Add add;
    Relu relu;

    auto start1 = std::chrono::high_resolution_clock::now();
    volatile int result1 = 0;
    for (int i = 0; i < N; ++i) {
        int sum = add(a, b);
        result1 = relu(sum);
    }
    auto end1 = std::chrono::high_resolution_clock::now();

    MixinClass<Add, Relu> fusedOp(Add{}, Relu{});
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