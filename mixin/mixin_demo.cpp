#include <iostream>
#include <utility>
#include <type_traits>

// 递归辅助展开 using operator()
template<typename... Ts> 
struct UsingOperator;

template<typename T>
struct UsingOperator<T> : T {
    using T::operator();
};

template<typename T, typename U, typename... Ts>
struct UsingOperator<T, U, Ts...> : UsingOperator<T>, UsingOperator<U, Ts...> {
    using UsingOperator<T>::operator();
    using UsingOperator<U, Ts...>::operator();
};

// MixinClass，继承所有 Mixins 并导出 operator()
template<typename... Mixins>
class MixinClass : public UsingOperator<Mixins...> {
public:

    template<typename... Args>
    MixinClass(Args&&... args) : UsingOperator<Mixins...>(std::forward<Args>(args)...) {}
};

// 两个简单算子
struct AddMixin {
    int operator()(int a, int b) const {
        return a + b;
    }
};

struct MulMixin {
    int operator()(int a, int b) const {
        return a * b;
    }
};

int main() {
    MixinClass<AddMixin, MulMixin> mixinObj;

    int a = 3, b = 4;

    // 明确指定调用哪个基类 operator()
    std::cout << "AddMixin: " << mixinObj.AddMixin::operator()(a, b) << "\n";
    std::cout << "MulMixin: " << mixinObj.MulMixin::operator()(a, b) << "\n";

    return 0;
}
