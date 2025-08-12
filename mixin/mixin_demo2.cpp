#include <iostream>
#include <type_traits>

template <typename Concrete, template <class> class H, template <class> class... Tail>
struct InheritChain {
    using result = typename InheritChain<Concrete, Tail...>::type;
    using type = H<result>;
};

template <typename Concrete, template <class> class H>
struct InheritChain<Concrete, H> {
    using type = H<Concrete>;
};

template <typename Derived>
struct MixinBase {
    using self_type = Derived;
    decltype(auto) self() & { return static_cast<self_type&>(*this); } // 只能在 lvalue 对象上调用
    decltype(auto) self() && { return static_cast<self_type&&>(*this); } // 只能在 rvalue 对象上调用
    decltype(auto) self() const& { return static_cast<self_type const&>(*this); } // 只能在 const lvalue 上调用
    decltype(auto) self() const&& { return static_cast<self_type const&&>(*this); } // 只能在 const rvalue 上调用
};

template <typename Concrete, template <class> class... Mixins>
using MixinImpl = typename InheritChain<MixinBase<Concrete>, Mixins...>::type;

template <typename Concrete, template <class> class... Mixins>
struct Mixin : public MixinImpl<Concrete, Mixins...> {
    template <typename... Rest>
    constexpr Mixin(Rest&&... rest)
            : MixinImpl<Concrete, Mixins...>(static_cast<decltype(rest)>(rest)...) {}

    Mixin() = default;
    Mixin(const Mixin&) = default;
    Mixin(Mixin&&) = default;
    Mixin& operator=(const Mixin&) = default;
    Mixin& operator=(Mixin&&) = default;
    ~Mixin() = default;
};

///// use demo

// template <typename Base>

template <typename Base>
struct A : public Base {
    using Base::Base; // 继承构造

    void fa() {
        std::cout << "A::fa()\n";
    }
};

template <typename Base>
struct B : public Base {
    using Base::Base;

    void fb() {
        std::cout << "B::fb()\n";
    }

    // 演示在基类调用 Derived 的方法
    void callDerivedHello() {
        this->self().hello(); // 保持值类别和类型正确
    }

    void checkType() {
        using Derived = typename Base::self_type;
        std::cout << typeid(decltype(this->self())).name() << " "
            << std::boolalpha
            << std::is_same_v<decltype(this->self()), Derived&>
            << "\n";
    }
};

template <typename Base>
struct C : public Base {
    using Base::Base;

    // C++20 写法
    // C(auto&&... rest) : Base(std::forward<decltype(rest)>(rest)...) {}

    // C++11 写法
    template<typename ... Rest>
    C(Rest&&... rest) : Base(std::forward<decltype(rest)>(rest)...) {}

    void fc() {
        std::cout << "C::fc()\n";
    }
};

// D 继承 A<B<C<MixinBase<D>>>>
// 调用 self 最终返回 D 的引用
struct D : public Mixin<D, A, B, C> {
    using Mixin<D, A, B, C>::Mixin; // 继承构造

    void hello() {
        std::cout << "D::hello()\n";
    }
};

int main() {
    D obj;  // A<B<C<MixinBase<D>>>>

    obj.fa();  // 来自 A
    obj.fb();  // 来自 B
    obj.fc();  // 来自 C

    obj.callDerivedHello(); // 在 B 中通过 self() 调用 D 的方法

    // 也可以直接访问 D 自己的方法
    obj.hello();

    // 测试 rvalue self() 调用
    D().callDerivedHello();

    obj.checkType();

    return 0;
}