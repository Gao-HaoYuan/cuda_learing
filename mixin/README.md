
* .cpp compile
```
g++ 1.cpp -O3 -std=c++17 -o a
```

* .cu compile
```
nvcc -O3 -std=c++17 --expt-relaxed-constexpr 1.cu -o a
```

生成汇编代码
```
nvcc -O3 -std=c++17 --expt-relaxed-constexpr apply_gpu.cu -arch=sm_80 -ptx -o fused.ptx

```

* apply demo
```c++
template <class F, class Tuple>
__host__ __device__
inline constexpr decltype(auto) apply(F&& f, Tuple&& t) {
  return std::apply(std::forward<F>(f), std::forward<Tuple>(t));
}
```