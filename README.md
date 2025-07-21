<h1> Overview </h1>
该 Repo 主要用来学习 CUDA 开发，最终应用场景是优化模型训练和推理，cuda 代码的测试 demo 位于 src 文件中。

目前，该文档共描述了三种 C to python module 的方法，三种方法用法都不一样，注意辨别。工作还是以第三种方法为主： [c_to_pytorch](#c_to_pytorch)

<h2> c_cuda_add </h2>
该 demo 演示了怎么使用 pybind11 写一个基本的 cuda add 的用法

<h2> c_register_kernel </h2>
该 demo 演示了怎么给 pytorch 注册一个 kernel，在工作中用处应该不会很大，主要结合 pytorch 源码学习。

- 这个地方注意，这个地方只是注册 kernel，没有编译生成 python module，所以不能直接使用 import

<h2> c_to_python </h2>
存在 c/cu to python 的需求，因此增加 c_to_python 的文件夹，来模拟整个流程，注意：

- 有三个地方的定义必须一致：
    - PyModuleDef：模块名称
    - PyInit_xxxx：xxxx （函数名）必须和模块名称一致
    - Extension：py 文件中的 Extension 类的 name 必须和模块名称一致
- c 代码输出的是 module， python 可以直接 <mark>import module</mark>，但注意 python 定义的包名不能和 module 名一样，否则两个符号的名字就一样了，会出现歧义，某种意义上讲 module 和 package 是等价的
    - 包（package）可以简单的理解为包含 \__init\__.py 的文件夹，或者说是 C++ 的头文件（即 .h）

## <h2> c_to_pytorch </h2>
该目录演示了怎么为 pytorch 添加一个拓展，需要用到 <mark>PYBIND11_MODULE</mark>

- 具体用法参考 : [pybind11](https://pybind11.readthedocs.io/en/stable/classes.html)

同时，注意 .cpp 和 .cu 文件不要同名，如下面的编译方法会报错，因为 add.cpp/add.cu 编译都会生成 add.o，上一个文件的符号会被覆盖，所以会有一些奇奇怪怪的错误，**推荐使用 make 编译，而且文件命名的时候最好也体现出他的功能，算是一种开发规范**

``` python
# 错误代码演示
no_make = CUDAExtension(
                    name="my_add",
                    sources=["./my_add/add.cpp", "./my_add/add.cu"],
                    include_dirs=["./my_add"]
                  )

setup(
    name ="my_add",
    ext_modules = [no_make],
    cmdclass = {"build_ext": BuildExtension}
)
```

<h2> cuda-smaples </h2>
是 NVIDIA 官方的 demo，以 submodule 的形式导入

<h2> pytorch profiler </h2>
pytorch profiler 工具，export_stacks 生成空文件，这个是一个 bug，pytorch 官方还未修复

参考 issues : [100253](https://github.com/pytorch/pytorch/issues/100253)

<h2> pytorch </h2>
pytorch 源码， 核心逻辑是：

* Tensor：
* AutoGrad: pytorch/torch/csrc/autograd
* Cublas Handle: pytorch/aten/src/ATen/cuda/CublasHandlePool.cpp

(学习的时候继续补充)

<h2> c_log_control </h2>
设计一个 log demo，用于 debug 错误，避免反复编译
