<h3> Overview </h3>
该 Repo 主要用来学习 CUDA 开发，最终应用场景是优化模型训练和推理，cuda 代码的测试 demo 位于 src 文件中

<h4> c_to_python </h4>
存在 c/cu to python 的需求，因此增加 c_to_python 的文件夹，来模拟整个流程，注意：

- 有三个地方的定义必须一致：
    - PyModuleDef：模块名称
    - PyInit_xxxx：xxxx （函数名）必须和模块名称一致
    - Extension：py 文件中的 Extension 类的 name 必须和模块名称一致
- c 代码输出的是 module， python 可以直接 <mark>import module</mark>，但注意 python 定义的包名不能和 module 名一样，否则两个符号的名字就一样了，会出现歧义，某种意义上讲 module 和 package 是等价的
    - 包（package）可以简单的理解为包含 \__init\__.py 的文件夹，或者说是 C++ 的头文件（即 .h）

<h4> cuda-smaples </h4>
是 NVIDIA 官方的 demo，以 submodule 的形式导入

<h4> pytorch 用于测试并优化相关模型性能 </h4>
pytorch profiler 工具，export_stacks 生成空文件，这个是一个 bug，pytorch 官方还未修复

参考 issues : [100253](https://github.com/pytorch/pytorch/issues/100253)
