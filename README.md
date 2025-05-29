该 Repo 用于 Cuda 学习测试

存在 c/cu to python 的需求，因此增加 c_to_python 的文件夹，来模拟整个流程，注意：
- 有三个地方的定义必须一致：
    - PyModuleDef：模块名称
    - PyInit_xxxx：xxxx （函数名）必须和模块名称一致
    - Extension：py 文件中的 Extension 类的 name 必须和模块名称一致
- c 代码输出的是 module， python 可以调用生成的 module，但注意 python 定义的包名不能和 module 名一样，否则两个符号的名字就一样了，会出现歧义，某种意义上讲 module 和 package 是等价的
    - 包（package）可以简单的理解为包含 \__init\__.py 的文件夹，或者说是 C++ 的头文件（.h）