## 说明

一个简单的 demo，用来介绍怎么用 torch 编一个 c lib.so，让另一个 c 工程调用

## 流程
1. 编 producer（Python / CUDAExtension）
```
cd torchclass_demo/producer
python setup.py build_ext --inplace
```
2. 编 consumer（C++ / LibTorch）
```
cd ../consumer
python setup.py build_ext --inplace
```
