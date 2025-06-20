#!/bin/bash

# git clone --recursive https://github.com/pytorch/pytorch
# cd pytorch
# git submodule sync
# git submodule update --init --recursive

# conda create -n pytorch-build python=3.10 -y
# conda activate pytorch-build
# pip install -r requirements.txt

export USE_ROCM=0
export USE_CUDA=1
export USE_CUDNN=1
export USE_NCCL=0	                # 启用分布式训练支持
export TORCH_CUDA_ARCH_LIST="8.6"   # RTX 4060 的 CUDA Compute Capability 是 8.6
export USE_MKL=0
export USE_MKLDNN=0                 # 用于 CPU 加速（Intel oneDNN）
export BUILD_PYTHON=0
export BUILD_TEST=0
export BUILD_EXAMPLES=0
export BUILD_BENCHMARK=0
export USE_DISTRIBUTED=0
export USE_XNNPACK=0
export USE_QNNPACK=0
export USE_GLOO=0                   # facebook 通信库
export USE_MPI=0
export USE_JIT=0
export USE_OPENMP=0
export USE_FLASH_ATTENTION=0
export USE_MEM_EFF_ATTENTION=0
export CMAKE_GENERATOR=Ninja        # 推荐使用 Ninja 加快构建
export MAX_JOBS=8                  # 根据 CPU 核心数设置（i7 适合 8~16）

python setup.py develop