#!/bin/bash

# git clone --recursive https://github.com/pytorch/pytorch
# cd pytorch
# git submodule sync
# git submodule update --init --recursive

# conda create -n pytorch-build python=3.10 -y
# conda activate pytorch-build
# pip install -r requirements.txt

set -euo pipefail

export USE_ROCM=0
export USE_CUDA=1
export USE_CUDNN=1          # 如果你没装好 cuDNN，把这行改成 0
export USE_NCCL=0

export TORCH_CUDA_ARCH_LIST="8.6"
export CMAKE_GENERATOR=Ninja
export MAX_JOBS=8

# 精简：保留 Python + Autograd 相关能力
export BUILD_TEST=0
export BUILD_EXAMPLES=0
export BUILD_BENCHMARK=0

export USE_DISTRIBUTED=0
export USE_GLOO=0
export USE_MPI=0

# 不建议关：先保证生成物完整（你要看的 autograd codegen）
unset BUILD_PYTHON          # ✅ 关键：别设 0
unset USE_JIT               # ✅ 关键：先别关

# CPU 侧你可以关（可选）
export USE_MKL=0
export USE_MKLDNN=0
export USE_OPENMP=0

# mobile/quant 加速相关（可关）
export USE_XNNPACK=0
export USE_QNNPACK=0

# attention 优化（可关）
export USE_FLASH_ATTENTION=0
export USE_MEM_EFF_ATTENTION=0

python -m pip install -U pip setuptools wheel ninja cmake
python setup.py develop

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.__file__)"