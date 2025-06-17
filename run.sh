#!/bin/bash

set -e  # 出错即退出

# 设置构建参数
BUILD_DIR="./build"
BUILD_TYPE="Release"
EXEC_NAME="cuda_test"
CUDA_PATH=$CUDA_HOME

## CUDA架构代码说明（根据GPU型号选择对应的代码）
# 60 - Pascal (GTX 10xx)
# 70 - Volta
# 75 - Turing (RTX 20xx)
# 80 - Ampere (RTX 30xx)
# 86 - Ampere (部分 30xx)
# 90 - Hopper (最新架构)
CUDA_ARCH=(70 75 80 86 89 90)

# 清理构建目录
rm -rf "$BUILD_DIR"
echo "Creating build directory: $BUILD_DIR."
mkdir $BUILD_DIR

CMAKE_COMMAND="-B $BUILD_DIR \
            -DEXEC_NAME=$EXEC_NAME \
            -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
            -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH \
            -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_PATH:-/usr/local/cuda} \
            -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"

# 构建项目
cmake ${CMAKE_COMMAND} .
cmake --build "$BUILD_DIR" -- -j$(nproc)

# 运行生成的可执行文件（如果存在）
EXEC_PATH="$BUILD_DIR/$EXEC_NAME"
echo "---------------------------------------------------------"
"$EXEC_PATH"
# ncu --set full --target-processes all -o my_report -f "$EXEC_PATH"