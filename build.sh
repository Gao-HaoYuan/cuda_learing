#!/bin/bash

set -e  # 出错即退出

# 设置构建目录
BUILD_DIR="./build"
EXEC_NAME="cuda_test"

## CUDA架构代码说明（根据GPU型号选择对应的代码）
# 60 - Pascal (GTX 10xx)
# 70 - Volta
# 75 - Turing (RTX 20xx)
# 80 - Ampere (RTX 30xx)
# 86 - Ampere (部分 30xx)
# 90 - Hopper (最新架构)
CUDA_ARCH=80

# 清理构建目录
rm -rf "$BUILD_DIR"

# 配置项目
CMAKE_COMMAND="-B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"

# 构建项目
cmake ${CMAKE_COMMAND} .
cmake --build "$BUILD_DIR" -- -j$(nproc)

# 运行生成的可执行文件（如果存在）
EXEC_PATH="$BUILD_DIR/$EXEC_NAME"
if [[ -f "$EXEC_PATH" ]]; then
  echo "---------------------------------------------------------"
  "$EXEC_PATH"
else
  echo "Error: Executable $EXEC_PATH not found."
  exit 1
fi
