#!/bin/bash

# 设置构建目录
BUILD_DIR="./build"

# 清理之前的构建
echo "Cleaning previous builds..."
rm -rf "$BUILD_DIR"

# 创建并构建项目
echo "Configuring project with CMake..."
cmake -B "$BUILD_DIR" || { echo "CMake configuration failed"; exit 1; }

# 编译项目
echo "Building the project..."
cmake --build "$BUILD_DIR" || { echo "Build failed"; exit 1; }

# 执行生成的可执行文件
EXEC_PATH="$BUILD_DIR/cuda_test"
if [[ -f "$EXEC_PATH" ]]; then
  echo "Running the program..."
  "$EXEC_PATH"
else
  echo "Error: Executable $EXEC_PATH not found."
  exit 1
fi
