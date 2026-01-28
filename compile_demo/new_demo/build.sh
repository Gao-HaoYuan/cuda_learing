#!/usr/bin/env bash
set -e

TORCH_PREFIX=$(python - <<'EOF'
import torch
print(torch.utils.cmake_prefix_path)
EOF
)

echo "[INFO] torch cmake prefix: $TORCH_PREFIX"

INSTALL_PREFIX="$(pwd)"
echo "[INFO] install prefix: $INSTALL_PREFIX"

mkdir -p build

# -S CMakeLists.txt 在的目录，-B 编译的目录
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" 
cmake --build build -j # -j 并行编译的线程数
cmake --install build

echo "[SUCCESS] consumer build finished"
