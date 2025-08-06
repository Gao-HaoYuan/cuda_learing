#!/bin/bash
set -e

BUILD_DIR="build"
BUILD_TYPE="Release"
INSTALL_DIR=$PWD
CUDA_ARCH="70;75;80;86;89;90"
ENABLE_DEBUG="OFF"
CUDA_PATH=${CUDA_HOME:-/usr/local/cuda}

while [[ $# -gt 0 ]]; do
  case $1 in
    --torch-toolchain) TORCH_TOOLCHAIN="$2"; shift 2 ;;
    --enable-debug) ENABLE_DEBUG="$2"; shift 2 ;;
    --clean) rm -rf "$BUILD_DIR"; echo "Cleaned."; exit 0 ;;
    *) shift ;;
  esac
done

echo "========= Build Configuration ========="
echo "Build Type      : $BUILD_TYPE"
echo "CUDA Path       : $CUDA_PATH"
echo "CUDA Archs      : $CUDA_ARCH"
echo "torch toolchain : $TORCH_TOOLCHAIN"
echo "Enable Debug    : $ENABLE_DEBUG"
echo "========================================"

if ! command -v ninja &> /dev/null; then
  echo "Error: Ninja build system is not installed." >&2
  exit 1
fi

mkdir -p "$BUILD_DIR"

CMAKE_ARGS=(
  -G Ninja
  -DCMAKE_BUILD_TYPE=$BUILD_TYPE
  -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR
  -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH
  -DENABLE_DEBUG=$ENABLE_DEBUG
  -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_PATH
  # -DTORCH_INCLUDE_DIR=$TORCH_INCLUDE
  # -DTORCH_LIBRARY_DIR=$TORCH_LIB
  -DCMAKE_PREFIX_PATH=$TORCH_TOOLCHAIN
)

pushd "$BUILD_DIR"
cmake "${CMAKE_ARGS[@]}" ..
ninja install -v -j$(nproc)
popd
