#!/bin/bash
set -e

BUILD_DIR="./build"
BUILD_TYPE="Release"
INSTALL_DIR=$PWD
CUDA_ARCH=(70 75 80 86 89 90)

# get cuda toolkit path
CUDA_PATH=$CUDA_HOME

if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating build directory: $BUILD_DIR."
    mkdir "$BUILD_DIR"
fi

CMAKE_COMMAND="-G Ninja \
            -B $BUILD_DIR \
            -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
            -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
            -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH \
            -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_PATH:-/usr/local/cuda} \
            -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"

cmake ${CMAKE_COMMAND} .
# make library
pushd .
cd "$BUILD_DIR"
ninja install -v
popd