# 这是CMakeLists.txt文件内容，关于现代CMake可以查看我的另一篇文章
cmake_minimum_required(VERSION 3.22)

# 指定CUDA标准版本
set(CMAKE_CUDA_STANDARD 17)
# 设置为发布模式
set(CMAKE_BUILD_TYPE Release)

project(cuda_test LANGUAGES CXX CUDA)

## CUDA架构代码说明（根据GPU型号选择对应的代码）
# 60 - Pascal (GTX 10xx)
# 70 - Volta
# 75 - Turing (RTX 20xx)
# 80 - Ampere (RTX 30xx)
# 86 - Ampere (部分 30xx)
# 90 - Hopper (最新架构)
set(CUDA_ARCHITECTURES 80)

# 添加src目录下源文件
file(GLOB_RECURSE SOURCES "./src/add/add.cu")

# 添加可执行文件
 add_executable(cuda_test ${SOURCES})
