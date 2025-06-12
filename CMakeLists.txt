# 这是 CMakeLists.txt 文件内容
cmake_minimum_required(VERSION 3.22)

project(cuda_test LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED ENV{CUDA_HOME})
  message(WARNING "CUDA_HOME not set, using default.")
  set(CUDA_HOME "/usr/local/cuda")
else()
  set(CUDA_HOME $ENV{CUDA_HOME})
endif()

# 如果未从外部设置 CUDA_ARCHITECTURES，则提供默认值
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

file(GLOB_RECURSE SOURCES "./src/clamp.cu")
add_executable(cuda_test ${SOURCES})
target_include_directories(cuda_test PRIVATE ${CUDA_HOME}/include())

add_custom_target(
  copy_database_compile_commands ALL
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${PROJECT_BINARY_DIR}/compile_commands.json
  ${PROJECT_SOURCE_DIR}/compile_commands.json
)