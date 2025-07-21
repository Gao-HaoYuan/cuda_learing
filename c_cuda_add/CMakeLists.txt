
cmake_minimum_required(VERSION 3.20)
project(MyAdd LANGUAGES CXX CUDA)

# Set C++/CUDA standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA compiler and host compiler
set(CMAKE_CXX_COMPILER clang++ CACHE STRING "C++ compiler")
set(CMAKE_CUDA_COMPILER nvcc)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

# Compilation flags
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC;-O3;--ptxas-options=-v")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
add_compile_options(
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC;-O3>"
    "$<$<COMPILE_LANGUAGE:CXX>:-O3>"
)

if(WIN32)
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/MD>"
    )
endif()

# Check CUDA_ARCHITECTURES
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(FATAL_ERROR "CUDA_ARCHITECTURES is not defined.")
endif()

# Check CUDA Path
if (NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
    message(FATAL_ERROR "CUDA_PATH is not defined.")
endif()

include_directories(
    ${CUDA_TOOLKIT_ROOT_DIR}/include
)

link_directories(
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)

# add_subdirectory(my_add)
file(GLOB CUDA_SOURCES ./*.cu)
add_library(my_add STATIC ${CUDA_SOURCES})
target_link_libraries(my_add PRIVATE
    cudart
    cublas
)

install(TARGETS my_add DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
