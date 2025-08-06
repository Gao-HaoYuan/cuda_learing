cmake_minimum_required(VERSION 3.20)
project(cuKernel LANGUAGES CXX CUDA)

# === Compiler and standard ===
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable position-independent code (-fPIC) for all c++ targets
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# === CUDA Arch ===
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES is not defined, Pass it from the command line")
endif()
message(STATUS "CMAKE_CUDA_ARCHITECTURES is ${CMAKE_CUDA_ARCHITECTURES}")

# === CUDA Toolkit path check ===
if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
    message(FATAL_ERROR "CUDA_TOOLKIT_ROOT_DIR is not defined")
endif()
message(STATUS "CUDA_TOOLKIT_ROOT_DIR is ${CUDA_TOOLKIT_ROOT_DIR}")

# === Set custom compilers (optional) ===
# Can override with -DCMAKE_CXX_COMPILER=... from CLI
set(CMAKE_CXX_COMPILER g++ CACHE STRING "C++ compiler")
set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc CACHE STRING "Full Path: NVCC Compiler.")
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

# === Check Required Dependency Libraries ===
function(check_dir_nonempty dir_path dir_desc)
    cmake_parse_arguments(ARG "" "" "FILE_NAMES" ${ARGN})

    if(NOT EXISTS "${dir_path}")
        message(FATAL_ERROR "${dir_desc} directory not found: ${dir_path}")
    endif()

    foreach(file_name IN LISTS ARG_FILE_NAMES)
        set(full_path "${dir_path}/${file_name}")
        if(NOT EXISTS "${full_path}")
            message(FATAL_ERROR "${dir_desc}: NOT found '${file_name}' in ${dir_path}")
        endif()
    endforeach()
endfunction()

check_dir_nonempty(
    "${CUDA_TOOLKIT_ROOT_DIR}/include" 
    "CUDA include" 
    FILE_NAMES "cuda.h;cublas_v2.h;cuda_runtime.h;cuda_fp16.h"
)
check_dir_nonempty(
    "${CUDA_TOOLKIT_ROOT_DIR}/lib64" 
    "CUDA lib64" 
    FILE_NAMES "libcudart.so;libcublas.so"
)

find_package(Torch REQUIRED)
find_package(Python COMPONENTS Development REQUIRED)

# === Debug macro ===
option(ENABLE_DEBUG "Enable debug logging for witin_nn." OFF)
message(STATUS "Debug mode enabled: ${ENABLE_DEBUG}")

if(ENABLE_DEBUG)
    add_definitions(-DWITIN_DEBUG)
endif()

# === Compilation flags ===
if(ENABLE_DEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -O0 -Xcompiler=-g")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wno-sign-compare")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --extended-lambda --expt-relaxed-constexpr -Xcompiler=-fPIC -Xcompiler=-Wno-sign-compare")
endif()

# === Source files ===
set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/witin_nn/cu/)
file(GLOB_RECURSE CUDA_SOURCES
    "${SOURCE_DIR}/src/*.cu"
    "${SOURCE_DIR}/src/*.cpp"
)

# === Library target ===
set(TARGET_NAME cuKernel)

add_library(${TARGET_NAME} STATIC ${CUDA_SOURCES})
target_include_directories(${TARGET_NAME} PRIVATE
    ${SOURCE_DIR}/include
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${TORCH_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
)
target_link_libraries(${TARGET_NAME} PRIVATE
    cudart
    ${TORCH_LIBRARIES}
)
target_link_directories(${TARGET_NAME} PRIVATE
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)

# === Install target ===
install(TARGETS ${TARGET_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
