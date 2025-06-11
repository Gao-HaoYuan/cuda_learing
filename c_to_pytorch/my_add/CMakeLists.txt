file(GLOB CUDA_SOURCES ./add_kernel.cu)
add_library(my_add INTERFACE ${CUDA_SOURCES})

target_link_libraries(my_add INTERFACE
    cudart
    cublas
)

target_include_directories(my_add INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
