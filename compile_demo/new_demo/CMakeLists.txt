cmake_minimum_required(VERSION 3.18)
project(consumer_demo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Torch REQUIRED)
find_package(Python COMPONENTS Development REQUIRED)

add_executable(consumer consumer.cpp)
target_include_directories(consumer PRIVATE 
    ${TORCH_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
)
target_link_libraries(consumer PUBLIC  ${TORCH_LIBRARIES})

set(TORCH_RPATH "${TORCH_INSTALL_PREFIX}/lib")
message(STATUS "Torch install prefix: ${TORCH_INSTALL_PREFIX}")
message(STATUS "Torch RPATH: ${TORCH_RPATH}")

set_target_properties(consumer PROPERTIES
    BUILD_RPATH "${TORCH_RPATH}"
    INSTALL_RPATH "${TORCH_RPATH}"
)

install(TARGETS consumer RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX})