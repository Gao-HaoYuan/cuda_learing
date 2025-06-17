#include <iostream>
#include <cuda_runtime.h>

#include "minitest.hpp"

void printDeviceProperties() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cout << "No CUDA-capable device detected." << std::endl;
        return;
    }

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        std::cout << "==== GPU Device " << device << " ====" << std::endl;
        std::cout << "Name: " << prop.name << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Multi-Processor Count (SMs): " << prop.multiProcessorCount << std::endl;
        std::cout << "Clock Rate (kHz): " << prop.clockRate << std::endl;
        std::cout << "Memory Clock Rate (kHz): " << prop.memoryClockRate << std::endl;
        std::cout << "Global Memory (MB): " << prop.totalGlobalMem / (1024 * 1024) << std::endl;
        std::cout << "Shared Memory Per Block (KB): " << prop.sharedMemPerBlock / 1024 << std::endl;
        std::cout << "Max Threads Per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Max Threads Dim: [" 
                  << prop.maxThreadsDim[0] << ", "
                  << prop.maxThreadsDim[1] << ", "
                  << prop.maxThreadsDim[2] << "]" << std::endl;
        std::cout << "Max Grid Size: [" 
                  << prop.maxGridSize[0] << ", "
                  << prop.maxGridSize[1] << ", "
                  << prop.maxGridSize[2] << "]" << std::endl;
        std::cout << "Warp Size: " << prop.warpSize << std::endl;
        std::cout << "L2 Cache Size (KB): " << prop.l2CacheSize / 1024 << std::endl;
        std::cout << "Async Engine Count: " << prop.asyncEngineCount << std::endl;
        std::cout << "Concurrent Kernels: " << (prop.concurrentKernels ? "Yes" : "No") << std::endl;
        std::cout << "ECC Enabled: " << (prop.ECCEnabled ? "Yes" : "No") << std::endl;
        std::cout << std::endl;
    }
}

TEST(CUDA, Prop) {
    printDeviceProperties();
}
