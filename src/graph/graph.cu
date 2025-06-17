#include <cuda_runtime.h>
#include <iostream>

#include "minitest.hpp"

// 检查报错
#define CHECK(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "CUDA error at " << __LINE__ << ": " << cudaGetErrorString(cudaGetLastError()) << std::endl; \
        exit(1); \
    }

const int N = 256;

__global__ void initKernel(int *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        data[idx] = idx;  // 将每个元素初始化为其索引
    }
}

__global__ void squareKernel(int *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        data[idx] *= data[idx];  // 将每个元素平方
    }
}

TEST(CUDA, Graph) {
    int *d_data, *h_data;
    cudaMalloc(&d_data, N * sizeof(int));
    h_data = (int*)malloc(N * sizeof(int));

    // 创建 graph
    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);

    // 设置通用的 kernel 参数
    dim3 block(32);
    dim3 grid((N + block.x - 1) / block.x);

    // ===== 创建第一个 kernel 节点：初始化 =====
    cudaKernelNodeParams initParams = {};
    void *initArgs[] = { &d_data };
    initParams.func = (void*)initKernel;
    initParams.gridDim = grid;
    initParams.blockDim = block;
    initParams.kernelParams = initArgs;
    initParams.sharedMemBytes = 0;

    cudaGraphNode_t initNode;
    cudaGraphAddKernelNode(&initNode, graph, nullptr, 0, &initParams);

    // ===== 创建第二个 kernel 节点：平方 =====
    cudaKernelNodeParams squareParams = {};
    void *squareArgs[] = { &d_data };
    squareParams.func = (void*)squareKernel;
    squareParams.gridDim = grid;
    squareParams.blockDim = block;
    squareParams.kernelParams = squareArgs;
    squareParams.sharedMemBytes = 0;

    cudaGraphNode_t squareNode;
    // 设置依赖关系：squareNode 依赖于 initNode
    cudaGraphAddKernelNode(&squareNode, graph, &initNode, 1, &squareParams);

    // 实例化并执行图
    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);

    // 拷贝结果回 host 并打印
    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; ++i) {
        std::cout << "h_data[" << i << "] = " << h_data[i] << std::endl;
    }

    // 清理资源
    cudaFree(d_data);
    free(h_data);
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(graphExec);
    cudaStreamDestroy(stream);
}
