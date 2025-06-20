#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cublasLt.h>
#include <cuda_runtime.h>

#include "perf.hpp"
#include "minitest.hpp"

void cpu_gemm(float* A, float* B, float* C, int M, int N, int K) {
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
                sum += A[m * K + k] * B[k * N + n];
            C[m * N + n] = sum;
        }
}

TEST(CUDA, LTBlas){
    const int M = 128, N = 256, K = 64;
    const float alpha = 1.0f, beta = 0.0f;

    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);

    // 分配主机内存
    float *hA = (float*)malloc(bytesA);
    float *hB = (float*)malloc(bytesB);
    float *hC = (float*)malloc(bytesC);
    float *hC_ref = (float*)malloc(bytesC);

    // 初始化矩阵，行主存储
    for (int i = 0; i < M * K; ++i) hA[i] = (float)(i % 13) / 13.0f;
    for (int i = 0; i < K * N; ++i) hB[i] = (float)(i % 7) / 7.0f;

    // 分配设备内存
    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB, bytesB));
    CHECK_CUDA(cudaMalloc(&dC, bytesC));

    // 复制数据到设备
    CHECK_CUDA(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));

    // 创建 cuBLASLt handle
    cublasLtHandle_t ltHandle;
    CHECK_CUBLAS(cublasLtCreate(&ltHandle));

    // 创建描述符
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;

    CHECK_CUBLAS(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, M, K, K));  // ld = K 行主
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, K, N, N));  // ld = N 行主
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, M, N, N));  // ld = N 行主

    // 设置矩阵顺序为行主
    int order_row = CUBLASLT_ORDER_ROW;
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_row, sizeof(order_row)));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_row, sizeof(order_row)));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_row, sizeof(order_row)));

    // 创建算法偏好
    cublasLtMatmulPreference_t preference = NULL;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));

    size_t workspaceSize = 4 * 1024 * 1024;
    void* dWorkspace;
    CHECK_CUDA(cudaMalloc(&dWorkspace, workspaceSize));

    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspaceSize, sizeof(workspaceSize)));

    // 获取算法
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    int returnedResults = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
        ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc,
        preference, 1, &heuristicResult, &returnedResults));

    if (returnedResults == 0) {
        fprintf(stderr, "No suitable cuBLASLt algorithm found.\n");
        exit(EXIT_FAILURE);
    }

    // 执行矩阵乘法
    CHECK_CUBLAS(cublasLtMatmul(
        ltHandle, operationDesc,
        &alpha, dA, Adesc,
        dB, Bdesc,
        &beta, dC, Cdesc,
        dC, Cdesc,
        &heuristicResult.algo,
        dWorkspace, workspaceSize, 0));

    // 拷贝结果回主机
    CHECK_CUDA(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost));

    // CPU 计算验证
    cpu_gemm(hA, hB, hC_ref, M, N, K);

    // 比较结果
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        float diff = fabs(hC[i] - hC_ref[i]);
        if (diff > 1e-3) {
            printf("Mismatch at %d: GPU %f vs CPU %f\n", i, hC[i], hC_ref[i]);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "✅ PASS: cuBLASLt GEMM is correct" : "❌ FAIL");

    // 释放资源
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtDestroy(ltHandle);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dWorkspace);
    free(hA);
    free(hB);
    free(hC);
    free(hC_ref);
}
