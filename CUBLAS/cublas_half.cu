#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <functional>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#define CHECK_CUDA(x) do{auto e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
#define CHECK_CUBLAS(x) do{auto s=(x); if(s!=CUBLAS_STATUS_SUCCESS){ \
  fprintf(stderr,"cuBLAS error %s:%d: %d\n",__FILE__,__LINE__,(int)s); exit(1);} }while(0)

template<typename T>
void host_fill(std::vector<T>& v, float lo=-128.f, float hi=127.f, unsigned seed=123){
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo,hi);
    for(auto& x: v) x = static_cast<T>(dist(rng));
}

float time_ms(std::function<void()> fn, int warm=5, int iters=20){
    cudaEvent_t a,b; CHECK_CUDA(cudaEventCreate(&a)); CHECK_CUDA(cudaEventCreate(&b));
    for(int i=0;i<warm;++i) fn();
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(a));
    for(int i=0;i<iters;++i) fn();
    CHECK_CUDA(cudaEventRecord(b)); CHECK_CUDA(cudaEventSynchronize(b));
    float ms=0; CHECK_CUDA(cudaEventElapsedTime(&ms,a,b));
    CHECK_CUDA(cudaEventDestroy(a)); CHECK_CUDA(cudaEventDestroy(b));
    return ms/iters;
}

// 与 FP32 基准对比的简易误差检查（FP16 输出 vs FP32 输出）
void check_close_f16_vs_f32(const __half* d_half, const float* d_f32, size_t n,
                            double atol=3e-3, double rtol=2e-2, const char* tag="check"){
    std::vector<__half> h_half(n); std::vector<float> h_f32(n);
    CHECK_CUDA(cudaMemcpy(h_half.data(), d_half, n*sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_f32.data(),  d_f32,  n*sizeof(float),  cudaMemcpyDeviceToHost));
    double max_abs=0, max_rel=0, num=0, den=0;
    for(size_t i=0;i<n;++i){
        float o = __half2float(h_half[i]);
        float r = h_f32[i];
        double d = fabs(o-r);
        max_abs = std::max(max_abs, d);
        double denom = fabs(r)>0?fabs(r):1.0;
        max_rel = std::max(max_rel, d/denom);
        num += (o-r)*(o-r); den += r*r;
    }
    double rel_l2 = sqrt(num)/(sqrt(den)+1e-30);
    bool ok = (max_abs<=atol) || (max_rel<=rtol);
    printf("[%s] max_abs=%.3e  max_rel=%.3e  rel_L2=%.3e  => %s\n",
           tag, max_abs, max_rel, rel_l2, ok?"OK":"MISMATCH");
}

int main(){
    // --- 选择一个 Tensor Core 友好的形状（8/16 倍数） ---
    int M=2048, N=2048, K=2048; // 列主
    int lda=M, ldb=K, ldc=M;
    size_t sizeA=(size_t)M*K, sizeB=(size_t)K*N, sizeC=(size_t)M*N;
    printf("Compare: cuBLAS FP32 (TF32 OFF)  vs  cuBLASLt FP16-in (FP32 accumulate)\n");
    printf("M=%d N=%d K=%d\n", M,N,K);

    // --- Host 初始化（FP32） ---
    std::vector<float> hA(sizeA), hB(sizeB);
    host_fill(hA); host_fill(hB);

    // --- Device 内存 ---
    float *dA=nullptr, *dB=nullptr, *dC_f32=nullptr;      // cuBLAS FP32 输出
    __half *dA16=nullptr, *dB16=nullptr, *dD_f16=nullptr; // cuBLASLt FP16 输出
    CHECK_CUDA(cudaMalloc(&dA, sizeof(float)*sizeA));
    CHECK_CUDA(cudaMalloc(&dB, sizeof(float)*sizeB));
    CHECK_CUDA(cudaMalloc(&dC_f32, sizeof(float)*sizeC));
    CHECK_CUDA(cudaMalloc(&dA16, sizeof(__half)*sizeA));
    CHECK_CUDA(cudaMalloc(&dB16, sizeof(__half)*sizeB));
    CHECK_CUDA(cudaMalloc(&dD_f16, sizeof(__half)*sizeC));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), sizeof(float)*sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), sizeof(float)*sizeB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC_f32, 0, sizeof(float)*sizeC));
    CHECK_CUDA(cudaMemset(dD_f16, 0, sizeof(__half)*sizeC));

    // --- 句柄 ---
    cublasHandle_t h;   CHECK_CUBLAS(cublasCreate(&h));
    cublasLtHandle_t lt;CHECK_CUBLAS(cublasLtCreate(&lt));

    // ===================== cuBLAS：纯 FP32（关闭 TF32） =====================
    CHECK_CUBLAS(cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH)); // 明确关 TF32
    float alpha_f32 = 1.f, beta_f32 = 0.f;

    auto run_cublas_fp32 = [&](){
        // 列主：C = A * B
        CHECK_CUBLAS(cublasSgemm(
            h, CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha_f32,
            dA, lda,
            dB, ldb,
            &beta_f32,
            dC_f32, ldc));
    };
    float t_fp32 = time_ms(run_cublas_fp32);
    double tflops_fp32 = (2.0 * M * (double)N * K) / (t_fp32 * 1e-3) / 1e12;
    printf("[cuBLAS  FP32] %.3f ms | %.2f TFLOP/s (TF32 OFF)\n", t_fp32, tflops_fp32);

    // ========== 用 cublasLtMatrixTransform: FP32 -> FP16（不写自定义 kernel） ==========
    cublasLtMatrixLayout_t A32_desc, A16_desc, B32_desc, B16_desc;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&A32_desc, CUDA_R_32F, M, K, lda));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&A16_desc, CUDA_R_16F, M, K, lda));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&B32_desc, CUDA_R_32F, K, N, ldb));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&B16_desc, CUDA_R_16F, K, N, ldb));
    cublasLtMatrixTransformDesc_t tDesc;
    CHECK_CUBLAS(cublasLtMatrixTransformDescCreate(&tDesc, CUDA_R_32F));
    float one=1.f, zero=0.f;
    CHECK_CUBLAS(cublasLtMatrixTransform(lt, tDesc, &one, dA, A32_desc, &zero, nullptr, nullptr, dA16, A16_desc, 0));
    CHECK_CUBLAS(cublasLtMatrixTransform(lt, tDesc, &one, dB, B32_desc, &zero, nullptr, nullptr, dB16, B16_desc, 0));

    // ===================== cuBLASLt：FP16 输入 + FP32 accumulate =====================
    cublasLtMatmulDesc_t opDesc;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_16F, CUDA_R_16F)); // FP32 累加（Tensor Core 路径）
    cublasOperation_t ta=CUBLAS_OP_N, tb=CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));

    cublasLtMatrixLayout_t aDesc,bDesc,cDesc,dDesc;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&aDesc, CUDA_R_16F, M, K, lda));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&bDesc, CUDA_R_16F, K, N, ldb));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&cDesc, CUDA_R_16F, M, N, ldc)); // 作为 C（beta=0）
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&dDesc, CUDA_R_16F, M, N, ldc)); // 输出 FP16

    float alpha = 1.f, beta = 0.f;
    auto run_lt_f16 = [&](){
        CHECK_CUBLAS(cublasLtMatmul(
            lt, opDesc,
            &alpha,
            dA16, aDesc,
            dB16, bDesc,
            &beta,
            dD_f16, cDesc,   // C (beta=0，不使用)
            dD_f16, dDesc,   // D
            nullptr, nullptr, 0, 0));
    };
    float t_lt = time_ms(run_lt_f16);
    double tflops_lt = (2.0 * M * (double)N * K) / (t_lt * 1e-3) / 1e12;
    printf("[cuBLASLt F16] %.3f ms | %.2f TFLOP/s (FP16 in, FP32 acc, FP16 out)\n", t_lt, tflops_lt);

    // ===================== 数值一致性（FP16 out vs FP32 out） =====================
    check_close_f16_vs_f32(dD_f16, dC_f32, sizeC, /*atol*/3e-3, /*rtol*/2e-2, "F16 vs F32");

    // ===== 可选：纯 FP16 累加（更快，但精度更差） =====
    // cublasLtMatmulDescDestroy(opDesc);
    // CHECK_CUBLAS(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_16F, CUDA_R_16F));
    // （其他设置不变）再次运行 run_lt_f16，即可对比 compute_16F 的速度/误差

    // --- 清理 ---
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(aDesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(bDesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(cDesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(dDesc));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(opDesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(A32_desc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(A16_desc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(B32_desc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(B16_desc));
    CHECK_CUBLAS(cublasLtMatrixTransformDescDestroy(tDesc));
    CHECK_CUBLAS(cublasLtDestroy(lt));
    CHECK_CUBLAS(cublasDestroy(h));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC_f32));
    CHECK_CUDA(cudaFree(dA16));
    CHECK_CUDA(cudaFree(dB16));
    CHECK_CUDA(cudaFree(dD_f16));
    return 0;
}
