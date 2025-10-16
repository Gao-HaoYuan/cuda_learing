#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <functional>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#define CHECK_CUDA(x) do{auto e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
#define CHECK_CUBLAS(x) do{auto s=(x); if(s!=CUBLAS_STATUS_SUCCESS){ \
  fprintf(stderr,"cuBLAS error %s:%d: %d\n",__FILE__,__LINE__,(int)s); exit(1);} }while(0)

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

template<typename T>
void host_fill(std::vector<T>& v, float lo=-1.f, float hi=1.f, unsigned seed=123){
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo,hi);
    for(auto& x : v) x = static_cast<T>(dist(rng));
}

void check_close_f32(const float* d_ref, const float* d_out, size_t n,
                     double atol, double rtol, const char* tag){
    std::vector<float> r(n), o(n);
    CHECK_CUDA(cudaMemcpy(r.data(), d_ref, n*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(o.data(), d_out, n*sizeof(float), cudaMemcpyDeviceToHost));
    double max_abs=0, max_rel=0, num=0, den=0;
    for(size_t i=0;i<n;++i){
        double R=r[i], O=o[i], d=fabs(O-R);
        max_abs = std::max(max_abs, d);
        double denom = fabs(R)>0 ? fabs(R) : 1.0;
        max_rel = std::max(max_rel, d/denom);
        num += (O-R)*(O-R); den += R*R;
    }
    double rel_l2 = sqrt(num)/(sqrt(den)+1e-30);
    bool ok = (max_abs<=atol) || (max_rel<=rtol);
    printf("[%s] max_abs=%.3e  max_rel=%.3e  rel_L2=%.3e  => %s\n",
           tag, max_abs, max_rel, rel_l2, ok?"OK":"MISMATCH");
}

int main(int argc, char** argv){
    bool use_tf32 = (argc>1 && std::strcmp(argv[1],"--tf32")==0);
    printf("Mode: %s\n", use_tf32 ? "TF32 (Tensor Core for FP32)" : "Pure FP32");

    // 选一组中等偏大的尺寸（列主）。可自行修改为你的常用形状
    int M=4096, N=7546, K=1896;
    int lda=M, ldb=K, ldc=M;
    size_t sizeA=(size_t)M*K, sizeB=(size_t)K*N, sizeC=(size_t)M*N;
    printf("GEMM (col-major): C[MxN] = A[MxK] * B[KxN]\nM=%d N=%d K=%d\n", M,N,K);

    // Host init (FP32)
    std::vector<float> hA(sizeA), hB(sizeB);
    host_fill(hA); host_fill(hB);

    // Device buffers
    float *dA=nullptr, *dB=nullptr, *dC_cublas=nullptr, *dD_lt=nullptr;
    CHECK_CUDA(cudaMalloc(&dA, sizeof(float)*sizeA));
    CHECK_CUDA(cudaMalloc(&dB, sizeof(float)*sizeB));
    CHECK_CUDA(cudaMalloc(&dC_cublas, sizeof(float)*sizeC));
    CHECK_CUDA(cudaMalloc(&dD_lt, sizeof(float)*sizeC));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), sizeof(float)*sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), sizeof(float)*sizeB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC_cublas, 0, sizeof(float)*sizeC));
    CHECK_CUDA(cudaMemset(dD_lt, 0, sizeof(float)*sizeC));

    // Handles
    cublasHandle_t h; CHECK_CUBLAS(cublasCreate(&h));
    cublasLtHandle_t lt; CHECK_CUBLAS(cublasLtCreate(&lt));

    // ========= cuBLAS: FP32 =========
    if(use_tf32){
        CHECK_CUBLAS(cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH)); // TF32 on Tensor Core
    }else{
        CHECK_CUBLAS(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));        // Pure FP32
    }

    float alpha=1.f, beta=0.f;
    auto run_cublas = [&](){
        // Sgemm 纯 FP32 路径（当 TF32 打开时，cuBLAS 也可能内部走 TF32/TC 快速路径）
        CHECK_CUBLAS(cublasSgemm(
            h, CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            dA, lda,
            dB, ldb,
            &beta,
            dC_cublas, ldc));
    };

    float t_blas = time_ms(run_cublas);
    double tflops_blas = (2.0 * (double)M * N * K) / (t_blas * 1e-3) / 1e12;
    printf("[cuBLAS ]  %.3f ms  | %.2f TFLOP/s\n", t_blas, tflops_blas);

    // ========= cuBLASLt: FP32 =========
    cublasLtMatmulDesc_t opDesc;
    if(use_tf32){
        // TF32（FP32 输入，Tensor Core 路径）
        CHECK_CUBLAS(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));
    }else{
        // 纯 FP32
        CHECK_CUBLAS(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    }
    cublasOperation_t ta=CUBLAS_OP_N, tb=CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));

    cublasLtMatrixLayout_t aDesc,bDesc,cDesc,dDesc;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&aDesc, CUDA_R_32F, M, K, lda));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&bDesc, CUDA_R_32F, K, N, ldb));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&cDesc, CUDA_R_32F, M, N, ldc));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&dDesc, CUDA_R_32F, M, N, ldc));

    
    //  ========= preference + heuristic ========= 
    cublasLtMatmulPreference_t pref;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
    size_t workspaceSize = 64<<20; // 64MB
    void* workspace = nullptr; 
    CHECK_CUDA(cudaMalloc(&workspace, workspaceSize));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));
    cublasLtMatmulHeuristicResult_t heur;
    int returnedResults = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(lt, opDesc, aDesc, bDesc, cDesc, dDesc, pref, 1, &heur, &returnedResults));

    auto run_lt = [&](){
        CHECK_CUBLAS(cublasLtMatmul(
            lt, opDesc,
            &alpha,
            dA, aDesc,
            dB, bDesc,
            &beta,
            dD_lt, cDesc,   // C（beta=0，不使用其值），这里直接复用输出缓冲
            dD_lt, dDesc,
            &heur.algo,        // algo: 用默认即可；要更快可做 heuristic 选优
            workspace, workspaceSize, 0));
    };

    float t_lt = time_ms(run_lt);
    double tflops_lt = (2.0 * (double)M * N * K) / (t_lt * 1e-3) / 1e12;
    printf("[cuBLASLt] %.3f ms  | %.2f TFLOP/s\n", t_lt, tflops_lt);

    // ========= 精度对齐（两边都是 FP32 输出） =========
    // 纯 FP32：使用严格容差；TF32：容差稍松
    double atol = use_tf32 ? 1e-4 : 1e-6;
    double rtol = use_tf32 ? 1e-3 : 1e-5;
    check_close_f32(dC_cublas, dD_lt, sizeC, atol, rtol,
                    use_tf32 ? "cuBLAS vs cuBLASLt (TF32)" : "cuBLAS vs cuBLASLt (FP32)");

    // 清理
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(aDesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(bDesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(cDesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(dDesc));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(opDesc));
    CHECK_CUBLAS(cublasLtDestroy(lt));
    CHECK_CUBLAS(cublasDestroy(h));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC_cublas));
    CHECK_CUDA(cudaFree(dD_lt));
    return 0;
}
