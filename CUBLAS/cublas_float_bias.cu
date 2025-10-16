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

template<typename T>
void host_fill(std::vector<T>& v, float lo=-128.f, float hi=128.f, unsigned seed=123){
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo,hi);
    for(auto& x : v) x = static_cast<T>(dist(rng));
}

// 列主：C(i,j) += bias(j)  （按列广播）
__global__ void add_bias_colmajor(float* __restrict__ C, const float* __restrict__ bias,
                                  int M, int N, int ldc){
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 列索引 j
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 行索引 i
    if(col < N && row < M){
        C[row + col * ldc] += bias[col];
    }
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

void check_close_f32(const float* d_ref, const float* d_out, size_t n,
                     double atol=1e-5, double rtol=1e-5, double l2tol=1e-6, double tiny=1e-6,
                     const char* tag="check"){
    std::vector<float> r(n), o(n);
    CHECK_CUDA(cudaMemcpy(r.data(), d_ref, n*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(o.data(), d_out, n*sizeof(float), cudaMemcpyDeviceToHost));
    double max_abs=0, max_rel=0, num=0, den=0;
    size_t rel_cnt=0;
    for(size_t i=0;i<n;++i){
        double R=r[i], O=o[i], d=fabs(O-R);
        if(d>max_abs) max_abs=d;
        if(fabs(R)>tiny){ double rel=d/fabs(R); if(rel>max_rel) max_rel=rel; rel_cnt++; }
        num+=(O-R)*(O-R); den+=R*R;
    }
    double rel_l2 = std::sqrt(num)/(std::sqrt(den)+1e-30);
    bool ok = (max_abs<=atol) || ((rel_cnt==0 || max_rel<=rtol) && rel_l2<=l2tol);
    printf("[%s] max_abs=%.3e  max_rel=%.3e  rel_L2=%.3e  => %s\n",
           tag, max_abs, max_rel, rel_l2, ok?"OK":"MISMATCH");
}

int main(int argc, char** argv){
    bool use_tf32 = (argc>1 && std::strcmp(argv[1],"--tf32")==0);
    printf("Mode: %s\n", use_tf32 ? "TF32 (Tensor Core)" : "Pure FP32");

    // 尺寸（列主）；N 是 bias 长度（按列广播）。可自行改成你的常用形状
    int M=2048, N=2048, K=1024;
    int lda=M, ldb=K, ldc=M, ldd=M;
    size_t sizeA=(size_t)M*K, sizeB=(size_t)K*N, sizeC=(size_t)M*N;

    printf("GEMM + Bias (col-major): D = A@B + bias(按列)\n");
    printf("M=%d N=%d K=%d\n", M,N,K);

    // Host 数据
    std::vector<float> hA(sizeA), hB(sizeB), hBias(N);
    host_fill(hA); host_fill(hB);
    host_fill(hBias, -0.1f, 0.1f); // bias 幅度小一点

    // Device 缓冲
    float *A=nullptr,*B=nullptr,*Bias=nullptr,*C_base=nullptr,*D_lt=nullptr;
    CHECK_CUDA(cudaMalloc(&A, sizeof(float)*sizeA));
    CHECK_CUDA(cudaMalloc(&B, sizeof(float)*sizeB));
    CHECK_CUDA(cudaMalloc(&Bias, sizeof(float)*N));
    CHECK_CUDA(cudaMalloc(&C_base, sizeof(float)*sizeC)); // baseline 输出
    CHECK_CUDA(cudaMalloc(&D_lt,   sizeof(float)*sizeC)); // Lt 输出
    CHECK_CUDA(cudaMemcpy(A, hA.data(), sizeof(float)*sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(B, hB.data(), sizeof(float)*sizeB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(Bias, hBias.data(), sizeof(float)*N, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(C_base, 0, sizeof(float)*sizeC));
    CHECK_CUDA(cudaMemset(D_lt,   0, sizeof(float)*sizeC));

    // 句柄
    cublasHandle_t blas;     CHECK_CUBLAS(cublasCreate(&blas));
    cublasLtHandle_t lt;     CHECK_CUBLAS(cublasLtCreate(&lt));
    if(use_tf32){
        CHECK_CUBLAS(cublasSetMathMode(blas, CUBLAS_TF32_TENSOR_OP_MATH));
    }else{
        CHECK_CUBLAS(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH));
    }

    float alpha=1.f, beta=0.f;

    // ===== Baseline：cublasSgemm + add_bias kernel =====
    auto run_baseline = [&](){
        // 1) GEMM
        CHECK_CUBLAS(cublasSgemm(
            blas, CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            A, lda,
            B, ldb,
            &beta,
            C_base, ldc));
        // 2) Bias add：按列广播
        dim3 block(32, 8);
        dim3 grid((N + block.x - 1)/block.x, (M + block.y - 1)/block.y);
        add_bias_colmajor<<<grid, block>>>(C_base, Bias, M, N, ldc);
    };
    float t_base = time_ms(run_baseline);
    double tflops_base = (2.0 * (double)M * N * K) / (t_base * 1e-3) / 1e12; // 只按 GEMM FLOPs 粗略算
    printf("[Baseline  GEMM+Bias]  %.3f ms  | GEMM-only %.2f TFLOP/s (不含Bias算术)\n", t_base, tflops_base);

    // ===== cuBLASLt：融合 Bias 的 matmul =====
    cublasLtMatmulDesc_t opDesc;
    if(use_tf32){
        CHECK_CUBLAS(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));
    }else{
        CHECK_CUBLAS(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    }
    cublasOperation_t ta=CUBLAS_OP_N, tb=CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));

    // 设置 epilogue = BIAS，并传入 bias 指针（按列广播）
    cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &Bias, sizeof(Bias)));

    // 各布局（列主，FP32）
    cublasLtMatrixLayout_t aDesc,bDesc,cDesc,dDesc;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&aDesc, CUDA_R_32F, M, K, lda));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&bDesc, CUDA_R_32F, K, N, ldb));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&cDesc, CUDA_R_32F, M, N, ldc));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&dDesc, CUDA_R_32F, M, N, ldd));

    auto run_lt = [&](){
        CHECK_CUBLAS(cublasLtMatmul(
            lt, opDesc,
            &alpha,
            A, aDesc,
            B, bDesc,
            &beta,
            D_lt, cDesc,   // C (beta=0)
            D_lt, dDesc,   // 输出
            nullptr, nullptr, 0, 0));
    };
    float t_lt = time_ms(run_lt);
    double tflops_lt = (2.0 * (double)M * N * K) / (t_lt * 1e-3) / 1e12; // 同样按 GEMM FLOPs 估算
    printf("[cuBLASLt GEMM+Bias]  %.3f ms  | GEMM-only %.2f TFLOP/s (融合Bias)\n", t_lt, tflops_lt);

    // ===== 精度对齐（两边都是 FP32 输出，且做了相同 bias）=====
    double atol = use_tf32 ? 1e-4 : 1e-5;
    double rtol = use_tf32 ? 1e-3 : 1e-5;
    check_close_f32(C_base, D_lt, (size_t)M*N, atol, rtol,
                    use_tf32 ? 1e-6 : 1e-6, 1e-6,
                    use_tf32 ? "FP32+Bias(TF32)" : "FP32+Bias");

    // 清理
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(aDesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(bDesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(cDesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(dDesc));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(opDesc));
    CHECK_CUBLAS(cublasLtDestroy(lt));
    CHECK_CUBLAS(cublasDestroy(blas));
    CHECK_CUDA(cudaFree(A));
    CHECK_CUDA(cudaFree(B));
    CHECK_CUDA(cudaFree(Bias));
    CHECK_CUDA(cudaFree(C_base));
    CHECK_CUDA(cudaFree(D_lt));
    return 0;
}
