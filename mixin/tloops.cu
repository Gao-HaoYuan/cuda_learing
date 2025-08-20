#include <cuda_runtime.h>
#include <iostream>

// 简单张量数据结构，存储二维张量数据（用一维数组模拟）
template<typename T>
struct Tensor2D {
    T* data;
    int rows, cols;

    __host__ __device__
    T& operator()(int i, int j) {
        return data[i * cols + j];
    }

    __host__ __device__
    const T& operator()(int i, int j) const {
        return data[i * cols + j];
    }
};

// 运算符标签，代表操作符
struct AddOp {
    __host__ __device__
    static float apply(float a, float b) { return a + b; }
};

struct MulOp {
    __host__ __device__
    static float apply(float a, float b) { return a * b; }
};

// 表达式模板基类
template<typename E>
struct Expr {
    __host__ __device__
    float operator()(int i, int j) const {
        // static_cast 到派生类
        return static_cast<const E&>(*this)(i,j);
    }
    __host__ __device__
    int rows() const { return static_cast<const E&>(*this).rows(); }
    __host__ __device__
    int cols() const { return static_cast<const E&>(*this).cols(); }
};

// 张量表达式包装
template<typename T>
struct TensorExpr : public Expr<TensorExpr<T>> {
    const Tensor2D<T>& t;

    TensorExpr(const Tensor2D<T>& t_) : t(t_) {}

    __host__ __device__
    float operator()(int i, int j) const { return t(i,j); }
    __host__ __device__
    int rows() const { return t.rows; }
    __host__ __device__
    int cols() const { return t.cols; }
};

// 二元运算表达式模板
template<typename LHS, typename RHS, typename Op>
struct BinaryExpr : public Expr<BinaryExpr<LHS,RHS,Op>> {
    const LHS& lhs;
    const RHS& rhs;

    BinaryExpr(const LHS& l, const RHS& r) : lhs(l), rhs(r) {}

    __host__ __device__
    float operator()(int i, int j) const {
        return Op::apply(lhs(i,j), rhs(i,j));
    }
    __host__ __device__
    int rows() const { return lhs.rows(); }
    __host__ __device__
    int cols() const { return lhs.cols(); }
};

// 操作符重载，构造表达式模板树
template<typename LHS, typename RHS>
BinaryExpr<LHS, RHS, AddOp> operator+(const Expr<LHS>& l, const Expr<RHS>& r) {
    return BinaryExpr<LHS, RHS, AddOp>(static_cast<const LHS&>(l), static_cast<const RHS&>(r));
}

template<typename LHS, typename RHS>
BinaryExpr<LHS, RHS, MulOp> operator*(const Expr<LHS>& l, const Expr<RHS>& r) {
    return BinaryExpr<LHS, RHS, MulOp>(static_cast<const LHS&>(l), static_cast<const RHS&>(r));
}

// CUDA kernel 计算张量表达式
template<typename ExprT>
__global__ void tensor_kernel(float* out, ExprT expr, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;

    int i = idx / cols;
    int j = idx % cols;

    out[idx] = expr(i,j);
}

int main() {
    int rows = 2, cols = 3;
    int size = rows * cols;

    float h_A[] = {1,2,3,4,5,6};
    float h_B[] = {10,20,30,40,50,60};
    float* d_A; float* d_B; float* d_out;
    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));
    cudaMalloc(&d_out, size * sizeof(float));
    cudaMemcpy(d_A, h_A, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size * sizeof(float), cudaMemcpyHostToDevice);

    Tensor2D<float> A{d_A, rows, cols};
    Tensor2D<float> B{d_B, rows, cols};

    auto expr = TensorExpr<float>(A) + TensorExpr<float>(B) * TensorExpr<float>(A);

    // 计算
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    tensor_kernel<<<blocks, threads>>>(d_out, expr, rows, cols);

    float h_out[size];
    cudaMemcpy(h_out, d_out, size * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Result:\n";
    for (int i=0; i < rows; i++) {
        for (int j=0; j < cols; j++) {
            std::cout << h_out[i*cols+j] << " ";
        }
        std::cout << "\n";
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_out);
    return 0;
}
