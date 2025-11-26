import torch
import cublaslt_gemm

# 维度
M, K, N = 128, 512, 256

# 生成随机 CUDA Tensor
A = torch.rand(M, K, device='cuda', dtype=torch.float32)
B = torch.rand(K, N, device='cuda', dtype=torch.float32)
bias = torch.rand(N, device='cuda', dtype=torch.float32)

# 1️⃣ PyTorch matmul + bias
D_ref = torch.matmul(A, B) + bias

# 2️⃣ 调用 cuBLASLt GEMM
D_gemm = cublaslt_gemm.register_gemm_float(A, B, bias=bias)

# 3️⃣ 精度对比
max_abs_error = (D_ref - D_gemm).abs().max().item()
mean_abs_error = (D_ref - D_gemm).abs().mean().item()
rmse = ((D_ref - D_gemm)**2).mean().sqrt().item()

print(f"Shape: {D_gemm.shape}")
print(f"Max absolute error: {max_abs_error:.6e}")
print(f"Mean absolute error: {mean_abs_error:.6e}")
print(f"RMSE: {rmse:.6e}")

# 可选：相对误差
rel_error = ((D_ref - D_gemm).abs() / (D_ref.abs() + 1e-6)).max().item()
print(f"Max relative error: {rel_error:.6e}")
