from itertools import combinations
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np

N = 100
total = N * (N - 1) * (N - 2) // 6  # 161700

# cubin
with open('kernel.cubin', 'rb') as f:
    cubin = f.read()
mod = cuda.module_from_buffer(cubin)

# ptx
# with open('kernel.ptx', 'r') as f:
#     ptx = f.read()
# mod = cuda.module_from_buffer(ptx.encode())

generate_combinations = mod.get_function("generate_combinations")

# 编译 CUDA kernel
generate_combinations = mod.get_function("generate_combinations")

# 分配 GPU 输出内存（int32）
out_gpu = cuda.mem_alloc(total * 3 * np.int32().nbytes)

# 启动 kernel（161700 个线程）
block_size = 256
grid_size = (total + block_size - 1) // block_size
generate_combinations(out_gpu  , block=(block_size, 1, 1), grid=(grid_size, 1))

# 拷贝回 CPU
out_host = np.empty((total, 3), dtype=np.int32)
cuda.memcpy_dtoh(out_host, out_gpu)

# CPU 参考输出
cpu_combs = np.array(list(combinations(range(100), 3)), dtype=np.int32)


gpu_set = set(map(tuple, out_host))
cpu_set = set(map(tuple, cpu_combs))

if gpu_set == cpu_set:
    print("✅ 所有组合唯一且完整。")
else:
    print("❌ 组合缺失或重复")
    print("GPU 多出的：", list(gpu_set - cpu_set)[:5])
    print("GPU 缺失的：", list(cpu_set - gpu_set)[:5])

