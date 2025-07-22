import cupy as cp

N = 100
total = N * (N - 1) * (N - 2) // 6  # 161700

kernel_code = r'''
extern "C" __global__
void generate_combinations(int* out) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= 161700) return;

    int count = 0;
    int i = 0;

    while (idx >= ((99 - i) * (98 - i)) / 2) {
        idx -= ((99 - i) * (98 - i)) / 2;
        ++i;
    }

    int j = i + 1;
    while (idx >= (99 - j)) {
        idx -= (99 - j);
        ++j;
    }

    int k = j + 1 + idx;

    int out_idx = blockDim.x * blockIdx.x + threadIdx.x;
    out[3 * out_idx + 0] = i;
    out[3 * out_idx + 1] = j;
    out[3 * out_idx + 2] = k;
}
'''

module = cp.RawModule(code=kernel_code)
generate = module.get_function("generate_combinations")

# 分配输出空间
out_gpu = cp.empty((total, 3), dtype=cp.int32)

# 启动 kernel
block_size = 256
grid_size = (total + block_size - 1) // block_size
generate((grid_size,), (block_size,), (out_gpu,))

# 验证
print("前 10 个组合：")
print(out_gpu[:10].get())
