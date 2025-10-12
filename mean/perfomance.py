# test_mean.py
import torch
import mean_hd
import time

def benchmark(func, *args, warmup=5, repeat=50):
    # GPU 测试必须同步，否则 time 不准确
    for _ in range(warmup):
        _ = func(*args)
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        _ = func(*args)
        torch.cuda.synchronize()
    t1 = time.time()
    return (t1 - t0) / repeat * 1e3  # 毫秒


def main():
    torch.manual_seed(0)
    device = 'cuda'
    x = torch.randn(512, 256, 64, 32, device=device, dtype=torch.float32)  # 稍大些看差距

    # ---------------------------
    # 功能正确性验证
    # ---------------------------
    y1 = mean_hd.mean_highdim_cuda(x, dim=(2,))
    ref1 = x.mean(dim=2)
    print("dim=(2,) allclose:", torch.allclose(y1, ref1, atol=1e-6))

    y2 = mean_hd.mean_highdim_cuda(x, (1,3))
    ref2 = x.mean(dim=(1,3))
    print("dim=(1,3) allclose:", torch.allclose(y2, ref2, atol=1e-6))

    y3 = mean_hd.mean_highdim_cuda(x)
    ref3 = x.mean()
    print("dim=None allclose:", torch.allclose(y3, ref3, atol=1e-6))

    # ---------------------------
    # 性能测试
    # ---------------------------
    print("\n=== Performance Benchmark ===")
    for dims in [(2,), (1,3), None]:
        # 自定义 kernel
        t_custom = benchmark(mean_hd.mean_highdim_cuda, x, dims)
        # PyTorch 原生 mean
        if dims is None:
            t_torch = benchmark(torch.mean, x)
        else:
            t_torch = benchmark(torch.mean, x, dims)

        print(f"dim={dims}: custom={t_custom:.3f} ms, torch={t_torch:.3f} ms, speedup={t_torch / t_custom:.2f}x")

if __name__ == "__main__":
    main()
