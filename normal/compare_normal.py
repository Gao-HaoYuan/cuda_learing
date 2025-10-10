# compare_normal.py
import math
import time
import torch

# 如果你是其他模块名，改这里
import AFP  

def moments(x):
    mean = x.mean()
    var = x.var(unbiased=False)
    std = var.sqrt()
    # 偏度/峰度（Fisher, excess kurtosis）
    z = (x - mean) / (std + 1e-12)
    skew = (z**3).mean()
    kurt = (z**4).mean() - 3.0
    return float(mean), float(std), float(skew), float(kurt)

@torch.inference_mode()
def hist_chi2(a, b, bins=200, range=(-6, 6)):
    # 直方图卡方距离（越小越接近）
    ha = torch.histc(a, bins=bins, min=range[0], max=range[1])
    hb = torch.histc(b, bins=bins, min=range[0], max=range[1])
    # 归一化
    ha = ha / ha.sum().clamp_min(1)
    hb = hb / hb.sum().clamp_min(1)
    num = (ha - hb) ** 2
    den = (hb + 1e-12)
    return float((num / den).sum())

@torch.inference_mode()
def ks_distance(x):
    # 与N(0,1)的KS距离（简化版）
    xs = torch.sort(x).values
    n = xs.numel()
    # 理论CDF
    def normal_cdf(t):
        return 0.5 * (1.0 + torch.erf(t / math.sqrt(2.0)))
    F_emp = torch.arange(1, n + 1, device=xs.device, dtype=xs.dtype) / n
    F_the = normal_cdf(xs)
    return float((F_emp - F_the).abs().max())

@torch.inference_mode()
def timed(func, warmup=3, iters=10):
    # 用 CUDA 事件计时
    for _ in range(warmup):
        func()
        torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    ms = 0.0
    for _ in range(iters):
        start.record()
        func()
        end.record()
        torch.cuda.synchronize()
        ms += start.elapsed_time(end)
    return ms / iters

def main(
    N=4_000_000,  # 样本数
    mean=0.0,
    std=1.0,
    seed=1234,
    plot=True,   # 如需图形，改为 True（需要 matplotlib）
):
    device = "cuda"
    dtype = torch.float32

    # --- 生成样本 ---
    torch.cuda.manual_seed(seed)

    # 用 torch.normal（或 randn_like 再仿射也行）
    x_torch = torch.normal(mean=torch.tensor(mean, device=device, dtype=dtype),
                           std=torch.tensor(std, device=device, dtype=dtype),
                           size=(N,), dtype=dtype, device=device)

    # cuRAND 扩展
    x_curand = torch.empty_like(x_torch)
    AFP.curand_normal_(x_curand, seed=seed, mean=mean, std=std)

    # --- 复现性检验（同 seed 是否一致） ---
    x_curand2 = torch.empty_like(x_curand)
    AFP.curand_normal_(x_curand2, seed=seed, mean=mean, std=std)
    reproducible = torch.allclose(x_curand, x_curand2)

    # --- 统计 ---
    m_t, s_t, skew_t, kurt_t = moments(x_torch)
    m_c, s_c, skew_c, kurt_c = moments(x_curand)

    chi2 = hist_chi2(x_torch, x_curand)
    # 与标准正态的KS距离（两者各自）
    ks_t = ks_distance((x_torch - mean) / (std + 1e-12))
    ks_c = ks_distance((x_curand - mean) / (std + 1e-12))

    # 两个样本的相关性（应该接近 0，若用同种子同序列映射也可能偏大）
    corr = torch.corrcoef(torch.stack([x_torch, x_curand]))[0, 1].item()

    # # --- 性能 ---
    # # 为避免额外分配，把生成函数写成闭包复用张量
    # def gen_torch():
    #     torch.normal(mean=torch.tensor(mean, device=device, dtype=dtype),
    #                  std=torch.tensor(std, device=device, dtype=dtype),
    #                  out=x_torch)

    # def gen_curand():
    #     AFP.curand_normal_(x_curand, seed=seed, mean=mean, std=std)

    # t_torch = timed(gen_torch)
    # t_curand = timed(gen_curand)

    # print("=== Distribution stats ===")
    # print(f"Torch : mean={m_t:.5f}, std={s_t:.5f}, skew={skew_t:.5f}, kurt={kurt_t:.5f}, KS={ks_t:.5f}")
    # print(f"cuRAND: mean={m_c:.5f}, std={s_c:.5f}, skew={skew_c:.5f}, kurt={kurt_c:.5f}, KS={ks_c:.5f}")
    # print(f"Chi2 distance (torch vs cuRAND hist): {chi2:.5f}")
    # print(f"Corr(x_torch, x_curand): {corr:.5f}")
    # print(f"Reproducible (cuRAND, same seed): {reproducible}")

    # print("\n=== Performance (ms per call) ===")
    # print(f"Torch.normal : {t_torch:.3f} ms")
    # print(f"AFP.curand   : {t_curand:.3f} ms")
    # print(f"Speedup (curand / torch): {t_curand / t_torch:.3f}x")

    if plot:
        import math
        import numpy as np
        import matplotlib
        # 无显示环境请启用无界面后端（放在 import pyplot 之前）
        matplotlib.use("Agg")  # 如果本机能弹窗，也可以删掉这一行
        import matplotlib.pyplot as plt

        # ---- 准备数据：展平 + 过滤 NaN/Inf ----
        xt = x_torch.detach().contiguous().view(-1).float().cpu().numpy()
        xc = x_curand.detach().contiguous().view(-1).float().cpu().numpy()

        xt = xt[np.isfinite(xt)]
        xc = xc[np.isfinite(xc)]

        print(f"[plot] xt size={xt.size}, xc size={xc.size}, xt mean/std={xt.mean():.4f}/{xt.std():.4f}, xc mean/std={xc.mean():.4f}/{xc.std():.4f}")

        # 若数组为空，直接跳过
        if xt.size == 0 or xc.size == 0:
            print("[plot] empty arrays after filtering; skip plotting")
        else:
            # ---- 直方图：固定范围更直观 ----
            bins = 120
            hist_range = (-6, 6)

            plt.figure(figsize=(6,4))
            plt.hist(xt, bins=bins, range=hist_range, density=True, alpha=0.6, label="torch")
            plt.hist(xc, bins=bins, range=hist_range, density=True, alpha=0.6, label="curand")
            plt.legend()
            plt.title("Histogram: torch vs cuRAND")
            plt.xlabel("x"); plt.ylabel("pdf")
            plt.tight_layout()
            plt.savefig("histogram_torch_vs_curand.png", dpi=150)
            print("[plot] saved histogram to histogram_torch_vs_curand.png")

            # ---- Q-Q 图：相对标准正态 ----
            # 标准化
            xs = ((xt - xt.mean()) / (xt.std() + 1e-12))
            cs = ((xc - xc.mean()) / (xc.std() + 1e-12))

            # 排序
            xs = np.sort(xs)
            cs = np.sort(cs)
            n  = min(xs.size, cs.size)
            if n == 0:
                print("[plot] empty arrays for QQ; skip")
            else:
                xs = xs[:n]
                cs = cs[:n]

                # 经验分位概率（避免 0 和 1）
                q = np.linspace(1.0/(n+1.0), n/(n+1.0), n, dtype=np.float64)
                # 理论分位数：Phi^{-1}(q) = sqrt(2)*erfinv(2q-1)
                theo = math.sqrt(2.0) * torch.erfinv(torch.from_numpy(2.0*q - 1.0)).double().cpu().numpy()

                step = max(1, n // 5000)  # 下采样，避免太密
                plt.figure(figsize=(6,4))
                plt.scatter(theo[::step], xs[::step], s=1, label="torch")
                plt.scatter(theo[::step], cs[::step], s=1, label="curand")
                plt.legend()
                plt.title("Q-Q vs N(0,1)")
                plt.xlabel("Theoretical quantiles"); plt.ylabel("Empirical quantiles")
                plt.tight_layout()
                plt.savefig("qqplot_torch_vs_curand.png", dpi=150)
                print("[plot] saved Q-Q plot to qqplot_torch_vs_curand.png")



if __name__ == "__main__":
    torch.cuda.init()
    main()
