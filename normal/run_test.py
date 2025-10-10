# run_test.py
import torch
import AFP

N = 1_000_000
x = torch.empty(N, device="cuda", dtype=torch.float32)

AFP.curand_normal_(x, seed=42, mean=0.0, std=1.0)

print(f"mean={x.mean().item():.4f}, std={x.std(unbiased=False).item():.4f}")
