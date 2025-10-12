# test_mean.py
import torch
import mean_hd

torch.manual_seed(0)
device = 'cuda'
x = torch.randn(2, 3, 4, 5, device=device, dtype=torch.float32)

# 单维
y1 = mean_hd.mean_highdim_cuda(x, dim=(2,))
ref1 = x.mean(dim=2)
print("dim=(2,) allclose:", torch.allclose(y1, ref1, atol=1e-6))

# 多维
y2 = mean_hd.mean_highdim_cuda(x, (1,3))
ref2 = x.mean(dim=(1,3))
print("dim=(1,3) allclose:", torch.allclose(y2, ref2, atol=1e-6))

# 全维（返回标量）
y3 = mean_hd.mean_highdim_cuda(x)  # dim=None
ref3 = x.mean()
print("dim=None (all) close:", torch.allclose(y3, ref3, atol=1e-6))

print("shapes:", y1.shape, y2.shape, y3.shape)
