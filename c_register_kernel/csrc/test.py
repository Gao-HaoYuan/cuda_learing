import torch
import traceable_autograd_demo  # noqa: F401

a = torch.randn(3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
c = 3.0

out = torch.ops.myops.scale_mul(a, b, c)
out.sum().backward()

print("max|grad_a| =", (a.grad), "b =", b*c)
print("max|grad_b| =", (b.grad), "a =", a*c)
