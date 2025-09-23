import torch
from torch.utils.cpp_extension import load

amax_ext = load(
    name="amax_axis_ext",
    sources=["/home/haoyuan.gao/Test/AFP/amax/amax.cu"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)

def amax_axis_keepdim(x: torch.Tensor, axis: int):
    if axis < 0:
        axis += x.dim()
    return amax_ext.amax_axis_keepdim(x.contiguous(), axis)

def quick_check(shape, axis, dtype=torch.float32):
    x = (torch.randn(shape, device="cuda", dtype=dtype) * 5)
    y_ref = torch.amax(x, dim=axis, keepdim=True)
    y = amax_axis_keepdim(x, axis)
    ok = torch.allclose(y, y_ref)
    print(f"shape={shape}, axis={axis}, dtype={dtype} -> match: {ok}")

    print(y)
    print(y_ref)

if __name__ == "__main__":
    torch.manual_seed(0)
    quick_check((2,3,4), 1, torch.float32)
    # quick_check((4,5,6,7), 2, torch.float32)
    # quick_check((4,5,6,7), -1, torch.float32)
    # quick_check((2,3,128,129), 1, torch.float32)
    # quick_check((2,32,1024), 1, torch.float64)
    print("done.")
