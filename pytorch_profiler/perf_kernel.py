import math
from typing import Callable

import torch
import torch.profiler
from torch.profiler import ProfilerActivity

def perf_profile(operator_fn, *args, **kwargs):
    # 同步 cuda 行为, 不执行同步 cuda 统计会出错，比如内存峰值
    def sync_op():
        torch.cuda.synchronize()
        operator_fn(*args, **kwargs)
        torch.cuda.synchronize()

    with torch.profiler.profile(
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule = torch.profiler.schedule(skip_first=3, wait=1, warmup=1, active=5, repeat=1),
        on_trace_ready = torch.profiler.tensorboard_trace_handler('./log/perf'),
        record_shapes = True,
        with_stack = True,
        profile_memory = True
    ) as prof:
        for _ in range(10):
            sync_op()
            prof.step()

    print(prof.key_averages().table(
        sort_by="cpu_time_total",  # cuda_memory_usage, self_cuda_memory_usage
        row_limit=10,
        max_name_column_width=30,   # 控制函数名最大列宽
        top_level_events_only=False,  # False: 显示所有事件， True: 限定顶层
        header = "CUDA Time and Memory Usage"
    ))

    torch.cuda.reset_peak_memory_stats()
    sync_op()
    peak_cuda_mem_MB = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"Peak GPU Memory Usage: {peak_cuda_mem_MB:.3f} MB\n")


def perf_accuracy(expected, compared, threshold=1.0):
    def compare(x, y):
        info = ""
        if torch.all(x == 0) and torch.all(y == 0):
            err = torch.tensor(0.0)
            corr = torch.tensor(1.0)
            info = "Note: Both tensors are all zeros."
            return err, corr, info

        err = torch.mean(torch.abs(x - y)) / torch.mean(torch.abs(x))
        corr = torch.cosine_similarity(x.contiguous().view(-1), y.contiguous().view(-1), dim=-1)
        return err, corr, info

    expected = expected if isinstance(expected, (list, tuple)) else (expected,)
    compared = compared if isinstance(compared, (list, tuple)) else (compared,)

    for i, (a, b) in enumerate(zip(expected, compared)):
        err, corr, info = compare(a, b)
        print(f"Output {i}: relative_error_mean {err:.6f}, corr: {corr:.6f}. {info}")
        assert math.isclose(corr.item(), threshold, rel_tol=1e-6)


def measure_step_memory(fn: Callable, desc="", device_id=0):
    """
    执行给定的函数，测量指定GPU设备执行时的显存使用情况。
    
    参数:
        fn (Callable): 要执行的函数或 lambda 表达式
        desc (str): 描述
        device_id (int): GPU设备号，默认0
    """
    torch.cuda.synchronize(device_id)
    fn()
    torch.cuda.synchronize(device_id)

    current = torch.cuda.memory_allocated(device=device_id) / 1024**2
    peak = torch.cuda.max_memory_allocated(device=device_id) / 1024**2
    print(f"[{desc:<12}][GPU{device_id}] Current: {current:.2f} MB; Peak: {peak:.2f} MB")


def print_tensor_info(tensor, name="tensor"):
    if not isinstance(tensor, torch.Tensor):
        print(f"[{name}] is not a torch.Tensor (type: {type(tensor)})\n")
        return

    print(f"===== {name} =====")
    print(f"ptr            : {tensor.data_ptr()}")
    print(f"Type           : {type(tensor)}")
    print(f"Shape          : {tensor.shape}")
    print(f"Dtype          : {tensor.dtype}")
    print(f"Device         : {tensor.device}")
    print(f"stride         : {tensor.stride()}")
    print(f"Requires Grad  : {tensor.requires_grad}")
    print(f"storage_offset : {tensor.storage_offset()}")
    print(f"Is Leaf        : {tensor.is_leaf}")
    print(f"Is contiguous  : {tensor.is_contiguous()}")
    print(f"Data (first 5) : {tensor.flatten()[:5].tolist()}")
    print("====================\n")