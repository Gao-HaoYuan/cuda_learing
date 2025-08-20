import math
import time
from functools import wraps
from contextlib import contextmanager
from typing import Callable

import torch
import torch.profiler
from torch.profiler import ProfilerActivity

def perf_profile(logdir, operator_fn, *args, **kwargs):
    trace_handler = torch.profiler.tensorboard_trace_handler(logdir) if logdir else None
    with torch.profiler.profile(
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule = torch.profiler.schedule(skip_first=3, wait=1, warmup=1, active=5, repeat=1),
        on_trace_ready = trace_handler if trace_handler else None,
        record_shapes = True,
        with_stack = True,
        profile_memory = True
    ) as prof:
        for _ in range(10):
            operator_fn(*args, **kwargs)
            prof.step()

    print(prof.key_averages().table(
        sort_by="cuda_time_total",  # cuda_memory_usage, self_cuda_memory_usage
        row_limit=10,
        max_name_column_width=30,   # 控制函数名最大列宽
        top_level_events_only=False,  # False: 显示所有事件， True: 限定顶层
        header = "CUDA Time and Memory Usage"
    ))


def perf_time(n_runs=10, prewarm=True):
    """装饰器：测量 CUDA 函数执行时间（毫秒），多次运行取平均"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if prewarm:
                # GPU 预热
                func(*args, **kwargs)
                torch.cuda.synchronize()

            times = []
            for _ in range(n_runs):
                start = time.time()
                out = func(*args, **kwargs)
                torch.cuda.synchronize()
                end = time.time()
                times.append((end - start) * 1000)

            avg_time = sum(times) / n_runs
            print(f"[{func.__name__}] Average CUDA execution time over {n_runs} runs: {avg_time:.3f} ms")
            return out
        return wrapper
    return decorator


def perf_accuracy(expected, compared, threshold = 1.0):
    """
    Compare two tensors for accuracy.
    Reports relative mean error and cosine similarity.
    """

    x = expected.flatten()
    y = compared.flatten()

    # Special case: both are all zeros
    if not x.any() and not y.any():
        raise AssertionError("Note: Both tensors are all zeros.")

    # Relative mean absolute error
    denom = torch.mean(torch.abs(x)).clamp_min(1e-12)
    err = torch.mean(torch.abs(x - y)) / denom
    # Cosine similarity
    corr = torch.dot(x, y) / (x.norm() * y.norm() + 1e-12)
    print(f"Relative mean absolute error {err.item():.2f}, Cosine similarity: {corr.item():.2f}")

    # assert
    assert abs(corr.item() - threshold) < 1e-5, f"cosine similarity error: {abs(corr.item() - threshold):.2f}"


@contextmanager
def perf_memory(desc="", device_id=0):
    torch.cuda.synchronize(device_id)
    curr = torch.cuda.memory_allocated(device=device_id)
    peak = torch.cuda.max_memory_allocated(device=device_id)
    try:
        yield
    finally:
        torch.cuda.synchronize(device_id)
        curr_end = torch.cuda.memory_allocated(device=device_id)
        peak_end = torch.cuda.max_memory_allocated(device=device_id)
        delta_curr = (curr_end - curr) / 1024**2
        delta_peak = (peak_end - peak) / 1024**2
        print(f"[{desc:<12}][GPU{device_id}] Alloc: {delta_curr:.2f} MB; Peak Increase: {delta_peak:.2f} MB")


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