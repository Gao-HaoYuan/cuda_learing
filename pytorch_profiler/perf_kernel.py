import torch
import torch.profiler
from torch.profiler import ProfilerActivity

def perf_profile(operator_fn, *args, **kwargs):
    # 同步 cuda 行为, 不执行同步 cuda 统计会出错，比如内存峰值
    def sync_op():
        torch.cuda.synchronize()
        _ = operator_fn(*args, **kwargs)
        torch.cuda.synchronize()

        return _

    with torch.profiler.profile(
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready = torch.profiler.tensorboard_trace_handler('./log/linear'),
        # record_shapes = True,
        # with_stack = True,
        profile_memory = True
    ) as prof:
        for _ in range(5):
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
    output = sync_op()
    peak_cuda_mem_MB = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"Peak GPU Memory Usage: {peak_cuda_mem_MB:.3f} MB\n")

    return output

def perf_accuracy(expected, compared):
    def compare(x, y):
        err = torch.mean(torch.abs(x - y)) / torch.mean(torch.abs(x))
        corr = torch.cosine_similarity(x.view(-1), y.contiguous().view(-1), dim=-1)
        return err, corr

    expected = expected if isinstance(expected, (list, tuple)) else (expected,)
    compared = compared if isinstance(compared, (list, tuple)) else (compared,)

    for i, (a, b) in enumerate(zip(expected, compared)):
        err, corr = compare(a, b)
        print(f"Output {i}: relative_error_mean {err:.6f}, corr: {corr:.6f}")


def measure_step_memory(fn, desc=""):
    torch.cuda.synchronize()
    result = fn()
    torch.cuda.synchronize()

    current = torch.cuda.memory_allocated() / 1024
    peak = torch.cuda.max_memory_allocated() / 1024

    print(f"[{desc:<12}] Δcurrent mem: {current:.2f} KB; Δpeak mem: {peak:.2f}")
    return result