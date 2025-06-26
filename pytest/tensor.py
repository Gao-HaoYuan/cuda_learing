import sys, os
sys.path.append(os.getcwd())

import torch
import torch.nn as nn

from pytorch_profiler import perf_kernel as perf

device = torch.device('cuda')
linear = nn.Linear(100, 128).to(device)
x = torch.randn(32, 100, device=device, requires_grad=True)

perf.measure_step_memory(lambda: None, "idle step")
y1 = perf.measure_step_memory(lambda: linear(x), desc="linear1(x)")
z = perf.measure_step_memory(lambda: y1.sum(), desc="y1.sum()")
perf.measure_step_memory(lambda: z.backward(), desc="backward()")
perf.measure_step_memory(lambda: None, "idle step")
