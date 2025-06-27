import sys, os
sys.path.append(os.getcwd())

import torch
import torch.nn as nn

from pytorch_profiler import perf_kernel as perf

device = torch.device('cuda:0')
linear1 = nn.Linear(100, 128).to(device)
linear2 = nn.Linear(100, 128).to(device)

x = torch.randn(32, 100, device=device, requires_grad=True)

def forward(input):
    out1 = linear1(input)
    out2 = linear2(input)
    return out1 + out2

perf.measure_step_memory(lambda: None, "idle step")
y1 = perf.measure_step_memory(lambda: forward(x), desc="forward(x)")
z = perf.measure_step_memory(lambda: y1.sum(), desc="y1.sum()")
perf.measure_step_memory(lambda: z.backward(), desc="backward()")
perf.measure_step_memory(lambda: None, "idle step")
