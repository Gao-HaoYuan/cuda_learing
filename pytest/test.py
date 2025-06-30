import sys, os
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

from torchinfo import summary
from pytorch_profiler import perf_kernel as perf

# prepare data input
X = torch.randn(1000, 3, 224, 224)
y = torch.randint(0, 10, (1000,))

train_dataset = TensorDataset(X[:800], y[:800])
val_dataset = TensorDataset(X[800:], y[800:])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)
batch_x, batch_y = next(iter(train_loader))
criterion = nn.CrossEntropyLoss()

perf.measure_step_memory(lambda: None, "begin")
model = models.resnet18(num_classes=10).to("cuda")

perf.measure_step_memory(lambda: None, "device model")
batch_x, batch_y = batch_x.to("cuda"), batch_y.to("cuda")
perf.measure_step_memory(lambda: None, "batch input")

optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer.zero_grad()
model.train()

outputs = model(batch_x)
perf.measure_step_memory(lambda: None, "model")
loss = criterion(outputs, batch_y)
perf.measure_step_memory(lambda: None, "criterion")
loss.backward()
perf.measure_step_memory(lambda: None, "backward")
optimizer.step()
perf.measure_step_memory(lambda: None, "optimizer")

# perf.perf_profile(model, batch_x)

summary(
    model, 
    input_size=(1, 3, 224, 224),  # (batch_size, channels, height, width)
    verbose=2,
    col_width=16,
    col_names=["kernel_size", "output_size", "num_params"],
    row_settings=["var_names"]
)