import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torchinfo import summary
from torch.profiler import profile, record_function, ProfilerActivity

class ThreeLayerFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        tmp = torch.cuda.memory_allocated()
        print(f"before linear1 : {(tmp) / 1024} KB")
        x = self.fc1(x)
        print(f"after  linear1 : {torch.cuda.memory_allocated() / 1024} KB")
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x

# 随机生成假数据：输入为 100 维，输出为 0-9 类别
X = torch.randn(1000, 100)         # 1000 个样本
y = torch.randint(0, 10, (1000,))  # 对应的标签，10 个类别

# 拆分训练集和验证集
train_dataset = TensorDataset(X[:800], y[:800])
val_dataset = TensorDataset(X[800:], y[800:])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# torch.cuda.reset_peak_memory_stats()
model = ThreeLayerFC().to("cuda")
# print(f"model totoal memory is {torch.cuda.memory_allocated() / 1024} KB")

summary(
    model, 
    input_size=(32, 100), 
    verbose=2,
    col_width=16,
    col_names=["kernel_size", "output_size", "num_params"],
    row_settings=["var_names"]
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练过程
for epoch in range(2):
    model.train()
    total_loss = 0
    print(f"model train:")
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to("cuda"), batch_y.to("cuda")

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    print()

    #     total_loss += loss.item()

    # avg_loss = total_loss / len(train_loader)

    # # 验证过程``
    # model.eval()
    # correct = total = 0
    # print()
    # print(f"model eval:")
    # with torch.no_grad():
    #     for batch_x, batch_y in val_loader:
    #         batch_x, batch_y = batch_x.to("cuda"), batch_y.to("cuda")
    #         outputs = model(batch_x)
    #         predicted = outputs.argmax(dim=1)
    #         correct += (predicted == batch_y).sum().item()
    #         total += batch_y.size(0)

    # accuracy = correct / total
    # print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val Acc: {accuracy:.2%}")
    # print()

