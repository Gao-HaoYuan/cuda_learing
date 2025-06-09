import torch
import my_add

x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
y = my_add.my_add(x)

print(y)  # Expect: [2.0, 3.0, 4.0]
