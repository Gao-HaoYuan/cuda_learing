import torch
import producer
import consumer

producer_so = "/home/haoyuangao/pinned/cuda_learing/torchclass_demo/producer/producer.cpython-311-x86_64-linux-gnu.so"

x = torch.randn(4).requires_grad_(True)
w = torch.randn(4).requires_grad_(True)

y = consumer.infer_by_producer_so(producer_so, 7, x, w)
print("y =", y)
print("x =", x)
print("w =", w)
print("x.grad =", x.grad)
print("w.grad =", w.grad)
