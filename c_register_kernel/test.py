import torch
import os

torch.ops.load_library(os.path.join(os.path.dirname(__file__), "./extension_cpp.cpython-310-x86_64-linux-gnu.so"))

def _backward(ctx, grad):
    a, b = ctx.saved_tensors
    grad_a, grad_b = None, None
    if ctx.needs_input_grad[0]:
        grad_a = grad * b
    if ctx.needs_input_grad[1]:
        grad_b = grad * a
    return grad_a, grad_b, None

def _setup_context(ctx, inputs, output):
    a, b, c = inputs
    saved_a, saved_b = None, None
    if ctx.needs_input_grad[0]:
        saved_b = b
    if ctx.needs_input_grad[1]:
        saved_a = a
    ctx.save_for_backward(saved_a, saved_b)

# This code adds training support for the operator. You must provide us
# the backward formula for the operator and a `setup_context` function
# to save values to be used in the backward.
torch.library.register_autograd("custom_linear::linear", _backward, setup_context=_setup_context)

# 使用已注册的 op
a = torch.randn(5, requires_grad=True)
b = torch.randn(5, requires_grad=True)
c = 3.0

out = torch.ops.custom_linear.linear(a, b, c)
print("Output:", out)

# 反向传播
out.sum().backward()

print("Grad a:", a.grad)
print("Grad b:", b.grad)
