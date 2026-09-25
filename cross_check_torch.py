"""
Cross-check the from-scratch Conv2D and Dense layers against PyTorch.

This is the strongest correctness evidence in the repo: rather than just
checking that training converges, we copy the SAME weights into both
implementations, feed the SAME input through both, and assert the outputs
and gradients match numerically.

NOTE: requires `pip install torch`. This wasn't run in the build sandbox
(no disk space for torch there) -- run it yourself and paste the output
into your README's results section.

Usage: python3 cross_check_torch.py
"""
import numpy as np
import torch
import torch.nn as nn

from layers.conv_im2col import Conv2D
from layers.basic import Dense


def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def check_conv2d():
    print("=== Conv2D cross-check vs nn.Conv2d ===")
    N, C, H, W = 4, 3, 16, 16
    out_channels, kernel_size, stride, padding = 8, 3, 1, 1

    x_np = np.random.randn(N, C, H, W).astype(np.float64)

    ours = Conv2D(C, out_channels, kernel_size, stride=stride, padding=padding)

    torch_conv = nn.Conv2d(C, out_channels, kernel_size, stride=stride, padding=padding, bias=True)
    torch_conv = torch_conv.double()
    with torch.no_grad():
        torch_conv.weight.copy_(torch.from_numpy(ours.W))
        torch_conv.bias.copy_(torch.from_numpy(ours.b))

    x_torch = torch.from_numpy(x_np).double().requires_grad_(True)

    out_ours = ours.forward(x_np)
    out_torch = torch_conv(x_torch)

    fwd_err = rel_error(out_ours, out_torch.detach().numpy())
    print(f"Forward output relative error: {fwd_err:.2e}")

    dout = np.random.randn(*out_ours.shape)
    dx_ours = ours.backward(dout)

    out_torch.backward(torch.from_numpy(dout))
    dx_torch = x_torch.grad.numpy()
    dW_torch = torch_conv.weight.grad.numpy()
    db_torch = torch_conv.bias.grad.numpy()

    print(f"dx relative error: {rel_error(dx_ours, dx_torch):.2e}")
    print(f"dW relative error: {rel_error(ours.dW, dW_torch):.2e}")
    print(f"db relative error: {rel_error(ours.db, db_torch):.2e}")


def check_dense():
    print("\n=== Dense cross-check vs nn.Linear ===")
    N, in_f, out_f = 8, 20, 10
    x_np = np.random.randn(N, in_f).astype(np.float64)

    ours = Dense(in_f, out_f)

    torch_linear = nn.Linear(in_f, out_f).double()
    with torch.no_grad():
        torch_linear.weight.copy_(torch.from_numpy(ours.W.T))  # torch stores (out,in)
        torch_linear.bias.copy_(torch.from_numpy(ours.b))

    x_torch = torch.from_numpy(x_np).requires_grad_(True)

    out_ours = ours.forward(x_np)
    out_torch = torch_linear(x_torch)
    print(f"Forward output relative error: {rel_error(out_ours, out_torch.detach().numpy()):.2e}")

    dout = np.random.randn(*out_ours.shape)
    dx_ours = ours.backward(dout)

    out_torch.backward(torch.from_numpy(dout))
    dx_torch = x_torch.grad.numpy()
    dW_torch = torch_linear.weight.grad.numpy().T
    db_torch = torch_linear.bias.grad.numpy()

    print(f"dx relative error: {rel_error(dx_ours, dx_torch):.2e}")
    print(f"dW relative error: {rel_error(ours.dW, dW_torch):.2e}")
    print(f"db relative error: {rel_error(ours.db, db_torch):.2e}")


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    check_conv2d()
    check_dense()
