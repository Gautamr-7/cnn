"""
Benchmark: naive loop-based Conv2D vs im2col-vectorized Conv2D.

This produces the centerpiece number for your README: "how much does
vectorization actually buy you, and why are frameworks fast."
"""
import time
import numpy as np
from layers.conv_naive import Conv2DNaive
from layers.conv_im2col import Conv2D


def time_forward_backward(layer, x, n_repeats=3):
    times = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        out = layer.forward(x)
        dout = np.random.randn(*out.shape)
        layer.backward(dout)
        times.append(time.perf_counter() - start)
    return min(times)  # best-of-N to reduce noise


def run_benchmark():
    configs = [
        {"batch": 4, "in_ch": 3, "hw": 16, "out_ch": 8, "k": 3},
        {"batch": 8, "in_ch": 8, "hw": 16, "out_ch": 16, "k": 3},
        {"batch": 16, "in_ch": 16, "hw": 28, "out_ch": 32, "k": 3},
    ]

    print(f"{'Config':<45}{'Naive (s)':>12}{'im2col (s)':>14}{'Speedup':>10}")
    print("-" * 81)

    for cfg in configs:
        x = np.random.randn(cfg["batch"], cfg["in_ch"], cfg["hw"], cfg["hw"])

        naive = Conv2DNaive(cfg["in_ch"], cfg["out_ch"], cfg["k"], stride=1, padding=1)
        fast = Conv2D(cfg["in_ch"], cfg["out_ch"], cfg["k"], stride=1, padding=1)
        fast.W, fast.b = naive.W.copy(), naive.b.copy()

        t_naive = time_forward_backward(naive, x)
        t_fast = time_forward_backward(fast, x)

        label = f"N={cfg['batch']},C={cfg['in_ch']}->{cfg['out_ch']},HW={cfg['hw']}"
        print(f"{label:<45}{t_naive:>12.4f}{t_fast:>14.4f}{t_naive / t_fast:>9.1f}x")


if __name__ == "__main__":
    run_benchmark()
