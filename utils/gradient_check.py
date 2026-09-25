"""
Gradient checking utilities.

The idea: if a layer's backward() is correct, the gradient it computes
analytically must match the gradient computed numerically via finite
differences. This is the single most important safety net when you're
deriving backprop by hand.
"""
import numpy as np


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Numerically compute the gradient of a function that takes an array
    and produces an array, using the centered finite-difference formula:

        d/dx_i f(x) ≈ (f(x + h*e_i) - f(x - h*e_i)) / (2h)

    Since f returns an array (not a scalar), we project the output
    difference onto the upstream gradient `df` (this is exactly what
    backward() does with an upstream gradient in real backprop).

    Args:
        f: function that takes x and returns an array (e.g. layer.forward)
        x: input array to compute the gradient with respect to
        df: upstream gradient, same shape as f(x)
        h: finite-difference step size

    Returns:
        numerical gradient, same shape as x
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index
        oldval = x[ix]

        x[ix] = oldval + h
        pos = f(x).copy()

        x[ix] = oldval - h
        neg = f(x).copy()

        x[ix] = oldval  # restore

        # chain rule: sum over all output elements of (d out / d x_ix) * df
        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


def rel_error(x, y):
    """
    Relative error between two arrays. This is the standard metric used
    for gradient checks (see CS231n notes on gradient checking).

    Guideline:
        < 1e-7   : excellent, essentially exact
        < 1e-4   : usually fine for float32 / non-smooth ops like ReLU
        > 1e-2   : something is almost certainly wrong
    """
    numerator = np.abs(x - y)
    denominator = np.maximum(1e-8, np.abs(x) + np.abs(y))
    return np.max(numerator / denominator)
