"""
Gradient checks for every layer.

Pattern used throughout: run forward(), get a random upstream gradient the
same shape as the output, run backward() to get the analytical dx, then
numerically estimate dx by perturbing the input and re-running forward().
They should match to a small relative error.

Run with:  pytest tests/test_gradients.py -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from utils.gradient_check import eval_numerical_gradient_array, rel_error
from layers.conv_naive import Conv2DNaive, MaxPool2DNaive
from layers.conv_im2col import Conv2D
from layers.basic import Dense, ReLU, softmax_loss
from layers.regularization import BatchNorm1D, Dropout

np.random.seed(0)
TOL = 1e-4  # acceptable relative error for float64 gradient checks


def test_conv_naive_gradient():
    x = np.random.randn(2, 3, 8, 8)
    layer = Conv2DNaive(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1)

    out = layer.forward(x)
    dout = np.random.randn(*out.shape)
    dx = layer.backward(dout)

    dx_num = eval_numerical_gradient_array(lambda xx: layer.forward(xx), x, dout)
    dW_num = eval_numerical_gradient_array(lambda w: _forward_with_W(layer, x, w), layer.W, dout)

    assert rel_error(dx, dx_num) < TOL, f"dx mismatch: {rel_error(dx, dx_num)}"
    assert rel_error(layer.dW, dW_num) < TOL, f"dW mismatch: {rel_error(layer.dW, dW_num)}"


def _forward_with_W(layer, x, w):
    old_W = layer.W
    layer.W = w
    out = layer.forward(x)
    layer.W = old_W
    return out


def test_conv_im2col_matches_naive():
    """The vectorized conv must produce IDENTICAL output and gradients to
    the naive version -- this is the correctness bar for the 'upgrade'."""
    x = np.random.randn(2, 3, 8, 8)

    naive = Conv2DNaive(3, 4, kernel_size=3, stride=1, padding=1)
    fast = Conv2D(3, 4, kernel_size=3, stride=1, padding=1)
    fast.W = naive.W.copy()
    fast.b = naive.b.copy()

    out_naive = naive.forward(x)
    out_fast = fast.forward(x)
    assert rel_error(out_naive, out_fast) < 1e-8, "im2col forward doesn't match naive forward"

    dout = np.random.randn(*out_naive.shape)
    dx_naive = naive.backward(dout)
    dx_fast = fast.backward(dout)

    assert rel_error(dx_naive, dx_fast) < 1e-8, "im2col backward (dx) doesn't match naive"
    assert rel_error(naive.dW, fast.dW) < 1e-8, "im2col backward (dW) doesn't match naive"
    assert rel_error(naive.db, fast.db) < 1e-8, "im2col backward (db) doesn't match naive"


def test_maxpool_gradient():
    x = np.random.randn(2, 3, 8, 8)
    layer = MaxPool2DNaive(pool_size=2, stride=2)

    out = layer.forward(x)
    dout = np.random.randn(*out.shape)
    dx = layer.backward(dout)

    dx_num = eval_numerical_gradient_array(lambda xx: layer.forward(xx), x, dout)
    assert rel_error(dx, dx_num) < TOL


def test_dense_gradient():
    x = np.random.randn(5, 20)
    layer = Dense(20, 10)

    out = layer.forward(x)
    dout = np.random.randn(*out.shape)
    dx = layer.backward(dout)

    dx_num = eval_numerical_gradient_array(lambda xx: layer.forward(xx), x, dout)
    dW_num = eval_numerical_gradient_array(lambda w: _dense_forward_with_W(layer, x, w), layer.W, dout)

    assert rel_error(dx, dx_num) < TOL
    assert rel_error(layer.dW, dW_num) < TOL


def _dense_forward_with_W(layer, x, w):
    old_W = layer.W
    layer.W = w
    out = layer.forward(x)
    layer.W = old_W
    return out


def test_relu_gradient():
    x = np.random.randn(10, 10)
    layer = ReLU()

    out = layer.forward(x)
    dout = np.random.randn(*out.shape)
    dx = layer.backward(dout)

    dx_num = eval_numerical_gradient_array(lambda xx: layer.forward(xx), x, dout)
    assert rel_error(dx, dx_num) < TOL


def test_batchnorm_gradient():
    x = np.random.randn(20, 10) * 5 + 3  # non-trivial mean/scale
    layer = BatchNorm1D(dim=10)
    layer.mode = "train"

    out = layer.forward(x)
    dout = np.random.randn(*out.shape)
    dx = layer.backward(dout)

    dx_num = eval_numerical_gradient_array(lambda xx: layer.forward(xx), x, dout)
    assert rel_error(dx, dx_num) < TOL


def test_softmax_loss_gradient():
    N, C = 10, 5
    scores = np.random.randn(N, C)
    y = np.random.randint(0, C, size=N)

    loss, dscores = softmax_loss(scores, y)

    def f(s):
        l, _ = softmax_loss(s, y)
        return np.array(l)  # numerical gradient util expects array-like output

    # here the "upstream gradient" is just 1.0 since loss is already scalar
    dscores_num = eval_numerical_gradient_array(f, scores, np.array(1.0))
    assert rel_error(dscores, dscores_num) < TOL


def test_dropout_forward_shapes_and_scaling():
    """Dropout doesn't have a meaningful numerical gradient check (it's
    stochastic), so instead we test the inverted-dropout invariant: expected
    value of output should approximately match input (scaling cancels out)."""
    np.random.seed(1)
    x = np.ones((1000, 500))
    layer = Dropout(keep_prob=0.6)
    layer.mode = "train"
    out = layer.forward(x)
    assert abs(out.mean() - x.mean()) < 0.05, "inverted dropout scaling is off"

    layer.mode = "test"
    out_test = layer.forward(x)
    assert np.allclose(out_test, x), "test mode should be a no-op"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
