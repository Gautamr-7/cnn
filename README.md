# CNN From Scratch (NumPy) — with Gradient Checking & im2col Benchmarking
<img width="1200" height="564" alt="image" src="https://github.com/user-attachments/assets/5dcf3932-b2f9-4345-a8e7-3837a8f3dd32" />


A convolutional neural network implemented entirely in NumPy — no autograd,
no `torch.nn` — built to actually understand backpropagation through
convolution and pooling, not just call a framework.

Reference: Yann LeCun's LeNet-5 (LeCun et al., 1998), the classic CONV-POOL-FC
architecture that established the modern CNN pattern.

## What problem this addresses

Frameworks like PyTorch hide two things people often use without
understanding: (1) how gradients actually flow backward through a
convolution and a max-pool, and (2) why convolution is implemented as a
matrix multiply (im2col) rather than literal nested loops, for speed. This
repo implements both versions and proves their equivalence.

<img width="1910" height="945" alt="image" src="https://github.com/user-attachments/assets/a09af18e-cec1-4638-bc61-0194cf08b1c7" />


## Core idea

- **Naive conv**: for every output position, take the dot product of the
  kernel with the corresponding input patch. Transparent, but O(output
  positions × kernel size) in pure Python — slow.
- **im2col conv**: unfold every patch the kernel will see into columns of a
  matrix, then compute the entire convolution as one matrix multiply. This
  is the same trick frameworks use to make convolution fast on CPU/GPU.

## Architecture

```
Input (1, 28, 28)
   ↓
Conv2D(3x3, 8 filters, pad=1) → ReLU
   ↓
Conv2D(3x3, 16 filters, pad=1) → ReLU
   ↓
MaxPool(2x2, stride=2)
   ↓
Flatten
   ↓
Dense(→128) → ReLU → [BatchNorm] → [Dropout]
   ↓
Dense(→10) → Softmax
```

BatchNorm and Dropout are optional (`--batchnorm` / `--dropout` flags) to
support the ablation experiments below.

## What's implemented from scratch vs. what isn't

| Component | From scratch? |
|---|---|
| Conv2D forward/backward (naive AND im2col) | ✅ NumPy only |
| MaxPool2D forward/backward | ✅ NumPy only |
| Dense, ReLU, Flatten, Softmax+CrossEntropy | ✅ NumPy only |
| BatchNorm, Dropout | ✅ NumPy only |
| SGD, SGD+Momentum, Adam | ✅ NumPy only |
| Gradient checking (finite differences) | ✅ NumPy only |
| Training loop, batching | ✅ NumPy only |
| MNIST idx-file parsing | ✅ Python stdlib (`struct`, `gzip`) |
| PyTorch | Used **only** in `cross_check_torch.py`, to verify correctness — never in the actual model or training path |

## Correctness: gradient checking

Every layer's analytical backward pass is checked against a numerical
(finite-difference) gradient before being trusted. All checks pass at
< 1e-4 relative error (most are < 1e-8):

```
$ pytest tests/test_gradients.py -v
tests/test_gradients.py::test_conv_naive_gradient PASSED
tests/test_gradients.py::test_conv_im2col_matches_naive PASSED
tests/test_gradients.py::test_maxpool_gradient PASSED
tests/test_gradients.py::test_dense_gradient PASSED
tests/test_gradients.py::test_relu_gradient PASSED
tests/test_gradients.py::test_batchnorm_gradient PASSED
tests/test_gradients.py::test_softmax_loss_gradient PASSED
tests/test_gradients.py::test_dropout_forward_shapes_and_scaling PASSED
======================== 8 passed ========================
```

## Correctness: PyTorch cross-check

`cross_check_torch.py` copies identical weights into the from-scratch layer
and the equivalent `nn.Conv2d`/`nn.Linear`, feeds identical input through
both, and compares forward outputs and gradients directly:

```
$ python3 cross_check_torch.py
=== Conv2D cross-check vs nn.Conv2d ===
Forward output relative error: <fill in after running>
dx relative error: <fill in after running>
dW relative error: <fill in after running>
db relative error: <fill in after running>
```

*(Run this on a machine with `torch` installed — not run in the build
environment due to sandbox disk constraints. Fill in the printed numbers
here; expect ~1e-10 to 1e-14.)*

## Performance: naive vs. im2col

```
Config                                          Naive (s)    im2col (s)   Speedup
---------------------------------------------------------------------------------
N=4,C=3->8,HW=16                                   0.0752        0.0004    178.6x
N=8,C=8->16,HW=16                                  0.3238        0.0018    177.2x
N=16,C=16->32,HW=28                                4.4073        0.0228    193.1x
```

Vectorizing convolution as a single matrix multiply gives a ~180x speedup
over naive nested loops at these sizes — this is the concrete answer to
"why do frameworks use im2col-style implementations."

## Ablation experiments

*(Fill in after running `train.py` on real MNIST — see Setup below)*

| Config | Train acc | Test acc | Notes |
|---|---|---|---|
| Baseline | | | |
| + BatchNorm | | | |
| + Dropout | | | |
| SGD vs SGD+Momentum vs Adam | | | |
| He init vs Xavier init | | | |

## Setup

```bash
pip install -r requirements.txt

# run gradient checks
pytest tests/test_gradients.py -v

# benchmark naive vs im2col
python3 benchmark.py

# train (uses synthetic data unless real MNIST idx files are present)
# download from http://yann.lecun.com/exdb/mnist/ into ./mnist_data/
python3 train.py --epochs 5 --batchnorm --dropout --optimizer adam

# cross-check against PyTorch (requires torch)
python3 cross_check_torch.py
```

##   Testing CONV NAIVE vs IM2COL

<img width="1523" height="389" alt="image" src="https://github.com/user-attachments/assets/27880ef8-098a-4678-811b-5bc00e19721a" />


## Reference

- LeCun et al., 1998 — [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) (LeNet-5)
- Ioffe & Szegedy, 2015 — [Batch Normalization](https://arxiv.org/abs/1502.03167)
- Srivastava et al., 2014 — [Dropout](https://jmlr.org/papers/v15/srivastava14a.html)
- He et al., 2015 — [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852) (He initialization)
- Kingma & Ba, 2014 — [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- CS231n course notes (Stanford) — theoretical reference for conv/pool backprop derivations
