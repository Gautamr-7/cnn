"""
Data loading. If real MNIST idx files are found at `data_dir`, load them.
Otherwise, fall back to synthetic random data so you can test the whole
pipeline (shapes, gradient flow, training loop) without needing a download.

To use real MNIST: download the 4 idx files from
http://yann.lecun.com/exdb/mnist/ (or the mirror at
https://ossci-datasets.s3.amazonaws.com/mnist/) into `data_dir` and this
loader will pick them up automatically.
"""
import os
import gzip
import struct
import numpy as np


def _read_idx(path):
    with gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb") as f:
        magic, = struct.unpack(">I", f.read(4))
        if magic == 2051:  # images
            n, rows, cols = struct.unpack(">III", f.read(12))
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)
        elif magic == 2049:  # labels
            n, = struct.unpack(">I", f.read(4))
            data = np.frombuffer(f.read(), dtype=np.uint8)
        else:
            raise ValueError(f"Unknown IDX magic number: {magic}")
        return data


def load_mnist(data_dir="./mnist_data", n_train=2000, n_test=500):
    """
    Returns (X_train, y_train, X_test, y_test).
    X shape: (N, 1, 28, 28), float32, normalized to [0, 1]
    y shape: (N,), int
    """
    files = {
        "train_images": "train-images-idx3-ubyte",
        "train_labels": "train-labels-idx1-ubyte",
        "test_images": "t10k-images-idx3-ubyte",
        "test_labels": "t10k-labels-idx1-ubyte",
    }
    paths = {k: os.path.join(data_dir, v) for k, v in files.items()}
    have_real_data = all(
        os.path.exists(p) or os.path.exists(p + ".gz") for p in paths.values()
    )

    if have_real_data:
        def find(p):
            return p if os.path.exists(p) else p + ".gz"

        X_train = _read_idx(find(paths["train_images"])).astype(np.float32) / 255.0
        y_train = _read_idx(find(paths["train_labels"])).astype(np.int64)
        X_test = _read_idx(find(paths["test_images"])).astype(np.float32) / 255.0
        y_test = _read_idx(find(paths["test_labels"])).astype(np.int64)

        X_train = X_train[:n_train, None, :, :]
        y_train = y_train[:n_train]
        X_test = X_test[:n_test, None, :, :]
        y_test = y_test[:n_test]
        print(f"Loaded real MNIST: {X_train.shape[0]} train, {X_test.shape[0]} test")
    else:
        print(f"MNIST idx files not found in '{data_dir}' -- using SYNTHETIC data.")
        print("Download real MNIST from http://yann.lecun.com/exdb/mnist/ to use actual data.")
        rng = np.random.RandomState(0)
        X_train = rng.rand(n_train, 1, 28, 28).astype(np.float32)
        y_train = rng.randint(0, 10, size=n_train)
        X_test = rng.rand(n_test, 1, 28, 28).astype(np.float32)
        y_test = rng.randint(0, 10, size=n_test)

    return X_train, y_train, X_test, y_test
