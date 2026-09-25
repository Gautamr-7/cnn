"""
Training loop for the from-scratch LeNet.

Usage:
    python3 train.py                          # baseline
    python3 train.py --batchnorm              # ablation: + BatchNorm
    python3 train.py --dropout                # ablation: + Dropout
    python3 train.py --optimizer sgd_momentum # ablation: different optimizer
"""
import argparse
import time
import numpy as np
from model import LeNetScratch
from data import load_mnist
import optim as optim_module


def get_batches(X, y, batch_size, shuffle=True):
    N = X.shape[0]
    idx = np.random.permutation(N) if shuffle else np.arange(N)
    for start in range(0, N, batch_size):
        batch_idx = idx[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]


def evaluate(model, X, y, batch_size=100):
    model.set_mode("test")
    correct = 0
    for xb, yb in get_batches(X, y, batch_size, shuffle=False):
        scores = model.forward(xb)
        preds = np.argmax(scores, axis=1)
        correct += np.sum(preds == yb)
    model.set_mode("train")
    return correct / X.shape[0]


def train(args):
    X_train, y_train, X_test, y_test = load_mnist(n_train=args.n_train, n_test=args.n_test)

    model = LeNetScratch(
        input_shape=(1, 28, 28),
        num_classes=10,
        use_batchnorm=args.batchnorm,
        use_dropout=args.dropout,
    )
    model.set_mode("train")

    update_fn = getattr(optim_module, args.optimizer)
    configs = {name: {"learning_rate": args.lr} for name in model.params}

    history = {"epoch": [], "train_loss": [], "train_acc": [], "test_acc": [], "time_sec": []}
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_losses = []
        for xb, yb in get_batches(X_train, y_train, args.batch_size):
            loss, grads = model.forward(xb, yb)
            epoch_losses.append(loss)

            for name, param in model.params.items():
                updated, configs[name] = update_fn(param, grads[name], configs[name])
                # params dict already holds references to the live layer
                # attributes (e.g. model.conv1.W), and update_fn mutates
                # in-place via -=, so nothing further to reassign here.

        train_acc = evaluate(model, X_train, y_train)
        test_acc = evaluate(model, X_test, y_test)
        elapsed = time.time() - start_time

        history["epoch"].append(epoch)
        history["train_loss"].append(float(np.mean(epoch_losses)))
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["time_sec"].append(elapsed)

        print(f"epoch {epoch+1}/{args.epochs}  "
              f"loss={np.mean(epoch_losses):.4f}  "
              f"train_acc={train_acc:.3f}  test_acc={test_acc:.3f}  "
              f"({elapsed:.1f}s elapsed)")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adam",
                         choices=["sgd", "sgd_momentum", "adam"])
    parser.add_argument("--batchnorm", action="store_true")
    parser.add_argument("--dropout", action="store_true")
    parser.add_argument("--n_train", type=int, default=2000)
    parser.add_argument("--n_test", type=int, default=500)
    args = parser.parse_args()

    train(args)
