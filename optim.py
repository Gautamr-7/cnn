"""
Hand-written optimizers. Each function takes a parameter, its gradient, and
a config dict holding any state the optimizer needs to persist between steps
(velocity for momentum, moving averages for Adam). Returns (updated_param,
updated_config) -- the caller is responsible for storing config back.
"""
import numpy as np


def sgd(w, dw, config=None):
    config = config or {"learning_rate": 1e-2}
    w -= config["learning_rate"] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    config = config or {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    v = config["momentum"] * v - config["learning_rate"] * dw
    w += v

    config["velocity"] = v
    return w, config


def adam(w, dw, config=None):
    """
    Adam (Kingma & Ba, 2014): keeps a running average of the gradient (m,
    like momentum) AND a running average of the squared gradient (v, giving
    each parameter its own adaptive learning rate). Bias-correction terms
    (mt, vt) fix the fact that m and v start at zero and are biased toward
    zero early in training.
    """
    config = config or {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("eps", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    config["t"] += 1
    config["m"] = config["beta1"] * config["m"] + (1 - config["beta1"]) * dw
    config["v"] = config["beta2"] * config["v"] + (1 - config["beta2"]) * (dw ** 2)

    mt = config["m"] / (1 - config["beta1"] ** config["t"])
    vt = config["v"] / (1 - config["beta2"] ** config["t"])

    w -= config["learning_rate"] * mt / (np.sqrt(vt) + config["eps"])
    return w, config
