import numpy as np


class BatchNorm1D:
    """
    Batch normalization for (N, D) inputs (i.e. after a Dense layer).

    Key idea (Ioffe & Szegedy, 2015): normalize each feature to zero mean /
    unit variance using BATCH statistics during training, then let the
    network learn a scale (gamma) and shift (beta) on top. At test time we
    don't have a "batch" (might even be a single example), so we use a
    running average of mean/variance accumulated during training instead.
    """

    def __init__(self, dim, momentum=0.9, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.dgamma = None
        self.dbeta = None

        self.momentum = momentum
        self.eps = eps
        self.running_mean = np.zeros(dim)
        self.running_var = np.ones(dim)

        self.mode = "train"  # or "test"
        self.cache = None

    def forward(self, x):
        if self.mode == "train":
            mu = np.mean(x, axis=0)
            var = np.var(x, axis=0)

            x_norm = (x - mu) / np.sqrt(var + self.eps)
            out = self.gamma * x_norm + self.beta

            # update running estimates for use at test time
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

            self.cache = (x, x_norm, mu, var)
        elif self.mode == "test":
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_norm + self.beta
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return out

    def backward(self, dout):
        x, x_norm, mu, var = self.cache
        N, D = x.shape
        std_inv = 1.0 / np.sqrt(var + self.eps)

        self.dgamma = np.sum(dout * x_norm, axis=0)
        self.dbeta = np.sum(dout, axis=0)

        dx_norm = dout * self.gamma
        dvar = np.sum(dx_norm * (x - mu) * -0.5 * std_inv**3, axis=0)
        dmu = np.sum(dx_norm * -std_inv, axis=0) + dvar * np.mean(-2 * (x - mu), axis=0)

        dx = dx_norm * std_inv + dvar * 2 * (x - mu) / N + dmu / N
        return dx


class Dropout:
    """
    Inverted dropout: scale during TRAINING (by 1/keep_prob) so that no
    change is needed at test time -- test time is simply "use everything".
    """

    def __init__(self, keep_prob=0.5):
        self.keep_prob = keep_prob
        self.mode = "train"
        self.mask = None

    def forward(self, x):
        if self.mode == "train":
            self.mask = (np.random.rand(*x.shape) < self.keep_prob) / self.keep_prob
            return x * self.mask
        elif self.mode == "test":
            return x
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def backward(self, dout):
        if self.mode == "train":
            return dout * self.mask
        return dout
