"""
LeNet-style CNN built entirely from the from-scratch layers.

Architecture: CONV -> RELU -> CONV -> RELU -> POOL -> FLATTEN -> DENSE ->
RELU -> [BATCHNORM] -> [DROPOUT] -> DENSE -> SOFTMAX

BatchNorm and Dropout are optional (use_batchnorm / use_dropout flags) so you
can run the ablation experiments (Step 8) by just toggling these.
"""
import numpy as np
from layers.conv_im2col import Conv2D
from layers.basic import Dense, ReLU, Flatten, softmax_loss
from layers.regularization import BatchNorm1D, Dropout


class LeNetScratch:
    def __init__(self, input_shape=(1, 28, 28), num_classes=10,
                 use_batchnorm=False, use_dropout=False, keep_prob=0.5):
        C, H, W = input_shape
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout

        self.conv1 = Conv2D(C, 8, kernel_size=3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.conv2 = Conv2D(8, 16, kernel_size=3, stride=1, padding=1)
        self.relu2 = ReLU()

        from layers.conv_naive import MaxPool2DNaive
        self.pool = MaxPool2DNaive(pool_size=2, stride=2)
        self.flatten = Flatten()

        flat_dim = 16 * (H // 2) * (W // 2)
        hidden_dim = 128

        self.fc1 = Dense(flat_dim, hidden_dim)
        self.relu3 = ReLU()
        self.bn1 = BatchNorm1D(hidden_dim) if use_batchnorm else None
        self.dropout1 = Dropout(keep_prob) if use_dropout else None
        self.fc2 = Dense(hidden_dim, num_classes)

        # collect params for the optimizer loop: {layer: [(paramname, param)]}
        self.params = {
            "conv1_W": self.conv1.W, "conv1_b": self.conv1.b,
            "conv2_W": self.conv2.W, "conv2_b": self.conv2.b,
            "fc1_W": self.fc1.W, "fc1_b": self.fc1.b,
            "fc2_W": self.fc2.W, "fc2_b": self.fc2.b,
        }
        if use_batchnorm:
            self.params["bn1_gamma"] = self.bn1.gamma
            self.params["bn1_beta"] = self.bn1.beta

    def set_mode(self, mode):
        """mode: 'train' or 'test' -- controls BatchNorm/Dropout behavior."""
        if self.bn1 is not None:
            self.bn1.mode = mode
        if self.dropout1 is not None:
            self.dropout1.mode = mode

    def forward(self, X, y=None):
        out = self.conv1.forward(X)
        out = self.relu1.forward(out)
        out = self.conv2.forward(out)
        out = self.relu2.forward(out)
        out = self.pool.forward(out)
        out = self.flatten.forward(out)
        out = self.fc1.forward(out)
        out = self.relu3.forward(out)
        if self.bn1 is not None:
            out = self.bn1.forward(out)
        if self.dropout1 is not None:
            out = self.dropout1.forward(out)
        scores = self.fc2.forward(out)

        if y is None:
            return scores

        loss, dscores = softmax_loss(scores, y)

        grads = {}
        dout = self.fc2.backward(dscores)
        grads["fc2_W"], grads["fc2_b"] = self.fc2.dW, self.fc2.db

        if self.dropout1 is not None:
            dout = self.dropout1.backward(dout)
        if self.bn1 is not None:
            dout = self.bn1.backward(dout)
            grads["bn1_gamma"], grads["bn1_beta"] = self.bn1.dgamma, self.bn1.dbeta

        dout = self.relu3.backward(dout)
        dout = self.fc1.backward(dout)
        grads["fc1_W"], grads["fc1_b"] = self.fc1.dW, self.fc1.db

        dout = self.flatten.backward(dout)
        dout = self.pool.backward(dout)
        dout = self.relu2.backward(dout)
        dout = self.conv2.backward(dout)
        grads["conv2_W"], grads["conv2_b"] = self.conv2.dW, self.conv2.db

        dout = self.relu1.backward(dout)
        dout = self.conv1.backward(dout)
        grads["conv1_W"], grads["conv1_b"] = self.conv1.dW, self.conv1.db

        return loss, grads
