import numpy as np

"""
A Dense layer takes some input features and produces some output neurons.
"""
class Dense:
    """Fully connected layer: out = x @ W + b"""

    def __init__(self, in_features, out_features):

        scale = np.sqrt(2.0 / in_features)  # He init
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros(out_features)

        self.dW = None
        self.db = None
        self.cache = None

    def forward(self, x):
        self.cache = x
        return x @ self.W + self.b

    def backward(self, dout):
        x = self.cache
        self.dW = x.T @ dout
        self.db = np.sum(dout, axis=0)
        dx = dout @ self.W.T
        return dx


class ReLU:
    def __init__(self):
        self.cache = None  #initially cache set to None   

    """before relu we need a hold to store the real value so we use cache"""
    """in forward store self.cache which was initially NULL as x store x into cache"""

    def forward(self, x):
        self.cache = x    #store current x value into cache
        return np.maximum(0, x)

    def backward(self, dout):
        x = self.cache
        dx = dout.copy()   #copys the original x value into dx
        dx[x <= 0] = 0
        return dx

"""
self.cache = (x, x_padded)
self.input_shape=x.shape
"""

class Flatten:
    """
    6,3,5,5--->6,75

    """
    
    def __init__(self):
        self.input_shape = None   #cache initialization 
  
    def forward(self, x):
        self.input_shape = x.shape  #store x.shape into input_shape which was set None initially
        return x.reshape(x.shape[0], -1)  #

    def backward(self, dout):
        return dout.reshape(self.input_shape)    #reshape into original format 


def softmax_loss(scores, y):
    """
    scores: (N, num_classes) raw, un-normalized class scores
    y: (N,) integer class labels

    Returns:
        loss: scalar, average cross-entropy loss over the batch
        dscores: (N, num_classes) gradient of loss w.r.t. scores
    """
    N = scores.shape[0]
    # subtract max for n
    # numerical stability before exponentiating
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    log_probs_correct = -np.log(probs[np.arange(N), y] + 1e-12)  #negative log likelihood
    loss = np.sum(log_probs_correct) / N

    dscores = probs.copy()
    dscores[np.arange(N), y] -= 1
    dscores /= N
    return loss, dscores

