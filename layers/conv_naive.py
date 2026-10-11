"""
Naive (loop-based) Conv2D.

This is deliberately slow. The point of writing it this way first is that
every line maps directly onto the math: for each output position, take the
dot product of the kernel with the corresponding input patch. Once you trust
this version (via gradient checking), you vectorize it with im2col and use
THIS version as the ground truth to check the vectorized version against.
"""
import numpy as np
"""
#a 2D convolutional layer is defined almost entirely by
#its filters (plus a bias per filter and the operational rules for sliding them)

#---------FILTER---------6,3,3,3
#6  → number of filters
#3  → number of input channels
#3  → kernel height
#3  → kernel width

#---------INPUT IMAGE------------
#For an RGB image:
#Input shape = (3, H, W)
#means:

#Channel 0 → Red   → H × W numbers
#Channel 1 → Green → H × W numbers
#Channel 2 → Blue  → H × W numbers

#----------F-------
#F = number of filters
#  = number of output channels
#  = number of feature maps
"""
class Conv2DNaive:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        #building filters for W say 6,3,5,5 6 filters(out),3 i/p channel, 5 kernel size
        self.stride = stride
        self.padding = padding
        #Kaiming He initialization: good default for ReLU networks
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))  #kaiming initialization
        #np.random.randn() generates values from a standard normal distribution:
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros(out_channels)
        self.dW = None
        self.db = None
        self.cache = None  #It explicitly stores a tuple containing:  1st → original x , 2nd → x_padded

    def forward(self, x):
        """x: (N, C, H, W) -> out: (N, F, H_out, W_out)   N is no of images in batch,input channels per image,h,w
        in out F is no of filters which also determine no of output channels For example, if we have 2 RGB images, the input shape is (2, 3, 8, 8)."""
        N, C, H, W = x.shape
        F, _, KH, KW = self.W.shape
        s, p = self.stride, self.padding

        x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")  #we only  pad 2,3 dimensions  ie h and w
        H_out = (H + 2 * p - KH) // s + 1
        W_out = (W + 2 * p - KW) // s + 1

        out = np.zeros((N, F, H_out, W_out))   #we already know the out dimensions 

        for n in range(N):  #n images
            for f in range(F):  #f filters 
                for i in range(H_out):      #i rows
                    for j in range(W_out):  #j columns

                        h_start, w_start = i * s, j * s
                        patch = x_padded[n, :, h_start:h_start + KH, w_start:w_start + KW]
                        out[n, f, i, j] = np.sum(patch * self.W[f]) + self.b[f]  #dot product of patch with filter added with bias per filter

        self.cache = (x, x_padded)  #store
        return out

    def backward(self, dout):
        """dout: (N, F, H_out, W_out) -> dx: (N, C, H, W)"""
        x, x_padded = self.cache  #recall whats stored
        N, C, H, W = x.shape
        F, _, KH, KW = self.W.shape
        s, p = self.stride, self.padding
        _, _, H_out, W_out = dout.shape

        #making pre-requisite arrays --- we gotta fill this in
        dx_padded = np.zeros_like(x_padded)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        for n in range(N):
            for f in range(F):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start, w_start = i * s, j * s
                        patch = x_padded[n, :, h_start:h_start + KH, w_start:w_start + KW]

                        self.dW[f] += dout[n, f, i, j] * patch
                        dx_padded[n, :, h_start:h_start + KH, w_start:w_start + KW] += (
                            dout[n, f, i, j] * self.W[f]
                        )
                self.db[f] += np.sum(dout[n, f])

        # remove padding to get dx back to original shape
        dx = dx_padded[:, :, p:p + H, p:p + W] if p > 0 else dx_padded  #ie if padding>0 remove padding 
        return dx


class MaxPool2DNaive:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size  #pool_size = means we use a 2 × 2 window to find the maximum value.
        self.stride = stride
        self.cache = None

    def forward(self, x):
        N, C, H, W = x.shape
        ps, s = self.pool_size, self.stride
        H_out = (H - ps) // s + 1
        W_out = (W - ps) // s + 1

        out = np.zeros((N, C, H_out, W_out))
        # remember WHERE the max came from -- that's the only place
        # gradient will flow during backward
        argmax = np.zeros((N, C, H_out, W_out, 2), dtype=int) #This creates an array to record where each maximum occurred inside its pooling window.

        for i in range(H_out):
            for j in range(W_out):
                h_start, w_start = i * s, j * s
                window = x[:, :, h_start:h_start + ps, w_start:w_start + ps]

                window_flat = window.reshape(N, C, -1) #flattens the matrix to a row
                idx = np.argmax(window_flat, axis=2)    #for dim -1 it finds max value's index
                out[:, :, i, j] = np.max(window_flat, axis=2) #output takes max value of flattened arrayS
                """
                we know that argmax shape is (n,c,hout,wout,2) here last 2, 0 is for idx//ps and 1 is idx is idx%ps together they give
                index location in matrix suppose idx is 3 then we get in matrix max located at (1,1)
                """
                argmax[:, :, i, j, 0] = idx // ps  
                argmax[:, :, i, j, 1] = idx % ps

        self.cache = (x.shape, argmax)  #keep track of where max in out came from 
        return out

    def backward(self, dout):
        x_shape, argmax = self.cache
        N, C, H, W = x_shape
        ps, s = self.pool_size, self.stride
        _, _, H_out, W_out = dout.shape

        dx = np.zeros(x_shape)
        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start, w_start = i * s, j * s
                        dh, dw = argmax[n, c, i, j]
                        # gradient only flows to the position that was the max
                        dx[n, c, h_start + dh, w_start + dw] += dout[n, c, i, j]
        return dx
