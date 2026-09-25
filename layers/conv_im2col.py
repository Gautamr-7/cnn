"""
Vectorized Conv2D using im2col.

The trick: convolution is just "take a patch, dot it with the kernel" repeated
many times. Instead of looping over every output position in Python, we:

  1. Extract every patch the kernel will ever see and stack them as COLUMNS
     of one big matrix (im2col) -- shape (C*KH*KW, N*H_out*W_out)
  2. Reshape the kernel weights into a matrix (F, C*KH*KW)
  3. The entire convolution becomes ONE matrix multiply:
         out = W_matrix @ columns
  4. Reshape the result back into (N, F, H_out, W_out)

This is exactly why frameworks are fast: they turn convolution into the thing
hardware (BLAS/GPUs) is already extremely good at -- matrix multiplication.
col2im does the reverse for the backward pass (scatter-add gradients back to
the original overlapping patch locations).
"""
import numpy as np
#kh=f here filters/kernal size

def get_output_shape(H, W, KH, KW, stride, padding):
    H_out = (H + 2 * padding - KH) // stride + 1
    W_out = (W + 2 * padding - KW) // stride + 1
    return H_out, W_out


def im2col(x, KH, KW, stride, padding):
    """x: (N, C, H, W) -> columns: (C*KH*KW, N*H_out*W_out)"""
    N, C, H, W = x.shape
    H_out, W_out = get_output_shape(H, W, KH, KW, stride, padding)

    x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")

    cols = np.zeros((C, KH, KW, N, H_out, W_out))
    for i in range(KH):
        i_max = i + stride * H_out
        for j in range(KW):
            j_max = j + stride * W_out
            # this slice grabs, for every output position at once, the pixel
            # that sits at offset (i,j) within that output position's patch
            cols[:, i, j, :, :, :] = x_padded[:, :, i:i_max:stride, j:j_max:stride].transpose(1, 0, 2, 3)

    cols = cols.reshape(C * KH * KW, N * H_out * W_out)
    return cols


def col2im(cols, x_shape, KH, KW, stride, padding):
    """Inverse of im2col: scatter-add columns back into an image-shaped gradient."""
    N, C, H, W = x_shape
    H_out, W_out = get_output_shape(H, W, KH, KW, stride, padding)
    H_padded, W_padded = H + 2 * padding, W + 2 * padding

    x_padded = np.zeros((N, C, H_padded, W_padded))
    cols_reshaped = cols.reshape(C, KH, KW, N, H_out, W_out)

    for i in range(KH):
        i_max = i + stride * H_out
        for j in range(KW):
            j_max = j + stride * W_out
            # note += , not = : overlapping patches must ACCUMULATE gradient
            x_padded[:, :, i:i_max:stride, j:j_max:stride] += cols_reshaped[:, i, j, :, :, :].transpose(1, 0, 2, 3)

    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.stride = stride
        self.padding = padding
        self.KH = self.KW = kernel_size
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros(out_channels)
        self.dW = None
        self.db = None
        self.cache = None

    def forward(self, x):
        N, C, H, W = x.shape
        F = self.W.shape[0]
        H_out, W_out = get_output_shape(H, W, self.KH, self.KW, self.stride, self.padding)

        cols = im2col(x, self.KH, self.KW, self.stride, self.padding)          # (C*KH*KW, N*H_out*W_out)
        W_matrix = self.W.reshape(F, -1)                                       # (F, C*KH*KW)

        out = W_matrix @ cols + self.b.reshape(-1, 1)                          # (F, N*H_out*W_out)
        out = out.reshape(F, N, H_out, W_out).transpose(1, 0, 2, 3)            # (N, F, H_out, W_out)

        self.cache = (x, cols, W_matrix)
        return out

    def backward(self, dout):
        x, cols, W_matrix = self.cache
        N, C, H, W = x.shape
        F = self.W.shape[0]
        _, _, H_out, W_out = dout.shape

        dout_reshaped = dout.transpose(1, 0, 2, 3).reshape(F, -1)              # (F, N*H_out*W_out)

        self.db = np.sum(dout_reshaped, axis=1)
        self.dW = (dout_reshaped @ cols.T).reshape(self.W.shape)

        dcols = W_matrix.T @ dout_reshaped                                     # (C*KH*KW, N*H_out*W_out)
        dx = col2im(dcols, x.shape, self.KH, self.KW, self.stride, self.padding)
        return dx
