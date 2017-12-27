import numpy as np


def softmax_func(x):
    """
    Numerically stable softmax function. For more details
    about numerically calculations please refer:
    http://www.deeplearningbook.org/slides/04_numerical.pdf
    :param x:
    :return:
    """
    stable_values = x - np.max(x, axis=1, keepdims=True)
    return np.exp(stable_values) / np.sum(np.exp(stable_values), axis=1, keepdims=True)


def log_sum_exp(x):
    """
    log_sum_exp is a very useful function in machine learning.
    It can be seen in many places including cross-entropy error.
    However, the naive implementation is numerically unstable.
    Therefore, we use the following implementation. For more details
    please refer: http://www.deeplearningbook.org/slides/04_numerical.pdf
    :param x:
    :return:
    """
    mx = np.max(x, axis=1, keepdims=True)
    safe = x - mx
    return mx + np.log(np.sum(np.exp(safe), axis=1, keepdims=True))


# TODO: (upul) replace im2col and col2im with high-performance Cython implementations.
def im2col(image, filter_size=(3, 3), padding=(0, 0), stride=(1, 1)):
    M, C, h, w, = image.shape
    filter_height = filter_size[0]
    filter_width = filter_size[1]
    padding_height = padding[0]
    padding_width = padding[1]
    stride_height = stride[0]
    stride_width = stride[1]
    x_padded = np.pad(image, ((0, 0),
                              (0, 0),
                              (padding_height, padding_height),
                              (padding_width, padding_width)),
                      mode='constant')
    h_new = int((h - filter_height + 2 * padding_height) / stride_height + 1)
    w_new = int((w - filter_width + 2 * padding_width) / stride_width + 1)

    out = np.zeros((filter_width * filter_height * C, M * h_new * w_new), dtype=image.dtype)

    itr = 0
    for i in range(h_new):
        for j in range(w_new):
            for m in range(M):
                start_i = stride_height * i
                end_i = stride_height * i + filter_width
                start_j = stride_width * j
                end_j = stride_width * j + filter_height
                out[:, itr] = x_padded[m, :, start_i:end_i, start_j:end_j].ravel()
                itr += 1
    return out


def col2im(cols, x_shape, filter_size=(3, 3), padding=(0, 0), stride=(1, 1)):
    N, C, H, W = x_shape
    filter_height = filter_size[0]
    filter_width = filter_size[1]
    padding_height = padding[0]
    padding_width = padding[1]
    stride_height = stride[0]
    stride_width = stride[1]

    H_padded, W_padded = H + 2 * padding_height, W + 2 * padding_width
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)

    idx = 0
    for i in range(0, H_padded - filter_height + 1, stride_height):
        for j in range(0, W_padded - filter_width + 1, stride_width):
            for m in range(N):
                col = cols[:, idx]
                col = col.reshape((C, filter_height, filter_width))
                x_padded[m, :, i:i + filter_height, j:j + filter_width] += col
                idx += 1
    if padding[0] or padding[1] > 0:
        return x_padded[:, :, padding_height:-padding_height, padding_width:-padding_width]
    else:
        return x_padded
