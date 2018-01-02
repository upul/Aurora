import numpy as np


def gradient_check_numpy_expr(func, x, output_gradient, h=1e-5):
    """
    This utility function calculates gradient of the function `func`
    at `x`.
    :param func:
    :param x:
    :param output_gradient:
    :param h:
    :return:
    """
    grad = np.zeros_like(x).astype(np.float32)
    iter = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not iter.finished:
        idx = iter.multi_index
        old_value = x[idx]

        # calculate positive value
        x[idx] = old_value + h
        pos = func(x).copy()

        # calculate negative value
        x[idx] = old_value - h
        neg = func(x).copy()

        # restore
        x[idx] = old_value

        # calculate gradient
        # Type of pos and neg will be memoryview if we are testing Cython functions.
        # Therefore, we create numpy arrays be performing - operation.
        # TODO: Don't we have an alternative method without creating numpy array from memoryview?
        grad[idx] = np.sum((np.array(pos) - np.array(neg)) * output_gradient) / (2 * h)
        iter.iternext()

    return grad
