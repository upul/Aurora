import numpy as np

def gradient_check_numpy_expr(func, x, dout, h=1e-5):
    """

    :param func:
    :param x:
    :param dout:
    :param h:
    :return:
    """
    grad = np.zeros_like(x)
    iter = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not iter.finished:
        idx = iter.multi_index
        old_value = x[idx]
        x[idx] = old_value + h
        pos = func(x).copy()

        x[idx] = old_value - h
        neg = func(x).copy()

