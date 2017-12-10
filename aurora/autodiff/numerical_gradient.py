import numpy as np
from .executor import Executor


def eval_numerical_grad(f, feed_dict, wrt, h=1e-5):
    grad = np.zeros_like(wrt)
    it = np.nditer(wrt, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old_val = wrt[ix]
        wrt[ix] = old_val + h
        executor = Executor([f])
        result_plus, = executor.run(feed_shapes=feed_dict)

        wrt[ix] = old_val - h
        executor = Executor([f])
        result_minus, = executor.run(feed_shapes=feed_dict)
        grad[ix] = np.sum((result_plus - result_minus) / (2 * h))
        wrt[ix] = old_val
        it.iternext()
    return grad
