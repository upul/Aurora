import numpy as np
from .executor import Executor


def eval_numerical_grad(f, feed_dict, wrt, h=1e-5):
    wrt_val = feed_dict[wrt]
    grad = np.zeros_like(wrt_val)

    it = np.nditer(wrt_val, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old_val = wrt_val[ix]
        wrt_val[ix] = old_val + h
        executor = Executor([f])
        feed_dict[wrt] = wrt_val

        result_plus, = executor.run(feed_shapes=feed_dict)
        wrt_val[ix] = old_val - h
        executor = Executor([f])

        feed_dict[wrt] = wrt_val
        result_minus, = executor.run(feed_shapes=feed_dict)

        grad[ix] = np.sum((result_plus - result_minus) / (2.0 * h))

        wrt_val[ix] = old_val
        feed_dict[wrt] = wrt_val
        it.iternext()
    return grad
