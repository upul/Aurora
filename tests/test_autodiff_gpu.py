import aurora as au
import aurora.autodiff as ad
import numpy as np
import numpy.testing as npt
from aurora.ndarray import ndarray, gpu_op


def test_identity():
    x2 = ad.Variable(name='x2')
    y = x2

    grad_x2, = ad.gradients(y, [x2])

    executor = ad.Executor([y, grad_x2], use_gpu=True)
    x2_val = 2 * np.ones(3)
    y_val, grad_x2_val = executor.run(feed_shapes={x2: x2_val})

    y_val_np = y_val.asnumpy()
    grad_x2_val_np = grad_x2_val.asnumpy()

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val_np, x2_val)
    assert np.array_equal(grad_x2_val_np, np.ones_like(x2_val))


def test_add_by_const():
    x2 = ad.Variable(name="x2")
    y = 5 + x2

    grad_x2, = ad.gradients(y, [x2])

    executor = ad.Executor([y, grad_x2], use_gpu=True)
    x2_val = 2 * np.ones(3)
    y_val, grad_x2_val = executor.run(feed_shapes={x2: x2_val})

    y_val = y_val.asnumpy()
    grad_x2_val = grad_x2_val.asnumpy()

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x2_val + 5)
    assert np.array_equal(grad_x2_val, np.ones_like(x2_val))


def test_softmax():
    ctx = ndarray.gpu(0)
    shape = (2, 2)
    x_val = np.random.uniform(-5, 5, shape).astype(np.float32)

    x2 = ad.Variable(name="x2")
    prob = au.nn.softmax(x2)
    executor = ad.Executor([prob], use_gpu=True)
    y, = executor.run(feed_shapes={x2: x_val})
    y = y.asnumpy()

    # arr_x = ndarray.array(x, ctx=ctx)
    # arr_y = ndarray.empty(shape, ctx=ctx)
    # gpu_op.softmax(arr_x, arr_y)
    # y = arr_y.asnumpy()
    np.testing.assert_allclose(au.nn.softmax_func(x_val), y, rtol=1e-5)
