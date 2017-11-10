import aurora as au
import aurora.autodiff as ad
import numpy as np
import numpy.testing as npt
from aurora.ndarray import ndarray, gpu_op


def test_dummy():
    assert 1 == 1

def test_array_set():
    ctx = ndarray.gpu(0)
    shape = (5000, 2000)
    # oneslike
    arr_x = ndarray.empty(shape, ctx=ctx)
    gpu_op.array_set(arr_x, 1.)
    x = arr_x.asnumpy()
    np.testing.assert_allclose(np.ones(shape), x)
    # zeroslike
    gpu_op.array_set(arr_x, 0.)
    x = arr_x.asnumpy()
    np.testing.assert_allclose(np.zeros(shape), x)


def test_broadcast_to():
    ctx = ndarray.gpu(0)
    shape = (200, 300)
    to_shape = (130, 200, 300)
    x = np.random.uniform(-1, 1, shape).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.empty(to_shape, ctx=ctx)
    gpu_op.broadcast_to(arr_x, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(np.broadcast_to(x, to_shape), y)


def test_reduce_sum_axis_zero():
    ctx = ndarray.gpu(0)
    shape = (500, 200, 100)
    to_shape = (200, 100)
    x = np.random.uniform(0, 20, shape).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.empty(to_shape, ctx=ctx)
    gpu_op.reduce_sum_axis_zero(arr_x, arr_y)
    y = arr_y.asnumpy()
    y_ = np.sum(x, axis=0)
    for index, _ in np.ndenumerate(y):
        v = y[index]
        v_ = y_[index]
        if abs((v - v_) / v_) > 1e-4:
            print(index, v, v_)
    np.testing.assert_allclose(np.sum(x, axis=0), y, rtol=1e-5)


def test_matrix_elementwise_add():
    ctx = ndarray.gpu(0)
    shape = (5000, 2000)
    x = np.random.uniform(0, 10, size=shape).astype(np.float32)
    y = np.random.uniform(0, 10, size=shape).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.array(y, ctx=ctx)
    arr_z = ndarray.empty(shape, ctx=ctx)
    gpu_op.matrix_elementwise_add(arr_x, arr_y, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(x + y, z, rtol=1e-5)


def test_matrix_elementwise_add_by_const():
    shape = (2000, 3000)
    ctx = ndarray.gpu(0)
    x = np.random.uniform(0, 10, size=shape).astype(np.float32)
    val = np.random.uniform(-5, 5)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.empty(shape, ctx=ctx)
    gpu_op.matrix_elementwise_add_by_const(arr_x, val, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(x + val, y, rtol=1e-5)


def test_matrix_elementwise_multiply():
    ctx = ndarray.gpu(0)
    shape = (500, 200)
    x = np.random.uniform(0, 10, size=shape).astype(np.float32)
    y = np.random.uniform(0, 10, size=shape).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.array(y, ctx=ctx)
    arr_z = ndarray.empty(shape, ctx=ctx)
    gpu_op.matrix_elementwise_multiply(arr_x, arr_y, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(x * y, z, rtol=1e-5)


def test_matrix_elementwise_multiply_by_const():
    shape = (2000, 3000)
    ctx = ndarray.gpu(0)
    x = np.random.uniform(0, 10, size=shape).astype(np.float32)
    val = np.random.uniform(-5, 5)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.empty(shape, ctx=ctx)
    gpu_op.matrix_elementwise_multiply_by_const(arr_x, val, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(x * val, y, rtol=1e-5)


def test_matrix_multiply():
    ctx = ndarray.gpu(0)
    x = np.random.uniform(0, 10, size=(500, 700)).astype(np.float32)
    y = np.random.uniform(0, 10, size=(700, 1000)).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.array(y, ctx=ctx)
    arr_z = ndarray.empty((500, 1000), ctx=ctx)
    gpu_op.matrix_multiply(arr_x, False, arr_y, False, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(np.dot(x, y), z, rtol=1e-5)

    x = np.random.uniform(0, 10, size=(1000, 500)).astype(np.float32)
    y = np.random.uniform(0, 10, size=(2000, 500)).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.array(y, ctx=ctx)
    arr_z = ndarray.empty((1000, 2000), ctx=ctx)
    gpu_op.matrix_multiply(arr_x, False, arr_y, True, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(np.dot(x, np.transpose(y)), z, rtol=1e-5)

    x = np.random.uniform(0, 10, size=(500, 1000)).astype(np.float32)
    y = np.random.uniform(0, 10, size=(2000, 500)).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.array(y, ctx=ctx)
    arr_z = ndarray.empty((1000, 2000), ctx=ctx)
    gpu_op.matrix_multiply(arr_x, True, arr_y, True, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(np.dot(np.transpose(x), np.transpose(y)), z,
                               rtol=1e-5)


def test_relu():
    shape = (2000, 2500)
    ctx = ndarray.gpu(0)
    x = np.random.uniform(-1, 1, shape).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.empty(shape, ctx=ctx)
    gpu_op.relu(arr_x, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(np.maximum(x, 0).astype(np.float32), y)


def test_relu_gradient():
    shape = (2000, 2500)
    ctx = ndarray.gpu(0)
    x = np.random.uniform(-1, 1, shape).astype(np.float32)
    grad_x = np.random.uniform(-5, 5, shape).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_grad_x = ndarray.array(grad_x, ctx=ctx)
    arr_y = ndarray.empty(shape, ctx=ctx)
    gpu_op.relu_gradient(arr_x, arr_grad_x, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(((x > 0) * grad_x).astype(np.float32), y)
