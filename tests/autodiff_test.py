import aurora.autodiff as ad
import numpy as np


def test_dummy():
    assert 1 == 1


def test_identity():
    x2 = ad.Variable(name="x2")
    y = x2

    grad_x2, = ad.gradients(y, [x2])

    executor = ad.Executor([y, grad_x2])
    x2_val = 2 * np.ones(3)
    y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})
    print(y_val)
    print(grad_x2_val)

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x2_val)
    assert np.array_equal(grad_x2_val, np.ones_like(x2_val))


def test_add_by_const():
    x2 = ad.Variable(name="x2")
    y = 5 + x2

    grad_x2, = ad.gradients(y, [x2])

    executor = ad.Executor([y, grad_x2])
    x2_val = 2 * np.ones(3)
    y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x2_val + 5)
    assert np.array_equal(grad_x2_val, np.ones_like(x2_val))


def test_mul_by_const():
    x2 = ad.Variable(name='x2')
    y = 3 * x2

    grad_x2, = ad.gradients(y, [x2])
    executor = ad.Executor([y, grad_x2])
    x2_val = 2 * np.ones(3)
    y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})

    # asserts
    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, 3 * x2_val)
    assert np.array_equal(grad_x2_val, 3 * np.ones_like(x2_val))


def test_mul_two_var():
    x2 = ad.Variable(name='x2')
    x3 = ad.Variable(name='x3')
    y = x2 * x3

    grad_x2, grad_x3 = ad.gradients(y, [x2, x3])
    executor = ad.Executor([y, grad_x2, grad_x3])
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict={x2: x2_val, x3: x3_val})

    # asserts
    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, 6 * np.ones(3))
    assert np.array_equal(grad_x2_val, x3_val)
    assert np.array_equal(grad_x3_val, x2_val)


def test_sub_two_vars():
    x2 = ad.Variable(name='x2')
    x3 = ad.Variable(name='x3')
    y = x2 - x3

    grad_x2, grad_x3 = ad.gradients(y, [x2, x3])
    executor = ad.Executor([y, grad_x2, grad_x3])
    x2_val = 4 * np.ones(3)
    x3_val = 3 * np.ones(3)
    y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict={x2: x2_val, x3: x3_val})

    # asserts
    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, 1 * np.ones(3))
    assert np.array_equal(grad_x2_val, np.ones(3))
    assert np.array_equal(grad_x3_val, -1 * np.ones(3))


def test_sub_by_const():
    x2 = ad.Variable(name='x2')
    y = x2 - 3

    grad_x2, = ad.gradients(y, [x2])
    executor = ad.Executor([y, grad_x2])
    x2_val = 2 * np.ones(3)
    y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})

    # asserts
    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, -1 * np.ones(3))
    assert np.array_equal(grad_x2_val, np.ones_like(x2_val))


def test_div_two_var():
    x2 = ad.Variable(name='x2')
    x3 = ad.Variable(name='x3')
    y = x2 / x3

    grad_x2, grad_x3 = ad.gradients(y, [x2, x3])
    executor = ad.Executor([y, grad_x2, grad_x3])
    x2_val = 4 * np.ones(3)
    x3_val = 2 * np.ones(3)
    y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict={x2: x2_val, x3: x3_val})

    # asserts
    assert isinstance(y, ad.Node)
    assert np.array_equal(grad_x2_val, 1.0 / x3_val)
    assert np.array_equal(grad_x3_val, -1.0 * x2_val / (x3_val * x3_val))


def test_div_by_const():
    x2 = ad.Variable(name='x2')
    y = x2 / 2.0

    grad_x2, = ad.gradients(y, [x2])
    executor = ad.Executor([y, grad_x2])
    x2_val = 2 * np.ones(3)
    y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})

    # asserts
    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x2_val / 2.0)
    assert np.array_equal(grad_x2_val, np.ones_like(x2_val) / 2.0)


def test_reduce_sum():
    x2 = ad.Variable(name='x2')
    y = ad.reduce_sum(x2)

    grad_x2, = ad.gradients(y, [x2])
    executor = ad.Executor([y, grad_x2])
    x2_val = np.array([[1, 2, 3], [4, 5, 6]])
    y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})

    # asserts
    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, np.array([5, 7, 9]))
    assert np.array_equal(grad_x2_val, np.array([1, 1, 1]))


def test_reduce_sum():
    x2 = ad.Variable(name='x2')
    x3 = ad.Variable(name='x3')
    y = ad.broadcast_to(x2, x3)

    grad_x2, grad_x3 = ad.gradients(y, [x2, x3])
    executor = ad.Executor([y, grad_x2, grad_x3])
    x2_val = np.array([[1, 2, 3]])
    x3_val = np.zeros((3, 3))
    y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict={x2: x2_val, x3: x3_val})

    # asserts
    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]))
    assert np.array_equal(grad_x2_val, np.array([3, 3, 3]))


def test_matmul_two_vars():
    x2 = ad.Variable(name='x2')
    x3 = ad.Variable(name='x3')
    y = ad.matmul(x2, x3)

    grad_x2, grad_x3 = ad.gradients(y, [x2, x3])
    executor = ad.Executor([y, grad_x2, grad_x3])
    x2_val = np.array([[1, 2], [3, 4], [5, 6]])  # 3x2
    x3_val = np.array([[7, 8, 9], [10, 11, 12]])  # 2x3

    y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict={x2: x2_val, x3: x3_val})

    expected_yval = np.matmul(x2_val, x3_val)
    expected_grad_x2_val = np.matmul(np.ones_like(expected_yval), np.transpose(x3_val))
    expected_grad_x3_val = np.matmul(np.transpose(x2_val), np.ones_like(expected_yval))

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, expected_yval)
    assert np.array_equal(grad_x2_val, expected_grad_x2_val)
    assert np.array_equal(grad_x3_val, expected_grad_x3_val)

def test_relu_test():
    x2 = ad.Variable(name='x2')
    y = ad.relu(x2)

    grad_x2, = ad.gradients(y, [x2])
    executor = ad.Executor([y, grad_x2])
    x2_val = np.array([[-1, 2, 3], [1, -2, 0]])
    y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})
    expected_y_val = np.array([[0, 2, 3], [1, 0, 0]])
    expected_x2_grad = np.array([[0, 1, 1], [1, 0, 0]])
    assert np.array_equal(y_val, expected_y_val)
    assert np.array_equal(grad_x2_val, expected_x2_grad)
