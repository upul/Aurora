import aurora as au
import aurora.autodiff as ad
import numpy as np
import numpy.testing as npt


def test_dummy():
    assert 1 == 1


def test_identity():
    x2 = ad.Variable(name="x2")
    y = x2

    grad_x2, = ad.gradients(y, [x2])

    executor = ad.Executor([y, grad_x2])
    x2_val = 2 * np.ones(3)
    y_val, grad_x2_val = executor.run(feed_shapes={x2: x2_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x2_val)
    assert np.array_equal(grad_x2_val, np.ones_like(x2_val))


def test_add_by_const():
    x2 = ad.Variable(name="x2")
    y = 5 + x2

    grad_x2, = ad.gradients(y, [x2])

    executor = ad.Executor([y, grad_x2])
    x2_val = 2 * np.ones(3)
    y_val, grad_x2_val = executor.run(feed_shapes={x2: x2_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x2_val + 5)
    assert np.array_equal(grad_x2_val, np.ones_like(x2_val))


def test_mul_by_const():
    x2 = ad.Variable(name='x2')
    y = 3 * x2

    grad_x2, = ad.gradients(y, [x2])
    executor = ad.Executor([y, grad_x2])
    x2_val = 2 * np.ones(3)
    y_val, grad_x2_val = executor.run(feed_shapes={x2: x2_val})

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
    y_val, grad_x2_val, grad_x3_val = executor.run(feed_shapes={x2: x2_val, x3: x3_val})

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
    y_val, grad_x2_val, grad_x3_val = executor.run(feed_shapes={x2: x2_val, x3: x3_val})

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
    y_val, grad_x2_val = executor.run(feed_shapes={x2: x2_val})

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
    y_val, grad_x2_val, grad_x3_val = executor.run(feed_shapes={x2: x2_val, x3: x3_val})

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
    y_val, grad_x2_val = executor.run(feed_shapes={x2: x2_val})

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
    y_val, grad_x2_val = executor.run(feed_shapes={x2: x2_val})

    # asserts
    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, np.array([5, 7, 9]))
    assert np.array_equal(grad_x2_val, np.array([1, 1, 1]))


def test_broadcast_to():
    x2 = ad.Variable(name='x2')
    x3 = ad.Variable(name='x3')
    y = ad.broadcast_to(x2, x3)

    grad_x2, grad_x3 = ad.gradients(y, [x2, x3])
    executor = ad.Executor([y, grad_x2, grad_x3])
    x2_val = np.array([[1, 2, 3]])
    x3_val = np.zeros((3, 3))
    y_val, grad_x2_val, grad_x3_val = executor.run(feed_shapes={x2: x2_val, x3: x3_val})

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

    y_val, grad_x2_val, grad_x3_val = executor.run(feed_shapes={x2: x2_val, x3: x3_val})

    expected_yval = np.matmul(x2_val, x3_val)
    expected_grad_x2_val = np.matmul(np.ones_like(expected_yval), np.transpose(x3_val))
    expected_grad_x3_val = np.matmul(np.transpose(x2_val), np.ones_like(expected_yval))

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, expected_yval)
    assert np.array_equal(grad_x2_val, expected_grad_x2_val)
    assert np.array_equal(grad_x3_val, expected_grad_x3_val)


def test_relu():
    x2 = ad.Variable(name='x2')
    y = au.nn.relu(x2)

    grad_x2, = ad.gradients(y, [x2])
    executor = ad.Executor([y, grad_x2])
    x2_val = np.array([[-1, 2, 3], [1, -2, 0]])
    y_val, grad_x2_val = executor.run(feed_shapes={x2: x2_val})
    expected_y_val = np.array([[0, 2, 3], [1, 0, 0]])
    expected_x2_grad = np.array([[0, 1, 1], [1, 0, 0]])
    assert np.array_equal(y_val, expected_y_val)
    assert np.array_equal(grad_x2_val, expected_x2_grad)


def test_cross_entropy():
    x2_pred = ad.Variable(name='x2_pred')
    x2_actu = ad.Variable(name='x2_actu')
    y = au.nn.cross_entropy_with_logits(x2_pred, x2_actu)

    x2_pred_grad, x2_actu_grad = ad.gradients(y, [x2_pred, x2_actu])

    x2_pred_val = np.array([[0.8, 0.01, 0.5], [0.8, 0.01, 0.5]])
    x2_actu_val = np.array([[1.0, 1.0, 0], [1.0, 1.0, 0]])

    executor = ad.Executor([y, x2_pred_grad, x2_actu_grad])
    y_val, x2_pred_grad_val, x2_actu_grad_val = executor.run(feed_shapes={x2_pred: x2_pred_val, x2_actu: x2_actu_val})
    # print(x2_actu_grad_val)
    assert True


def test_matmul_var_and_param():
    x2 = ad.Variable(name="x2")
    w2_val = np.array([[7, 8, 9], [10, 11, 12]])  # 2x3
    w2 = ad.Parameter(name="w2", init=w2_val)
    y = ad.matmul(x2, w2)

    grad_x2, grad_w2 = ad.gradients(y, [x2, w2])

    executor = ad.Executor([y, grad_x2, grad_w2])
    x2_val = np.array([[1, 2], [3, 4], [5, 6]])  # 3x2

    y_val, grad_x2_val, grad_w2_val = executor.run(feed_shapes={x2: x2_val})

    expected_yval = np.matmul(x2_val, w2_val)
    expected_grad_x2_val = np.matmul(np.ones_like(expected_yval), np.transpose(w2_val))
    expected_grad_x3_val = np.matmul(np.transpose(x2_val), np.ones_like(expected_yval))

    assert isinstance(y, ad.Node)
    # assert np.array_equal(y_val, expected_yval)
    # assert np.array_equal(grad_x2_val, expected_grad_x2_val)
    # assert np.array_equal(grad_w2_val, expected_grad_x3_val)


def test_sigmoid_activation():
    x2 = ad.Variable(name='x2')
    y = au.nn.sigmoid(x2)

    x2_val = np.array([-100, 0, 100])
    grad_x2, = ad.gradients(y, [x2])
    executor = ad.Executor([y, grad_x2])
    y_val, grad_x2_val = executor.run(feed_shapes={x2: x2_val})
    npt.assert_array_almost_equal(np.array([0.000, 0.500, 1.0]), y_val)
    npt.assert_array_almost_equal(np.array([0, 0.25, 0]), grad_x2_val)


def test_conv2d():
    x2 = ad.Variable(name='x2')
    w2 = ad.Variable(name='w2')
    b2 = ad.Variable(name='b2')

    y = au.nn.conv2d(input=x2, filter=w2, bias=b2)

    grad_x2, grad_w2, grad_b2 = ad.gradients(y, [x2, w2, b2])
    executor = ad.Executor([y, grad_x2, grad_w2, grad_b2])
    x2_val = np.random.randn(1, 2, 4, 4)
    w2_val = np.random.randn(2, 2, 3, 3)
    b2_val = np.random.randn(2,)

    y_val, grad_x2_val, grad_w2_val, grad_b2_val = executor.run(feed_shapes={x2: x2_val,
                                                                             w2: w2_val,
                                                                             b2: b2_val})

    numerical_grad_w2 = ad.eval_numerical_grad(y,
                                               feed_dict={x2: x2_val,
                                                          w2: w2_val,
                                                          b2: b2_val},
                                               wrt=w2_val)
    numerical_grad_x2 = ad.eval_numerical_grad(y,
                                               feed_dict={x2: x2_val,
                                                          w2: w2_val,
                                                          b2: b2_val},
                                               wrt=x2_val)
    numerical_grad_b2 = ad.eval_numerical_grad(y,
                                               feed_dict={x2: x2_val,
                                                          w2: w2_val,
                                                          b2: b2_val},
                                               wrt=b2_val)

    assert isinstance(y, ad.Node)
    npt.assert_array_almost_equal(numerical_grad_w2, grad_w2_val)
    npt.assert_array_almost_equal(numerical_grad_x2, grad_x2_val)
    npt.assert_array_almost_equal(numerical_grad_b2, grad_b2_val)

    x2 = ad.Variable(name='x2')
    w2 = ad.Parameter(name='w2', init=w2_val)
    b2 = ad.Parameter(name='b2', init=b2_val)
    y = au.nn.conv2d(x2, w2, b2)

    grad_x2, grad_w2, grad_b2 = ad.gradients(y, [x2, w2, b2])
    executor = ad.Executor([y, grad_x2, grad_w2, grad_b2])
    y_val, grad_x2_val, grad_w2_val, grad_b2_val = executor.run(feed_shapes={x2: x2_val})

    assert isinstance(y, ad.Node)
    npt.assert_array_almost_equal(numerical_grad_w2, grad_w2_val)
    npt.assert_array_almost_equal(numerical_grad_b2, grad_b2_val)
    npt.assert_array_almost_equal(numerical_grad_x2, grad_x2_val)


def test_reshape():
    x2 = ad.Variable(name='x2')
    y = ad.reshape(x2, newshape=(1, 4))

    grad_x2, = ad.gradients(y, [x2])
    executor = ad.Executor([y, grad_x2])
    x2_val = np.random.randn(2, 2)
    y_val, grad_x2_val = executor.run(feed_shapes={x2: x2_val})

    assert isinstance(y, ad.Node)
    assert y_val.shape == (1, 4)
    npt.assert_array_equal(grad_x2_val, np.ones((2, 2)))

    # x2 = ad.Variable(name='x2')
    # y = ad.reshape(x2, newshape=(2, 1, 2, 3))
    # grad_x2, = ad.gradients(y, [x2])
    # executor = ad.Executor([y, grad_x2])
    # x2_val = np.random.randn(2, 6)
    # y_val, grad_x2_val = executor.run(feed_shapes={x2: x2_val})
    #
    # assert isinstance(y, ad.Node)
    # assert y_val.shape == (2, 1, 2, 3)
    # npt.assert_array_equal(grad_x2_val, np.ones((2, 1, 2, 3)))
