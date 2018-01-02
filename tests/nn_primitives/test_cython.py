import numpy as np
import numpy.testing as npt
from aurora.nn.pyx.fast_pooling import max_pool_forward
from aurora.nn.pyx.fast_pooling import max_pool_backward


# Testing Pooling layers
def test_max_pooling_forward():
    data = np.array([[[[0.12, -1.23, 0.01, 2.45],
                       [5.00, -10.01, 1.09, 4.66],
                       [4.56, 6.78, 3.45, 3.33],
                       [0.01, 1.00, 3.56, 3.39]]]])

    # Test Case: 1
    # filter = (2, 2) stride = (2, 2)
    result = max_pool_forward(data, 2, 2, 2, 2)
    expected = np.array([[[[5.00, 4.66],
                           [6.78, 3.56]]]])
    assert result.shape == expected.shape
    npt.assert_array_almost_equal(result, expected)

    # Test Case: 2
    # filter = (2, 2) stride = (1, 1)
    result = max_pool_forward(data, 2, 2, 1, 1)
    expected = np.array([[
        [[5.00, 1.09, 4.66],
         [6.78, 6.78, 4.66],
         [6.78, 6.78, 3.56]]]])
    assert result.shape == expected.shape
    npt.assert_array_almost_equal(expected, result)

    # Test Case: 3
    # filter = (2, 2), stride = (2, 2)
    shape = (2, 3, 4, 4)
    data = np.linspace(-0.3, 0.4, num=np.prod(shape)).reshape(shape)
    result = max_pool_forward(data, 2, 2, 2, 2)
    expected = np.array([[[[-0.26315789, -0.24842105],
                           [-0.20421053, -0.18947368]],
                          [[-0.14526316, -0.13052632],
                           [-0.08631579, -0.07157895]],
                          [[-0.02736842, -0.01263158],
                           [0.03157895, 0.04631579]]],
                         [[[0.09052632, 0.10526316],
                           [0.14947368, 0.16421053]],
                          [[0.20842105, 0.22315789],
                           [0.26736842, 0.28210526]],
                          [[0.32631579, 0.34105263],
                           [0.38526316, 0.4]]]])
    npt.assert_array_almost_equal(result, expected)


def test_max_pooling_backward():
    data = np.array([[[[0.12, -1.23, 0.01, 2.45],
                       [5.00, -10.01, 1.09, 4.66],
                       [4.56, 6.78, 3.45, 3.33],
                       [0.01, 1.00, 3.56, 3.39]]]])
    output_grad = np.array([[[[1.0, 1.0],
                             [1.0, 1.0]]]])
    # Test Case: 1
    # filter = (2, 2) stride = (2, 2)
    expected = np.array([[[[0.0, 0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0, 1.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0]]]])
    result = max_pool_backward(output_grad, data,
                               filter_height=2, filter_width=2,
                               stride_height=2, stride_width=2)
    npt.assert_array_almost_equal(result, expected)

    # Test Case: 2
    # filter = (2, 2) stride = (2, 2)
    # different output_grad
    output_grad = np.array([[[[0.0, 5.10],
                              [0.12, 0.20]]]])
    expected = np.array([[[[0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 5.10],
                           [0.0, 0.12, 0.0, 0.0],
                           [0.0, 0.0, 0.20, 0.0]]]])
    result = max_pool_backward(output_grad, data,
                               filter_height=2, filter_width=2,
                               stride_height=2, stride_width=2)
    npt.assert_array_almost_equal(result, expected)

    # Test Case: 3
    # filter = (2, 2) stride = (1, 1)
    #output_grad = np.array([[[[1.0, 1.0],
                              #[1.0, 1.0]]]])