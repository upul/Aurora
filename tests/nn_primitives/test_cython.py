import numpy as np
import numpy.testing as npt
from aurora.nn.pyx.fast_pooling import max_pool_forward
from aurora.nn.pyx.fast_pooling import max_pool_backward
from aurora.nn.pyx.im2col import im2col
from aurora.nn.pyx.im2col import col2im
from tests.utils.gradient_check import gradient_check_numpy_expr


# Testing Max Pooling Layers

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

    # calculate numerical gradient
    numerical = gradient_check_numpy_expr(lambda d: max_pool_forward(d, 2, 2, 2, 2), data, output_grad)
    npt.assert_array_almost_equal(numerical, expected, decimal=3)

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

    # calculate numerical gradient
    numerical = gradient_check_numpy_expr(lambda d: max_pool_forward(d, 2, 2, 2, 2), data, output_grad)
    npt.assert_array_almost_equal(numerical, expected, decimal=2)

    # Test Case: 3
    # filter = (2, 2) stride = (1, 1)
    output_grad = np.array([[[[1.0, 1.0, 1.0],
                              [1.0, 1.0, 1.0],
                              [1.0, 1.0, 1.0]]]])
    result = max_pool_backward(output_grad, data,
                               filter_height=2, filter_width=2,
                               stride_height=1, stride_width=1)
    numerical = gradient_check_numpy_expr(lambda x: max_pool_forward(x, 2, 2, 1, 1), data, output_grad)
    npt.assert_array_almost_equal(numerical, result, decimal=2)

    # Test Case: 4
    # filter = (2, 2) stride = (2, 2)
    # input shape = (2, 2, 6, 6)
    data = np.random.normal(scale=0.01, size=(2, 2, 6, 6))
    output_grad = np.ones((2, 2, 3, 3))
    result = max_pool_backward(output_grad, data,
                               filter_height=2, filter_width=2,
                               stride_height=2, stride_width=2)
    numerical = gradient_check_numpy_expr(lambda d: max_pool_forward(d, 2, 2, 2, 2), data, output_grad)
    npt.assert_array_almost_equal(numerical, result, decimal=4)


# Testing Image to Column operations
def test_im2col():
    data = np.arange(16).reshape((1, 1, 4, 4)).astype(np.float64)
    # one image in the batch 2 by 2 kernel with stride = 1
    result = im2col(data, filter_height=2, filter_width=2,
                    padding_height=0, padding_width=0,
                    stride_height=1, stride_width=1)

    expected = np.array([[0, 1, 2, 4, 5, 6, 8, 9, 10],
                         [1, 2, 3, 5, 6, 7, 9, 10, 11],
                         [4, 5, 6, 8, 9, 10, 12, 13, 14],
                         [5, 6, 7, 9, 10, 11, 13, 14, 15]]).astype(np.float64)
    npt.assert_array_almost_equal(result, expected)

    # one image in the batch 2 by 2 kernel with stride = 2
    result = im2col(data, filter_height=2, filter_width=2,
                    padding_height=0, padding_width=0,
                    stride_height=2, stride_width=2)
    expected = np.array([[0, 2, 8, 10],
                         [1, 3, 9, 11],
                         [4, 6, 12, 14],
                         [5, 7, 13, 15]]).astype(np.float64)
    npt.assert_array_almost_equal(result, expected)

    # one image in the batche 2 by 2 kernel with stride = 1 and padding  = 1
    data = np.arange(9).reshape(1, 1, 3, 3).astype(np.float64)
    result = im2col(data, filter_height=2, filter_width=2,
                    padding_height=1, padding_width=1,
                    stride_height=1, stride_width=1)
    expected = np.array([[0, 0, 0, 0, 0, 0, 1, 2, 0, 3, 4, 5, 0, 6, 7, 8],
                         [0, 0, 0, 0, 0, 1, 2, 0, 3, 4, 5, 0, 6, 7, 8, 0],
                         [0, 0, 1, 2, 0, 3, 4, 5, 0, 6, 7, 8, 0, 0, 0, 0],
                         [0, 1, 2, 0, 3, 4, 5, 0, 6, 7, 8, 0, 0, 0, 0, 0]]).astype(np.float64)
    npt.assert_array_almost_equal(result, expected)

    # more than one color channels
    # kernel 2 by 2 stride = 1
    data = np.arange(18).reshape(1, 2, 3, 3).astype(np.float64)
    result = im2col(data, filter_height=2, filter_width=2,
                    padding_height=0, padding_width=0,
                    stride_height=1, stride_width=1)
    expected = np.array([[0, 1, 3, 4],
                         [1, 2, 4, 5],
                         [3, 4, 6, 7],
                         [4, 5, 7, 8],
                         [9, 10, 12, 13],
                         [10, 11, 13, 14],
                         [12, 13, 15, 16],
                         [13, 14, 16, 17]])
    npt.assert_array_almost_equal(result, expected)

    # more than one batch and color chennel
    # kernel 2 by 2 with stride of 1
    data = np.arange(36).reshape(2, 2, 3, 3).astype(np.float64)
    result = im2col(data, filter_height=2, filter_width=2,
                    padding_height=0, padding_width=0,
                    stride_height=1, stride_width=1)
    expected = np.array([[0, 18, 1, 19, 3, 21, 4, 22],
                         [1, 19, 2, 20, 4, 22, 5, 23],
                         [3, 21, 4, 22, 6, 24, 7, 25],
                         [4, 22, 5, 23, 7, 25, 8, 26],
                         [9, 27, 10, 28, 12, 30, 13, 31],
                         [10, 28, 11, 29, 13, 31, 14, 32],
                         [12, 30, 13, 31, 15, 33, 16, 34],
                         [13, 31, 14, 32, 16, 34, 17, 35]]).astype(np.float64)
    print(np.array(result))
    npt.assert_array_almost_equal(result, expected)

    # TODO: (upul) test several kernel sizes and different stride, kernel size and padding
    #     : in different directions


def test_col2im():
    # batch size 1, color channels 1, 3 by 3 image. Stride 1, filter 2 by 2 and no padding
    data = np.arange(9).reshape((1, 1, 3, 3)).astype(np.float64)
    i2c_result = im2col(data, filter_height=2, filter_width=2,
                        padding_height=0, padding_width=0,
                        stride_height=1, stride_width=1)
    result = col2im(i2c_result, 1, 1, 3, 3,
                    2, 2,
                    0, 0,
                    1, 1)
    expected = np.array([[[[0., 2., 2.],
                           [6., 16., 10.],
                           [6., 14., 8.]]]]).astype(np.float64)
    npt.assert_array_almost_equal(result, expected)

    # batch size 1, color channels 1, 4 by 4 image. Stride 2, filter 2 by 2 and no padding
    data = np.arange(16).reshape((1, 1, 4, 4)).astype(np.float64)
    i2c_result = im2col(data, filter_height=2, filter_width=2,
                        padding_height=0, padding_width=0,
                        stride_height=2, stride_width=2)
    result = col2im(i2c_result,
                    1, 1,  # batch size and color channels
                    4, 4,  # img width and height
                    2, 2,  # kernel
                    0, 0,  # padding
                    2, 2)  # stride
