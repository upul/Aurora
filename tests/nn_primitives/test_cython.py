import numpy as np
import numpy.testing as npt
from aurora.nn.pyx.fast_pooling import max_pool_forward


# Testing Pooling layers
def test_max_pooling_forward():
    # case 1
    data = np.array([
        [
            [[0.12, -1.23, 0.01, 2.45],
             [5.00, -10.01, 1.09, 4.66],
             [4.56, 6.78, 3.45, 3.33],
             [0.01, 1.00, 3.56, 3.39]]
        ]
    ])
    # filter = (2, 2) stride = (2, 2)
    result = max_pool_forward(data, 2, 2, 2, 2)
    expected = np.array([
        [
            [[5.00, 4.66],
             [6.78, 3.56]]
        ]
    ])
    assert result.shape == expected.shape
    npt.assert_array_almost_equal(result, expected)

    # filter = (2, 2) stride = (1, 1)
    result = max_pool_forward(data, 2, 2, 1, 1)
    expected = np.array([
        [
            [[5.00, 1.09, 4.66],
             [6.78, 6.78, 4.66],
             [6.78, 6.78, 3.56]]
        ]
    ])
    assert result.shape == expected.shape
    npt.assert_array_almost_equal(expected, result)


def test_max_pooling_backward():
    pass
