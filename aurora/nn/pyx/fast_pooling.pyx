cimport cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def max_pool_forward(np.float64_t[:, :, :, :] data,
                     int filter_height, int filter_width,
                     int stride_height, int stride_width):
    """

    :param data:
    :param filter_height:
    :param filter_width:
    :param stride_height:
    :param stride_width:
    :return:
    """

    cdef int batch_size = data.shape[0]
    cdef int input_channels = data.shape[1]
    cdef int height = data.shape[2]
    cdef int width = data.shape[3]

    # Define the dimensions of the output
    cdef int n_H = int(1 + (height - filter_height) / stride_height)
    cdef int n_W = int(1 + (width - filter_width) / stride_width)
    cdef int n_C = input_channels

    # Initialize output matrix
    cdef np.float64_t[:, :, :, :] output = np.zeros((batch_size, n_C, n_H, n_W))

    cdef int i, c, h, w
    cdef int vert_start, vert_end, horiz_start, horiz_end

    cdef float max_in_grid = -1e20
    cdef int ii, jj

    for i in range(batch_size):       # loop over the training examples
        for c in range (n_C):         # loop over the channels of the output volume
            for h in range(n_H):      # loop on the vertical axis of the output volume
                for w in range(n_W):  # loop on the horizontal axis of the output volume

                    # Find the corners of the current "slice"
                    vert_start = h*stride_height
                    vert_end = h*stride_height + filter_height
                    horiz_start = w*stride_width
                    horiz_end = w*stride_width + filter_width

                    max_in_grid = -1e20
                    for ii in range(vert_start, vert_end):
                        for jj in range(horiz_start, horiz_end):
                            if data[i, c, ii, jj] > max_in_grid:
                                max_in_grid = data[i, c, ii, jj]
                    output[i, c, h, w] = max_in_grid
    return output

@cython.boundscheck(False)
@cython.wraparound(False)
def max_pool_backward(np.float64_t[:, :, :, :] output_grad,
                      np.float64_t[:, :, :, :] input_data,
                      int filter_height=2, int filter_width=2,
                      int stride_height=2, int stride_width=2):
    """

    :param output_grad:
    :param input_data:
    :param filter_height:
    :param filter_width:
    :param stride_height:
    :param stride_width:
    :return:
    """
    batch_size = output_grad.shape[0]
    channels = output_grad.shape[1]
    height = output_grad.shape[2]
    width = output_grad.shape[3]

    return _max_pool_backward_inner(output_grad, input_data,
                                   batch_size, channels,height,
                                   width, filter_height,
                                   filter_width, stride_height,
                                   stride_width)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _max_pool_backward_inner(np.float64_t[:, :, :, :] output_grad,
                             np.float64_t[:, :, :, :] input_data,
                             int batch_size, int
                             channels,
                             int height, int width,
                             int filter_height, int filter_width,
                             int stride_height, int stride_width):
    """
    
    :param output_grad: 
    :param input_data: 
    :param batch_size: 
    :param channels: 
    :param height: 
    :param width: 
    :param filter_height: 
    :param filter_width: 
    :param stride_height: 
    :param stride_width: 
    :return: 
    """

    grad_input = np.zeros_like(input_data)

    cdef np.float64_t[:, :, :]  cct_example
    cdef int h, w, c, vert_start, vert_end, horiz_start, horiz_end

    cdef int slice_height, slice_width
    cdef int max_i, max_j
    cdef float max_value
    cdef float cct_value

    # loop over the training examples
    for i in range(batch_size):

        # pick the current training example
        cct_example = input_data[i, :, :, :]

        for h in range(height):             # loop on the vertical axis
            for w in range(width):          # loop on the horizontal axis
                for c in range(channels):   # loop over the channels (depth)

                    # Find the corners of the current slice.
                    vert_start = h*stride_height
                    vert_end = h*stride_height + filter_height
                    horiz_start = w*stride_width
                    horiz_end = w*stride_width + filter_width

                    # Compute the backward propagation in both modes.
                    max_value = -1.0e20
                    for slice_height in range(vert_start, vert_end):
                        for slice_width in range(horiz_start, horiz_end):
                            cct_value = cct_example[c, slice_height, slice_width]
                            if cct_value > max_value:
                                max_value = cct_value
                                max_i = slice_height
                                max_j = slice_width
                    grad_input[i, c, max_i, max_j] += output_grad[i, c, h, w]
    return grad_input
