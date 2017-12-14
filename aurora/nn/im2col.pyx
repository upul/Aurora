cimport cython

import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _im2col_helper(np.float64_t[:, :, :, :] x_padded, np.float64_t[:, :] out, int h_new, int w_new, int C, int M,
                 int filter_height, int filter_width, int stride_height, int stride_width):
    cdef int itr = 0
    
    cdef int start_i
    cdef int end_i
    cdef int start_j
    cdef int end_j
    
    cdef int i
    cdef int j
    cdef int m 
    
    cdef int k
    cdef int c
    cdef int p_h
    cdef int p_w
                
    for i in range(h_new):
        for j in range(w_new):
            for m in range(M):
                start_i = stride_height * i
                end_i = stride_height * i + filter_width
                start_j = stride_width * j
                end_j = stride_width * j + filter_height              
                
                k = 0
                for c in range(C):
                    for p_h in range(start_i, end_i):
                        for p_w in range(start_j, end_j):
                            out[k, itr] = x_padded[m, c, p_h, p_w]
                            k += 1
                itr += 1
                

@cython.boundscheck(False)
@cython.wraparound(False)
def im2col(np.float64_t[:, :, :, :] image,
                  int filter_height=3,
                  int filter_width=3, 
                  int padding_height=0, 
                  int padding_width=0, 
                  int stride_height=1,
                  int stride_width=1):
    
    cdef int M = image.shape[0]
    cdef int C = image.shape[1]
    cdef int h = image.shape[2]
    cdef int w = image.shape[3]
    
    cdef np.float64_t[:, :, :, :]  x_padded = np.pad(image, ((0, 0),
                              (0, 0),
                              (padding_height, padding_height),
                              (padding_width, padding_width)),
                      mode='constant')
    
    cdef int h_new = int((h - filter_height + 2 * padding_height) / stride_height + 1)
    cdef int w_new = int((w - filter_width + 2 * padding_width) / stride_width + 1)

    cdef int col_height = filter_width * filter_height * C
    cdef int col_width = M * h_new * w_new
    
    cdef np.float64_t[:, :] out = np.zeros((col_height, col_width))

    _im2col_helper(x_padded, out, h_new, w_new, C, M, filter_height, filter_width, stride_height, stride_width)
    
    return out