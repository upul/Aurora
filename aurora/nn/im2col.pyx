cimport cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cdef im2col_inner(np.float64_t[:, :, :, :] x_padded, np.float64_t[:, :] out, int h_new, int w_new, int C, int M,
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
cdef col2img_inner(np.float64_t[:, :] cols, np.float64_t[:, :, :, :] x_padded, int filter_height, int filter_width,
                   int N, int C, int H, int W, int H_padded, int W_padded, int padding_height, int padding_width,
                   int stride_height, int stride_width):
    cdef int idx = 0
    cdef int i, j, m, c, sh, sw
    cdef int start_height, start_width, k
    cdef np.float64_t[:] col

    cdef int p = H_padded - filter_height + 1
    cdef int q = W_padded - filter_width + 1
    i =0
    while i < p:
        j = 0
        while j < q:
            for m in range(N):
                col = cols[:, idx]
                start_height = i
                start_width = j
                k = 0
                for c in range(C):
                    for sh in range(start_height, start_height + filter_height):
                        for sw in range(start_width, start_width + filter_width):
                            x_padded[m, c, sh, sw] += col[k]
                            k += 1
                idx += 1
            j += stride_width
        i += stride_height
    if padding_height > 0 or padding_width >0:
        return x_padded[:, :, padding_height:-padding_height, padding_width:-padding_width]
    else:
        return x_padded

@cython.boundscheck(False)
@cython.wraparound(False)
def im2col(np.float64_t[:, :, :, :] image,
           int filter_height=3, int filter_width=3,
           int padding_height=0, int padding_width=0,
           int stride_height=1, int stride_width=1):

    cdef int imgs_per_batch = image.shape[0]
    cdef int n_channels = image.shape[1]
    cdef int img_h = image.shape[2]
    cdef int img_w = image.shape[3]

    cdef np.float64_t[:, :, :, :]  x_padded = np.pad(image, ((0, 0),
                              (0, 0),
                              (padding_height, padding_height),
                              (padding_width, padding_width)),
                      mode='constant')

    cdef int new_h = int((img_h - filter_height + 2 * padding_height) / stride_height + 1)
    cdef int new_w = int((img_w - filter_width + 2 * padding_width) / stride_width + 1)

    cdef int col_height = filter_width * filter_height * n_channels
    cdef int col_width = imgs_per_batch * new_h * new_w

    cdef np.float64_t[:, :] result = np.zeros((col_height, col_width))
    im2col_inner(x_padded, result, new_h, new_w, n_channels, imgs_per_batch,
                 filter_height, filter_width, stride_height, stride_width)

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def col2im(np.float64_t[:, :] col2img_converted,  int batch_size,
           int no_channels, int image_height, int image_width,
           int filter_height=3, int filter_width=3,
           int padding_height=0, int padding_width=0,
           int stride_height=1, int stride_width=1):

    cdef int padded_h = image_height + 2 * padding_height
    cdef int padded_w = image_width + 2 * padding_width
    cdef np.float64_t[:, :, :, :]  result = np.zeros((batch_size, no_channels, padded_h, padded_w))

    col2img_inner(col2img_converted, result, filter_height,
                  filter_width, batch_size, no_channels,
                  image_height, image_width, padded_h,
                  padded_w, padding_height, padding_width,
                  stride_height, stride_width)
    return result
