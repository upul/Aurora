/*!
 *  Copyright (c) 2017 by Contributors
 * \file c_runtime_api.h
 * \brief DL runtime library.
 *
 */

#ifndef DLSYS_RUNTIME_C_RUNTIME_API_H_
#define DLSYS_RUNTIME_C_RUNTIME_API_H_

#ifdef __cplusplus
#define DLSYS_EXTERN_C extern "C"
#else
#define DLSYS_EXTERN_C
#endif

#include "dlarray.h"
#include <stddef.h>
#include <stdint.h>

DLSYS_EXTERN_C {
/*! \brief type of array index. */
typedef int64_t index_t;

/*! \brief the array handle */
typedef DLArray *DLArrayHandle;
/*!
 * \brief The stream that is specific to device
 * can be NULL, which indicates the default one.
 */
typedef void *DLStreamHandle;

// Array related apis for quick proptying
/*!
 * \brief Allocate a nd-array's memory,
 *  including space of shape, of given spec.
 *
 * \param shape The shape of the array, the data content will be copied to out
 * \param ndim The number of dimension of the array.
 * \param ctx The ctx this array sits on.
 * \param out The output handle.
 * \return 0 when success, -1 when failure happens
 */
int DLArrayAlloc(const index_t *shape, index_t ndim, DLContext ctx,
                 DLArrayHandle *out);

/*!
 * \brief Free the DL Array.
 * \param handle The array handle to be freed.
 * \return 0 when success, -1 when failure happens
 */
int DLArrayFree(DLArrayHandle handle);

/*!
 * \brief Copy the array, both from and to must be valid during the copy.
 * \param from The array to be copied from.
 * \param to The target space.
 * \param stream The stream where the copy happens, can be NULL.
 * \return 0 when success, -1 when failure happens
 */
int DLArrayCopyFromTo(DLArrayHandle from, DLArrayHandle to,
                      DLStreamHandle stream);

/*!
 * \brief Set all array elements to given value.
 * \param arr The array to be Set.
 * \param value The target value.
 * \return 0 when success, -1 when failure happens
 */
int DLGpuArraySet(DLArrayHandle arr, float value);


int DLArrayReshape(const DLArrayHandle handle, const index_t *new_shape, index_t new_dim);

/*!
 * \brief Broadcast input array to output array.
 * \param input The input array.
 * \param output The output array.
 * \return 0 when success, -1 when failure happens
 */
int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output);

/*!
 * \brief Reduce sum input array by axis=0 and store to output.
 * \param input The input array.
 * \param output The output array.
 * \return 0 when success, -1 when failure happens
 */
int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output);

/*!
 * \brief Elementwise add two matrices and store to output.
 * \param matA The left input array.
 * \param matB The right input array.
 * \param output The output array.
 * \return 0 when success, -1 when failure happens
 */
int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output);

/*!
 * \brief Add matrix by const and store to output.
 * \param input The input array.
 * \param val The constant.
 * \param output The output array.
 * \return 0 when success, -1 when failure happens
 */
int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output);


int DLGpuMatrixElementwiseSubtract(const DLArrayHandle matA,
                                   const DLArrayHandle matB, DLArrayHandle output);

int DLGpuMatrixElementwiseSubtractByConst(const DLArrayHandle input, float val,
                                          DLArrayHandle output);

/*!
 * \brief Elementwise multiply two matrices and store to output.
 * \param matA The left input array.
 * \param matB The right input array.
 * \param output The output array.
 * \return 0 when success, -1 when failure happens
 */
int DLGpuMatrixElementwiseMultiply(
        const DLArrayHandle matA, const DLArrayHandle matB, DLArrayHandle output);

/*!
 * \brief Multiply matrix by const and store to output.
 * \param input The input array.
 * \param val The constant.
 * \param output The output array.
 * \return 0 when success, -1 when failure happens
 */
int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output);


// TODO: (upul) documentation
int DLGpuMatrixElementwiseDiv(const DLArrayHandle matA,
                              const DLArrayHandle matB,
                              DLArrayHandle output);

// TODO: (upul) documentation
int DLGpuMatrixElementwiseDivByConst(const DLArrayHandle matA, float val,
                                     DLArrayHandle output);

/*!
 * \brief Matrix multiply two matrices and store to output.
 * \param matA The left input array.
 * \param transposeA Whether matA needs to be transposed
 * \param matB The right input array.
 * \param transposeB Whether matB needs to be transposed
 * \param output The output array.
 * \return 0 when success, -1 when failure happens
 */
int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC);

/*!
 * \brief Compute relu on all array elements, and store to output.
 * \param input The input array.
 * \param output The output value.
 * \return 0 when success, -1 when failure happens
 */
int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output);

/*!
 * \brief Compute relu gradient, and store to output.
 * \param input The input array.
 * \param in_grad The input gradients value.
 * \param output The output array.
 * \return 0 when success, -1 when failure happens
 */
int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output);

/*!
 * \brief Compute softmax on matrix, and store to output.
 * \param input The input array.
 * \param output The output value.
 * \return 0 when success, -1 when failure happens
 */
int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output);

/*!
 * \brief Compute softmax_cross_entropy.
 *  np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
 * \param input_a The y array.
 * \param input_b The y_ array.
 * \param output The output value.
 * \return 0 when success, -1 when failure happens
 */
int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                             const DLArrayHandle input_b,
                             DLArrayHandle output);

int DLGpuMatrixElementwiseSqrt(const DLArrayHandle input_a, DLArrayHandle output);

/*
* CUDNN....
*/
int cudnnReLUForward(const DLArrayHandle input, DLArrayHandle output);

int cudnnConv2DForward(const DLArrayHandle input,
                       const DLArrayHandle filter,
                       const DLArrayHandle bias,
                       const int stride_height,
                       const int stride_width,
                       const int padding_height,
                       const int padding_width,
                       DLArrayHandle output);

int cudnnPoolForward(const DLArrayHandle input,
                     const int pooling_height,
                     const int pooling_width,
                     const int stride_height,
                     const int stride_width,
                     const char *mode,
                     DLArrayHandle output);

int cudnnPoolBackward(const DLArrayHandle input,
                      const DLArrayHandle output_grads,
                      const DLArrayHandle output,
                      const int pooling_height,
                      const int pooling_width,
                      const int stride_height,
                      const int stride_width,
                      const char *mode,
                      DLArrayHandle pool_grad);

int cudnnConv2DBackwardFilter(const DLArrayHandle input,
                              const DLArrayHandle output_grads,
                              const int stride_height,
                              const int stride_width,
                              const int padding_height,
                              const int padding_width,
                              DLArrayHandle filter_grad);

int cudnnConv2DBackwardData(const DLArrayHandle filter,
                            const DLArrayHandle output_grads,
                            const int stride_height,
                            const int stride_width,
                            const int padding_height,
                            const int padding_width,
                            DLArrayHandle data_grad);

int cudnnConv2DBackwardBias(const DLArrayHandle output_grads,
                            DLArrayHandle bias_grads);

} // DLSYS_EXTERN_C

#endif // DLSYS_RUNTIME_C_RUNTIME_API_H_
