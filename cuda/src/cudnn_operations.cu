#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

#define checkCUDNN(expression)                                  \
{                                                               \
    cudnnStatus_t status = (expression);                        \
    if (status != CUDNN_STATUS_SUCCESS) {                       \
        std::cerr  << "Error on line " << __LINE__ << ": "      \
                   << cudnnGetErrorString(status) << std::endl; \
        std::exit(EXIT_FAILURE);                                \
    }                                                           \
}

int setTensorDescriptor(cudnnTensorDescriptor_t activationDesc,
                           const int numDim,
                           const long shape[]){
    int batchSize = 0;
    int channels = 0;
    switch(numDim){
        case 2:
            batchSize = shape[0];
            channels = shape[1];
            checkCUDNN(cudnnSetTensor4dDescriptor(activationDesc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          batchSize,
                                          channels, 1, 1));
            break;

        case 4:
            batchSize = shape[0];
            channels = shape[1];
            int height = shape[2];
            int width = shape[3];
            checkCUDNN(cudnnSetTensor4dDescriptor(activationDesc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          batchSize,
                                          channels,
                                          height,
                                          width));
            break;
        // TODO: handle other cases and errors

    }
    return 0;
}

int cudnnReLUForward(const DLArrayHandle input, DLArrayHandle output) {
	const float *input_data = (const float *) input->data;
	float *output_data = (float *) output->data;

	assert(input->shape[0] == output->shape[0]);
	assert(input->shape[1] == output->shape[1]);

    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    cudnnActivationDescriptor_t activation_descriptor;
    checkCUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
    checkCUDNN(cudnnSetActivationDescriptor(activation_descriptor,
                                           CUDNN_ACTIVATION_RELU, // type of activation
                                           CUDNN_PROPAGATE_NAN, // reluNanOpt
                                           0));  //relu_coef

    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    setTensorDescriptor(input_descriptor, input->ndim, input->shape);

    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    setTensorDescriptor(output_descriptor, output->ndim, output->shape);

    const float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnActivationForward(cudnn,
                                      activation_descriptor,
                                      &alpha,
                                      input_descriptor,
                                      input_data,
                                      &beta,
                                      output_descriptor,
                                      output_data));

    cudnnDestroyActivationDescriptor(activation_descriptor);
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);

	return 0;
}

int cudnnConv2DForward(const DLArrayHandle input,
                       const DLArrayHandle filter,
                       const DLArrayHandle bias,
                       const int stride_height,
                       const int stride_width,
                       const int padding_height,
                       const int padding_width,
                       DLArrayHandle output){

    const int input_dim = input->ndim;
    const int output_dim = output->ndim;
    assert(input_dim == 4);
    assert(output_dim == 4);

    const int filter_shape = filter->ndim;
    assert(filter_shape == 4);
    const int num_filters = filter->shape[0];
    const int num_outputs = filter->shape[1];
    const int filter_height = filter->shape[2];
    const int filter_width = filter->shape[3];

    const float *input_data = (const float *) input->data;
    const float *filter_date = (const float *) filter->data;
	float *output_data = (float *) output->data;

    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);


    // creating input and output tensors
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    setTensorDescriptor(input_descriptor, input->ndim, input->shape);

    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    setTensorDescriptor(output_descriptor, output->ndim, output->shape);

    // create filter tensors
    cudnnFilterDescriptor_t filter_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&filter_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(filter_descriptor,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*out_channels=*/num_outputs,
                                        /*in_channels=*/num_filters,
                                        /*kernel_height=*/filter_height,
                                        /*kernel_width=*/filter_width));
    // create convolution tensor
    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                             /*pad_height=*/padding_height,
                                             /*pad_width=*/padding_width,
                                             /*vertical_stride=*/stride_height,
                                             /*horizontal_stride=*/stride_width,
                                             /*dilation_height=*/1,
                                             /*dilation_width=*/1,
                                             /*mode=*/CUDNN_CROSS_CORRELATION,
                                             /*computeType=*/CUDNN_DATA_FLOAT));

    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn,
                                          input_descriptor,
                                          filter_descriptor,
                                          convolution_descriptor,
                                          output_descriptor,
                                          CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                          /*memoryLimitInBytes=*/0,
                                          &convolution_algorithm));

    size_t workspace_bytes{0};
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                     input_descriptor,
                                                     filter_descriptor,
                                                     convolution_descriptor,
                                                     output_descriptor,
                                                     convolution_algorithm,
                                                     &workspace_bytes));
    //std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB" << std::endl;
    assert(workspace_bytes > 0);

    void* d_workspace{nullptr};
    cudaMalloc(&d_workspace, workspace_bytes);

    const float alpha = 1.0f, beta = 0.0f;

    checkCUDNN(cudnnConvolutionForward(cudnn,
                                     &alpha,
                                     input_descriptor,
                                     input_data,
                                     filter_descriptor,
                                     filter_date,
                                     convolution_descriptor,
                                     convolution_algorithm,
                                     d_workspace,
                                     workspace_bytes,
                                     &beta,
                                     output_descriptor,
                                     output_data));

    cudaFree(d_workspace);

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);

    cudnnDestroy(cudnn);

    return 0;
}


int cudnnMaxPoolingForward(const DLArrayHandle input,
                        const int pooling_height,
                        const int pooling_width,
                        const int stride_height,
                        const int stride_width,
                        const int mode,
                        DLArrayHandle output){

    const int input_dim = input->ndim;
    const int output_dim = output->ndim;
    assert(input_dim == 4);
    assert(output_dim == 4);

    const float *input_data = (const float *) input->data;
    float *output_data = (float *) output->data;

    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    // creating input and output tensors
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    setTensorDescriptor(input_descriptor, input->ndim, input->shape);

    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    setTensorDescriptor(output_descriptor, output->ndim, output->shape);

    cudnnPoolingDescriptor_t pooling_descriptor;
    checkCUDNN(cudnnCreatePoolingDescriptor(&pooling_descriptor));
    checkCUDNN(cudnnSetPooling2dDescriptor(pooling_descriptor,
                                CUDNN_POOLING_MAX,
                                CUDNN_PROPAGATE_NAN,
                                pooling_height,
                                pooling_width,
                                0, // TODO: parameterize vertical padding
                                0, // TODO: parameterize horizontal padding
                                stride_height,
                                stride_width));


    const float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnPoolingForward(cudnn,
                                   pooling_descriptor,
                                   &alpha,
                                   input_descriptor,
                                   input_data,
                                   &beta,
                                   output_descriptor,
                                   output_data));

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyPoolingDescriptor(pooling_descriptor);

    cudnnDestroy(cudnn);
    return 0;
}

