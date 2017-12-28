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
            checkCUDNN(cudnnCreateTensorDescriptor(&activationDesc));
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
                                        /*mode=*/CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/0));

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

