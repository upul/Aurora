#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/* TODO: Your code here */
/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
		const float *input_a, const float *input_b, float *output) {
	// Dynamic shared memory, size provided at kernel launch.
	extern __shared__ float loss_per_row[];
	// Two dimensional thread blocks.
	int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x
			+ threadIdx.x;
	if (y >= nrow) {
		return;
	}
	input_a += y * ncol;
	input_b += y * ncol;
	float maxval = *input_a;
	// Find max for a row.
	for (int x = 1; x < ncol; ++x) {
		maxval = max(maxval, input_a[x]);
	}
	// Deduct by max for a row, and raise to exp.
	float sum = 0;
	for (int x = 0; x < ncol; ++x) {
		sum += exp(input_a[x] - maxval);
	}
	// Compute per-row loss.
	float loss = 0;
	for (int x = 0; x < ncol; ++x) {
		loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
	}
	loss_per_row[y] = loss;
	__syncthreads();
	// Compute reduce_mean across rows.
	float mean_loss = 0;
	// Use a single thread to reduce mean across rows.
	if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
		for (int i = 0; i < nrow; ++i) {
			mean_loss += loss_per_row[i];
		}
		mean_loss /= nrow;
		output[0] = mean_loss;
	}
}


__global__ void array_set_kernel(float* array, float value, int n) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < n) {
		array[index] = value;
	}
}


int DLGpuArraySet(DLArrayHandle arr, float value) { /* TODO: Your code here */
	int n = 1;
	for (int i = 0; i < arr->ndim; i++) {
		n = n * arr->shape[i];
	}

	float* array_data = (float*) arr->data;

	int threads_per_block = 1024;
	int num_blocks = (n + threads_per_block - 1) / threads_per_block;

	array_set_kernel<<<num_blocks, threads_per_block>>>(array_data, value, n);
	return 0;
}


__global__ void broadcast_to_kernel(const float *input_data,
                                    float *output_data,
                                    index_t input_n,
                                    index_t output_n) {
    index_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < output_n) {
        output_data[idx] = input_data[idx % input_n];
    }
}


int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
	/* TODO: Your code here */
    index_t input_n = 1;
    for (int i = 0; i < input->ndim; i++)
        input_n *= input->shape[i];

    index_t output_n = 1;
    for (int i = 0; i < output->ndim; i++)
        output_n *= output->shape[i];

    const float *input_data = (const float *) input->data;
    float *output_data = (float *) output->data;

    int thread_per_block = 512;
    int n_blocks = (output_n + thread_per_block - 1) / thread_per_block;
    broadcast_to_kernel<<<n_blocks, thread_per_block>>>(input_data, output_data,
            input_n, output_n);
    return 0;
}

__global__ void reduced_sum_axis_zero(const float* input_data, float* output_data, int input_n, int output_n){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < output_n) {
        output_data[idx] = 0.0;
        for (int i = 0; i < input_n / output_n; i++) {
            output_data[idx] += input_data[i * output_n + idx];
        }
    }
}

int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
	/* TODO: Your code here */
    int input_n = 1;
    for(int i=0; i<input->ndim;i++){
        input_n *= input->shape[i];
    }

    int output_n = 1;
    for(int i=0; i<output->ndim; i++){
        output_n *= output->shape[i];
    }

    const float* input_data = (const float*) input->data;
    float* output_data = (float*) output->data;

    int thread_per_block = 1024;
    int n_blocks = (output_n + thread_per_block - 1) / thread_per_block;

    reduced_sum_axis_zero<<<n_blocks, thread_per_block>>>(input_data, output_data, input_n, output_n);
	return 0;
}

__global__ void matrix_elementwise_add(const float *a, const float *b, float *c,
		int n) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < n) {
		c[index] = a[index] + b[index];
	}
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
		const DLArrayHandle matB, DLArrayHandle output) {
	/* TODO: Your code here */
	int n = 1;
	for (int i = 0; i < output->ndim; i++) {
		n = n * output->shape[i];
	}
	const float* data_A = (const float*) matA->data;
	const float* data_B = (const float*) matB->data;
	float* data_output = (float*) output->data;

	int threads_per_block = 1024;
	int num_blocks = (n + threads_per_block - 1) / threads_per_block;

	matrix_elementwise_add<<<num_blocks, threads_per_block>>>(data_A, data_B,
			data_output, n);
	return 0;
}

__global__ void matrix_elementwise_add_by_const_kernal(const float *d_in,
		float *d_out, float val, int n) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < n) {
		d_out[index] = d_in[index] + val;
	}
}

int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
		DLArrayHandle output) {
	/* TODO: Your code here */
	int n = 1;
	for (int i = 0; i < output->ndim; i++) {
		n = n * output->shape[i];
	}
	const float* input_data = (const float*) input->data;
	float* output_data = (float*) output->data;
	int threads_per_block = 1024;
	int num_blocks = (n + threads_per_block - 1) / threads_per_block;
	matrix_elementwise_add_by_const_kernal<<<num_blocks, threads_per_block>>>(
			input_data, output_data, val, n);
	return 0;
}

__global__ void elementwise_mul_kernel(const float* data_a, const float* data_b,
		float* output, int n) {

	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < n){
		output[index] = data_a[index] * data_b[index];
	}
}

int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
		const DLArrayHandle matB, DLArrayHandle output) {
	/* TODO: Your code here */
	int n = 1;
	for (int i = 0; i < output->ndim; i++) {
		n = n * output->shape[i];
	}

	int threads_per_block = 1024;
	int num_blocks = (n + threads_per_block - 1) / threads_per_block;

	const float* mat_a_data = (const float*) matA->data;
	const float* mat_b_data = (const float*) matB->data;
	float* output_data = (float*) output->data;

	elementwise_mul_kernel<<<num_blocks, threads_per_block>>>(mat_a_data,
			mat_b_data, output_data, n);

	return 0;
}

__global__ void marix_multiply_by_const(const float* d_input, float* d_output,
		float val, int n) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < n) {
		d_output[index] = d_input[index] * val;
	}
}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
		DLArrayHandle output) {
	/* TODO: Your code here */
	int n = 1;
	for (int i = 0; i < input->ndim; i++) {
		n *= input->shape[i];
	}

	const float *input_data = (const float *) input->data;
	float *output_data = (float *) output->data;
	int threads_per_block = 1024;
	int num_blocks = (n + threads_per_block - 1) / threads_per_block;
	marix_multiply_by_const<<<num_blocks, threads_per_block>>>(input_data,
			output_data, val, n);
	return 0;
}

int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
		const DLArrayHandle matB, bool transposeB, DLArrayHandle matC) {
	/* TODO: Your code here */
	// Hint: use cublas
	// cublas assume matrix is column major
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS)
        printf("CUBLAS initialization failed\n");

    const float *matA_data = (const float *) matA->data;
    const float *matB_data = (const float *) matB->data;
    float *matC_data = (float *) matC->data;

    cublasOperation_t transa = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;

    int m = transposeB ? matB->shape[0] : matB->shape[1];
    int n = transposeA ? matA->shape[1] : matA->shape[0];
    int k = transposeA ? matA->shape[0] : matA->shape[1];

    float alpha = 1.0f;
    float beta = 0.0f;
    stat = cublasSgemm(handle, transb, transa,
                       m, n, k,
                       &alpha, matB_data, matB->shape[1],
                       matA_data, matA->shape[1],
                       &beta, matC_data, m);

    if (stat != CUBLAS_STATUS_SUCCESS)
        printf("CUBLAS kernel execution error.\n");

    stat = cublasDestroy(handle);
    if (stat != CUBLAS_STATUS_SUCCESS)
        printf("CUBLAS shutdown error\n");

    return 0;
}

__global__ void relu_kernel(const float *input, float *output, int n) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < n) {
		float element = input[index];
		if (element <= 0) {
			output[index] = 0;
		} else {
			output[index] = element;
		}
	}
}

int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
	/* TODO: Your code here */
	int n = 1;
	for (int i = 0; i < input->ndim; i++) {
		n *= input->shape[i];
	}

	const float *input_data = (const float *) input->data;
	float *output_data = (float *) output->data;
	int threads_per_block = 1024;
	int num_blocks = (n + threads_per_block - 1) / threads_per_block;
	relu_kernel<<<num_blocks, threads_per_block>>>(input_data, output_data, n);
	return 0;
}

__global__ void relu_gradient_kernel(const float* input, float* output,
		const float* in_grad, int n) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < n) {
		float element = input[index];
		if (element <= 0) {
			output[index] = 0;
		} else {
			output[index] = in_grad[index];
		}
	}
}

int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
		DLArrayHandle output) {
	/* TODO: Your code here */
	int n = 1;
	for (int i = 0; i < input->ndim; i++) {
		n *= input->shape[i];
	}

	const float *input_data = (const float *) input->data;
	float *output_data = (float *) output->data;
	const float* in_grad_data = (const float *) in_grad->data;
	int threads_per_block = 1024;
	int num_blocks = (n + threads_per_block - 1) / threads_per_block;

	relu_gradient_kernel<<<num_blocks, threads_per_block>>>(input_data,
			output_data, in_grad_data, n);
	return 0;
}

__global__ void softmax_kernel(int64_t nrow, int64_t ncol,
                               const float *input_data,
                               float *output_data) {

// two dimensional thread blocks.
    int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    if (y >= nrow){
        return;
    }
    // y_th row of input data
    input_data += y * ncol;
    output_data += y * ncol;
    // find max for a row.
    float maxval = *input_data;
    for (int x = 1; x < ncol; ++x){
        maxval = max(maxval, input_data[x]);
    }
    // Deduct by max for a row, and raise to exp.
    // in case of too large of exp, and the result will not be affected
    float sum = 0;
    for (int x = 0; x < ncol; ++x) {
        sum += exp(input_data[x] - maxval);
    }
    // Compute per-row softmax.
    for (int x = 0; x < ncol; ++x) {
        output_data[x] = exp(input_data[x] - maxval) / sum;
    }
}


int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
	/* TODO: Your code here */
    assert(input->ndim == 2);
    assert(output->ndim == 2);
    int64_t nrow = input->shape[0];
    int64_t ncol = input->shape[1];
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    dim3 threads;
    if (nrow < 1024){
        threads.x = nrow;
    }
    else{
        threads.x = 1024;
        threads.y = (nrow + 1023) / 1024;
    }
    softmax_kernel<<<1, threads>>>(nrow, ncol, input_data, output_data);
    return 0;
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
		const DLArrayHandle input_b, DLArrayHandle output) {
	assert(input_a->ndim == 2);
	assert(input_b->ndim == 2);
	assert(output->ndim == 1);
	assert(
			input_a->shape[0] == input_b->shape[0]
					&& input_a->shape[1] == input_b->shape[1]);
	int nrow = input_a->shape[0];
	// Maximum x- or y-dimension of a block = 1024
	// But we need 'nrow' shared memory, and max shared memory is 48KB.
	// Conservatively allow max 16KB shared memory.
	assert(nrow <= 1024 * 4);
	int ncol = input_a->shape[1];
	const float *input_data_a = (const float *) input_a->data;
	const float *input_data_b = (const float *) input_b->data;
	float *output_data = (float *) output->data;
	dim3 threads;
	if (nrow <= 1024) {
		threads.x = nrow;
	} else {
		threads.x = 1024;
		threads.y = (nrow + 1023) / 1024;
	}
	// 1 block, each block with 'threads' number of threads with 'nrow' shared
	// memory size
	matrix_softmax_cross_entropy_kernel<<<1, threads, nrow * sizeof(float)>>>(
			nrow, ncol, input_data_a, input_data_b, output_data);
	return 0;
}
