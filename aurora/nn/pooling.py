from aurora.autodiff.autodiff import Op
from aurora.nn.pyx.fast_pooling import max_pool_forward, max_pool_backward


class MaxPoolOp(Op):
    def __call__(self, input, filter=(2, 2), strides=(2, 2)):
        new_node = Op.__call__(self)
        new_node.inputs = [input]
        new_node.filter = filter
        new_node.strides = strides
        new_node.name = 'MaxPoolOp({})'.format(input.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1

        filter_height = node.filter[0]
        filter_width = node.filter[1]
        stride_height = node.strides[0]
        stride_width = node.strides[1]

        if use_numpy:
            output_val[:] = max_pool_forward(input_vals[0],
                                             filter_height=filter_height,
                                             filter_width=filter_width,
                                             stride_height=stride_height,
                                             stride_width=stride_width)
        else:
            raise NotImplementedError('GPU version of MaxPoolOp not yet implemented')

    def gradient(self, node, output_grads):
        return [maxPoolBack(node.inputs[0], output_grads)]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1

        filter_height = node.filter[0]
        filter_width = node.filter[1]
        stride_height = node.strides[0]
        stride_width = node.strides[1]

        input_batch_size = input_shapes[0][0]
        input_n_channels = input_shapes[0][1]
        input_height = input_shapes[0][2]
        input_width = input_shapes[0][3]

        new_height = int((input_height - filter_height) / stride_height) + 1
        new_width = int((input_width - filter_width) / stride_width) + 1
        return input_batch_size, input_n_channels, new_height, new_width


class MaxGradientOp(Op):
    def __call__(self, node_A, node_B, filter=(2, 2), strides=(2, 2)):
        new_node = Op.__call__(self)
        # node_B is the output_grad
        new_node.inputs = [node_A, node_B]
        new_node.filter = filter
        new_node.strides = strides
        new_node.name = 'AverageGradientOp(%s)' % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2

        filter_height = node.filter[0]
        filter_width = node.filter[1]
        stride_height = node.strides[0]
        stride_width = node.strides[1]

        data = input_vals[0]
        output_grad = input_vals[1]
        if use_numpy:
            output_val[:] = max_pool_backward(output_grad,
                                              data,
                                              filter_height=filter_height,
                                              filter_width=filter_width,
                                              stride_height=stride_height,
                                              stride_width=stride_width
                                              )
        else:
            raise NotImplementedError('GPU version of AverageGradientOp not yet implemented')

    def gradient(self, node, output_grads):
        raise NotImplementedError('Gradient of AverageGradientOp is not implemented')

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        return input_shapes[0]


# Global singleton operators
maxPool = MaxPoolOp()
maxPoolBack = MaxGradientOp()
