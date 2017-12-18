from aurora.autodiff.autodiff import Op
from aurora.nn.im2col import im2col, col2im


# TODO: (upul) The numpy version of the Conv2dOp, X_col is calculated twice.
#       One in compute() of Conv2dOp and the second time inside the compute() of
#       Conv2dBackwardFilter node. Check the feasibility of caching.

class Conv2dOp(Op):
    def __call__(self, input, filter, bias, strides=(1, 1), padding=(0, 0)):
        new_node = Op.__call__(self)
        # input: 4-D data, (batch_size, depth, height, width)
        # filter: 4-D kernel (num_filters, depth, kernel_height, kernel_width)
        new_node.inputs = [input, filter, bias]
        new_node.strides = strides
        new_node.padding = padding
        new_node.name = 'Conv2d({0:s}, {1:s})'.format(input.name, filter.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 3

        X = input_vals[0]
        h = X.shape[2]
        w = X.shape[3]
        batch_size = X.shape[0]

        W = input_vals[1]
        filter_height = W.shape[2]
        filter_width = W.shape[3]
        n_filters = W.shape[0]

        b = input_vals[2].reshape(n_filters, -1)

        padding_height = node.padding[0]
        padding_width = node.padding[1]
        stride_height = node.strides[0]
        stride_width = node.strides[1]

        if use_numpy:
            h_new = int((h - filter_height + 2 * padding_height) / stride_height + 1)
            w_new = int((w - filter_width + 2 * padding_width) / stride_width + 1)
            X_col = im2col(X, filter_height, filter_width, padding_height, padding_width,
                           stride_height, stride_width)
            W_col = W.reshape(n_filters, -1)
            out = W_col @ X_col + b
            out = out.reshape(n_filters, h_new, w_new, batch_size)
            output_val[:] = out.transpose(3, 0, 1, 2)
        else:
            raise NotImplementedError('GPU version of Conv2dOp not yet implemented')

    def gradient(self, node, output_grads):
        #
        filter_node = node.inputs[1]
        data_node = node.inputs[0]
        return [conv2dBackData(data_node, filter_node, output_grads),
                conv2dBackFilter(data_node, filter_node, output_grads),
                conv2dBackBias(output_grads)]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 3

        X_shape = input_shapes[0]
        h = X_shape[2]
        w = X_shape[3]

        W_shape = input_shapes[1]
        filter_height = W_shape[2]
        filter_width = W_shape[3]

        padding_height = node.padding[0]
        padding_width = node.padding[1]
        stride_height = node.strides[0]
        stride_width = node.strides[1]

        h_new = int((h - filter_height + 2 * padding_height) / stride_height + 1)
        w_new = int((w - filter_width + 2 * padding_width) / stride_width + 1)
        d_new = W_shape[0]
        batch_size = X_shape[0]
        return batch_size, d_new, h_new, w_new


class Conv2dBackwardFilter(Op):
    def __call__(self, node_A, node_B, output_grad, strides=(1, 1), padding=(0, 0)):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B, output_grad]
        new_node.strides = strides
        new_node.padding = padding
        new_node.name = "Conv2dBackwardFilter(%s, %s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 3

        X = input_vals[0]  # data
        W = input_vals[1]  # filter

        assert len(X.shape) == 4
        assert len(W.shape) == 4

        filter_height = W.shape[2]
        filter_width = W.shape[3]
        n_filters = W.shape[0]
        out_grad = input_vals[2]

        padding_height = node.padding[0]
        padding_width = node.padding[1]
        stride_height = node.strides[0]
        stride_width = node.strides[1]

        if use_numpy:
            X_col = im2col(X, filter_height, filter_width, padding_height, padding_width,
                           stride_height, stride_width)
            dout_reshaped = out_grad.transpose(1, 2, 3, 0).reshape(n_filters, -1)
            dW = dout_reshaped @ X_col.T
            output_val[:] = dW.reshape(W.shape)

        else:
            raise NotImplementedError('GPU version of Conv2dBackwardFilter not yet implemented')

    def gradient(self, node, output_grads):
        raise NotImplementedError('Gradient of ReluGradientOp not implemented')

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 3
        W_size = input_shapes[1]
        return W_size


class Conv2dBackwardData(Op):
    def __call__(self, node_A, node_B, output_grad, strides=(1, 1), padding=(0, 0)):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B, output_grad]
        new_node.strides = strides
        new_node.padding = padding
        new_node.name = "Conv2dBackwardData(%s, %s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 3
        X = input_vals[0]  # data
        W = input_vals[1]  # filter

        assert len(X.shape) == 4
        assert len(W.shape) == 4

        filter_height = W.shape[2]
        filter_width = W.shape[3]
        n_filters = W.shape[0]

        if use_numpy:
            W_reshape = W.reshape(n_filters, -1)
            dout_reshaped = input_vals[2].transpose(1, 2, 3, 0).reshape(n_filters, -1)

            dX_col = W_reshape.T @ dout_reshaped
            batch_size, n_channels, img_height, img_width = X.shape
            padding_height, padding_width = node.padding
            stride_height, stride_width = node.strides
            output_val[:] = col2im(dX_col, batch_size, n_channels,
                                   img_height, img_width, filter_height, filter_width,
                                   padding_height, padding_width,
                                   stride_height, stride_width)
        else:
            raise NotImplementedError('GPU version of Conv2dBackwardData not yet implemented')

    def gradient(self, node, output_grads):
        raise NotImplementedError('Gradient of ReluGradientOp not implemented')

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 3
        X_size = input_shapes[0]
        return X_size


class Conv2dBackwardBias(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Conv2dBackwardBiase(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1

        if use_numpy:
            # db = input_vals[0].sum(axis=(0, 2, 3))
            output_val[:] = input_vals[0].sum(axis=(0, 2, 3))
        else:
            raise NotImplementedError('GPU version of Conv2dBackwardBias not yet implemented')

    def gradient(self, node, output_grads):
        raise NotImplementedError('Gradient of ReluGradientOp not implemented')

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        # size of the input_shape[0] = (batch_size, num_filters, filter_height, filter_width)
        return (input_shapes[0][1],)


# Global singleton operators
conv2d = Conv2dOp()
conv2dBackFilter = Conv2dBackwardFilter()
conv2dBackData = Conv2dBackwardData()
conv2dBackBias = Conv2dBackwardBias()
