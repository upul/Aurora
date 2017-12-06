from aurora.autodiff.autodiff import Op
from .utils import im2col


class Conv2dOp(Op):
    def __call__(self, node_A, node_B, strides=(1, 1), padding=(0, 0)):
        new_node = Op.__call__(self)
        # node_A: 4-D data, (batch_size, depth, height, width)
        # node_B: 4-D kernel (num_filters, depth, kernel_height, kernel_width)
        new_node.inputs = [node_A, node_B]
        new_node.strides = strides
        new_node.padding = padding
        new_node.name = 'Conv2d({0:s}, {1:s})'.format(node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        padding_height = node.padding[0]
        padding_width = node.padding[1]
        stride_height = node.strides[0]
        stride_width = node.strides[1]

        X = input_vals[0]
        h = X.shape[2]
        w = X.shape[3]
        batch_size = X.shape[0]

        W = input_vals[1]
        filter_height = W.shape[2]
        filter_width = W.shape[3]
        n_filters = W.shape[0]

        padding_height = node.padding[0]
        padding_width = node.padding[1]
        stride_height = node.strides[0]
        stride_width = node.strides[1]

        if use_numpy:
            h_new = int((h - filter_height + 2 * padding_height) / stride_height + 1)
            w_new = int((w - filter_width + 2 * padding_width) / stride_width + 1)
            X_col = im2col(X, filter_size=(filter_height, filter_width),
                           padding=node.padding, stride=node.strides)
            W_col = W.reshape(n_filters, -1)
            out = W_col @ X_col
            out = out.reshape(n_filters, h_new, w_new, batch_size)
            output_val[:] = out.transpose(3, 0, 1, 2)
        else:
            raise NotImplementedError('GPU version of Conv2dOp not yet implemented')

    def gradient(self, node, output_grads):
        return

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2

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
        return d_new, h_new, w_new

# Global singleton operators
conv2d = Conv2dOp()