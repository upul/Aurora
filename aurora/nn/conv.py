from aurora.autodiff.autodiff import Op


class Conv2dOp(Op):
    def __call__(self, node_A, node_B, strides=(1, 1), padding=(0, 0)):
        new_node = Op.__call__(self)
        # node_A: 4-D data, (batch_size, depth, height, width)
        # node_B: 4-D kernel (num_filters, depth, kernel_height, kernel_width)
        new_node.inputs = [node_A, node_B]
        new_node.strides = strides
        new_node.padding = padding
        new_node.name = 'Conv2d({0:s}, {1:s})'.format(node_A.name)
        return new_node

    def compute(self, node, input_vals):
        pass

    def gradient(self, node, output_grads):
        pass
