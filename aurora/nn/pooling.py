from aurora.autodiff.autodiff import Op


class AveragePoolOp(Op):
    def __call__(self, input, filter=(2, 2), strides=(2, 2)):
        new_node = Op.__call__(self)
        new_node.inputs = [input]
        new_node.filter = filter
        new_node.strides = strides
        new_node.name = 'AveragePoolOp({})'.format(input.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1

        filter_height = node.filter[0]
        filter_width = node.filter[1]
        stride_height = node.strides[0]
        stride_width = node.strides[1]

        if use_numpy:
            pass
        else:
            raise NotImplementedError('GPU version of AveragePoolOp not yet implemented')

    def gradient(self, node, output_grads):
        pass

    def infer_shape(self, node, input_shapes):
        pass
