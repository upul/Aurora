import numpy as np

from aurora.autodiff.autodiff import Op


class TanhOp(Op):
    """
    Tanh Activation function

    """

    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = 'Tanh({0:s})'.format(node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            output_val[:] = np.tanh(input_vals[0])
        else:
            raise NotImplementedError('GPU version of TanhOp not yet implemented')

    def gradient(self, node, output_grads):
        x = node.inputs[0]
        g = 1 - (tanh(x) * tanh(x))
        return [g * output_grads]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes)
        return input_shapes[0]

# Global singleton operations
tanh = TanhOp()

