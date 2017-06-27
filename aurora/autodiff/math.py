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

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.tanh(input_vals[0])

    def gradient(self, node, output_grads):
        x = node.inputs[0]
        g = 1 - (tanh(x) * tanh(x))
        return [g * output_grads]


class SigmoidOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = 'Sigmoid({0:s})'.format(node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return 1 / (1 + np.exp(-1 * input_vals[0]))

    def gradient(self, node, output_grads):
        x = node.inputs[0]
        g = sigmoid(x) * (1 - sigmoid(x))
        return [g * output_grads]


# Global singleton operations
tanh = TanhOp()
sigmoid = SigmoidOp()
