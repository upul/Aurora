import numpy as np
from aurora.autodiff.autodiff import Op, zeros_like
from .utils import softmax_func
from .activations import softmax


class CrossEntropyOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = 'CrossEntropy({0:s}, {1:s})'.format(node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        pred = softmax_func(input_vals[0])
        actual = input_vals[1]
        return np.mean(-np.sum(actual * np.log(pred), axis=1), keepdims=True)

    def gradient(self, node, output_grads):
        grad_A = (softmax(node.inputs[0]) + -1 * node.inputs[1]) * output_grads
        grad_B = zeros_like(node.inputs[1])
        return [grad_A, grad_B]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        return (1, )


# TODO (upul) MSE
# TODO (upul) RMSE

# Global singleton operations
cross_entropy_with_logits = CrossEntropyOp()