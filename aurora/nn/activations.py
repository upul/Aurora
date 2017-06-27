import numpy as np
from aurora.autodiff.autodiff import Op


class ReluOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = 'Relu({0:s})'.format(node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.maximum(input_vals[0], 0)

    def gradient(self, node, output_grads):
        return [relu_grad(node.inputs[0]) * output_grads]


class ReluGradOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = 'ReluGrad({0:s})'.format(node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.sign(np.maximum(input_vals[0], 0))

    def gradient(self, node, output_grads):
        raise NotImplementedError

#
relu = ReluOp()
relu_grad = ReluGradOp()