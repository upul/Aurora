import numpy as np
from aurora.autodiff.autodiff import Op
from aurora.nn.utils import softmax_func


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


class SoftmaxOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = 'SoftmaxOp({0:s})'.format(node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return softmax_func(input_vals[0])

    def gradient(self, node, output_grads):
        raise NotImplementedError('Not yet implemented, Please use CrossEntropy operator')

# TODO (upul)

#
relu = ReluOp()
relu_grad = ReluGradOp()
sigmoid = SigmoidOp()
softmax = SoftmaxOp()