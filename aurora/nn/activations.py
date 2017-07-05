import numpy as np
from aurora.autodiff.autodiff import Op
from aurora.nn.utils import softmax_func


class ReluOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = 'Relu({0:s})'.format(node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        return np.maximum(input_vals[0], 0)

    def gradient(self, node, output_grads):
        return [relu_grad(node.inputs[0]) * output_grads]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


class ReluGradOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = 'ReluGrad({0:s})'.format(node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        return np.sign(np.maximum(input_vals[0], 0))

    def gradient(self, node, output_grads):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


class SigmoidOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = 'Sigmoid({0:s})'.format(node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        return 1 / (1 + np.exp(-1 * input_vals[0]))

    def gradient(self, node, output_grads):
        x = node.inputs[0]
        g = sigmoid(x) * (1 - sigmoid(x))
        return [g * output_grads]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes)
        return input_shapes[0]


class SoftmaxOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = 'SoftmaxOp({0:s})'.format(node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        return softmax_func(input_vals[0])

    def gradient(self, node, output_grads):
        raise NotImplementedError('Not yet implemented, Please use CrossEntropy operator')

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


# TODO (upul)

# Global singleton operators
relu = ReluOp()
relu_grad = ReluGradOp()
sigmoid = SigmoidOp()
softmax = SoftmaxOp()
