import numpy as np
from aurora.autodiff.autodiff import Op
from aurora.nn.utils import softmax_func
from config import sys_configs

if sys_configs['use_gpu']:
    from aurora.ndarray import ndarray, gpu_op


# class ReluOp(Op):
#     def __call__(self, node_A):
#         new_node = Op.__call__(self)
#         new_node.inputs = [node_A]
#         new_node.name = 'Relu({0:s})'.format(node_A.name)
#         return new_node
#
#     def compute(self, node, input_vals, output_val, use_numpy=True):
#         assert len(input_vals) == 1
#         if use_numpy:
#             output_val[:] = np.maximum(input_vals[0], 0)
#         else:
#             gpu_op.relu_gradient(input_vals[0], input_vals[1], output_val)
#
#     def gradient(self, node, output_grads):
#         return [relu_grad(node.inputs[0], output_grads)]
#
#     def infer_shape(self, node, input_shapes):
#         assert len(input_shapes) == 1
#         return input_shapes[0]
#
#
# class ReluGradOp(Op):
#     def __call__(self, node_A, node_B):
#         new_node = Op.__call__(self)
#         new_node.inputs = [node_A, node_B]
#         new_node.name = 'ReluGrad({0:s})'.format(node_A.name)
#         return new_node
#
#     def compute(self, node, input_vals, output_val, use_numpy=True):
#         assert len(input_vals) == 2
#         if use_numpy:
#             output_val[:] = (np.sign(input_vals[0]) + 1) * 0.5 * input_vals[1]
#         else:
#             gpu_op.relu_gradient(input_vals[0], input_vals[1], output_val)
#
#     def gradient(self, node, output_grads):
#         raise NotImplementedError
#
#     def infer_shape(self, node, input_shapes):
#         assert len(input_shapes) == 2
#         assert input_shapes[0] == input_shapes[1]
#         return input_shapes[0]

class ReluOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Relu(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            output_val[:] = np.maximum(input_vals[0], 0)
        else:
            gpu_op.relu(input_vals[0], output_val)

    def gradient(self, node, output_grad):
        return [relu_grad(node.inputs[0], output_grad)]

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""
        assert len(input_shapes) == 1
        return input_shapes[0]


class ReluGradientOp(Op):
    def __call__(self, node_A, node_B):
        """node_B is output_grad"""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "ReluGradient(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        if use_numpy:
            output_val[:] = np.sign(np.maximum(input_vals[0], 0)) * input_vals[1]
        else:
            gpu_op.relu_gradient(input_vals[0], input_vals[1], output_val)

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""
        assert len(input_shapes) == 2
        assert input_shapes[0] == input_shapes[1]
        return input_shapes[0]


class SigmoidOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = 'Sigmoid({0:s})'.format(node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            output_val[:] = 1 / (1 + np.exp(-1 * input_vals[0]))
        else:
            raise NotImplementedError('GPU version not yet implemented')

    def gradient(self, node, output_grads):
        x = node.inputs[0]
        # g = sigmoid(x) * (1 - sigmoid(x))
        # TODO: (upul) obove g failed in unit testing, need to check it.
        g = sigmoid(x) - sigmoid(x) * sigmoid(x)
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
        if use_numpy:
            output_val[:] = softmax_func(input_vals[0])
        else:
            gpu_op.softmax(input_vals[0], output_val)

    def gradient(self, node, output_grads):
        raise NotImplementedError('Not yet implemented, Please use CrossEntropy operator')

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


# TODO (upul)

# Global singleton operators
relu = ReluOp()
relu_grad = ReluGradientOp()
sigmoid = SigmoidOp()
softmax = SoftmaxOp()
