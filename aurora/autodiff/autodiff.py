import numpy as np
from .utils import softmax_func


class Node(object):
    """ Node object represents a node in the computational graph"""

    def __init__(self):
        """ New node will be created by Op objects __call__ method"""
        # list of inputs to this node
        self.inputs = []
        # operator
        self.op = None
        # constants
        self.const = None
        # name of the node mainly use for debugging
        self.name = ""

    def __add__(self, other):
        """ Adding two nodes and returns a new node"""
        if isinstance(other, Node):
            return add(self, other)
        else:
            return add_const(self, other)

    def __sub__(self, other):
        if isinstance(other, Node):
            return sub(self, other)
        else:
            return sub_const(self, other)

    def __rsub__(self, other):
        return ref_sub_const(self, other)

    def __mul__(self, other):
        if isinstance(other, Node):
            return mul(self, other)
        else:
            return mul_const(self, other)

    def __truediv__(self, other):
        if isinstance(other, Node):
            return div(self, other)
        else:
            return div_const(self, other)

    # Allow left-hand-side add and multiply.
    __radd__ = __add__
    __rmul__ = __mul__
    __rdiv__ = __truediv__


class Op(object):
    """ Op class represents operations perform on nodes"""

    def __call__(self):
        """
        Create a new node which represents operations perform on the graph

        Parameters
        ----------
        None

        Returns
        -------
        Node
            The new node object
        """
        new_node = Node()
        new_node.op = self
        return new_node

    def compute(self, node, input_vals):
        """
        Given the values of input nodes, compute the output value

        Parameters
        ----------
        :param node: Node that performs the computation
        :param input_vals: Values of input node

        Returns
        -------
        :return: The output value of the node
        """
        raise NotImplementedError

    def gradient(self, node, output_grads):
        """
        Given the value of output gradients this operation calculate the
        gradient contribution of each input node

        Parameters
        ----------
        :param node:
        :param output_grads:

        Returns
        -------
        :return: A list of gradient contribution to each input node respectively
        """
        raise NotImplementedError


class AddOp(Op):
    """

    """

    def __call__(self, nodeA, nodeB):
        """
        This Operator element-wise two nodes

        Parameters
        ----------
        :param nodeA: LHS operand
        :param nodeB: RHS operand

        Returns
        -------
        :return: A new Node which represents the element-wise plus operation
        """
        new_node = Op.__call__(self)
        new_node.inputs = [nodeA, nodeB]
        new_node.name = '({}+{})'.format(nodeA.name, nodeB.name)
        return new_node

    def compute(self, node, input_vals):
        """
        Given values of two input nodes, return result of element-wise addition.
        Parameters
        ----------
        :param node:
        :param input_vals: List of two input nodes

        Returens
        --------
        :return:  The result of the element-wise addition operation
        """
        assert len(input_vals) == 2
        return input_vals[0] + input_vals[1]

    def gradient(self, node, output_grads):
        """
        Given the values of output gradients, calculate the gradients of input nodes

        Parameters
        ----------
        :param node:
        :param output_grads: Gradient contribution of output nodes

        Returns
        -------
        :return: A list of gradient contribution of output nodes
        """
        return [output_grads, output_grads]


class AddByConstOp(Op):
    """
    Operator represents the element-wise addition of a node and a const
    """

    def __call__(self, node_A, const_val):
        """

        :param node:
        :param const_val:
        :return:
        """
        new_node = Op.__call__(self)
        new_node.const = const_val
        new_node.inputs = [node_A]
        new_node.name = '({0:s}+{1:f})'.format(node_A.name, const_val)
        return new_node

    def compute(self, node, input_vals):
        """

        :param node:
        :param input_vals:
        :return:
        """
        assert len(input_vals) == 1
        return node.const + input_vals[0]

    def gradient(self, node, output_grads):
        """

        :param node:
        :param output_grads:
        :return:
        """
        return [output_grads]


class SubOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = '({0:s}-{1:s})'.format(node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        return input_vals[0] - input_vals[1]

    def gradient(self, node, output_grads):
        return [output_grads, -1 * output_grads]


class SubByConstOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.const = const_val
        new_node.name = '({0:s}-{1:f})'.format(node_A.name, const_val)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return input_vals[0] - node.const

    def gradient(self, node, output_grads):
        return [output_grads]


class ReflectedSubByConstOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.const = const_val
        new_node.name = '({0:f}-{1:s})'.format(const_val, node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return node.const - input_vals[0]

    def gradient(self, node, output_grads):
        return [-1 * output_grads]


class OnesLikeOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = 'Oneslike({})'.format(node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        assert isinstance(input_vals[0], np.ndarray)
        return np.ones(input_vals[0].shape)

    def gradient(self, node, output_grads):
        return [zeros_like(node.inputs[0])]


class ZerosLikeOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = 'Zeroslike({})'.format(node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        assert isinstance(input_vals[0], np.ndarray)
        return np.zeros(input_vals[0].shape)

    def gradient(self, node, output_grads):
        return [zeros_like(node.inputs[0])]


class MulOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = '({0:s}*{1:s})'.format(node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        return input_vals[0] * input_vals[1]

    def gradient(self, node, output_grads):
        return [node.inputs[1] * output_grads, node.inputs[0] * output_grads]


class MulByConstOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.const = const_val
        new_node.name = '({0:s}*{1:f})'.format(node_A.name, const_val)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return node.const * input_vals[0]

    def gradient(self, node, output_grads):
        return [node.const * output_grads]


class DivOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = '({0:s}/{1:s})'.format(node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        return input_vals[0] / input_vals[1]

    def gradient(self, node, output_grads):
        return [output_grads / node.inputs[1], -1.0 * output_grads * node.inputs[0] / (node.inputs[1] * node.inputs[1])]


class DivByConstOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.const = const_val
        new_node.name = '({0:s}/{1:f})'.format(node_A.name, const_val)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return input_vals[0] / node.const

    def gradient(self, node, output_grads):
        return [output_grads / node.const]


class PlaceholderOp(Op):
    """Op to feed value to a nodes."""

    def __call__(self):
        """Creates a variable node."""
        new_node = Op.__call__(self)
        return new_node

    def compute(self, node, input_vals):
        """No compute function since node value is fed directly in Executor."""
        assert False, "placeholder values provided by feed_dict"

    def gradient(self, node, output_grad):
        """No gradient function since node has no inputs."""
        return None


class ReduceSumOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = 'ReduceSum({0:s})'.format(node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.sum(input_vals[0], axis=0)

    def gradient(self, node, output_grads):
        return [output_grads]


class BroadcastToOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = 'BroadcastTo({0:s}, {1:s}.shape)'.format(node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        return np.broadcast_to(input_vals[0], input_vals[1].shape)

    def gradient(self, node, output_grads):
        grad_A = reduce_sum(output_grads)
        grad_B = zeros_like(node.inputs[1])
        return [grad_A, grad_B]


class MatMulOp(Op):
    def __call__(self, node_A, node_B, trans_A=False, trans_B=False):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.trans_A = trans_A
        new_node.trans_B = trans_B
        new_node.name = 'MatMul({0:s}, {1:s}'.format(node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        if node.trans_A:
            input_vals[0] = input_vals[0].T
        if node.trans_B:
            input_vals[1] = input_vals[1].T
        return np.dot(input_vals[0], input_vals[1])

    def gradient(self, node, output_grads):
        grad_A = matmul(output_grads, node.inputs[1], trans_A=False, trans_B=True)
        grad_B = matmul(node.inputs[0], output_grads, trans_A=True, trans_B=False)
        return [grad_A, grad_B]


def Variable(name):
    """User defined variables in an expression.
        e.g. x = Variable(name = "x")
    """
    placeholder_node = placeholder()
    placeholder_node.name = name
    return placeholder_node


def Parameter(name, init):
    """
    example: w = Parameter(name='w', state=...)
    :param name:
    :param init:
    :return:
    """
    parameter_node = placeholder()
    parameter_node.name = name
    parameter_node.const = init
    return parameter_node


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


class CrossEntropyOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = 'CrossEntropy({0:s}, {1:s})'.format(node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        pred = softmax_func(input_vals[0])
        actual = input_vals[1]
        return np.mean(-np.sum(actual * np.log(pred), axis=1), keepdims=True)

    def gradient(self, node, output_grads):
        grad_A = (softmax(node.inputs[0]) + -1 * node.inputs[1]) * output_grads
        grad_B = zeros_like(node.inputs[1])
        return [grad_A, grad_B]


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
        g = 1 - (tanh(x)*tanh(x))
        return [g * output_grads]


# Global singleton operations
add = AddOp()
add_const = AddByConstOp()
sub = SubOp()
sub_const = SubByConstOp()
ref_sub_const = ReflectedSubByConstOp()
mul = MulOp()
mul_const = MulByConstOp()
div = DivOp()
div_const = DivByConstOp()
zeros_like = ZerosLikeOp()
ones_like = OnesLikeOp()
reduce_sum = ReduceSumOp()
broadcast_to = BroadcastToOp()
matmul = MatMulOp()
relu = ReluOp()
relu_grad = ReluGradOp()
softmax = SoftmaxOp()
cross_entropy = CrossEntropyOp()
placeholder = PlaceholderOp()
sigmoid = SigmoidOp()
tanh = TanhOp()
