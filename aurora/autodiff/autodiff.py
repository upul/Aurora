import numpy as np
from aurora.ndarray import ndarray
from aurora.ndarray import gpu_op


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

    def compute(self, node, input_vals, use_gpu=False):
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

    def infer_shape(self, node, input_shapes):
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

    def compute(self, node, input_vals, output_val, use_numpy=True):
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

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        assert input_shapes[0] == input_shapes[1]
        return input_shapes[0]


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

    def compute(self, node, input_vals, output_val, use_numpy=True):
        """

        :param node:
        :param input_vals:
        :return:
        """
        assert len(input_vals) == 1
        if use_numpy:
            output_val[:] = node.const + input_vals[0]
        else:
            gpu_op.matrix_elementwise_add_by_const(
                input_vals[0], node.const, output_val)

    def gradient(self, node, output_grads):
        """

        :param node:
        :param output_grads:
        :return:
        """
        return [output_grads]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        assert node.const.shape == input_shapes[0]
        return input_shapes[0]


class SubOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = '({0:s}-{1:s})'.format(node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        return input_vals[0] - input_vals[1]

    def gradient(self, node, output_grads):
        return [output_grads, -1 * output_grads]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        assert input_shapes[0] == input_shapes[1]
        return input_shapes[0]


class SubByConstOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.const = const_val
        new_node.name = '({0:s}-{1:f})'.format(node_A.name, const_val)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        return input_vals[0] - node.const

    def gradient(self, node, output_grads):
        return [output_grads]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


class ReflectedSubByConstOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.const = const_val
        new_node.name = '({0:f}-{1:s})'.format(const_val, node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        return node.const - input_vals[0]

    def gradient(self, node, output_grads):
        return [-1 * output_grads]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


class OnesLikeOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = 'Oneslike({})'.format(node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        assert isinstance(input_vals[0], np.ndarray)
        return np.ones(input_vals[0].shape)

    def gradient(self, node, output_grads):
        return [zeros_like(node.inputs[0])]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        if input_shapes[0] == 1:  # TODO (upul) do we need this if ?
            return (1,)
        else:
            return input_shapes[0]


class ZerosLikeOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = 'Zeroslike({})'.format(node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        assert isinstance(input_vals[0], np.ndarray)
        return np.zeros(input_vals[0].shape)

    def gradient(self, node, output_grads):
        return [zeros_like(node.inputs[0])]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        if input_shapes[0] == 1:  # TODO (upul) do we need this if ?
            return (1,)
        else:
            return input_shapes[0]


class MulOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = '({0:s}*{1:s})'.format(node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        return input_vals[0] * input_vals[1]

    def gradient(self, node, output_grads):
        return [node.inputs[1] * output_grads, node.inputs[0] * output_grads]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        if input_shapes[0] == (1,):
            return input_shapes[1]
        elif input_shapes[1] == (1,):
            return input_shapes[0]
        elif input_shapes[0] == input_shapes[1]:
            return input_shapes[0]
        else:
            stmt = 'Invalid dimensions {0:s}, (1:s)'.format(input_shapes[0], input_shapes[1])
            raise RuntimeError(stmt)


class MulByConstOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.const = const_val
        new_node.name = '({0:s}*{1:f})'.format(node_A.name, const_val)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        return node.const * input_vals[0]

    def gradient(self, node, output_grads):
        return [node.const * output_grads]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


class DivOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = '({0:s}/{1:s})'.format(node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        return input_vals[0] / input_vals[1]

    def gradient(self, node, output_grads):
        grad_A = output_grads / node.inputs[1]
        grad_B = -1.0 * output_grads * node.inputs[0] / (node.inputs[1] * node.inputs[1])
        return [grad_A, grad_B]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        assert input_shapes[0] == input_shapes[1]
        return input_shapes[0]


class DivByConstOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.const = const_val
        new_node.name = '({0:s}/{1:f})'.format(node_A.name, const_val)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        return input_vals[0] / node.const

    def gradient(self, node, output_grads):
        return [output_grads / node.const]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


class PlaceholderOp(Op):
    """Op to feed value to a nodes."""

    def __call__(self):
        """Creates a variable node."""
        new_node = Op.__call__(self)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
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

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            assert isinstance(output_val, np.ndarray)
            output_val[:] = np.sum(input_vals[0], axis=0)
        else:
            gpu_op.reduce_sum_axis_zero(input_vals[0], output_val)

    def gradient(self, node, output_grads):
        return [output_grads]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes[0]) == 1
        if len(input_shapes[0]) == 1:
            return (1,)
        else:
            tuple(input_shapes[0][i] for i in range(1, len(input_shapes[0])))


class BroadcastToOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = 'BroadcastTo({0:s}, {1:s}.shape)'.format(node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        return np.broadcast_to(input_vals[0], input_vals[1].shape)

    def gradient(self, node, output_grads):
        grad_A = reduce_sum(output_grads)
        grad_B = zeros_like(node.inputs[1])
        return [grad_A, grad_B]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        return input_shapes[1]


class MatMulOp(Op):
    def __call__(self, node_A, node_B, trans_A=False, trans_B=False):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.trans_A = trans_A
        new_node.trans_B = trans_B
        new_node.name = 'MatMul({0:s}, {1:s}'.format(node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
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
placeholder = PlaceholderOp()
