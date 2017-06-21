import numpy as np
from .helper import find_topo_sort
from .helper import sum_node_list
from .helper import softmax_func


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
            if isinstance(self.op, ParameterOp):
                self.const += other
                return self
            else:
                return add_const(self, other)

    def __sub__(self, other):
        if isinstance(other, Node):
            return sub(self, other)
        else:
            return sub_const(self, other)

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
    __rsub__ = __sub__
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


class ParameterOp(Op):
    def __call__(self):
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


def Parameter(name, state):
    """
    example: w = Parameter(name='w', state=...)
    :param name:
    :param state:
    :return:
    """
    parameter_node = parameter()
    parameter_node.name = name
    parameter_node.const = state
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


# Global singleton operations
add = AddOp()
add_const = AddByConstOp()
sub = SubOp()
sub_const = SubByConstOp()
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
parameter = ParameterOp()

def gradients(output_node, node_list):
    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list = {}
    # Special note on initializing gradient of output_node as oneslike_op(output_node):
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_node] = [ones_like(output_node)]
    # a map from node to the gradient of that node
    node_to_output_grad = {}
    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
    reverse_topo_order = reversed(find_topo_sort([output_node]))
    """TODO: Your code here"""
    for node in reverse_topo_order:
        output_grad = sum_node_list(node_to_output_grads_list[node])
        node_to_output_grad[node] = output_grad

        input_grads_list = node.op.gradient(node, output_grad)
        for i in range(len(node.inputs)):
            if node.inputs[i] not in node_to_output_grads_list:
                node_to_output_grads_list[node.inputs[i]] = []
            node_to_output_grads_list[node.inputs[i]].append(input_grads_list[i])

    # Collect results for gradients requested.
    grad_node_list = [node_to_output_grad[node] for node in node_list]
    return grad_node_list


class Executor:
    """

    """

    def __init__(self, eval_list):
        """
        Executor computes values for a given subset of nodes in a computation graph.

        Parameters:
        -----------
        :param eval_list: Values of the nodes of this list need to be computed
        """
        self.eval_list = eval_list

    def run(self, feed_dict):
        """
        Values of the nodes given in eval_list are evaluated against feed_dict

        Parameters
        ----------
        :param feed_dict: A dictionary of nodes who values are specified by the user

        Returns
        -------
        :return: Values of the nodes specified by the eval_list
        """
        node_to_eval_map = dict(feed_dict)
        topo_order = find_topo_sort(self.eval_list)
        for node in topo_order:
            if node in feed_dict:
                continue

            if isinstance(node.op, ParameterOp):
                node_to_eval_map[node] = node.const
                continue

            inputs = [node_to_eval_map[n] for n in node.inputs]
            value = node.op.compute(node, inputs)
            node_to_eval_map[node] = value

        # select values of nodes given in feed_dicts
        return [node_to_eval_map[node] for node in self.eval_list]
